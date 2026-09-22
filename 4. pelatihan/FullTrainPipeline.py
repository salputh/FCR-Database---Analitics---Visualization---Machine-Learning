# -*- coding: utf-8 -*-
"""
Skrip pipeline lengkap untuk:
1. Menggabungkan data dari beberapa file CSV periode.
2. Melakukan rekayasa fitur (feature engineering) canggih pada data yang digabungkan.
3. Melakukan normalisasi fitur dengan RobustScaler.
4. Melatih model hibrida yang terdiri dari TCN dan XGBoost dengan SELURUH DATA sebagai training.
5. Menambahkan analisis korelasi fitur temporal terhadap fitur lain.

Alur Kerja:
- Langkah 1: Penggabungan data mentah.
- Langkah 2: Rekayasa fitur dari data gabungan.
- Langkah 3: Normalisasi fitur yang telah direkayasa.
- Langkah 4: Pelatihan model hibrida dan analisis korelasi fitur temporal.
"""

# Tambahkan import untuk analisis statistik
import pandas as pd
import numpy as np
import os
import warnings
import logging
import json
import pickle
from typing import Dict
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats

### --- IMPLEMENTASI BARU --- ###
# Impor untuk Model Hibrida
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tcn import TCN
    import xgboost as xgb
    from sklearn.model_selection import KFold
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score,
    )
except ImportError as e:
    print(f"Error impor: {e}")
    print(
        "Harap pasang library yang diperlukan: pip install tensorflow keras-tcn xgboost optuna scikit-learn seaborn"
    )
    exit()

# Impor kelas pipeline dari file terpisah.
try:
    import sys

    sys.path.append("1. Normalisasi")
    from Feature.FeatureEngineeringPipeline import FeatureEngineeringPipeline
except ImportError:
    print("ERROR: File 'FeatureEngineeringPipeline.py' tidak ditemukan.")
    exit()

# =============================================================================
# KONFIGURASI DAN SETUP AWAL
# =============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Mengabaikan peringatan (warnings) agar output lebih bersih
warnings.filterwarnings("ignore")

# Konfigurasi untuk rekayasa fitur
FEATURE_CONFIG = {
    "lag_periods": [1, 2, 3, 7, 14],
    "rolling_windows": [3, 7, 14],
    "health_weights": {"mati": 0.5, "deplesi": 0.5},
    "variance_threshold": 0.01,
    "correlation_threshold": 0.98,
    "decimal_places": 3,
}

MULTI_OUTPUT_CONFIG = {
    "output_strategy": "multi_head",
    "horizons": [1, 3],
    "output_dims": [95, 95],
    "horizon_weights": [0.5, 0.5],
}


# =============================================================================
# LANGKAH 1: FUNGSI PENGGABUNGAN DATA
# =============================================================================
def merge_data_files(folder_path: str, output_file: str):
    print("=" * 80)
    print("MEMULAI LANGKAH 1: PENGGABUNGAN DATA")
    print("=" * 80)

    dataframes = []
    for i in range(1, 15):
        file_name = f"NORMALISASI_PERIODE_{i}.csv"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            print(f"Membaca file: {file_name}")
            try:
                df = pd.read_csv(file_path, sep=";")
                if "PERIODE" not in df.columns:
                    df["PERIODE"] = i
                else:
                    df["PERIODE"] = df["PERIODE"].fillna(i)
                df["PERIODE"] = df["PERIODE"].astype(int)
                dataframes.append(df)
            except Exception as e:
                print(f"  - ERROR membaca {file_name}: {str(e)}\n")
        else:
            print(f"File tidak ditemukan: {file_name}")

    if not dataframes:
        logger.error("GAGAL: Tidak ada file yang berhasil dibaca. Pipeline dihentikan.")
        return False

    combined_df = pd.concat(dataframes, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, sep=";", index=False)
    print(
        f"\nBERHASIL: Penggabungan data selesai. File disimpan sebagai: {output_file}\n"
    )
    return True


# =============================================================================
# LANGKAH 2: FUNGSI REKAYASA FITUR
# =============================================================================
def run_feature_engineering_step(
    input_file: str,
    output_data_path: str,
    output_pipeline_path: str,
    metadata_path: str,
    config: Dict,
) -> bool:
    print(
        "\n"
        + "=" * 80
        + "\nMEMULAI LANGKAH 2: REKAYASA FITUR & PENYIMPANAN PIPELINE\n"
        + "=" * 80
    )
    if not os.path.exists(input_file):
        logger.critical(f"GAGAL: File input '{input_file}' tidak ditemukan.")
        return False

    df_raw = pd.read_csv(input_file, sep=";")
    logger.info(f"Data mentah dimuat untuk rekayasa fitur: {df_raw.shape}")

    feature_pipeline = FeatureEngineeringPipeline(config=config)

    logger.info("Menjalankan fit_transform pada FeatureEngineeringPipeline...")
    df_enhanced = feature_pipeline.fit_transform(
        df_raw, metadata_output_dir=metadata_path
    )

    df_enhanced.to_csv(output_data_path, sep=";", index=False)
    logger.info(f"Data dengan fitur enhanced berhasil disimpan di: {output_data_path}")

    with open(output_pipeline_path, "wb") as f:
        pickle.dump(feature_pipeline, f)
    logger.info(f"Model rekayasa fitur (pipeline) disimpan di: {output_pipeline_path}")

    return True


# =============================================================================
# LANGKAH 3: FUNGSI NORMALISASI FITUR
# =============================================================================
def normalize_features(
    input_file: str,
    output_file: str,
    info_file: str,
    robust_scaler_path: str,
) -> bool:
    print("\n" + "=" * 80)
    print("MEMULAI LANGKAH 3: NORMALISASI FITUR")
    print("=" * 80)

    if not os.path.exists(input_file):
        logger.error(
            f"File input '{input_file}' tidak ditemukan. Langkah 3 dihentikan."
        )
        return False

    df = pd.read_csv(input_file, delimiter=";")
    logger.info(
        f"Data untuk normalisasi dimuat: {df.shape[0]} baris, {df.shape[1]} kolom"
    )

    non_normalized_cols = ["TANGGAL", "PERIODE", "FCR_ACT"]

    all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features_to_normalize = [
        col for col in all_numeric_cols if col not in non_normalized_cols
    ]

    logger.info(f"Ditemukan {len(features_to_normalize)} fitur untuk dinormalisasi.")
    logger.info(f"Semua fitur akan dinormalisasi dengan RobustScaler.")

    robust_scaler = RobustScaler()
    df_normalized = df.copy()

    if features_to_normalize:
        robust_data = df[features_to_normalize].fillna(
            df[features_to_normalize].median()
        )
        df_normalized[features_to_normalize] = robust_scaler.fit_transform(robust_data)

        with open(robust_scaler_path, "wb") as f:
            pickle.dump(robust_scaler, f)
        logger.info(f"Model RobustScaler disimpan ke: {robust_scaler_path}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_normalized.to_csv(output_file, index=False, sep=";")

    normalization_info = {
        "robust_features": features_to_normalize,
        "non_normalized": [
            col for col in non_normalized_cols if col in df_normalized.columns
        ],
        "robust_scaler_params": {
            "center_": robust_scaler.center_.tolist(),
            "scale_": robust_scaler.scale_.tolist(),
            "feature_names": features_to_normalize,
        },
        "final_columns": df_normalized.columns.tolist(),
    }
    with open(info_file, "w") as f:
        json.dump(normalization_info, f, indent=2)

    print(f"\nBERHASIL: Normalisasi fitur selesai.")
    print(f" - Fitur dinormalisasi dengan Robust Scaler: {len(features_to_normalize)}")
    print(f" - File data ternormalisasi disimpan di: {output_file}")
    print(f" - Info normalisasi disimpan di: {info_file}")
    return True


# =============================================================================
# LANGKAH 4: PELATIHAN MODEL HIBRIDA TCN-XGBOOST DAN ANALISIS KORELASI
# =============================================================================
def asymmetric_objective(y_true, y_pred):
    residual = y_pred - y_true
    grad = np.where(residual > 0, 2.0 * 1.5 * residual, 4.0 * residual)
    hess = np.where(residual > 0, 2.0 * 1.5, 4.0)
    return grad, hess


def create_sequences_multi_output(X, y_dict, sequence_length):
    X_seq = []
    y_seq_dict = {target: [] for target in y_dict.keys()}

    for i in range(len(X) - sequence_length):
        X_seq.append(X.iloc[i : (i + sequence_length)].values)
        for target in y_dict.keys():
            target_value = y_dict[target].iloc[i + sequence_length]
            if isinstance(target_value, str):
                try:
                    target_value = float(target_value.replace(",", "."))
                except (ValueError, AttributeError):
                    target_value = 0.0
            elif pd.isna(target_value):
                target_value = 0.0
            y_seq_dict[target].append(float(target_value))

    X_seq_array = np.array(X_seq, dtype=np.float32)
    y_seq_arrays = {
        target: np.array(y_seq_dict[target], dtype=np.float32)
        for target in y_dict.keys()
    }
    return X_seq_array, y_seq_arrays


def create_tcn_model_multi_head(input_shape, tcn_params, multi_output_config):
    input_layer = Input(shape=input_shape)
    tcn_output = TCN(
        nb_filters=tcn_params["nb_filters"],
        kernel_size=tcn_params["kernel_size"],
        nb_stacks=tcn_params["nb_stacks"],
        dilations=[1, 2, 4, 8],
        use_skip_connections=True,
        dropout_rate=tcn_params["dropout_rate"],
        activation="relu",
        return_sequences=False,
    )(input_layer)

    outputs = []
    horizons = multi_output_config["horizons"]
    for i, horizon in enumerate(horizons):
        head_output = Dense(64, activation="relu", name=f"head_{horizon}d_dense")(
            tcn_output
        )
        head_output = Dense(1, name=f"fcr_next_{horizon}d")(head_output)
        outputs.append(head_output)

    model = Model(inputs=input_layer, outputs=outputs)
    losses = {f"fcr_next_{h}d": "mae" for h in horizons}
    loss_weights = {
        f"fcr_next_{h}d": w
        for h, w in zip(horizons, multi_output_config["horizon_weights"])
    }
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tcn_params["learning_rate"]),
        loss=losses,
        loss_weights=loss_weights,
        metrics=["mae", "mse"],
    )
    return model


def analyze_temporal_correlation(X, y_dict, tcn_encoder, sequence_length, output_path):
    """Analisis korelasi fitur temporal terhadap fitur tabular dan target dengan Pearson correlation."""
    logger.info(
        "Analisis korelasi fitur temporal dengan Pearson correlation dimulai..."
    )
    X_seq, y_seq_dict = create_sequences_multi_output(X, y_dict, sequence_length)
    temporal_features = tcn_encoder.predict(X_seq, verbose=0)

    # Ambil fitur tabular dan target untuk indeks yang sesuai
    df_tabular = X.iloc[sequence_length:].reset_index(drop=True)
    # Gunakan hanya kunci yang valid dari y_dict (harus sesuai horizons)
    valid_targets = [
        key for key in y_dict.keys() if key in [f"FCR_NEXT_{h}D" for h in [1, 3]]
    ]  # Sesuaikan dengan horizons
    df_target = pd.DataFrame(
        {target: y_dict[target].iloc[sequence_length:] for target in valid_targets}
    )

    df_temporal = pd.DataFrame(
        temporal_features,
        columns=[f"temporal_{i}" for i in range(temporal_features.shape[1])],
    )

    # Gabungkan semua data
    df_combined = pd.concat([df_temporal, df_tabular, df_target], axis=1)

    # === ANALISIS PEARSON CORRELATION YANG DITINGKATKAN ===

    # 1. Hitung matriks korelasi Pearson
    pearson_correlation_matrix = df_combined.corr(method="pearson")

    # 2. Hitung matriks korelasi Spearman untuk perbandingan
    spearman_correlation_matrix = df_combined.corr(method="spearman")

    # 3. Hitung p-values untuk korelasi Pearson
    def calculate_pearson_pvalues(df):
        """Menghitung p-values untuk korelasi Pearson."""
        n_cols = df.shape[1]
        p_matrix = np.zeros((n_cols, n_cols))

        for i in range(n_cols):
            for j in range(n_cols):
                if i != j:
                    try:
                        _, p_val = pearsonr(
                            df.iloc[:, i].dropna(), df.iloc[:, j].dropna()
                        )
                        p_matrix[i, j] = p_val
                    except:
                        p_matrix[i, j] = np.nan
                else:
                    p_matrix[i, j] = 0.0

        return pd.DataFrame(p_matrix, index=df.columns, columns=df.columns)

    logger.info("Menghitung p-values untuk korelasi Pearson...")
    pearson_pvalues = calculate_pearson_pvalues(df_combined)

    # 4. Identifikasi korelasi signifikan (p < 0.05)
    significant_correlations = pearson_pvalues < 0.05

    # === ANALISIS KORELASI FITUR TEMPORAL ===

    temporal_cols = [f"temporal_{i}" for i in range(temporal_features.shape[1])]

    # Korelasi temporal dengan target
    temporal_target_corr = pearson_correlation_matrix.loc[
        temporal_cols, df_target.columns
    ]
    temporal_target_pvals = pearson_pvalues.loc[temporal_cols, df_target.columns]

    # Korelasi temporal dengan fitur tabular
    tabular_cols = df_tabular.columns.tolist()
    temporal_tabular_corr = pearson_correlation_matrix.loc[temporal_cols, tabular_cols]
    temporal_tabular_pvals = pearson_pvalues.loc[temporal_cols, tabular_cols]

    # === ANALISIS STATISTIK LANJUTAN ===

    # Top korelasi temporal-target
    logger.info("\n=== ANALISIS KORELASI PEARSON FITUR TEMPORAL ===\n")

    for target in df_target.columns:
        logger.info(f"\nKorelasi Temporal dengan {target}:")
        target_corrs = temporal_target_corr[target].abs().sort_values(ascending=False)
        target_pvals = temporal_target_pvals[target]

        logger.info("Top 10 korelasi terkuat:")
        for i, (temporal_feature, corr_val) in enumerate(target_corrs.head(10).items()):
            p_val = target_pvals[temporal_feature]
            significance = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            logger.info(
                f"  {i+1:2d}. {temporal_feature}: {corr_val:.4f} (p={p_val:.4f}) {significance}"
            )

    # Analisis distribusi korelasi
    all_temporal_target_corrs = temporal_target_corr.values.flatten()
    all_temporal_target_corrs = all_temporal_target_corrs[
        ~np.isnan(all_temporal_target_corrs)
    ]

    logger.info(f"\nStatistik Korelasi Temporal-Target:")
    logger.info(f"  Mean: {np.mean(all_temporal_target_corrs):.4f}")
    logger.info(f"  Std: {np.std(all_temporal_target_corrs):.4f}")
    logger.info(f"  Min: {np.min(all_temporal_target_corrs):.4f}")
    logger.info(f"  Max: {np.max(all_temporal_target_corrs):.4f}")
    logger.info(
        f"  Korelasi |r| > 0.3: {np.sum(np.abs(all_temporal_target_corrs) > 0.3)}"
    )
    logger.info(
        f"  Korelasi |r| > 0.5: {np.sum(np.abs(all_temporal_target_corrs) > 0.5)}"
    )

    # === SIMPAN HASIL ANALISIS ===

    # Simpan matriks korelasi Pearson
    pearson_correlation_matrix.to_csv(
        output_path.replace(".json", "_pearson_correlation_matrix.csv")
    )
    logger.info(
        f"Matriks korelasi Pearson disimpan di: {output_path.replace('.json', '_pearson_correlation_matrix.csv')}"
    )

    # Simpan matriks korelasi Spearman
    spearman_correlation_matrix.to_csv(
        output_path.replace(".json", "_spearman_correlation_matrix.csv")
    )
    logger.info(
        f"Matriks korelasi Spearman disimpan di: {output_path.replace('.json', '_spearman_correlation_matrix.csv')}"
    )

    # Simpan p-values
    pearson_pvalues.to_csv(output_path.replace(".json", "_pearson_pvalues.csv"))
    logger.info(
        f"P-values Pearson disimpan di: {output_path.replace('.json', '_pearson_pvalues.csv')}"
    )

    # Simpan analisis korelasi temporal
    temporal_analysis = {
        "temporal_target_correlations": temporal_target_corr.to_dict(),
        "temporal_target_pvalues": temporal_target_pvals.to_dict(),
        "correlation_statistics": {
            "mean_correlation": float(np.mean(all_temporal_target_corrs)),
            "std_correlation": float(np.std(all_temporal_target_corrs)),
            "min_correlation": float(np.min(all_temporal_target_corrs)),
            "max_correlation": float(np.max(all_temporal_target_corrs)),
            "strong_correlations_03": int(
                np.sum(np.abs(all_temporal_target_corrs) > 0.3)
            ),
            "strong_correlations_05": int(
                np.sum(np.abs(all_temporal_target_corrs) > 0.5)
            ),
        },
    }

    with open(
        output_path.replace(".json", "_temporal_correlation_analysis.json"), "w"
    ) as f:
        json.dump(temporal_analysis, f, indent=2)
    logger.info(
        f"Analisis korelasi temporal disimpan di: {output_path.replace('.json', '_temporal_correlation_analysis.json')}"
    )

    # === VISUALISASI YANG DITINGKATKAN ===

    # 1. Heatmap korelasi Pearson lengkap
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(pearson_correlation_matrix, dtype=bool))
    sns.heatmap(
        pearson_correlation_matrix,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Heatmap Korelasi Pearson - Fitur Temporal, Tabular, dan Target", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(
        output_path.replace(".json", "_pearson_correlation_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Heatmap khusus temporal-target
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        temporal_target_corr,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".3f",
        square=False,
        linewidths=0.5,
    )
    plt.title("Korelasi Pearson: Fitur Temporal vs Target", fontsize=14)
    plt.xlabel("Target Variables")
    plt.ylabel("Temporal Features")
    plt.tight_layout()
    plt.savefig(
        output_path.replace(".json", "_temporal_target_correlation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Distribusi korelasi
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Histogram korelasi temporal-target
    axes[0, 0].hist(
        all_temporal_target_corrs,
        bins=30,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    axes[0, 0].axvline(
        np.mean(all_temporal_target_corrs),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_temporal_target_corrs):.3f}",
    )
    axes[0, 0].set_title("Distribusi Korelasi Temporal-Target")
    axes[0, 0].set_xlabel("Korelasi Pearson")
    axes[0, 0].set_ylabel("Frekuensi")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot korelasi per target
    target_corr_data = [
        temporal_target_corr[col].values for col in temporal_target_corr.columns
    ]
    axes[0, 1].boxplot(target_corr_data, labels=temporal_target_corr.columns)
    axes[0, 1].set_title("Distribusi Korelasi per Target")
    axes[0, 1].set_ylabel("Korelasi Pearson")
    axes[0, 1].grid(True, alpha=0.3)

    # Scatter plot: Pearson vs Spearman correlation
    pearson_vals = pearson_correlation_matrix.loc[
        temporal_cols, df_target.columns
    ].values.flatten()
    spearman_vals = spearman_correlation_matrix.loc[
        temporal_cols, df_target.columns
    ].values.flatten()
    valid_idx = ~(np.isnan(pearson_vals) | np.isnan(spearman_vals))

    axes[1, 0].scatter(pearson_vals[valid_idx], spearman_vals[valid_idx], alpha=0.6)
    axes[1, 0].plot([-1, 1], [-1, 1], "r--", alpha=0.8)
    axes[1, 0].set_xlabel("Korelasi Pearson")
    axes[1, 0].set_ylabel("Korelasi Spearman")
    axes[1, 0].set_title("Pearson vs Spearman Correlation")
    axes[1, 0].grid(True, alpha=0.3)

    # Bar plot: Korelasi terkuat
    top_corrs = (
        temporal_target_corr.abs().max(axis=1).sort_values(ascending=False).head(10)
    )
    axes[1, 1].barh(range(len(top_corrs)), top_corrs.values)
    axes[1, 1].set_yticks(range(len(top_corrs)))
    axes[1, 1].set_yticklabels(top_corrs.index)
    axes[1, 1].set_xlabel("|Korelasi Pearson|")
    axes[1, 1].set_title("Top 10 Fitur Temporal (Korelasi Terkuat)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path.replace(".json", "_correlation_analysis_plots.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Significance heatmap
    plt.figure(figsize=(12, 8))
    significance_mask = temporal_target_pvals > 0.05
    sns.heatmap(
        temporal_target_corr,
        mask=significance_mask,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".3f",
        cbar_kws={"label": "Korelasi Pearson (p < 0.05)"},
    )
    plt.title("Korelasi Signifikan: Fitur Temporal vs Target (p < 0.05)", fontsize=14)
    plt.xlabel("Target Variables")
    plt.ylabel("Temporal Features")
    plt.tight_layout()
    plt.savefig(
        output_path.replace(".json", "_significant_correlations.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(f"Visualisasi korelasi Pearson disimpan:")
    logger.info(
        f"  - Heatmap lengkap: {output_path.replace('.json', '_pearson_correlation_heatmap.png')}"
    )
    logger.info(
        f"  - Temporal-Target: {output_path.replace('.json', '_temporal_target_correlation.png')}"
    )
    logger.info(
        f"  - Analisis distribusi: {output_path.replace('.json', '_correlation_analysis_plots.png')}"
    )
    logger.info(
        f"  - Korelasi signifikan: {output_path.replace('.json', '_significant_correlations.png')}"
    )

    # Tampilkan ringkasan korelasi utama
    logger.info("\n=== RINGKASAN KORELASI PEARSON ===\n")
    logger.info("Korelasi fitur temporal dengan target (nilai absolut):")
    print(temporal_target_corr.abs())

    return {
        "pearson_correlation_matrix": pearson_correlation_matrix,
        "temporal_target_correlations": temporal_target_corr,
        "correlation_statistics": temporal_analysis["correlation_statistics"],
    }


def train_tcn_xgboost_hybrid_with_kfold(
    data_file: str, tcn_encoder_path: str, xgb_model_path: str, hybrid_config_path: str
) -> bool:
    print("\n" + "=" * 80)
    print(
        "MEMULAI LANGKAH 4: PELATIHAN MODEL HIBRIDA TCN-XGBOOST DENGAN 9-FOLD CROSS-VALIDATION"
    )
    print("=" * 80)

    OPTIMAL_HYPERPARAMS = {
        "sequence_length": 3,
        "batch_size": 8,
        "multi_output_config": MULTI_OUTPUT_CONFIG,
        "tcn_params": {
            "nb_filters": 32,
            "kernel_size": 3,
            "nb_stacks": 3,
            "dilations": [1, 2, 4],
            "dropout_rate": 0.4,
            "spatial_dropout": 0.2,
            "kernel_regularizer": "l2",
            "kernel_regularizer_strength": 1e-4,
            "learning_rate": 1e-4,
            "optimizer": "adamw",
            "weight_decay": 1e-3,
            "gradient_clip_norm": 0.5,
        },
        "xgb_params": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 150,
            "learning_rate": 0.100,
            "max_depth": 7,
            "subsample": 0.98,
            "colsample_bytree": 0.90,
            "reg_alpha": 0.1,
            "reg_lambda": 0.25,
            "random_state": 42,
            "n_jobs": -1,
        },
        "training_params": {
            "early_stopping": {"patience": 25, "monitor": "val_mae"},
            "lr_scheduler": {"type": "reduce_on_plateau", "patience": 15},
        },
    }

    logger.info("Langkah 4.1: Memuat dan Mempersiapkan Data untuk Multi-Horizon...")
    df = pd.read_csv(data_file, sep=";")

    df["FCR_ACT"] = df["FCR_ACT"].astype(str).str.replace(",", ".", regex=False)
    df["FCR_ACT"] = pd.to_numeric(df["FCR_ACT"], errors="coerce")
    if df["FCR_ACT"].isna().sum() > 0:
        logger.warning(
            f"Found {df['FCR_ACT'].isna().sum()} non-numeric values in FCR_ACT, filling with median"
        )
        df["FCR_ACT"] = df["FCR_ACT"].fillna(df["FCR_ACT"].median())

    horizons = OPTIMAL_HYPERPARAMS["multi_output_config"]["horizons"]
    for horizon in horizons:
        target_col = f"FCR_NEXT_{horizon}D"
        df[target_col] = df.groupby("PERIODE")["FCR_ACT"].shift(-horizon)
        df[target_col] = df[target_col].astype(str).str.replace(",", ".", regex=False)
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        if df[target_col].isna().sum() > 0:
            logger.warning(
                f"Found {df[target_col].isna().sum()} non-numeric values in {target_col}"
            )

    target_cols_to_check = [f"FCR_NEXT_{h}D" for h in horizons]
    logger.info(f"Data shape before dropping NA targets: {df.shape}")
    df.dropna(subset=target_cols_to_check, inplace=True)
    logger.info(f"Data shape after dropping NA targets: {df.shape}")

    TARGETS = [f"FCR_NEXT_{h}D" for h in horizons]
    features_to_drop = ["TANGGAL", "FCR_ACT"]
    df = df.drop(columns=features_to_drop, errors="ignore")

    features = [c for c in df.columns if c not in TARGETS + ["PERIODE"]]
    X = df[features]
    y = {target: df[target].astype(np.float32) for target in TARGETS}
    if any(y[target].isna().any() for target in TARGETS):
        for target in TARGETS:
            y[target] = y[target].fillna(y[target].median())

    logger.info(f"Data untuk cross-validation: {X.shape}")
    logger.info(f"Multi-horizon targets: {TARGETS}")

    X_seq, y_seq_dict = create_sequences_multi_output(
        X, y, OPTIMAL_HYPERPARAMS["sequence_length"]
    )
    valid_indices = np.arange(len(X_seq))

    kfold = KFold(n_splits=9, shuffle=True, random_state=42)
    cv_results = {
        "fold_scores": [],
        "tcn_scores": [],
        "xgb_scores": {horizon: [] for horizon in horizons},
        "hybrid_scores": {horizon: [] for horizon in horizons},
    }

    fold_num = 1
    best_models = {"tcn": None, "xgb": {}}
    best_score = float("inf")

    for train_idx, val_idx in kfold.split(valid_indices):
        logger.info(f"\n=== FOLD {fold_num}/9 ===")
        X_train_fold = X_seq[train_idx]
        X_val_fold = X_seq[val_idx]
        y_train_fold = {target: y_seq_dict[target][train_idx] for target in TARGETS}
        y_val_fold = {target: y_seq_dict[target][val_idx] for target in TARGETS}

        logger.info(f"Fold {fold_num}: Melatih TCN Model...")
        input_shape = (OPTIMAL_HYPERPARAMS["sequence_length"], X.shape[1])
        tcn_model = create_tcn_model_multi_head(
            input_shape,
            OPTIMAL_HYPERPARAMS["tcn_params"],
            OPTIMAL_HYPERPARAMS["multi_output_config"],
        )

        y_train_fold_fixed = {
            f"fcr_next_{h}d": y_train_fold[f"FCR_NEXT_{h}D"] for h in horizons
        }
        y_val_fold_fixed = {
            f"fcr_next_{h}d": y_val_fold[f"FCR_NEXT_{h}D"] for h in horizons
        }

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
            )
        ]
        history = tcn_model.fit(
            X_train_fold,
            y_train_fold_fixed,
            validation_data=(X_val_fold, y_val_fold_fixed),
            epochs=100,
            batch_size=OPTIMAL_HYPERPARAMS["batch_size"],
            verbose=0,
            callbacks=callbacks,
        )

        tcn_val_loss = min(history.history["val_loss"])
        cv_results["tcn_scores"].append(tcn_val_loss)

        tcn_encoder = Model(inputs=tcn_model.input, outputs=tcn_model.layers[1].output)

        logger.info(f"Fold {fold_num}: Melatih XGBoost Models...")
        temporal_features_train = tcn_encoder.predict(X_train_fold, verbose=0)
        temporal_features_val = tcn_encoder.predict(X_val_fold, verbose=0)

        tabular_train = X.iloc[
            train_idx + OPTIMAL_HYPERPARAMS["sequence_length"]
        ].values
        tabular_val = X.iloc[val_idx + OPTIMAL_HYPERPARAMS["sequence_length"]].values

        X_hybrid_train = np.concatenate(
            [tabular_train, temporal_features_train], axis=1
        )
        X_hybrid_val = np.concatenate([tabular_val, temporal_features_val], axis=1)

        fold_xgb_models = {}
        for horizon in horizons:
            target_key = f"FCR_NEXT_{horizon}D"
            xgb_params_custom = OPTIMAL_HYPERPARAMS["xgb_params"].copy()
            xgb_params_custom["objective"] = asymmetric_objective
            xgb_model = xgb.XGBRegressor(**xgb_params_custom)
            xgb_model.fit(
                X_hybrid_train,
                y_train_fold[target_key],
                eval_set=[(X_hybrid_val, y_val_fold[target_key])],
                verbose=False,
            )
            fold_xgb_models[f"horizon_{horizon}d"] = xgb_model

            val_pred = xgb_model.predict(X_hybrid_val)
            val_mae = mean_absolute_error(y_val_fold[target_key], val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val_fold[target_key], val_pred))
            val_mape = mean_absolute_percentage_error(y_val_fold[target_key], val_pred)
            val_r2 = r2_score(y_val_fold[target_key], val_pred)

            cv_results["xgb_scores"][horizon].append(
                {"mae": val_mae, "rmse": val_rmse, "mape": val_mape, "r2": val_r2}
            )
            cv_results["hybrid_scores"][horizon].append(
                {"mae": val_mae, "rmse": val_rmse, "mape": val_mape, "r2": val_r2}
            )

        avg_mae = np.mean([cv_results["xgb_scores"][h][-1]["mae"] for h in horizons])
        if avg_mae < best_score:
            best_score = avg_mae
            best_models["tcn"] = tcn_model
            best_models["xgb"] = fold_xgb_models.copy()
            best_tcn_encoder = tcn_encoder

        logger.info(f"Fold {fold_num} selesai - Avg MAE: {avg_mae:.4f}")
        fold_num += 1

    logger.info("\n=== HASIL 9-FOLD CROSS-VALIDATION ===")
    tcn_mean = np.mean(cv_results["tcn_scores"])
    tcn_std = np.std(cv_results["tcn_scores"])
    logger.info(f"TCN Validation Loss: {tcn_mean:.4f} ± {tcn_std:.4f}")

    for horizon in horizons:
        horizon_scores = cv_results["xgb_scores"][horizon]
        mae_scores = [score["mae"] for score in horizon_scores]
        rmse_scores = [score["rmse"] for score in horizon_scores]
        mape_scores = [score["mape"] for score in horizon_scores]
        r2_scores = [score["r2"] for score in horizon_scores]
        logger.info(f"\nHorizon {horizon}D Results:")
        logger.info(f"  MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
        logger.info(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        logger.info(f"  MAPE: {np.mean(mape_scores):.4f} ± {np.std(mape_scores):.4f}")
        logger.info(f"  R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    logger.info("\nMenyimpan model terbaik dari cross-validation...")
    best_tcn_encoder.save(tcn_encoder_path)
    logger.info(f"TCN Encoder terbaik disimpan di: {tcn_encoder_path}")

    for horizon in horizons:
        horizon_model_path = xgb_model_path.replace(".pkl", f"_horizon_{horizon}d.pkl")
        with open(horizon_model_path, "wb") as f:
            pickle.dump(best_models["xgb"][f"horizon_{horizon}d"], f)
        logger.info(
            f"XGBoost model horizon {horizon}d disimpan di: {horizon_model_path}"
        )

    logger.info("Menyimpan konfigurasi dengan hasil cross-validation...")
    hybrid_config = {
        "sequence_length": OPTIMAL_HYPERPARAMS["sequence_length"],
        "batch_size": OPTIMAL_HYPERPARAMS["batch_size"],
        "multi_output_config": OPTIMAL_HYPERPARAMS["multi_output_config"],
        "tcn_params": OPTIMAL_HYPERPARAMS["tcn_params"],
        "xgb_params": OPTIMAL_HYPERPARAMS["xgb_params"],
        "training_params": OPTIMAL_HYPERPARAMS["training_params"],
        "feature_names": features,
        "target_names": TARGETS,
        "horizons": horizons,
        "model_type": "TCN-XGBoost Multi-Output Hybrid with 9-Fold CV",
        "cross_validation_results": {
            "n_folds": 9,
            "tcn_validation_loss": {
                "mean": float(tcn_mean),
                "std": float(tcn_std),
                "scores": [float(score) for score in cv_results["tcn_scores"]],
            },
            "horizon_results": {},
        },
    }

    for horizon in horizons:
        horizon_scores = cv_results["xgb_scores"][horizon]
        mae_scores = [score["mae"] for score in horizon_scores]
        rmse_scores = [score["rmse"] for score in horizon_scores]
        mape_scores = [score["mape"] for score in horizon_scores]
        r2_scores = [score["r2"] for score in horizon_scores]
        hybrid_config["cross_validation_results"]["horizon_results"][
            f"horizon_{horizon}d"
        ] = {
            "mae": {
                "mean": float(np.mean(mae_scores)),
                "std": float(np.std(mae_scores)),
            },
            "rmse": {
                "mean": float(np.mean(rmse_scores)),
                "std": float(np.std(rmse_scores)),
            },
            "mape": {
                "mean": float(np.mean(mape_scores)),
                "std": float(np.std(mape_scores)),
            },
            "r2": {"mean": float(np.mean(r2_scores)), "std": float(np.std(r2_scores))},
        }

    with open(hybrid_config_path, "w") as f:
        json.dump(hybrid_config, f, indent=4)
    logger.info(f"Konfigurasi dengan hasil CV disimpan di: {hybrid_config_path}")

    logger.info("Membuat visualisasi hasil cross-validation...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("9-Fold Cross-Validation Results", fontsize=16)
    metrics = ["mae", "rmse", "mape", "r2"]
    metric_names = ["MAE", "RMSE", "MAPE", "R²"]
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        for horizon in horizons:
            scores = [score[metric] for score in cv_results["xgb_scores"][horizon]]
            ax.plot(range(1, 10), scores, marker="o", label=f"Horizon {horizon}D")
        ax.set_xlabel("Fold")
        ax.set_ylabel(name)
        ax.set_title(f"{name} across Folds")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, len(horizons), figsize=(15, 6))
    if len(horizons) == 1:
        axes = [axes]
    for idx, horizon in enumerate(horizons):
        mae_scores = [score["mae"] for score in cv_results["xgb_scores"][horizon]]
        rmse_scores = [score["rmse"] for score in cv_results["xgb_scores"][horizon]]
        axes[idx].boxplot([mae_scores, rmse_scores], labels=["MAE", "RMSE"])
        axes[idx].set_title(f"Horizon {horizon}D Score Distribution")
        axes[idx].set_ylabel("Score")
        axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    logger.info("Menganalisis korelasi fitur temporal...")
    analyze_temporal_correlation(
        X,
        y,
        best_tcn_encoder,
        OPTIMAL_HYPERPARAMS["sequence_length"],
        hybrid_config_path,
    )

    print("\n=== RINGKASAN 9-FOLD CROSS-VALIDATION ===")
    print(f"Model divalidasi menggunakan 9-fold cross-validation")
    print(f"TCN Validation Loss: {tcn_mean:.4f} ± {tcn_std:.4f}")
    for horizon in horizons:
        horizon_scores = cv_results["xgb_scores"][horizon]
        mae_scores = [score["mae"] for score in horizon_scores]
        print(
            f"Horizon {horizon}D - MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}"
        )
    print("\nModel terbaik telah disimpan berdasarkan performa cross-validation.")

    return True


# =============================================================================
# BLOK EKSEKUSI UTAMA
# =============================================================================
if __name__ == "__main__":
    base_folder_path = os.getcwd()

    data_input_folder = os.path.join(base_folder_path, "3. non-normalize")
    intermediate_data_folder = os.path.join(
        base_folder_path, "1. Normalisasi", "Data", "Data_Rekayasa"
    )
    final_output_folder = os.path.join(
        base_folder_path, "2. Hasil Normalisasi", "Terbaru"
    )
    models_folder = os.path.join(base_folder_path, "5. Model", "models_terbaru")

    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(intermediate_data_folder, exist_ok=True)
    os.makedirs(final_output_folder, exist_ok=True)

    merged_file = os.path.join(intermediate_data_folder, "File_Gabungan.csv")
    enhanced_file = os.path.join(
        intermediate_data_folder, "File_Gabungan_Enhanced_Features.csv"
    )
    normalized_file = os.path.join(
        final_output_folder, "Data_Gabungan_Fitur_Rekayasa_Normalized.csv"
    )

    fe_pipeline_file = os.path.join(models_folder, "feature_engineering_pipeline.pkl")
    robust_scaler_file = os.path.join(models_folder, "robust_scaler.pkl")
    feature_metadata_file = os.path.join(final_output_folder, "feature_metadata.json")
    norm_info_file = os.path.join(final_output_folder, "normalization_info.json")

    tcn_encoder_model_path = os.path.join(models_folder, "tcn_encoder.keras")
    xgb_hybrid_model_path = os.path.join(models_folder, "xgboost_hybrid_model.pkl")
    hybrid_config_file_path = os.path.join(models_folder, "hybrid_model_config.json")

    try:
        if not merge_data_files(folder_path=data_input_folder, output_file=merged_file):
            raise Exception("Gagal pada langkah penggabungan data.")
        if not run_feature_engineering_step(
            input_file=merged_file,
            output_data_path=enhanced_file,
            output_pipeline_path=fe_pipeline_file,
            metadata_path=final_output_folder,
            config=FEATURE_CONFIG,
        ):
            raise Exception("Gagal pada langkah rekayasa fitur.")
        if not normalize_features(
            input_file=enhanced_file,
            output_file=normalized_file,
            info_file=norm_info_file,
            robust_scaler_path=robust_scaler_file,
        ):
            raise Exception("Gagal pada langkah normalisasi.")
        if not train_tcn_xgboost_hybrid_with_kfold(
            data_file=normalized_file,
            tcn_encoder_path=tcn_encoder_model_path,
            xgb_model_path=xgb_hybrid_model_path,
            hybrid_config_path=hybrid_config_file_path,
        ):
            raise Exception("Gagal pada langkah pelatihan model hibrida.")

        print(
            "\n\n>>> SEMUA LANGKAH PIPELINE (TERMASUK PELATIHAN MODEL HIBRIDA) SELESAI DENGAN SUKSES! <<<"
        )
        print("Model-model Anda siap untuk diimplementasikan.")
        print(f"Folder 'models' kini berisi: {os.listdir(models_folder)}")
        print("\nCATATAN PENTING:")
        print("- Model dilatih menggunakan SELURUH data yang tersedia")
        print("- Tidak ada pembagian data untuk validasi/testing")
        print("- Evaluasi ditampilkan berdasarkan performa pada data training")
        print(
            "- Untuk evaluasi objektif, gunakan data baru yang belum pernah dilihat model"
        )

    except Exception as e:
        logger.critical(f"Terjadi kesalahan fatal pada pipeline: {e}", exc_info=True)
