# -*- coding: utf-8 -*-
"""
Skrip pipeline lengkap untuk:
1. Menggabungkan data dari beberapa file CSV periode.
2. Melakukan rekayasa fitur (feature engineering) canggih pada data yang digabungkan.
3. Melakukan normalisasi fitur dengan RobustScaler.
4. Melatih model hibrida yang terdiri dari TCN dan XGBoost dengan SELURUH DATA sebagai training.

Alur Kerja:
- Langkah 1: Penggabungan data mentah.
- Langkah 2: Rekayasa fitur dari data gabungan.
- Langkah 3: Normalisasi fitur yang telah direkayasa.
- Langkah 4: Pelatihan model hibrida menggunakan seluruh data.

"""

import pandas as pd
import numpy as np
import os
import warnings
import logging
import json
import pickle
from typing import Dict
from sklearn.preprocessing import RobustScaler

### --- IMPLEMENTASI BARU --- ###
# Impor untuk Model Hibrida
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tcn import TCN
    import xgboost as xgb
    import optuna
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score,
    )
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error impor: {e}")
    print(
        "Harap pasang library yang diperlukan: pip install tensorflow keras-tcn xgboost optuna scikit-learn"
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
    "health_weights": {"mati": 0.5, "deplesi": 0.2},
    "variance_threshold": 0.01,
    "correlation_threshold": 0.98,
    "decimal_places": 3,
}

MULTI_OUTPUT_CONFIG = {
    "output_strategy": "multi_head",  # Separate heads untuk each horizon
    "horizons": [1, 3],
    "output_dims": [95, 95],  # Same features untuk each horizon
    "horizon_weights": [0.6, 0.4],  # Weight lebih tinggi untuk 1-day
}

# =============================================================================
# LANGKAH 1: FUNGSI PENGGABUNGAN DATA
# =============================================================================


def merge_data_files(folder_path: str, output_file: str):
    """
    Membaca semua file CSV dari folder, menggabungkannya, dan menyimpannya.
    """
    print("=" * 80)
    print("MEMULAI LANGKAH 1: PENGGABUNGAN DATA")
    print("=" * 80)

    dataframes = []
    for i in range(1, 14):
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
# LANGKAH 2: FUNGSI REKAYASA FITUR (IMPLEMENTASI BARU)
# =============================================================================
def run_feature_engineering_step(
    input_file: str,
    output_data_path: str,
    output_pipeline_path: str,
    metadata_path: str,
    config: Dict,
) -> bool:
    """
    Menggunakan kelas FeatureEngineeringPipeline untuk memproses data dan menyimpan pipeline.
    """
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

    # 1. Inisialisasi pipeline rekayasa fitur dengan konfigurasi
    feature_pipeline = FeatureEngineeringPipeline(config=config)

    # 2. Lakukan 'fit_transform' pada data mentah. Ini akan:
    #    a. "Melatih" pipeline (mempelajari kolom mana yang akan dihapus).
    #    b. Mentransformasi data menjadi data dengan fitur-fitur baru.
    #    c. Menyimpan metadata proses ke file JSON.
    logger.info("Menjalankan fit_transform pada FeatureEngineeringPipeline...")
    df_enhanced = feature_pipeline.fit_transform(
        df_raw, metadata_output_dir=metadata_path
    )

    # 3. Simpan data yang sudah direkayasa fiturnya (hasil dari pipeline)
    df_enhanced.to_csv(output_data_path, sep=";", index=False)
    logger.info(f"Data dengan fitur enhanced berhasil disimpan di: {output_data_path}")

    # 4. Simpan objek pipeline yang sudah di-'fit' ke file .pkl
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
    """
    Menerapkan normalisasi menggunakan RobustScaler pada data.
    """
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

    # Kolom yang tidak dinormalisasi (termasuk kolom target)
    non_normalized_cols = ["TANGGAL", "PERIODE", "FCR_ACT"]

    # 1. Identifikasi semua kolom numerik yang tersedia
    all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # 2. Tentukan fitur mana yang akan dinormalisasi
    features_to_normalize = [
        col for col in all_numeric_cols if col not in non_normalized_cols
    ]

    logger.info(f"Ditemukan {len(features_to_normalize)} fitur untuk dinormalisasi.")
    logger.info(f"Semua fitur akan dinormalisasi dengan RobustScaler.")

    robust_scaler = RobustScaler()
    df_normalized = df.copy()

    # Terapkan normalisasi dengan RobustScaler
    if features_to_normalize:
        robust_data = df[features_to_normalize].fillna(
            df[features_to_normalize].median()
        )
        df_normalized[features_to_normalize] = robust_scaler.fit_transform(robust_data)

        # Simpan model RobustScaler
        with open(robust_scaler_path, "wb") as f:
            pickle.dump(robust_scaler, f)
        logger.info(f"Model RobustScaler disimpan ke: {robust_scaler_path}")

    # Simpan hasil
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_normalized.to_csv(output_file, index=False, sep=";")

    # Simpan informasi normalisasi
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
# LANGKAH 4: PELATIHAN MODEL HIBRIDA TCN-XGBOOST (SELURUH DATA SEBAGAI TRAINING)
# =============================================================================
def asymmetric_objective(y_true, y_pred):
    """
    Fungsi loss kustom untuk XGBoost yang memberikan penalti lebih besar
    pada over-prediction (prediksi > aktual).

    Args:
        y_true (np.array): Nilai aktual.
        y_pred (np.array): Nilai prediksi.

    Returns:
        tuple: Gradien dan Hessian.
    """
    residual = y_pred - y_true
    # Beri penalti 0.5x lebih besar untuk over-prediction (residual > 0)
    grad = np.where(residual > 0, 2.0 * 0.5 * residual, 4.0 * residual)
    hess = np.where(residual > 0, 2.0 * 0.5, 4.0)
    return grad, hess


def train_tcn_xgboost_hybrid(
    data_file: str, tcn_encoder_path: str, xgb_model_path: str, hybrid_config_path: str
) -> bool:
    """
    Melatih seluruh pipeline model hibrida TCN-XGBoost menggunakan SELURUH DATA sebagai training
    dengan hyperparameter yang sudah dioptimasi.
    """
    print("\n" + "=" * 80)
    print(
        "MEMULAI LANGKAH 4: PELATIHAN MODEL HIBRIDA TCN-XGBOOST (SELURUH DATA) DENGAN HYPERPARAMETER OPTIMAL"
    )
    print("=" * 80)

    # --- Hyperparameter yang sudah dioptimasi ---
    OPTIMAL_HYPERPARAMS = {
        "sequence_length": 11,
        "batch_size": 64,
        "multi_output_config": MULTI_OUTPUT_CONFIG,
        "tcn_params": {
            # Core Architecture (Reduced to prevent overfitting)
            "nb_filters": 32,  # Dikurangi dari 120 → 32
            "kernel_size": 3,  # Dikurangi dari 4 → 3
            "nb_stacks": 1,  # Dikurangi untuk simplify
            "dilations": [1, 2, 4],  # Short dilations untuk weekly patterns
            # Regularization (Aggressive)
            "dropout_rate": 0.4,  # Dinaikkan dari 0.25 → 0.4
            "spatial_dropout": 0.2,  # Tambahan regularization
            "kernel_regularizer": "l2",
            "kernel_regularizer_strength": 1e-4,
            # Training (Conservative)
            "learning_rate": 1e-4,  # Dikurangi dari 2.5e-4
            "optimizer": "adamw",
            "weight_decay": 1e-3,
            "gradient_clip_norm": 0.5,
        },
        "xgb_params": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 775,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.98,
            "colsample_bytree": 0.90,
            "reg_alpha": 0.18459690345484073,
            "reg_lambda": 0.70524808585881,
            "random_state": 42,
            "n_jobs": -1,
        },
        "training_params": {
            "early_stopping": {"patience": 25, "monitor": "val_mae"},
            "lr_scheduler": {"type": "reduce_on_plateau", "patience": 15},
        },
    }

    # --- Langkah 1: Persiapan Data ---
    logger.info("Langkah 4.1: Memuat dan Mempersiapkan Data untuk Multi-Horizon...")
    df = pd.read_csv(data_file, sep=";")

    # PERBAIKAN: Ensure FCR_ACT is numeric before creating targets
    # Convert FCR_ACT with more aggressive cleaning
    logger.info(f"FCR_ACT original dtype: {df['FCR_ACT'].dtype}")
    logger.info(f"FCR_ACT sample values before conversion: {df['FCR_ACT'].head()}")

    # Clean FCR_ACT data
    df["FCR_ACT"] = (
        df["FCR_ACT"].astype(str).str.replace(",", ".", regex=False)
    )  # Replace comma with dot if any
    df["FCR_ACT"] = pd.to_numeric(df["FCR_ACT"], errors="coerce")

    # Check for any remaining non-numeric values
    non_numeric_count = df["FCR_ACT"].isna().sum()
    if non_numeric_count > 0:
        logger.warning(
            f"Found {non_numeric_count} non-numeric values in FCR_ACT, filling with median"
        )
        df["FCR_ACT"] = df["FCR_ACT"].fillna(df["FCR_ACT"].median())

    logger.info(f"FCR_ACT dtype after conversion: {df['FCR_ACT'].dtype}")
    logger.info(f"FCR_ACT sample values after conversion: {df['FCR_ACT'].head()}")

    # Target adalah FCR untuk multiple horizons
    horizons = OPTIMAL_HYPERPARAMS["multi_output_config"]["horizons"]

    # PERBAIKAN: Buat target untuk setiap horizon dengan cleaning yang lebih baik
    for horizon in horizons:
        target_col = f"FCR_NEXT_{horizon}D"
        df[target_col] = df.groupby("PERIODE")["FCR_ACT"].shift(-horizon)

        # Aggressive cleaning for target columns
        logger.info(f"Creating target {target_col}...")
        logger.info(f"Target {target_col} original dtype: {df[target_col].dtype}")
        logger.info(
            f"Target {target_col} sample values before conversion: {df[target_col].head()}"
        )

        # Convert to string first, clean, then to numeric
        df[target_col] = df[target_col].astype(str).str.replace(",", ".", regex=False)
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        # Check for remaining non-numeric values
        target_non_numeric = df[target_col].isna().sum()
        if target_non_numeric > 0:
            logger.warning(
                f"Found {target_non_numeric} non-numeric values in {target_col}"
            )

        logger.info(
            f"Target {target_col} dtype after conversion: {df[target_col].dtype}"
        )
        logger.info(
            f"Target {target_col} sample values after conversion: {df[target_col].head()}"
        )

    # Hapus baris yang tidak memiliki target untuk horizon terpanjang
    max_horizon = max(horizons)
    target_cols_to_check = [f"FCR_NEXT_{h}D" for h in horizons]

    # Log data before dropping
    logger.info(f"Data shape before dropping NA targets: {df.shape}")
    df.dropna(subset=target_cols_to_check, inplace=True)
    logger.info(f"Data shape after dropping NA targets: {df.shape}")

    TARGETS = [f"FCR_NEXT_{h}D" for h in horizons]

    # Buang kolom yang tidak diperlukan untuk training
    features_to_drop = ["TANGGAL", "FCR_ACT"]
    df = df.drop(columns=features_to_drop, errors="ignore")

    # MENGGUNAKAN SELURUH DATA UNTUK TRAINING
    features = [c for c in df.columns if c not in TARGETS + ["PERIODE"]]
    X_train = df[features]

    # PERBAIKAN: Ensure all target values are properly converted to float32
    y_train = {}
    for target in TARGETS:
        # Final cleaning and conversion
        target_series = df[target].copy()

        # Remove any remaining non-numeric values
        if target_series.dtype == "object":
            target_series = pd.to_numeric(target_series, errors="coerce")

        # Fill any remaining NaN values with median
        if target_series.isna().any():
            target_series = target_series.fillna(target_series.median())

        # Convert to float32
        y_train[target] = target_series.astype(np.float32)

        logger.info(f"Final target {target} dtype: {y_train[target].dtype}")
        logger.info(f"Final target {target} shape: {y_train[target].shape}")
        logger.info(f"Final target {target} sample values: {y_train[target].head()}")

    logger.info(f"Menggunakan SELURUH DATA untuk training: {X_train.shape}")
    logger.info(f"Multi-horizon targets: {TARGETS}")

    # --- Fungsi Helper untuk Sequences ---
    def create_sequences_multi_output(X, y_dict, sequence_length):
        X_seq = []
        y_seq_dict = {target: [] for target in y_dict.keys()}

        for i in range(len(X) - sequence_length):
            X_seq.append(X.iloc[i : (i + sequence_length)].values)
            for target in y_dict.keys():
                # PERBAIKAN: Ensure individual target values are numeric
                target_value = y_dict[target].iloc[i + sequence_length]
                if isinstance(target_value, str):
                    try:
                        target_value = float(target_value.replace(",", "."))
                    except (ValueError, AttributeError):
                        target_value = 0.0  # fallback value
                elif pd.isna(target_value):
                    target_value = 0.0  # fallback for NaN

                y_seq_dict[target].append(float(target_value))

        # PERBAIKAN: Ensure all arrays are float32
        X_seq_array = np.array(X_seq, dtype=np.float32)
        y_seq_arrays = {}
        for target in y_dict.keys():
            y_seq_arrays[target] = np.array(y_seq_dict[target], dtype=np.float32)
            logger.info(
                f"Sequence target {target} final dtype: {y_seq_arrays[target].dtype}"
            )
            logger.info(
                f"Sequence target {target} sample values: {y_seq_arrays[target][:5]}"
            )

        return X_seq_array, y_seq_arrays

    # --- Fungsi untuk Membuat TCN Model dengan Parameter Optimal ---
    def create_tcn_model_multi_head(input_shape, tcn_params, multi_output_config):
        """Membuat model TCN dengan multi-head output untuk multiple horizons."""
        input_layer = Input(shape=input_shape)

        # Shared TCN layers
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

        # Multiple output heads untuk each horizon
        outputs = []
        horizons = multi_output_config["horizons"]

        for i, horizon in enumerate(horizons):
            head_output = Dense(64, activation="relu", name=f"head_{horizon}d_dense")(
                tcn_output
            )
            head_output = Dense(1, name=f"fcr_next_{horizon}d")(head_output)
            outputs.append(head_output)

        model = Model(inputs=input_layer, outputs=outputs)

        # Compile dengan weighted loss untuk multiple outputs
        losses = {f"fcr_next_{h}d": "mae" for h in horizons}
        loss_weights = {
            f"fcr_next_{h}d": w
            for h, w in zip(horizons, multi_output_config["horizon_weights"])
        }

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=tcn_params["learning_rate"]
            ),
            loss=losses,
            loss_weights=loss_weights,
            metrics=["mae", "mse"],
        )
        return model

    # --- Langkah 2: Persiapan Data untuk TCN ---
    logger.info(
        "Langkah 4.2: Mempersiapkan Data untuk TCN Multi-Output dengan Parameter Optimal..."
    )

    optimal_sequence_length = OPTIMAL_HYPERPARAMS["sequence_length"]
    X_train_seq, y_train_seq_dict = create_sequences_multi_output(
        X_train, y_train, optimal_sequence_length
    )

    logger.info(f"Menggunakan sequence length optimal: {optimal_sequence_length}")
    logger.info(f"Data sequences shape: {X_train_seq.shape}")
    for target, y_seq in y_train_seq_dict.items():
        logger.info(f"Target {target} shape: {y_seq.shape}")

    # --- Langkah 3: Latih TCN Multi-Head dengan Parameter Optimal ---
    logger.info("Langkah 4.3: Melatih TCN Multi-Head dengan Parameter Optimal...")

    # Buat TCN model dengan multi-head output
    final_input_shape = (optimal_sequence_length, X_train.shape[1])
    tcn_model = create_tcn_model_multi_head(
        final_input_shape,
        OPTIMAL_HYPERPARAMS["tcn_params"],
        OPTIMAL_HYPERPARAMS["multi_output_config"],
    )

    # Latih model final dengan multi-output
    logger.info("Melatih TCN model multi-head dengan seluruh data...")

    # Prepare training params
    training_params = OPTIMAL_HYPERPARAMS["training_params"]
    callbacks = [
        EarlyStopping(
            monitor=training_params["early_stopping"]["monitor"],
            patience=training_params["early_stopping"]["patience"],
            restore_best_weights=True,
        )
    ]

    # Add learning rate scheduler if specified
    y_train_seq_dict_fixed = {}
    for target in y_train_seq_dict.keys():
        # Convert FCR_NEXT_1D -> fcr_next_1d, FCR_NEXT_3D -> fcr_next_3d
        model_key = target.lower()
        y_train_seq_dict_fixed[model_key] = y_train_seq_dict[target]
        logger.info(
            f"Mapped {target} -> {model_key}, dtype: {y_train_seq_dict[target].dtype}"
        )

    # Use the fixed dictionary for training
    history = tcn_model.fit(
        X_train_seq,
        y_train_seq_dict_fixed,  # Use the fixed dictionary
        epochs=150,
        batch_size=OPTIMAL_HYPERPARAMS["batch_size"],
        verbose=1,
        callbacks=callbacks,
    )

    # Buat TCN encoder (tanpa output heads)
    # Ambil layer sebelum output heads
    shared_layers = [
        layer for layer in tcn_model.layers if not layer.name.startswith("fcr_next_")
    ]
    tcn_encoder_output = None
    for layer in shared_layers:
        if "dense" not in layer.name.lower():
            tcn_encoder_output = layer.output

    if tcn_encoder_output is None:
        # Fallback: ambil output dari TCN layer
        for layer in tcn_model.layers:
            if hasattr(layer, "nb_filters"):
                tcn_encoder_output = layer.output
                break

    tcn_encoder = Model(inputs=tcn_model.input, outputs=tcn_encoder_output)
    tcn_encoder.save(tcn_encoder_path)
    logger.info(f"TCN Encoder berhasil dilatih dan disimpan di: {tcn_encoder_path}")

    # --- Langkah 4: Persiapan Data untuk XGBoost Multi-Output ---
    logger.info("Langkah 4.4: Mempersiapkan Data untuk XGBoost Multi-Output...")

    # Buat sequences untuk XGBoost
    X_train_xgb_seq, y_train_xgb_dict = create_sequences_multi_output(
        X_train, y_train, optimal_sequence_length
    )

    # Ekstrak fitur temporal dari TCN encoder
    temporal_features_train = tcn_encoder.predict(X_train_xgb_seq)

    # Gabungkan fitur tabular dan temporal
    tabular_features_train = X_train.iloc[optimal_sequence_length:].values
    X_train_hybrid = np.concatenate(
        [tabular_features_train, temporal_features_train], axis=1
    )

    # --- Langkah 5: Latih XGBoost untuk setiap horizon ---
    logger.info("Langkah 4.5: Melatih XGBoost untuk setiap horizon...")

    xgb_models = {}
    horizons = OPTIMAL_HYPERPARAMS["multi_output_config"]["horizons"]

    for horizon in horizons:
        target_key = f"FCR_NEXT_{horizon}D"
        logger.info(f"Melatih XGBoost untuk horizon {horizon} hari...")

        # Salin parameter dan ganti objective dengan fungsi loss kustom
        # untuk mengatasi over-prediction.
        xgb_params_custom = OPTIMAL_HYPERPARAMS["xgb_params"].copy()
        xgb_params_custom["objective"] = asymmetric_objective

        xgb_model = xgb.XGBRegressor(**xgb_params_custom)
        xgb_model.fit(X_train_hybrid, y_train_xgb_dict[target_key], verbose=True)

        xgb_models[f"horizon_{horizon}d"] = xgb_model

        # Simpan model XGBoost untuk setiap horizon
        horizon_model_path = xgb_model_path.replace(".pkl", f"_horizon_{horizon}d.pkl")
        with open(horizon_model_path, "wb") as f:
            pickle.dump(xgb_model, f)
        logger.info(
            f"Model XGBoost Horizon {horizon}d disimpan di: {horizon_model_path}"
        )

    # --- Langkah 6: Simpan Konfigurasi Multi-Output ---
    logger.info("Langkah 4.6: Menyimpan Konfigurasi Model Multi-Output...")

    # Simpan konfigurasi lengkap
    hybrid_config = {
        "sequence_length": optimal_sequence_length,
        "batch_size": OPTIMAL_HYPERPARAMS["batch_size"],
        "multi_output_config": OPTIMAL_HYPERPARAMS["multi_output_config"],
        "tcn_params": OPTIMAL_HYPERPARAMS["tcn_params"],
        "xgb_params": OPTIMAL_HYPERPARAMS["xgb_params"],
        "training_params": OPTIMAL_HYPERPARAMS["training_params"],
        "feature_names": features,
        "target_names": TARGETS,
        "horizons": horizons,
        "training_data_shape": X_train.shape,
        "sequences_shape": X_train_seq.shape,
        "hybrid_features_shape": X_train_hybrid.shape,
        "model_type": "TCN-XGBoost Multi-Output Hybrid with Optimal Hyperparameters",
    }

    with open(hybrid_config_path, "w") as f:
        json.dump(hybrid_config, f, indent=4)
    logger.info(
        f"Konfigurasi model hibrida multi-output disimpan di: {hybrid_config_path}"
    )

    # --- Langkah 7: Evaluasi Model Multi-Output pada Data Training ---
    logger.info("Langkah 4.7: Evaluasi Model Multi-Output pada Data Training...")

    print(
        "\n--- Performa Model Hibrida TCN-XGBOOST Multi-Output pada Data Training ---"
    )

    for horizon in horizons:
        target_key = f"FCR_NEXT_{horizon}D"
        model_key = f"horizon_{horizon}d"

        train_preds = xgb_models[model_key].predict(X_train_hybrid)
        actual_values = y_train_xgb_dict[target_key]

        print(f"\n=== Horizon {horizon} Hari ===")
        print(f"MAE: {mean_absolute_error(actual_values, train_preds):.4f}")
        print(f"MSE: {mean_squared_error(actual_values, train_preds):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(actual_values, train_preds)):.4f}")
        print(f"MAPE: {mean_absolute_percentage_error(actual_values, train_preds):.4f}")
        print(f"R-squared: {r2_score(actual_values, train_preds):.4f}")

    print("-" * 50)

    print("\n--- Hyperparameter yang Digunakan ---")
    print(f"Sequence Length: {optimal_sequence_length}")
    print(f"Batch Size: {OPTIMAL_HYPERPARAMS['batch_size']}")
    print(f"TCN Filters: {OPTIMAL_HYPERPARAMS['tcn_params']['nb_filters']}")
    print(f"TCN Kernel Size: {OPTIMAL_HYPERPARAMS['tcn_params']['kernel_size']}")
    print(f"TCN Stacks: {OPTIMAL_HYPERPARAMS['tcn_params']['nb_stacks']}")
    print(f"XGBoost Estimators: {OPTIMAL_HYPERPARAMS['xgb_params']['n_estimators']}")
    print(f"XGBoost Max Depth: {OPTIMAL_HYPERPARAMS['xgb_params']['max_depth']}")
    print("-" * 50)

    # Plot hasil training untuk horizon pertama sebagai contoh
    horizon_1_target = f"FCR_NEXT_{horizons[0]}D"
    train_preds_sample = xgb_models[f"horizon_{horizons[0]}d"].predict(X_train_hybrid)
    actual_values_sample = y_train_xgb_dict[horizon_1_target]

    plt.figure(figsize=(15, 7))
    plt.plot(
        actual_values_sample[:100], label="Aktual", marker="o", markersize=3
    )  # Hanya 100 titik pertama untuk visualisasi
    plt.plot(
        train_preds_sample[:100],
        label=f"Prediksi Hibrida Horizon {horizons[0]}D (Optimal Hyperparams)",
        linestyle="--",
        marker="s",
        markersize=3,
    )
    plt.title(
        f"Performa Model Hibrida TCN-XGBoost Horizon {horizons[0]}D dengan Hyperparameter Optimal (100 titik pertama)"
    )
    plt.xlabel("Sampel")
    plt.ylabel("FCR Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot scatter untuk melihat korelasi prediksi vs aktual
    plt.figure(figsize=(10, 8))
    plt.scatter(actual_values_sample, train_preds_sample, alpha=0.6)
    plt.plot(
        [actual_values_sample.min(), actual_values_sample.max()],
        [actual_values_sample.min(), actual_values_sample.max()],
        "r--",
        lw=2,
    )
    plt.xlabel("Nilai Aktual")
    plt.ylabel("Nilai Prediksi")
    plt.title(
        f"Scatter Plot: Prediksi vs Aktual Horizon {horizons[0]}D (Data Training)"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(
        f"\nModel berhasil dilatih menggunakan {len(actual_values_sample)} sampel data!"
    )
    print("CATATAN: Model dilatih dengan seluruh data yang tersedia.")
    print("Model menggunakan hyperparameter yang sudah dioptimasi sebelumnya.")

    return True


# =============================================================================
# BLOK EKSEKUSI UTAMA
# =============================================================================
if __name__ == "__main__":
    base_folder_path = os.getcwd()

    # Definisi path
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

    # Nama file untuk setiap tahap
    merged_file = os.path.join(intermediate_data_folder, "File_Gabungan.csv")
    enhanced_file = os.path.join(
        intermediate_data_folder, "File_Gabungan_Enhanced_Features.csv"
    )
    normalized_file = os.path.join(
        final_output_folder, "Data_Gabungan_Fitur_Rekayasa_Normalized.csv"
    )

    # Path untuk menyimpan semua model (.pkl) dan info (.json)
    fe_pipeline_file = os.path.join(models_folder, "feature_engineering_pipeline.pkl")
    robust_scaler_file = os.path.join(models_folder, "robust_scaler.pkl")
    feature_metadata_file = os.path.join(final_output_folder, "feature_metadata.json")
    norm_info_file = os.path.join(final_output_folder, "normalization_info.json")

    # Path untuk model Hibrida
    tcn_encoder_model_path = os.path.join(models_folder, "tcn_encoder.keras")
    xgb_hybrid_model_path = os.path.join(models_folder, "xgboost_hybrid_model.pkl")
    hybrid_config_file_path = os.path.join(models_folder, "hybrid_model_config.json")

    try:
        # Menjalankan pipeline secara berurutan
        # Langkah 1: Gabungkan data mentah
        if not merge_data_files(folder_path=data_input_folder, output_file=merged_file):
            raise Exception("Gagal pada langkah penggabungan data.")

        # Langkah 2: Lakukan rekayasa fitur dan simpan model pipeline-nya
        if not run_feature_engineering_step(
            input_file=merged_file,
            output_data_path=enhanced_file,
            output_pipeline_path=fe_pipeline_file,
            metadata_path=final_output_folder,
            config=FEATURE_CONFIG,
        ):
            raise Exception("Gagal pada langkah rekayasa fitur.")

        # Langkah 3: Lakukan normalisasi dan simpan model scaler-nya
        if not normalize_features(
            input_file=enhanced_file,
            output_file=normalized_file,
            info_file=norm_info_file,
            robust_scaler_path=robust_scaler_file,
        ):
            raise Exception("Gagal pada langkah normalisasi.")

        # Langkah 4: Pelatihan Model Hibrida dengan Seluruh Data
        if not train_tcn_xgboost_hybrid(
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
