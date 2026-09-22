# -*- coding: utf-8 -*-
"""
Skrip pipeline lengkap untuk:
1. Menggabungkan data dari beberapa file CSV periode.
2. Melakukan rekayasa fitur (feature engineering) canggih pada data yang digabungkan.
3. Melakukan normalisasi fitur dengan metode campuran (MinMax dan Robust Scaler).

Alur Kerja:
- Langkah 1: Penggabungan data mentah.
- Langkah 2: Rekayasa fitur dari data gabungan.
- Langkah 3: Normalisasi fitur yang telah direkayasa.
"""

import pandas as pd
import numpy as np
import os
import warnings
import logging
import json
from typing import List, Dict, Tuple
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, RobustScaler

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
    "health_weights": {"mati": 0.5, "afkir": 0.3, "deplesi": 0.2},
    "variance_threshold": 0.01,
    "correlation_threshold": 0.95,
    "decimal_places": 3,
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
# LANGKAH 2: FUNGSI REKAYASA FITUR (FEATURE ENGINEERING)
# =============================================================================


class FeatureEngineer:
    """
    Kelas untuk rekayasa fitur canggih pada data peternakan ayam broiler.
    """

    def __init__(self, config: Dict = None):
        self.config = config or FEATURE_CONFIG
        self.feature_names = {
            "lagged": [],
            "rolling": [],
            "interaction": [],
            "trend": [],
            "domain": [],
        }

    def safe_numeric_conversion(self, df: pd.DataFrame, column: str) -> pd.Series:
        try:
            if df[column].dtype == "object":
                return pd.to_numeric(
                    df[column].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
            return df[column]
        except Exception as e:
            logger.warning(f"Gagal mengonversi kolom {column}: {e}")
            return df[column]

    # ... (Semua metode create_*_features, handle_missing_values, dll. dari skrip sebelumnya tetap sama) ...
    def create_lagged_features(
        self, df: pd.DataFrame, target_columns: List[str], lag_periods: List[int] = None
    ) -> pd.DataFrame:
        lag_periods = lag_periods or self.config["lag_periods"]
        df_result = df.copy()
        logger.info(f"Membuat fitur lagged untuk {len(target_columns)} kolom")
        df_result = df_result.sort_values(["PERIODE", "AGE"]).reset_index(drop=True)
        for column in target_columns:
            if column not in df_result.columns:
                continue
            for lag in lag_periods:
                lag_column_name = f"{column}_lag_{lag}"
                try:
                    df_result[lag_column_name] = df_result.groupby("PERIODE")[
                        column
                    ].shift(lag)
                    self.feature_names["lagged"].append(lag_column_name)
                except Exception as e:
                    logger.error(f"Error membuat fitur lag {lag_column_name}: {e}")
        return df_result

    def create_rolling_features(
        self, df: pd.DataFrame, target_columns: List[str], windows: List[int] = None
    ) -> pd.DataFrame:
        windows = windows or self.config["rolling_windows"]
        df_result = df.copy()
        logger.info(f"Membuat fitur rolling untuk {len(target_columns)} kolom")
        df_result = df_result.sort_values(["PERIODE", "AGE"]).reset_index(drop=True)
        for column in target_columns:
            if column not in df_result.columns:
                continue
            for window in windows:
                try:
                    grouped = df_result.groupby("PERIODE")[column]
                    rolling_mean = (
                        grouped.rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    rolling_std = (
                        grouped.rolling(window=window, min_periods=1)
                        .std()
                        .reset_index(level=0, drop=True)
                    )
                    rolling_min = (
                        grouped.rolling(window=window, min_periods=1)
                        .min()
                        .reset_index(level=0, drop=True)
                    )
                    rolling_max = (
                        grouped.rolling(window=window, min_periods=1)
                        .max()
                        .reset_index(level=0, drop=True)
                    )
                    df_result[f"{column}_rolling_mean_{window}"] = rolling_mean
                    df_result[f"{column}_rolling_std_{window}"] = rolling_std
                    df_result[f"{column}_rolling_min_{window}"] = rolling_min
                    df_result[f"{column}_rolling_max_{window}"] = rolling_max
                    self.feature_names["rolling"].extend(
                        [
                            f"{column}_rolling_mean_{window}",
                            f"{column}_rolling_std_{window}",
                            f"{column}_rolling_min_{window}",
                            f"{column}_rolling_max_{window}",
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"Error membuat fitur rolling untuk {column}, window {window}: {e}"
                    )
        return df_result

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        logger.info("Membuat fitur interaksi")
        if all(col in df_result.columns for col in ["PAKAN_ACT_GR/EK", "DG_ACT"]):
            df_result["PAKAN_DG_RATIO"] = df_result["PAKAN_ACT_GR/EK"] / (
                df_result["DG_ACT"] + 1e-8
            )
            self.feature_names["interaction"].append("PAKAN_DG_RATIO")
        if all(col in df_result.columns for col in ["SUHU_MAX", "KELEMBABAN"]):
            df_result["SUHU_KELEMBABAN_INTERACTION"] = (
                df_result["SUHU_MAX"] * df_result["KELEMBABAN"]
            )
            self.feature_names["interaction"].append("SUHU_KELEMBABAN_INTERACTION")
        if all(
            col in df_result.columns for col in ["SUHU_MAX", "SUHU_MIN", "KELEMBABAN"]
        ):
            df_result["STRESS_INDEX"] = (
                df_result["SUHU_MAX"] - df_result["SUHU_MIN"]
            ) * df_result["KELEMBABAN"]
            self.feature_names["interaction"].append("STRESS_INDEX")
        if "AGE" in df_result.columns:
            df_result["AGE_SQUARED"] = df_result["AGE"] ** 2
            df_result["AGE_CUBED"] = df_result["AGE"] ** 3
            self.feature_names["interaction"].extend(["AGE_SQUARED", "AGE_CUBED"])
        if all(
            col in df_result.columns for col in ["SUHU_MAX", "SUHU_MIN", "KELEMBABAN"]
        ):
            temp_avg = (df_result["SUHU_MAX"] + df_result["SUHU_MIN"]) / 2
            temp_deviation = np.abs(temp_avg - 22)
            humidity_factor = np.where(
                (df_result["KELEMBABAN"] >= 60) & (df_result["KELEMBABAN"] <= 70),
                1.0,
                0.8,
            )
            df_result["COMFORT_INDEX"] = humidity_factor / (1 + temp_deviation * 0.1)
            self.feature_names["interaction"].append("COMFORT_INDEX")
        return df_result

    def create_trend_features(
        self, df: pd.DataFrame, target_columns: List[str]
    ) -> pd.DataFrame:
        df_result = df.copy()
        logger.info(f"Membuat fitur tren untuk {len(target_columns)} kolom")
        df_result = df_result.sort_values(["PERIODE", "AGE"]).reset_index(drop=True)
        for column in target_columns:
            if column not in df_result.columns or column == "FCR_ACT":
                continue
            try:
                grouped = df_result.groupby("PERIODE")[column]
                first_diff = grouped.diff(1)
                df_result[f"{column}_momentum_1"] = first_diff
                self.feature_names["trend"].append(f"{column}_momentum_1")
                second_diff = first_diff.groupby(df_result["PERIODE"]).diff(1)
                df_result[f"{column}_acceleration"] = second_diff
                self.feature_names["trend"].append(f"{column}_acceleration")
                volatility = (
                    first_diff.groupby(df_result["PERIODE"])
                    .rolling(window=3, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
                df_result[f"{column}_volatility_3"] = volatility
                self.feature_names["trend"].append(f"{column}_volatility_3")
                slope_3 = (
                    grouped.rolling(window=3, min_periods=3)
                    .apply(
                        lambda x: (
                            np.polyfit(range(len(x)), x, 1)[0]
                            if len(x) == 3
                            else np.nan
                        )
                    )
                    .reset_index(level=0, drop=True)
                )
                df_result[f"{column}_trend_slope_3"] = slope_3
                self.feature_names["trend"].append(f"{column}_trend_slope_3")
            except Exception as e:
                logger.error(f"Error membuat fitur tren untuk {column}: {e}")
        return df_result

    def create_domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        logger.info("Membuat fitur domain-spesifik")
        health_cols = ["MATI", "AFKIR", "JML_DEPLESI"]
        if all(col in df_result.columns for col in health_cols):
            weights = self.config["health_weights"]
            df_result["HEALTH_STATUS_SCORE"] = (
                weights["mati"] * df_result["MATI"]
                + weights["afkir"] * df_result["AFKIR"]
                + weights["deplesi"] * df_result["JML_DEPLESI"]
            )
            self.feature_names["domain"].append("HEALTH_STATUS_SCORE")
        if all(
            col in df_result.columns
            for col in ["DG_ACT_rolling_std_7", "DG_ACT_rolling_mean_7"]
        ):
            df_result["GROWTH_CONSISTENCY_INDEX"] = df_result[
                "DG_ACT_rolling_std_7"
            ] / (df_result["DG_ACT_rolling_mean_7"] + 1e-8)
            self.feature_names["domain"].append("GROWTH_CONSISTENCY_INDEX")
        if "FCR_ACT_lag_1" in df_result.columns:
            df_result["FEED_EFFICIENCY_lag_1"] = 1.0 / (
                df_result["FCR_ACT_lag_1"] + 1e-8
            )
            self.feature_names["domain"].append("FEED_EFFICIENCY_lag_1")
            if all(
                col in df_result.columns
                for col in ["SUHU_MAX", "SUHU_MIN", "KELEMBABAN"]
            ):
                temp_factor = 1 - 0.05 * (df_result["SUHU_MAX"] - df_result["SUHU_MIN"])
                humidity_factor = 1 + 0.02 * (df_result["KELEMBABAN"] - 65) / 65
                df_result["FCR_EFFICIENCY_SCORE_lag_1"] = (
                    df_result["FEED_EFFICIENCY_lag_1"] * temp_factor * humidity_factor
                )
                self.feature_names["domain"].append("FCR_EFFICIENCY_SCORE_lag_1")
        if "AGE" in df_result.columns:
            df_result["GROWTH_STAGE"] = pd.cut(
                df_result["AGE"],
                bins=[0, 7, 14, 21, 28, 35, float("inf")],
                labels=[1, 2, 3, 4, 5, 6],
                include_lowest=True,
            ).astype(float)
            self.feature_names["domain"].append("GROWTH_STAGE")
        if "PERIODE" in df_result.columns:
            try:
                # This part might not be effective if PERIODE is just an integer ID
                # If PERIODE was a date, it would work. We keep it for potential future use.
                pass
            except Exception as e:
                logger.warning(f"Tidak dapat membuat fitur musiman: {e}")
        return df_result

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        logger.info("Menangani nilai yang hilang (missing values)")
        for periode in df_result["PERIODE"].unique():
            periode_mask = df_result["PERIODE"] == periode
            df_result.loc[periode_mask] = df_result.loc[periode_mask].ffill()
            df_result.loc[periode_mask] = df_result.loc[periode_mask].bfill()
        numeric_columns = df_result.select_dtypes(include=[np.number]).columns
        columns_to_fill = [
            col for col in numeric_columns if col not in ["PERIODE", "AGE"]
        ]
        for column in columns_to_fill:
            for periode in df_result["PERIODE"].unique():
                periode_mask = df_result["PERIODE"] == periode
                if df_result.loc[periode_mask, column].isnull().any():
                    median_val = df_result.loc[periode_mask, column].median()
                    if pd.isna(median_val):
                        median_val = df_result[column].median()
                    df_result.loc[periode_mask, column] = df_result.loc[
                        periode_mask, column
                    ].fillna(median_val)
        return df_result

    def remove_low_variance_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [
            col for col in numeric_columns if col not in ["PERIODE", "AGE", "FCR_ACT"]
        ]
        if not feature_columns:
            return df, []
        selector = VarianceThreshold(threshold=self.config["variance_threshold"])
        feature_data = df[feature_columns].fillna(0)
        try:
            selector.fit(feature_data)
            selected_features = [
                feature_columns[i]
                for i in range(len(feature_columns))
                if selector.variances_[i] > self.config["variance_threshold"]
            ]
            removed_features = [
                col for col in feature_columns if col not in selected_features
            ]
            logger.info(
                f"Menghapus {len(removed_features)} fitur dengan varians rendah"
            )
            keep_columns = [
                col for col in df.columns if col not in feature_columns
            ] + selected_features
            return df[keep_columns], removed_features
        except Exception as e:
            logger.error(f"Error pada seleksi ambang batas varians: {e}")
            return df, []

    def remove_highly_correlated_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [
            col for col in numeric_columns if col not in ["PERIODE", "AGE", "FCR_ACT"]
        ]
        if not feature_columns:
            return df, []
        try:
            corr_matrix = df[feature_columns].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_remove = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > self.config["correlation_threshold"])
            ]
            logger.info(f"Menghapus {len(to_remove)} fitur dengan korelasi tinggi")
            keep_columns = [col for col in df.columns if col not in to_remove]
            return df[keep_columns], to_remove
        except Exception as e:
            logger.error(f"Error pada analisis korelasi: {e}")
            return df, []

    def save_feature_metadata(
        self, output_dir: str, removed_features: Dict[str, List[str]]
    ):
        metadata = {
            "config": self.config,
            "created_features": self.feature_names,
            "removed_features": removed_features,
        }
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        with open(output_path / "feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(
            f"Metadata fitur disimpan di {output_path / 'feature_metadata.json'}"
        )


def advanced_feature_engineering(
    input_file: str, output_file: str, config: Dict = None
) -> bool:
    """
    Fungsi utama untuk menjalankan pipeline rekayasa fitur canggih.
    """
    print("\n" + "=" * 80)
    print("MEMULAI LANGKAH 2: REKAYASA FITUR (FEATURE ENGINEERING)")
    print("=" * 80)

    config = config or FEATURE_CONFIG
    engineer = FeatureEngineer(config)

    if not os.path.exists(input_file):
        logger.error(
            f"File input '{input_file}' tidak ditemukan. Langkah 2 dihentikan."
        )
        return False

    df = pd.read_csv(input_file, sep=";", decimal=".")
    for col in df.columns:
        df[col] = engineer.safe_numeric_conversion(df, col)

    base_cols = [
        "PAKAN_ACT_GR/EK",
        "DG_ACT",
        "SUHU_MAX",
        "SUHU_MIN",
        "KELEMBABAN",
        "MATI",
        "AFKIR",
        "JML_DEPLESI",
        "ABW_ACT",
    ]
    delta_cols = ["DELTA_FCR", "DELTA_PAKAN_GR/EKOR", "DELTA_DG"]

    lagged_columns = ["FCR_ACT"] + base_cols + delta_cols
    rolling_columns = base_cols + delta_cols
    trend_columns = base_cols + delta_cols

    lagged_columns = [col for col in lagged_columns if col in df.columns]
    rolling_columns = [col for col in rolling_columns if col in df.columns]
    trend_columns = [col for col in trend_columns if col in df.columns]

    df_enhanced = engineer.create_lagged_features(df, lagged_columns)
    df_enhanced = engineer.create_rolling_features(df_enhanced, rolling_columns)
    df_enhanced = engineer.create_interaction_features(df_enhanced)
    df_enhanced = engineer.create_trend_features(df_enhanced, trend_columns)
    df_enhanced = engineer.create_domain_specific_features(df_enhanced)
    df_enhanced = engineer.handle_missing_values(df_enhanced)

    removed_features = {}
    df_enhanced, low_var_removed = engineer.remove_low_variance_features(df_enhanced)
    removed_features["low_variance"] = low_var_removed
    df_enhanced, corr_removed = engineer.remove_highly_correlated_features(df_enhanced)
    removed_features["high_correlation"] = corr_removed

    numeric_columns = df_enhanced.select_dtypes(include=[np.number]).columns
    columns_to_round = [col for col in numeric_columns if col not in ["PERIODE", "AGE"]]
    for col in columns_to_round:
        df_enhanced[col] = df_enhanced[col].round(config["decimal_places"])

    df_enhanced.to_csv(output_file, sep=";", decimal=".", index=False)
    engineer.save_feature_metadata(str(Path(output_file).parent), removed_features)

    print(f"\nBERHASIL: Rekayasa fitur selesai. File disimpan sebagai: {output_file}")
    return True


# =============================================================================
# LANGKAH 3: FUNGSI NORMALISASI FITUR
# =============================================================================


def normalize_features(input_file: str, output_file: str, info_file: str):
    """
    Menerapkan strategi normalisasi metode campuran pada data.
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

    # Definisi fitur untuk setiap scaler
    minmax_features = []

    robust_features = []

    # Aturan untuk RobustScaler (jika nama mengandung keyword ini)
    robust_keywords = ["DELTA_", "_std_", "momentum", "acceleration", "volatility"]

    for feature in features_to_normalize:
        if any(keyword in feature for keyword in robust_keywords):
            robust_features.append(feature)
        else:
            minmax_features.append(feature)

    logger.info(f"Ditemukan {len(features_to_normalize)} fitur untuk dinormalisasi.")
    logger.info(f"Dialokasikan {len(minmax_features)} fitur ke MinMaxScaler.")
    logger.info(f"Dialokasikan {len(robust_features)} fitur ke RobustScaler.")

    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()

    df_normalized = df.copy()

    # 4. Terapkan normalisasi
    if minmax_features:
        minmax_data = df[minmax_features].fillna(df[minmax_features].median())
        df_normalized[minmax_features] = minmax_scaler.fit_transform(minmax_data)

    if robust_features:
        robust_data = df[robust_features].fillna(df[robust_features].median())
        df_normalized[robust_features] = robust_scaler.fit_transform(robust_data)

    # Simpan hasil
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_normalized.to_csv(output_file, index=False, sep=";")

    # Simpan informasi normalisasi
    normalization_info = {
        "minmax_features": minmax_features,
        "robust_features": robust_features,
        "non_normalized": [
            col for col in non_normalized_cols if col in df_normalized.columns
        ],
        "minmax_scaler_params": {
            "min_": minmax_scaler.min_.tolist(),
            "scale_": minmax_scaler.scale_.tolist(),
            "feature_names": minmax_features,
        },
        "robust_scaler_params": {
            "center_": robust_scaler.center_.tolist(),
            "scale_": robust_scaler.scale_.tolist(),
            "feature_names": robust_features,
        },
        "final_columns": df_normalized.columns.tolist(),
    }
    with open(info_file, "w") as f:
        json.dump(normalization_info, f, indent=2)

    print(f"\nBERHASIL: Normalisasi fitur selesai.")
    print(f" - Fitur dinormalisasi dengan MinMax Scaler: {len(minmax_features)}")
    print(f" - Fitur dinormalisasi dengan Robust Scaler: {len(robust_features)}")
    print(f" - File data ternormalisasi disimpan di: {output_file}")
    print(f" - Info normalisasi disimpan di: {info_file}")
    return True


# =============================================================================
# BLOK EKSEKUSI UTAMA
# =============================================================================

if __name__ == "__main__":
    # --- Konfigurasi Path ---
    base_folder_path = os.getcwd()  # Menggunakan direktori kerja saat ini

    # Path untuk data mentah (input langkah 1)
    data_input_folder = os.path.join(base_folder_path, "non-normalize")

    # Path untuk data antara (setelah digabung dan direkayasa)
    intermediate_data_folder = os.path.join(
        base_folder_path, "Normalisasi", "Data Copy"
    )

    # Path untuk hasil akhir (setelah normalisasi)
    final_output_folder = os.path.join(base_folder_path, "Hasil Normalisasi copy")

    # Definisi nama file
    merged_file = os.path.join(intermediate_data_folder, "File_Gabungan.csv")
    enhanced_file = os.path.join(
        intermediate_data_folder, "File_Gabungan_Enhanced_Features.csv"
    )
    normalized_file = os.path.join(
        final_output_folder, "Data_Gabungan_Fitur_Rekayasa_Normalized.csv"
    )
    norm_info_file = os.path.join(final_output_folder, "normalization_info.json")

    # --- Menjalankan Pipeline ---
    try:
        # LANGKAH 1: Menggabungkan data
        if not merge_data_files(folder_path=data_input_folder, output_file=merged_file):
            raise Exception("Gagal pada langkah penggabungan data.")

        # LANGKAH 2: Rekayasa fitur
        custom_config = FEATURE_CONFIG.copy()
        custom_config["lag_periods"] = [1, 2, 3, 7, 14, 21]
        custom_config["rolling_windows"] = [3, 7, 14, 21]
        if not advanced_feature_engineering(
            input_file=merged_file, output_file=enhanced_file, config=custom_config
        ):
            raise Exception("Gagal pada langkah rekayasa fitur.")

        # LANGKAH 3: Normalisasi fitur
        if not normalize_features(
            input_file=enhanced_file,
            output_file=normalized_file,
            info_file=norm_info_file,
        ):
            raise Exception("Gagal pada langkah normalisasi fitur.")

        print("\n\n>>> SEMUA LANGKAH PIPELINE SELESAI DENGAN SUKSES! <<<")
        print(f"Output akhir (data siap latih) tersedia di: {normalized_file}")

    except Exception as e:
        logger.error(f"Terjadi kesalahan fatal pada pipeline: {e}", exc_info=False)
