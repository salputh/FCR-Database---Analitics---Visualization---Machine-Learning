# -*- coding: utf-8 -*-
"""
Pipeline Terpadu: Data Processing, Model Training, dan XAI Managerial Recommendations

Pipeline ini mengintegrasikan:
1. Penggabungan data dari multiple CSV files
2. Feature engineering dengan FeatureEngineeringPipeline
3. Normalisasi fitur dengan metode campuran (MinMax dan Robust Scaler)
4. Pelatihan model hibrida TCN-XGBoost dengan optimasi hyperparameter (Refactored)
5. Sistem rekomendasi manajerial berbasis XAI (SHAP)
6. Analisis XAI komprehensif dan laporan harian

Author: AI Assistant
Date: 2024 (Refactored: 2025)
"""

import pandas as pd
import numpy as np
import os
import warnings
import logging
import json
import pickle
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

# Impor untuk model dan XAI
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tcn import TCN
    import xgboost as xgb
    import optuna
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except ImportError as e:
    print(f"Error impor: {e}")
    print("Harap pasang library yang diperlukan")
    exit()

try:
    import sys

    sys.path.append("1. Normalisasi")
    from Feature.FeatureEngineeringPipeline import FeatureEngineeringPipeline
except ImportError:
    print("ERROR: File 'FeatureEngineeringPipeline.py' tidak ditemukan.")
    exit()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# =============================================================================
# KONFIGURASI SISTEM
# =============================================================================
class UnifiedPipelineConfig:
    """Konfigurasi untuk pipeline terpadu"""

    # (Konten Konfigurasi tidak berubah, dibiarkan sama seperti aslinya)
    # ...
    # Konfigurasi Feature Engineering
    FEATURE_CONFIG = {
        "lag_periods": [1, 2, 3, 7, 14],
        "rolling_windows": [3, 7, 14],
        "health_weights": {"mati": 0.5, "afkir": 0.3, "deplesi": 0.2},
        "variance_threshold": 0.05,
        "correlation_threshold": 0.95,
        "decimal_places": 3,
    }

    # Threshold FCR untuk klasifikasi kondisi
    FCR_THRESHOLDS = {"excellent": 1.4, "good": 1.6, "warning": 1.8, "critical": 2.0}

    # Threshold SHAP values untuk menentukan signifikansi fitur
    SHAP_SIGNIFICANCE_THRESHOLD = 0.05

    # Mapping fitur ke kategori manajerial
    FEATURE_CATEGORIES = {
        "performance": [
            "DG_ACT",
            "DELTA_DG",
            "DELTA_ABW",
            "FCR_ACT",
            "DELTA_FCR",
            "DG_ACT_lag_1",
            "DG_ACT_lag_2",
            "DG_ACT_lag_3",
            "DG_ACT_rolling_std_3",
            "DG_ACT_rolling_std_7",
            "DG_ACT_rolling_std_14",
            "DG_ACT_rolling_min_14",
            "DG_ACT_momentum_1",
            "DG_ACT_acceleration",
            "DG_ACT_volatility_3",
            "DG_ACT_trend_slope_3",
        ],
        "consumption": [
            "DELTA_PAKAN_ZAK",
            "DELTA_PAKAN_CUM_GR/EKOK",
            "PAKAN_DG_RATIO",
            "DELTA_PAKAN_GR/EKOR_lag_1",
            "DELTA_PAKAN_GR/EKOR_lag_2",
            "DELTA_PAKAN_GR/EKOR_lag_3",
            "DELTA_PAKAN_GR/EKOR_lag_7",
            "DELTA_PAKAN_GR/EKOR_lag_14",
            "PAKAN_ACT_GR/EK_rolling_std_3",
            "PAKAN_ACT_GR/EK_rolling_std_7",
            "PAKAN_ACT_GR/EK_rolling_std_14",
            "DELTA_PAKAN_GR/EKOR_rolling_std_3",
            "DELTA_PAKAN_GR/EKOR_rolling_max_3",
            "DELTA_PAKAN_GR/EKOR_rolling_mean_7",
            "DELTA_PAKAN_GR/EKOR_rolling_std_7",
            "DELTA_PAKAN_GR/EKOR_rolling_max_7",
            "DELTA_PAKAN_GR/EKOR_rolling_mean_14",
            "DELTA_PAKAN_GR/EKOR_rolling_std_14",
            "DELTA_PAKAN_GR/EKOR_rolling_max_14",
            "PAKAN_ACT_GR/EK_momentum_1",
            "PAKAN_ACT_GR/EK_acceleration",
            "PAKAN_ACT_GR/EK_volatility_3",
            "PAKAN_ACT_GR/EK_trend_slope_3",
            "DELTA_PAKAN_GR/EKOR_momentum_1",
            "DELTA_PAKAN_GR/EKOR_acceleration",
            "DELTA_PAKAN_GR/EKOR_volatility_3",
            "DELTA_PAKAN_GR/EKOR_trend_slope_3",
        ],
        "health": [
            "MATI",
            "JML_DEPLESI",
            "STD_CUM_DEPLESI_%",
            "MATI_lag_1",
            "MATI_lag_2",
            "MATI_lag_3",
            "MATI_lag_7",
            "MATI_lag_14",
            "JML_DEPLESI_lag_14",
            "MATI_rolling_mean_3",
            "MATI_rolling_std_3",
            "MATI_rolling_min_3",
            "MATI_rolling_max_3",
            "MATI_rolling_mean_7",
            "MATI_rolling_std_7",
            "MATI_rolling_min_7",
            "MATI_rolling_max_7",
            "MATI_rolling_mean_14",
            "MATI_rolling_std_14",
            "MATI_rolling_min_14",
            "MATI_rolling_max_14",
            "JML_DEPLESI_rolling_std_3",
            "JML_DEPLESI_rolling_std_7",
            "JML_DEPLESI_rolling_std_14",
            "MATI_momentum_1",
            "MATI_acceleration",
            "MATI_volatility_3",
            "MATI_trend_slope_3",
            "JML_DEPLESI_acceleration",
            "JML_DEPLESI_volatility_3",
        ],
        "stability": [
            "PAKAN_ACT_GR/EK_rolling_std_3",
            "PAKAN_ACT_GR/EK_rolling_std_7",
            "PAKAN_ACT_GR/EK_rolling_std_14",
            "DG_ACT_rolling_std_3",
            "DG_ACT_rolling_std_7",
            "DG_ACT_rolling_std_14",
            "SUHU_MAX_rolling_std_3",
            "SUHU_MAX_rolling_std_7",
            "SUHU_MAX_rolling_std_14",
            "MATI_rolling_std_3",
            "MATI_rolling_std_7",
            "MATI_rolling_std_14",
            "JML_DEPLESI_rolling_std_3",
            "JML_DEPLESI_rolling_std_7",
            "JML_DEPLESI_rolling_std_14",
            "DELTA_PAKAN_GR/EKOR_rolling_std_3",
            "DELTA_PAKAN_GR/EKOR_rolling_std_7",
            "DELTA_PAKAN_GR/EKOR_rolling_std_14",
            "PAKAN_ACT_GR/EK_volatility_3",
            "DG_ACT_volatility_3",
            "SUHU_MAX_volatility_3",
            "MATI_volatility_3",
            "JML_DEPLESI_volatility_3",
            "DELTA_PAKAN_GR/EKOR_volatility_3",
        ],
        "trends": [
            "DELTA_DG_rolling_mean_3",
            "DELTA_DG_rolling_mean_7",
            "DELTA_DG_rolling_mean_14",
            "DELTA_DG_rolling_min_3",
            "DELTA_DG_rolling_max_3",
            "DELTA_DG_rolling_std_7",
            "DELTA_DG_rolling_min_7",
            "DELTA_DG_rolling_max_7",
            "DELTA_DG_rolling_min_14",
            "DELTA_PAKAN_GR/EKOR_rolling_mean_7",
            "DELTA_PAKAN_GR/EKOR_rolling_mean_14",
            "PAKAN_ACT_GR/EK_trend_slope_3",
            "DG_ACT_trend_slope_3",
            "MATI_trend_slope_3",
            "DELTA_PAKAN_GR/EKOR_trend_slope_3",
            "PAKAN_ACT_GR/EK_momentum_1",
            "PAKAN_ACT_GR/EK_acceleration",
            "DG_ACT_momentum_1",
            "DG_ACT_acceleration",
            "SUHU_MAX_momentum_1",
            "SUHU_MAX_acceleration",
            "MATI_momentum_1",
            "MATI_acceleration",
            "JML_DEPLESI_acceleration",
            "DELTA_PAKAN_GR/EKOR_momentum_1",
            "DELTA_PAKAN_GR/EKOR_acceleration",
        ],
        "temporal": [
            "FCR_ACT_lag_14",
            "DG_ACT_lag_1",
            "DG_ACT_lag_2",
            "DG_ACT_lag_3",
            "MATI_lag_1",
            "MATI_lag_2",
            "MATI_lag_3",
            "MATI_lag_7",
            "MATI_lag_14",
            "JML_DEPLESI_lag_14",
            "DELTA_PAKAN_GR/EKOR_lag_1",
            "DELTA_PAKAN_GR/EKOR_lag_2",
            "DELTA_PAKAN_GR/EKOR_lag_3",
            "DELTA_PAKAN_GR/EKOR_lag_7",
            "DELTA_PAKAN_GR/EKOR_lag_14",
            "DELTA_DG_lag_1",
            "DELTA_DG_lag_2",
            "DELTA_DG_lag_3",
            "DELTA_DG_lag_7",
            "DELTA_DG_lag_14",
            "FEED_EFFICIENCY_lag_1",
        ],
        "contextual": ["AGE", "PERIODE", "JUMLAH_AYAM_AKHIR", "IP_STD", "FCR_STD"],
    }

    # Template rekomendasi berdasarkan fitur dan kondisi
    RECOMMENDATION_TEMPLATES = {
        "DG_ACT_low": {
            "priority": "HIGH",
            "category": "Performa Pertumbuhan",
            "issue": "Daily Gain (DG_ACT) rendah",
            "actions": [
                "Lakukan inspeksi visual menyeluruh di seluruh kandang",
                "Cek tanda-tanda penyakit: ayam lesu, kedinginan, ngorok, mencret",
                "Panggil teknisi kesehatan hewan jika diperlukan",
                "Lakukan penimbangan sampel acak untuk validasi data DG_ACT",
                "Periksa distribusi bobot - apakah penurunan merata atau terlokalisir",
            ],
            "monitoring": "Pantau DG_ACT harian dan bandingkan dengan standar usia",
        },
        "PAKAN_volatility_high": {
            "priority": "MEDIUM",
            "category": "Stabilitas Konsumsi",
            "issue": "Konsumsi pakan tidak stabil/fluktuatif tinggi",
            "actions": [
                "Audit sistem pakan: periksa ketersediaan di semua feeder",
                "Pastikan tidak ada feeder yang kosong atau tersumbat",
                "Cek sistem air minum - dehidrasi mengurangi nafsu makan",
                "Evaluasi faktor stres lingkungan: suhu, ventilasi, kebisingan",
                "Periksa kualitas pakan - apakah ada perubahan tekstur/aroma",
            ],
            "monitoring": "Monitor konsumsi pakan per blok kandang setiap 4 jam",
        },
        "MATI_increase": {
            "priority": "CRITICAL",
            "category": "Kesehatan Populasi",
            "issue": "Peningkatan angka kematian (MATI) harian",
            "actions": [
                "SEGERA lakukan nekropsi pada ayam yang baru mati",
                "Identifikasi penyebab kematian: infeksi, keracunan, stress",
                "Perketat biosekuriti: batasi akses, desinfeksi peralatan",
                "Isolasi ayam yang menunjukkan gejala sakit",
                "Konsultasi dengan dokter hewan untuk treatment",
                "Dokumentasikan lokasi dan waktu kematian untuk pola analisis",
            ],
            "monitoring": "Monitoring ketat setiap 2 jam, catat semua kematian",
        },
        "FCR_trend_negative": {
            "priority": "HIGH",
            "category": "Efisiensi Konversi",
            "issue": "Tren FCR memburuk berdasarkan momentum dan acceleration",
            "actions": [
                "Analisis komprehensif: bandingkan dengan periode sebelumnya",
                "Evaluasi perubahan manajemen dalam 1 minggu terakhir",
                "Cek kualitas pakan: kandungan nutrisi, freshness",
                "Review jadwal pemberian pakan dan frekuensi",
                "Evaluasi kondisi lingkungan: suhu, kelembaban, ventilasi",
            ],
            "monitoring": "Hitung FCR harian dan plot tren mingguan",
        },
        "DELTA_PAKAN_anomaly": {
            "priority": "MEDIUM",
            "category": "Manajemen Pakan",
            "issue": "Anomali dalam delta pakan per ekor",
            "actions": [
                "Verifikasi akurasi pencatatan konsumsi pakan",
                "Cek sistem distribusi pakan di seluruh kandang",
                "Evaluasi perubahan formulasi atau supplier pakan",
                "Analisis pola konsumsi berdasarkan waktu pemberian",
                "Periksa kondisi storage pakan - kelembaban, kontaminasi",
            ],
            "monitoring": "Catat konsumsi pakan per blok setiap pemberian",
        },
        "performance_excellent": {
            "priority": "INFO",
            "category": "Optimasi Performa",
            "issue": "Performa sangat baik - pertahankan kondisi",
            "actions": [
                "Dokumentasikan semua parameter manajemen saat ini",
                "Catat kondisi lingkungan sebagai best practice",
                "Simpan sampel pakan untuk referensi kualitas",
                "Hindari perubahan drastis pada sistem yang berjalan",
                "Evaluasi kemungkinan optimasi minor",
            ],
            "monitoring": "Monitoring rutin untuk mempertahankan performa",
        },
    }


# =============================================================================
# KELAS UTAMA PIPELINE TERPADU
# =============================================================================


class UnifiedXAIPipeline:
    """Pipeline terpadu untuk data processing, model training, dan XAI recommendations"""

    def __init__(self, base_folder_path: str, config: UnifiedPipelineConfig = None):
        self.base_folder_path = base_folder_path
        self.config = config or UnifiedPipelineConfig()

        # Setup paths
        self._setup_paths()

        # Initialize components
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.hybrid_feature_names = (
            []
        )  # REFACTORED: Added for storing combined feature names
        self.shap_explainer = None
        self.feature_pipeline = None

    def _setup_paths(self):
        """Setup semua path yang diperlukan"""
        self.data_input_folder = os.path.join(self.base_folder_path, "3. non-normalize")
        self.intermediate_data_folder = os.path.join(
            self.base_folder_path, "1. Normalisasi", "Data"
        )
        self.final_output_folder = os.path.join(
            self.base_folder_path, "2. Hasil Normalisasi", "Terbaru"
        )
        self.models_folder = os.path.join(self.base_folder_path, "5. Model")
        self.xai_output_dir = os.path.join(self.models_folder, "xai_analysis")
        self.reports_dir = os.path.join(self.base_folder_path, "xai_managerial_reports")

        # Create directories
        for folder in [
            self.models_folder,
            self.intermediate_data_folder,
            self.final_output_folder,
            self.xai_output_dir,
            self.reports_dir,
            os.path.join(
                self.models_folder, "optimization_results"
            ),  # REFACTORED: Added path for optuna results
        ]:
            os.makedirs(folder, exist_ok=True)

        # File paths
        self.merged_file = os.path.join(
            self.intermediate_data_folder, "File_Gabungan.csv"
        )
        self.enhanced_file = os.path.join(
            self.intermediate_data_folder, "File_Gabungan_Enhanced_Features.csv"
        )
        self.normalized_file = os.path.join(
            self.final_output_folder, "Data_Gabungan_Fitur_Rekayasa_Normalized.csv"
        )

        # Model paths
        self.fe_pipeline_file = os.path.join(
            self.models_folder, "feature_engineering_pipeline.pkl"
        )
        self.minmax_scaler_file = os.path.join(self.models_folder, "minmax_scaler.pkl")
        self.robust_scaler_file = os.path.join(self.models_folder, "robust_scaler.pkl")
        self.tcn_encoder_path = os.path.join(self.models_folder, "tcn_encoder.keras")
        self.xgb_model_path = os.path.join(
            self.models_folder, "xgboost_hybrid_model.pkl"
        )
        self.hybrid_config_path = os.path.join(
            self.models_folder, "hybrid_model_config.json"
        )

        # Info files
        self.feature_metadata_file = os.path.join(
            self.final_output_folder, "feature_metadata.json"
        )
        self.norm_info_file = os.path.join(
            self.final_output_folder, "normalization_info.json"
        )

    # =============================================================================
    # LANGKAH 1: PENGGABUNGAN DATA
    # =============================================================================

    def merge_data_files(self) -> bool:
        """Membaca semua file CSV dari folder, menggabungkannya, dan menyimpannya."""
        logger.info("MEMULAI LANGKAH 1: PENGGABUNGAN DATA")
        # ... (Kode tidak berubah) ...
        dataframes = []
        for i in range(1, 14):
            file_name = f"NORMALISASI_PERIODE_{i}.csv"
            file_path = os.path.join(self.data_input_folder, file_name)
            if os.path.exists(file_path):
                logger.info(f"Membaca file: {file_name}")
                try:
                    df = pd.read_csv(file_path, sep=";")
                    if "PERIODE" not in df.columns:
                        df["PERIODE"] = i
                    else:
                        df["PERIODE"] = df["PERIODE"].fillna(i)
                    df["PERIODE"] = df["PERIODE"].astype(int)
                    dataframes.append(df)
                except Exception as e:
                    logger.error(f"ERROR membaca {file_name}: {str(e)}")
            else:
                logger.warning(f"File tidak ditemukan: {file_name}")

        if not dataframes:
            logger.error(
                "GAGAL: Tidak ada file yang berhasil dibaca. Pipeline dihentikan."
            )
            return False

        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(self.merged_file, sep=";", index=False)
        logger.info(
            f"BERHASIL: Penggabungan data selesai. File disimpan sebagai: {self.merged_file}"
        )
        return True

    # =============================================================================
    # LANGKAH 2: REKAYASA FITUR
    # =============================================================================

    def run_feature_engineering_step(self) -> bool:
        """Menggunakan kelas FeatureEngineeringPipeline untuk memproses data dan menyimpan pipeline."""
        logger.info("MEMULAI LANGKAH 2: REKAYASA FITUR & PENYIMPANAN PIPELINE")
        # ... (Kode tidak berubah) ...
        if not os.path.exists(self.merged_file):
            logger.critical(f"GAGAL: File input '{self.merged_file}' tidak ditemukan.")
            return False

        df_raw = pd.read_csv(self.merged_file, sep=";")
        logger.info(f"Data mentah dimuat untuk rekayasa fitur: {df_raw.shape}")

        # Inisialisasi pipeline rekayasa fitur dengan konfigurasi
        self.feature_pipeline = FeatureEngineeringPipeline(
            config=self.config.FEATURE_CONFIG
        )

        # Lakukan 'fit_transform' pada data mentah
        logger.info("Menjalankan fit_transform pada FeatureEngineeringPipeline...")
        df_enhanced = self.feature_pipeline.fit_transform(
            df_raw, metadata_output_dir=self.final_output_folder
        )

        # Simpan data yang sudah direkayasa fiturnya
        df_enhanced.to_csv(self.enhanced_file, sep=";", index=False)
        logger.info(
            f"Data dengan fitur enhanced berhasil disimpan di: {self.enhanced_file}"
        )

        # Simpan objek pipeline yang sudah di-'fit' ke file .pkl
        with open(self.fe_pipeline_file, "wb") as f:
            pickle.dump(self.feature_pipeline, f)
        logger.info(
            f"Model rekayasa fitur (pipeline) disimpan di: {self.fe_pipeline_file}"
        )

        return True

    # =============================================================================
    # LANGKAH 3: NORMALISASI FITUR
    # =============================================================================

    def normalize_features(self) -> bool:
        """Menerapkan strategi normalisasi metode campuran pada data."""
        logger.info("MEMULAI LANGKAH 3: NORMALISASI FITUR")
        # ... (Kode tidak berubah) ...
        if not os.path.exists(self.enhanced_file):
            logger.error(
                f"File input '{self.enhanced_file}' tidak ditemukan. Langkah 3 dihentikan."
            )
            return False

        df = pd.read_csv(self.enhanced_file, delimiter=";")
        logger.info(
            f"Data untuk normalisasi dimuat: {df.shape[0]} baris, {df.shape[1]} kolom"
        )

        # Kolom yang tidak dinormalisasi (termasuk kolom target)
        non_normalized_cols = ["TANGGAL", "PERIODE", "FCR_ACT"]

        # Identifikasi semua kolom numerik yang tersedia
        all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Tentukan fitur mana yang akan dinormalisasi
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

        logger.info(
            f"Ditemukan {len(features_to_normalize)} fitur untuk dinormalisasi."
        )
        logger.info(f"Dialokasikan {len(minmax_features)} fitur ke MinMaxScaler.")
        logger.info(f"Dialokasikan {len(robust_features)} fitur ke RobustScaler.")

        minmax_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()

        df_normalized = df.copy()

        # Terapkan normalisasi
        if minmax_features:
            minmax_data = df[minmax_features].fillna(df[minmax_features].median())
            df_normalized[minmax_features] = minmax_scaler.fit_transform(minmax_data)

            # Simpan model MinMaxScaler
            with open(self.minmax_scaler_file, "wb") as f:
                pickle.dump(minmax_scaler, f)
            logger.info(f"Model MinMaxScaler disimpan ke: {self.minmax_scaler_file}")
            self.scalers["minmax"] = minmax_scaler

        if robust_features:
            robust_data = df[robust_features].fillna(df[robust_features].median())
            df_normalized[robust_features] = robust_scaler.fit_transform(robust_data)

            # Simpan model RobustScaler
            with open(self.robust_scaler_file, "wb") as f:
                pickle.dump(robust_scaler, f)
            logger.info(f"Model RobustScaler disimpan ke: {self.robust_scaler_file}")
            self.scalers["robust"] = robust_scaler

        # Simpan hasil
        df_normalized.to_csv(self.normalized_file, index=False, sep=";")

        # Simpan informasi normalisasi
        normalization_info = {
            "minmax_features": minmax_features,
            "robust_features": robust_features,
            "non_normalized": [
                col for col in non_normalized_cols if col in df_normalized.columns
            ],
            "minmax_scaler_params": (
                {
                    "min_": minmax_scaler.min_.tolist(),
                    "scale_": minmax_scaler.scale_.tolist(),
                    "feature_names": minmax_features,
                }
                if minmax_features
                else {}
            ),
            "robust_scaler_params": (
                {
                    "center_": robust_scaler.center_.tolist(),
                    "scale_": robust_scaler.scale_.tolist(),
                    "feature_names": robust_features,
                }
                if robust_features
                else {}
            ),
            "final_columns": df_normalized.columns.tolist(),
        }
        with open(self.norm_info_file, "w") as f:
            json.dump(normalization_info, f, indent=2)

        logger.info(f"BERHASIL: Normalisasi fitur selesai.")
        logger.info(
            f" - Fitur dinormalisasi dengan MinMax Scaler: {len(minmax_features)}"
        )
        logger.info(
            f" - Fitur dinormalisasi dengan Robust Scaler: {len(robust_features)}"
        )
        logger.info(f" - File data ternormalisasi disimpan di: {self.normalized_file}")
        logger.info(f" - Info normalisasi disimpan di: {self.norm_info_file}")
        return True

    # =============================================================================
    # LANGKAH 4: PELATIHAN MODEL HIBRIDA (REFACTORED)
    # =============================================================================
    # REFACTORED: Seluruh blok training model telah dikonsolidasikan menjadi satu
    # metode yang bersih dan menggunakan fungsi-fungsi helper yang terpusat.

    def optimize_hyperparameters_and_train(
        self, n_trials: int = 50, timeout: int = 3600
    ) -> bool:
        """
        Optimasi hyperparameter dan pelatihan model hibrida TCN-XGBoost.
        Versi ini adalah konsolidasi dan perbaikan dari duplikasi kode sebelumnya.
        """
        logger.info(
            "MEMULAI LANGKAH 4: OPTIMASI HYPERPARAMETER & PELATIHAN MODEL HIBRIDA (VERSI REFACTORED)"
        )

        if not os.path.exists(self.normalized_file):
            logger.error(
                f"File input '{self.normalized_file}' tidak ditemukan. Langkah 4 dihentikan."
            )
            return False

        # 1. Memuat dan Mempersiapkan Data
        df = pd.read_csv(self.normalized_file, delimiter=";")
        logger.info(
            f"Data untuk pelatihan dimuat: {df.shape[0]} baris, {df.shape[1]} kolom"
        )

        df["FCR_NEXT_DAY"] = df.groupby("PERIODE")["FCR_ACT"].shift(-1)
        df = df.dropna(subset=["FCR_NEXT_DAY"])

        self.feature_names = [
            col
            for col in df.columns
            if col not in ["TANGGAL", "PERIODE", "FCR_ACT", "FCR_NEXT_DAY"]
        ]
        X = df[self.feature_names]
        y = df["FCR_NEXT_DAY"]

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        logger.info(
            f"Data training: {X_train.shape[0]}, Data validation: {X_val.shape[0]}"
        )

        # 2. Fungsi Objective untuk Optuna
        def objective(trial):
            try:
                # Hyperparameters
                sequence_length = 7
                batch_size = 32  # Anda bisa mengoptimasi ini jika mau

                tcn_model = self._create_tcn_model(trial)

                X_train_seq, y_train_seq = self._create_sequences(
                    X_train, y_train, sequence_length
                )
                X_val_seq, y_val_seq = self._create_sequences(
                    X_val, y_val, sequence_length
                )

                # FIXED: Pengecekan data validasi yang lebih kuat
                if len(X_val_seq) < 1:
                    logger.warning(
                        f"Trial dilewati karena data validasi tidak cukup: {len(X_val_seq)} sampel."
                    )
                    return float("inf")

                # =====================================================================
                # DEBUGGING: Tambahkan blok ini untuk memeriksa bentuk data & model
                # =====================================================================
                print("\n" + "=" * 25 + " DEBUGGING INFO " + "=" * 25)
                print(f"Tipe X_train_seq: {type(X_train_seq)}")
                print(f"Bentuk X_train_seq: {X_train_seq.shape}")
                print(f"\nTipe y_train_seq: {type(y_train_seq)}")
                print(f"Bentuk y_train_seq: {y_train_seq.shape}")
                print(f"\nTipe X_val_seq: {type(X_val_seq)}")
                print(f"Bentuk X_val_seq: {X_val_seq.shape}")
                print(f"\nTipe y_val_seq: {type(y_val_seq)}")
                print(f"Bentuk y_val_seq: {y_val_seq.shape}")
                print("\nRingkasan Model (tcn_model):")
                tcn_model.summary()
                print("=" * 70 + "\n")
                # =====================================================================

                # Training TCN
                # tcn_model.fit(
                #     X_train_seq,
                #     y_train_seq,
                #     validation_data=(X_val_seq, y_val_seq),
                #     epochs=50,
                #     batch_size=batch_size,
                #     callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                #     verbose=0,
                # )

                hybrid_features_train = self._extract_hybrid_features(
                    tcn_model, X_train, sequence_length
                )
                hybrid_features_val = self._extract_hybrid_features(
                    tcn_model, X_val, sequence_length
                )

                # Training XGBoost
                xgb_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "random_state": 42,
                }
                xgb_model = xgb.XGBRegressor(**xgb_params)

                # Align y_train_seq untuk XGBoost
                y_train_for_xgb = self._create_sequences(
                    X_train, y_train, sequence_length
                )[1]
                xgb_model.fit(hybrid_features_train, y_train_for_xgb)

                y_pred = xgb_model.predict(hybrid_features_val)
                y_val_for_xgb = self._create_sequences(X_val, y_val, sequence_length)[1]

                # Pastikan prediksi tidak kosong
                if len(y_pred) == 0:
                    return float("inf")

                rmse = np.sqrt(mean_squared_error(y_val_for_xgb, y_pred))
                return rmse
            except Exception as e:
                logger.error(
                    f"Error dalam trial: {e}", exc_info=False
                )  # exc_info=False agar tidak terlalu verbose
                return float("inf")

        # 3. Menjalankan Optimasi
        logger.info("Memulai optimasi hyperparameter dengan Optuna...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        logger.info(f"Optimasi selesai. Best RMSE: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        self._save_optuna_results(study)

        # 4. Training Model Final
        logger.info("Training model final dengan best parameters...")
        sequence_length = 7

        # Buat dan latih TCN final
        final_tcn_model = self._create_tcn_model(fixed_params=best_params)
        X_full_seq, y_full_seq = self._create_sequences(X, y, sequence_length)

        final_tcn_model.fit(
            X_full_seq,
            y_full_seq,
            epochs=100,
            batch_size=32,
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=1,
        )

        # Buat TCN encoder untuk disimpan
        tcn_encoder = Model(
            inputs=final_tcn_model.input, outputs=final_tcn_model.layers[-2].output
        )

        # Ekstrak fitur hibrida untuk seluruh data
        hybrid_features_full = self._extract_hybrid_features(
            final_tcn_model, X, sequence_length
        )

        # Latih XGBoost final
        final_xgb_params = {
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "learning_rate": best_params["learning_rate"],
            "subsample": best_params["subsample"],
            "colsample_bytree": best_params["colsample_bytree"],
            "random_state": 42,
        }
        final_xgb_model = xgb.XGBRegressor(**final_xgb_params)
        final_xgb_model.fit(hybrid_features_full, y_full_seq)

        # 5. Menyimpan Model dan Konfigurasi
        tcn_encoder.save(self.tcn_encoder_path)
        with open(self.xgb_model_path, "wb") as f:
            pickle.dump(final_xgb_model, f)

        # REFACTORED: Sentralisasi pembuatan nama fitur hibrida
        self.hybrid_feature_names = self._get_hybrid_feature_names(
            self.feature_names, tcn_encoder
        )

        hybrid_config = {
            "sequence_length": sequence_length,
            "feature_names": self.feature_names,
            "hybrid_feature_names": self.hybrid_feature_names,
            "tcn_params": {
                key: val
                for key, val in best_params.items()
                if key.startswith(("nb_", "kernel_", "max_dilation", "dropout_"))
            },
            "xgb_params": final_xgb_params,
            "optimization_results": {
                "best_rmse": study.best_value,
                "best_params": best_params,
            },
        }
        with open(self.hybrid_config_path, "w") as f:
            json.dump(hybrid_config, f, indent=2)

        self.models["tcn_encoder"] = tcn_encoder
        self.models["xgboost"] = final_xgb_model

        logger.info(f"Model TCN encoder disimpan: {self.tcn_encoder_path}")
        logger.info(f"Model XGBoost disimpan: {self.xgb_model_path}")

        # 6. Evaluasi Final
        self._evaluate_and_plot_final_model(
            final_xgb_model, hybrid_features_full, y_full_seq
        )

        return True

    # =============================================================================
    # FUNGSI-FUNGSI HELPER UNTUK TRAINING (REFACTORED)
    # =============================================================================

    def _create_sequences(self, data_x, data_y, sequence_length):
        """
        Membuat sequences. Jika data_y diberikan, kembalikan X dan y.
        Jika data_y adalah None, kembalikan hanya X.
        """
        if isinstance(data_x, pd.DataFrame):
            data_x = data_x.values

        sequences_x = []
        for i in range(len(data_x) - sequence_length + 1):
            sequences_x.append(data_x[i : i + sequence_length])

        # FIXED: Hanya proses data_y jika ada (bukan None)
        if data_y is not None:
            if isinstance(data_y, pd.Series):
                data_y = data_y.values
            sequences_y = []
            for i in range(len(data_x) - sequence_length + 1):
                sequences_y.append(data_y[i + sequence_length - 1])
            return np.array(sequences_x), np.array(sequences_y)
        else:
            return np.array(sequences_x)

    def _create_tcn_model(self, trial=None, fixed_params=None):
        """
        Membuat model TCN sederhana menggunakan lapisan Conv1D asli dari Keras
        untuk menghindari bug/inkompatibilitas dari library eksternal.
        """
        sequence_length = 7
        n_features = len(self.feature_names)

        # --- Parameter Hyperparameter ---
        if trial:
            params = {
                "filters": trial.suggest_int("nb_filters", 32, 128),
                "kernel_size": trial.suggest_int(
                    "kernel_size", 2, 3
                ),  # Kernel lebih kecil lebih umum di TCN
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
                "num_blocks": trial.suggest_int(
                    "nb_stacks", 1, 2
                ),  # Jumlah blok residual
            }
        elif fixed_params:
            # Sesuaikan nama parameter
            params = {
                "filters": fixed_params.get("nb_filters", 64),
                "kernel_size": fixed_params.get("kernel_size", 2),
                "dropout_rate": fixed_params.get("dropout_rate", 0.2),
                "num_blocks": fixed_params.get("nb_stacks", 1),
            }
        else:
            raise ValueError("Harus menyediakan 'trial' atau 'fixed_params'")

        # --- Arsitektur Model ---
        from tensorflow.keras.layers import (
            Conv1D,
            Dropout,
            Add,
            LayerNormalization,
            GlobalAveragePooling1D,
        )

        input_layer = Input(shape=(sequence_length, n_features))
        x = input_layer

        for i in range(params["num_blocks"]):
            # Simpan koneksi residual
            residual = x

            # Blok Konvolusi 1
            x = Conv1D(
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                dilation_rate=2**i,  # Laju dilatasi bertambah per blok
                padding="causal",
                activation="relu",
            )(x)
            x = LayerNormalization()(x)
            x = Dropout(params["dropout_rate"])(x)

            # Blok Konvolusi 2
            x = Conv1D(
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                dilation_rate=2**i,
                padding="causal",
                activation="relu",
            )(x)
            x = LayerNormalization()(x)
            x = Dropout(params["dropout_rate"])(x)

            # Koneksi residual. Sesuaikan dimensi jika perlu.
            if residual.shape[-1] != x.shape[-1]:
                residual = Conv1D(params["filters"], kernel_size=1, padding="same")(
                    residual
                )

            x = Add()([residual, x])

        # Lapisan setelah blok TCN
        x = GlobalAveragePooling1D()(x)  # Mengagregasi output sequence
        output_layer = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer="adam", loss="mse")
        return model

    def _extract_hybrid_features(self, tcn_model, X_data, sequence_length):
        """Mengekstrak fitur temporal dan menggabungkannya dengan fitur tabular."""
        tcn_encoder = Model(inputs=tcn_model.input, outputs=tcn_model.layers[-2].output)

        # REFACTORED: Hanya membuat sequence untuk X
        X_seq = self._create_sequences(
            X_data, data_y=None, sequence_length=sequence_length
        )[0]

        temporal_features = tcn_encoder.predict(X_seq, verbose=0)

        # Align tabular features
        tabular_features = X_data.iloc[
            sequence_length - 1 : sequence_length - 1 + len(X_seq)
        ].values

        return np.concatenate([temporal_features, tabular_features], axis=1)

    def _get_hybrid_feature_names(self, tabular_names, tcn_encoder):
        """Membuat daftar nama fitur gabungan."""
        n_temporal_features = tcn_encoder.output_shape[1]
        temporal_feature_names = [f"tcn_feat_{i}" for i in range(n_temporal_features)]
        return temporal_feature_names + tabular_names

    def _save_optuna_results(self, study):
        """Menyimpan hasil dan plot dari studi optimasi Optuna."""
        optimization_file = os.path.join(
            self.models_folder, "optimization_results", "optimization_results.json"
        )
        with open(optimization_file, "w") as f:
            json.dump(
                {"best_params": study.best_params, "best_value": study.best_value},
                f,
                indent=2,
            )

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_image(
                os.path.join(
                    self.models_folder,
                    "optimization_results",
                    f"opt_history_{timestamp}.png",
                )
            )

            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_image(
                os.path.join(
                    self.models_folder,
                    "optimization_results",
                    f"opt_importance_{timestamp}.png",
                )
            )
            logger.info(f"Plot optimasi disimpan di folder 'optimization_results'")
        except Exception as e:
            logger.warning(
                f"Gagal membuat plot Optuna (mungkin perlu 'pip install plotly kaleido'): {e}"
            )

    def _evaluate_and_plot_final_model(self, model, X, y):
        """Mengevaluasi model final dan membuat plot hasilnya."""
        y_pred = model.predict(X)
        logger.info(f"Evaluasi Model Final:")
        logger.info(
            f"MAE: {mean_absolute_error(y, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}, R²: {r2_score(y, y_pred):.4f}"
        )

        try:
            plt.figure(figsize=(10, 5))
            plt.scatter(y, y_pred, alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
            plt.xlabel("Actual FCR"), plt.ylabel("Predicted FCR"), plt.title(
                "Actual vs Predicted FCR"
            )
            plt.tight_layout()
            plot_file = os.path.join(
                self.xai_output_dir,
                f"prediction_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            plt.savefig(plot_file, dpi=300)
            plt.close()
            logger.info(f"Plot prediksi disimpan: {plot_file}")
        except Exception as e:
            logger.warning(f"Gagal membuat plot prediksi: {e}")

    # =============================================================================
    # LANGKAH 5: ANALISIS XAI KOMPREHENSIF
    # =============================================================================
    def run_comprehensive_xai_analysis(self) -> bool:
        """Menjalankan analisis XAI komprehensif."""
        logger.info("MEMULAI LANGKAH 5: ANALISIS XAI KOMPREHENSIF")
        try:
            if not self.models:
                self.load_trained_models()

            df = pd.read_csv(self.normalized_file, delimiter=";")
            df["FCR_NEXT_DAY"] = df.groupby("PERIODE")["FCR_ACT"].shift(-1)
            df = df.dropna(subset=["FCR_NEXT_DAY"])
            X = df[self.feature_names]
            y = df["FCR_NEXT_DAY"]

            sequence_length = 7  # Sesuai dengan config training
            hybrid_features = self._extract_hybrid_features(
                self.models["tcn_encoder"], X, sequence_length
            )
            y_actual = self._create_sequences(X, y, sequence_length)[1]

            # REFACTORED: Menggunakan nama fitur hybrid yang sudah tersimpan
            feature_names = self.hybrid_feature_names

            self._analyze_feature_importance_xgb(feature_names)
            self._analyze_shap_values(hybrid_features, feature_names)

            logger.info("Analisis XAI komprehensif selesai")
            return True
        except Exception as e:
            logger.error(f"Error dalam analisis XAI: {e}", exc_info=True)
            return False

    def _analyze_feature_importance_xgb(self, feature_names: List[str]):
        """Analisis feature importance dari XGBoost."""
        # ... (Kode tidak berubah, tetapi sekarang menggunakan feature_names yang benar) ...
        try:
            importance = self.models["xgboost"].feature_importances_
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importance}
            ).sort_values("importance", ascending=False)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            importance_file = os.path.join(
                self.xai_output_dir, f"xgb_feature_importance_{timestamp}.csv"
            )
            importance_df.to_csv(importance_file, index=False)

            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            sns.barplot(x="importance", y="feature", data=top_features, orient="h")
            plt.xlabel("Feature Importance")
            plt.title("Top 20 XGBoost Feature Importances")
            plt.tight_layout()

            plot_file = os.path.join(
                self.xai_output_dir, f"xgb_feature_importance_{timestamp}.png"
            )
            plt.savefig(plot_file, dpi=300)
            plt.close()

            logger.info(f"XGBoost feature importance disimpan: {importance_file}")
        except Exception as e:
            logger.error(f"Error dalam analisis feature importance XGBoost: {e}")

    def _analyze_shap_values(self, X_hybrid, feature_names: List[str]):
        """Analisis SHAP values."""
        # ... (Kode tidak berubah, tetapi sekarang menggunakan feature_names yang benar) ...
        try:
            if not self.shap_explainer:
                self.shap_explainer = shap.TreeExplainer(self.models["xgboost"])

            sample_size = min(1000, X_hybrid.shape[0])
            X_sample = X_hybrid[:sample_size]
            shap_values = self.shap_explainer.shap_values(X_sample)

            X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Plot SHAP summary
            shap.summary_plot(shap_values, X_sample_df, show=False)
            summary_plot_file = os.path.join(
                self.xai_output_dir, f"shap_summary_plot_{timestamp}.png"
            )
            plt.savefig(summary_plot_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"SHAP summary plot disimpan: {summary_plot_file}")
        except Exception as e:
            logger.error(f"Error dalam analisis SHAP: {e}")

    # =============================================================================
    # Sisa fungsi (load models, rekomendasi, dll) tidak diubah secara signifikan
    # karena tidak mengandung duplikasi mayor.
    # =============================================================================
    def load_trained_models(self):
        """Load semua model yang sudah dilatih untuk XAI analysis"""
        # ... (Kode tidak berubah) ...
        try:
            # Load XGBoost model
            with open(self.xgb_model_path, "rb") as f:
                self.models["xgboost"] = pickle.load(f)

            # Load TCN encoder
            self.models["tcn_encoder"] = tf.keras.models.load_model(
                self.tcn_encoder_path, custom_objects={"TCN": TCN}
            )

            # Load scalers
            with open(self.minmax_scaler_file, "rb") as f:
                self.scalers["minmax"] = pickle.load(f)
            with open(self.robust_scaler_file, "rb") as f:
                self.scalers["robust"] = pickle.load(f)

            # Load feature engineering pipeline
            with open(self.fe_pipeline_file, "rb") as f:
                self.feature_pipeline = pickle.load(f)

            # Load model config untuk feature names
            model_config = self.load_model_config()
            self.feature_names = model_config.get("all_feature_names", [])

            # Initialize SHAP explainer
            self.shap_explainer = shap.TreeExplainer(self.models["xgboost"])

            logger.info("Semua model berhasil dimuat untuk XAI analysis")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    # ... (Sisa fungsi seperti generate_recommendations, run_complete_pipeline, dll)

    def run_complete_pipeline(self) -> bool:
        """Menjalankan pipeline lengkap dari data processing hingga XAI analysis."""
        logger.info("=" * 80)
        logger.info("MEMULAI PIPELINE TERPADU: DATA PROCESSING + MODEL TRAINING + XAI")
        logger.info("=" * 80)

        try:
            if not self.merge_data_files():
                return False
            if not self.run_feature_engineering_step():
                return False
            if not self.normalize_features():
                return False
            if not self.optimize_hyperparameters_and_train():
                return False
            if not self.run_comprehensive_xai_analysis():
                return False

            logger.info("=" * 80)
            logger.info("PIPELINE TERPADU BERHASIL DISELESAIKAN!")
            logger.info("=" * 80)
            return True
        except Exception as e:
            logger.error(f"Error kritis dalam pipeline: {e}", exc_info=True)
            return False


# =============================================================================
# CONTOH PENGGUNAAN
# =============================================================================

if __name__ == "__main__":
    # Ganti dengan path folder proyek Anda
    BASE_FOLDER = "."  # Menggunakan folder saat ini sebagai default

    logger.info("Memulai Pipeline Terpadu: Data Processing + Model Training + XAI")

    pipeline = UnifiedXAIPipeline(BASE_FOLDER)
    success = pipeline.run_complete_pipeline()

    if success:
        print("\n" + "=" * 80)
        print("PIPELINE TERPADU BERHASIL DISELESAIKAN!")
        print("=" * 80)
    else:
        print("\nPipeline gagal. Periksa log 'INFO' dan 'ERROR' di atas untuk detail.")
