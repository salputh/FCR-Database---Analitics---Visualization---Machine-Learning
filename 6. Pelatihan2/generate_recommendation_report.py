# -*- coding: utf-8 -*-
"""
Skrip Inferensi dan Rekomendasi Manajerial Berbasis XAI

Tujuan:
- Memuat model TCN-XGBoost, scaler, dan pipeline feature engineering yang sudah dilatih.
- Mengambil data mentah baru sebagai input.
- Menghasilkan prediksi FCR untuk hari berikutnya.
- Memberikan penjelasan (XAI) dan rekomendasi manajerial yang dapat ditindaklanjuti.

Cara Penggunaan:
1. Pastikan semua artifak model dari FullTrainPipeline.py sudah ada di folder yang benar.
2. Siapkan data mentah baru dalam format CSV yang sesuai.
3. Jalankan skrip ini dengan menunjuk ke folder model dan file data baru.
"""

import pandas as pd
import numpy as np
import os
import warnings
import logging
import json
import pickle
from typing import Dict
from sklearn.preprocessing import MinMaxScaler, RobustScaler

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

# Configure logging
# Create logger
logger = logging.getLogger("FCR_Recommender")
logger.setLevel(logging.INFO)

# Create console handler and set level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create file handler and set level
file_handler = logging.FileHandler("fcr_recommender.log")
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Prevent logging from propagating to the root logger
logger.propagate = False


# =============================================================================
# KONFIGURASI SISTEM (RULES MANAJERIAL)
# Diambil dari XAI_Managerial_Recommendations.py
# =============================================================================
class RecommendationConfig:
    """Konfigurasi yang berisi rules dan template untuk rekomendasi manajerial."""

    # Threshold FCR untuk klasifikasi kondisi
    FCR_THRESHOLDS = {
        "Sangat Baik": 1.5,
        "Baik": 1.65,
        "Peringatan": 1.8,
        "Kritis": 2.0,
    }

    # Threshold SHAP values untuk menentukan signifikansi fitur
    SHAP_SIGNIFICANCE_THRESHOLD = 0.02

    # Mapping fitur inti ke template rekomendasi
    # Kunci di sini harus berupa bagian dari nama fitur yang signifikan
    FEATURE_TO_TEMPLATE_KEY = {
        "AGE": "AGE_critical",
        "MATI": "MATI_increase",
        "JML_DEPLESI": "DEPLESI_high",
        "DELTA_PAKAN_ZAK": "PAKAN_supply_anomaly",
        "DELTA_PAKAN_CUM_GR/EKOK": "PAKAN_cumulative_high",
        "DG_ACT": "DG_ACT_low",
        "DELTA_ABW": "ABW_change_anomaly",
        "DELTA_DG": "DG_trend_negative",
        "FCR_ACT": "FCR_trend_negative",
        "DELTA_FCR": "FCR_delta_high",
        "IP_STD": "IP_performance_low",
        "PAKAN_ACT_GR/EK_rolling_std": "PAKAN_volatility_high",
        "DG_ACT_rolling_std": "DG_volatility_high",
        "MATI_rolling_std": "MATI_volatility_high",
        "DELTA_PAKAN_GR/EKOR_rolling_std": "PAKAN_intake_volatile",
    }

    # Template rekomendasi berdasarkan fitur dan kondisi
    RECOMMENDATION_TEMPLATES = {
        "DG_ACT_low": {
            "priority": "TINGGI",
            "category": "Performa Pertumbuhan",
            "issue": "Pertumbuhan Harian (Daily Gain) rendah atau menurun.",
            "actions": [
                "Lakukan inspeksi visual menyeluruh di seluruh kandang, cari tanda-tanda ayam lesu atau sakit.",
                "Validasi data dengan melakukan penimbangan sampel acak di beberapa titik kandang.",
                "Periksa apakah ada masalah pada distribusi pakan atau kepadatan kandang yang tidak merata.",
            ],
            "monitoring": "Pantau DG harian dan bandingkan dengan standar bobot untuk usia saat ini.",
        },
        "PAKAN_volatility_high": {
            "priority": "SEDANG",
            "category": "Stabilitas Konsumsi",
            "issue": "Konsumsi pakan sangat tidak stabil/fluktuatif (naik turun drastis).",
            "actions": [
                "Audit sistem pakan dan air minum. Pastikan ketersediaan dan aksesibilitasnya merata.",
                "Evaluasi faktor stres lingkungan: fluktuasi suhu ekstrem, kebisingan, atau ventilasi buruk.",
                "Periksa kualitas pakan (tekstur, aroma, gumpalan) yang mungkin menurunkan nafsu makan.",
            ],
            "monitoring": "Monitor konsumsi pakan per blok kandang setiap periode pemberian pakan.",
        },
        "MATI_increase": {
            "priority": "KRITIS",
            "category": "Kesehatan Populasi",
            "issue": "Peningkatan signifikan pada angka kematian harian.",
            "actions": [
                "SEGERA lakukan nekropsi (bedah bangkai) pada ayam yang baru mati untuk identifikasi penyebab.",
                "Perketat biosekuriti secara drastis: batasi akses, desinfeksi, dan ganti alas kaki.",
                "Isolasi ayam yang menunjukkan gejala sakit dan konsultasikan dengan teknisi kesehatan hewan.",
            ],
            "monitoring": "Lakukan pemantauan intensif setiap 2-3 jam dan catat semua kematian dengan lokasi.",
        },
        "FCR_trend_negative": {
            "priority": "TINGGI",
            "category": "Efisiensi Pakan",
            "issue": "FCR Harian sebelumnya sudah tinggi, menunjukkan tren efisiensi yang memburuk.",
            "actions": [
                "Analisis komprehensif data 3 hari terakhir untuk menemukan akar masalah (kesehatan, pakan, lingkungan).",
                "Evaluasi apakah ada perubahan manajemen, pakan, atau lingkungan dalam seminggu terakhir.",
                "Review kembali program kesehatan dan jadwal vaksinasi.",
            ],
            "monitoring": "Hitung FCR harian secara akurat dan plot trennya dalam grafik sederhana.",
        },
        "DELTA_PAKAN_anomaly": {
            "priority": "SEDANG",
            "category": "Manajemen Pakan",
            "issue": "Terjadi anomali pada jumlah pakan yang diberikan.",
            "actions": [
                "Verifikasi akurasi pencatatan dan penimbangan pakan harian.",
                "Pastikan tidak ada pakan yang tumpah atau terbuang sia-sia.",
                "Evaluasi apakah formula atau jenis pakan baru saja diganti.",
            ],
            "monitoring": "Catat sisa pakan (jika ada) sebelum pemberian pakan berikutnya untuk akurasi.",
        },
        "performance_excellent": {
            "priority": "INFO",
            "category": "Performa Optimal",
            "issue": "Performa diprediksi sangat baik. Pertahankan kondisi saat ini.",
            "actions": [
                "Dokumentasikan semua parameter manajemen saat ini (suhu, kelembaban, jadwal pakan) sebagai 'Best Practice'.",
                "Hindari melakukan perubahan drastis pada sistem yang sedang berjalan optimal.",
                "Jaga konsistensi kualitas pakan dan air minum.",
            ],
            "monitoring": "Lanjutkan monitoring rutin untuk memastikan performa tetap stabil.",
        },
    }


class ManagerialRecommender:
    """Kelas untuk memuat model dan menghasilkan rekomendasi manajerial."""

    def __init__(self, models_base_path: str, base_dir: str = None):
        self.models_base_path = models_base_path
        self.base_dir = base_dir or os.path.dirname(models_base_path)
        self.config = RecommendationConfig()
        self.artifacts = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Memuat semua artifak yang diperlukan: model, scaler, pipeline, config."""
        logger.info("Memuat artifak model...")
        try:
            # Muat model
            with open(
                os.path.join(self.models_base_path, "xgboost_hybrid_model.pkl"), "rb"
            ) as f:
                self.artifacts["xgb_model"] = pickle.load(f)
            self.artifacts["tcn_encoder"] = tf.keras.models.load_model(
                os.path.join(self.models_base_path, "tcn_encoder.keras")
            )

            # Muat pipeline & scaler
            with open(
                os.path.join(self.models_base_path, "feature_engineering_pipeline.pkl"),
                "rb",
            ) as f:
                self.artifacts["fe_pipeline"] = pickle.load(f)
            with open(
                os.path.join(self.models_base_path, "minmax_scaler.pkl"), "rb"
            ) as f:
                self.artifacts["minmax_scaler"] = pickle.load(f)
            with open(
                os.path.join(self.models_base_path, "robust_scaler.pkl"), "rb"
            ) as f:
                self.artifacts["robust_scaler"] = pickle.load(f)

            # Muat config model & normalisasi
            with open(
                os.path.join(self.models_base_path, "hybrid_model_config.json"), "r"
            ) as f:
                self.artifacts["model_config"] = json.load(f)
            # Load normalization configuration
            with open(
                os.path.join(
                    self.base_dir,
                    "2. Hasil Normalisasi/Terbaru/normalization_info.json",
                ),
                "r",
            ) as f:
                self.artifacts["norm_config"] = json.load(f)

            logger.info("Semua artifak berhasil dimuat.")
        except FileNotFoundError as e:
            logger.error(
                f"GAGAL memuat artifak: File tidak ditemukan - {e}. Pastikan path '{self.models_base_path}' benar."
            )
            raise
        except Exception as e:
            logger.error(f"Terjadi error saat memuat artifak: {e}")
            raise

    def _create_sequences(self, data_x, sequence_length):
        """Membuat sekuens HANYA untuk data X."""
        if isinstance(data_x, pd.DataFrame):
            data_x = data_x.values

        sequences_x = []
        for i in range(len(data_x) - sequence_length + 1):
            sequences_x.append(data_x[i : i + sequence_length])

        return np.array(sequences_x)

    def _extract_hybrid_features(self, processed_data: pd.DataFrame) -> np.ndarray:
        """Mengekstrak fitur gabungan dari TCN dan data tabular untuk input XGBoost."""
        model_config = self.artifacts["model_config"]
        sequence_length = model_config["sequence_length"]
        feature_names = model_config["feature_names"]
        tcn_encoder = self.artifacts["tcn_encoder"]

        # Pastikan kolom data sesuai urutan
        data_for_sequence = processed_data[feature_names]

        # Buat sekuens
        X_seq = self._create_sequences(data_for_sequence, sequence_length)
        if len(X_seq) == 0:
            raise ValueError(
                f"Data input tidak cukup untuk membuat sekuens dengan panjang {sequence_length}. Butuh minimal {sequence_length} baris data."
            )

        # Ekstrak fitur temporal dari TCN
        temporal_features = tcn_encoder(X_seq, training=False)

        # Ambil data tabular terbaru yang sesuai dengan sekuens terakhir
        last_tabular_features = data_for_sequence.iloc[-1].values.reshape(1, -1)

        # Gabungkan fitur temporal (dari sekuens terakhir) dan tabular (dari hari terakhir)
        hybrid_features = np.concatenate(
            [temporal_features[-1:], last_tabular_features], axis=1
        )
        return hybrid_features

    def generate_recommendations(self, raw_data_df: pd.DataFrame) -> str:
        """
        Pipeline lengkap dari data mentah hingga laporan rekomendasi.

        Args:
            raw_data_df: DataFrame pandas berisi data mentah historis termasuk hari ini.
                         Harus memiliki cukup baris (minimal sequence_length) untuk membuat fitur.
        """
        # 1. Rekayasa Fitur
        logger.info("Menerapkan rekayasa fitur pada data baru...")
        # Gunakan .transform(), bukan .fit_transform()
        df_enhanced = self.artifacts["fe_pipeline"].transform(raw_data_df)

        # 2. Normalisasi Fitur
        logger.info("Menerapkan normalisasi pada data baru...")
        norm_config = self.artifacts["norm_config"]
        minmax_features = norm_config["minmax_features"]
        robust_features = norm_config["robust_features"]

        # Pastikan kolom ada sebelum normalisasi
        df_processed = df_enhanced.copy()
        df_processed[minmax_features] = self.artifacts["minmax_scaler"].transform(
            df_enhanced[minmax_features]
        )
        df_processed[robust_features] = self.artifacts["robust_scaler"].transform(
            df_enhanced[robust_features]
        )

        # 3. Ekstrak Fitur Hibrida untuk Prediksi
        logger.info("Mengekstrak fitur hibrida untuk prediksi...")
        hybrid_features = self._extract_hybrid_features(df_processed)

        # 4. Buat Prediksi
        logger.info("Membuat prediksi FCR...")
        xgb_model = self.artifacts["xgb_model"]
        prediction = xgb_model.predict(hybrid_features)[0]

        # 5. Jelaskan Prediksi dengan SHAP
        logger.info("Menghasilkan penjelasan XAI dengan SHAP...")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(hybrid_features)

        # 6. Terjemahkan ke Laporan Manajerial
        logger.info("Menyusun laporan manajerial...")
        report = self._format_report(prediction, shap_values, explainer.expected_value)

        return report

    def _format_report(
        self, prediction: float, shap_values: np.ndarray, base_value: float
    ) -> str:
        """Menerjemahkan output numerik menjadi laporan teks yang mudah dibaca."""

        # Klasifikasi status FCR
        pred_status = "Kritis"
        for status, threshold in sorted(
            self.config.FCR_THRESHOLDS.items(), key=lambda item: item[1]
        ):
            if prediction <= threshold:
                pred_status = status
                break

        # Header Laporan
        report_lines = [
            "===========================================================",
            f"LAPORAN REKOMENDASI MANAJERIAL - {datetime.now().strftime('%d %B %Y')}",
            "===========================================================",
            f"\nPrediksi FCR untuk Besok: {prediction:.3f} (Status: {pred_status})\n",
            f"Analisis: Prediksi ini didasarkan pada FCR rata-rata (base value) sebesar {base_value:.3f} dan dipengaruhi oleh faktor-faktor berikut:",
        ]

        # Analisis Faktor Pendorong dari SHAP
        hybrid_feature_names = self.artifacts["model_config"]["hybrid_feature_names"]
        shap_df = pd.DataFrame(
            {"feature": hybrid_feature_names, "shap_value": shap_values[0]}
        ).sort_values(by="shap_value", key=abs, ascending=False)

        significant_drivers = []
        used_templates = set()

        if pred_status in ["Peringatan", "Kritis"]:
            report_lines.append("\n--- FAKTOR UTAMA PENYEBAB FCR TINGGI ---\n")
            top_drivers = shap_df[
                shap_df["shap_value"] > self.config.SHAP_SIGNIFICANCE_THRESHOLD
            ].head(3)

            for _, row in top_drivers.iterrows():
                feature_name = row["feature"]
                for key, template_name in self.config.FEATURE_TO_TEMPLATE_KEY.items():
                    if key in feature_name and template_name not in used_templates:
                        template = self.config.RECOMMENDATION_TEMPLATES[template_name]
                        significant_drivers.append(template)
                        used_templates.add(template_name)
                        report_lines.append(
                            f"-> {template['issue']} (Kontribusi SHAP: {row['shap_value']:.3f})"
                        )
                        break
        else:  # Baik atau Sangat Baik
            significant_drivers.append(
                self.config.RECOMMENDATION_TEMPLATES["performance_excellent"]
            )

        # Bagian Rekomendasi
        report_lines.append("\n--- REKOMENDASI TINDAKAN MANAJERIAL ---\n")
        if not significant_drivers:
            report_lines.append(
                "Tidak ada faktor risiko signifikan yang terdeteksi. Lanjutkan praktik manajemen standar."
            )
        else:
            for template in significant_drivers:
                report_lines.append(
                    f"Prioritas: {template['priority']} | Kategori: {template['category']}"
                )
                for action in template["actions"]:
                    report_lines.append(f"  - {action}")
                report_lines.append(f"  > Monitoring: {template['monitoring']}\n")

        report_lines.append(
            "==========================================================="
        )
        report_lines.append(
            "Laporan ini dihasilkan secara otomatis oleh Sistem Prediksi FCR & XAI."
        )

        return "\n".join(report_lines)


# =============================================================================
# CONTOH PENGGUNAAN
# =============================================================================
if __name__ == "__main__":
    # 1. Tentukan path ke folder model Anda
    # Ganti path ini sesuai dengan struktur folder Anda
    # Get the current script directory
    BASE_DIR = os.getcwd()
    MODELS_FOLDER_PATH = os.path.join(BASE_DIR, "5. Model", "models_terbaru")

    # 2. Siapkan data mentah baru
    #    Dalam skenario nyata, Anda akan memuat file CSV terbaru.
    #    Di sini, kita akan memuat data training yang sudah ada dan mengambil
    #    beberapa baris terakhir sebagai simulasi data baru.
    #    PASTIKAN JUMLAH BARIS >= sequence_length (misal, 7 hari terakhir)
    try:
        # Ganti path ini ke data mentah gabungan Anda
        ALL_RAW_DATA_PATH = "./3. non-normalize/NORMALISASI_PERIODE_14.csv"
        df_all_raw = pd.read_csv(ALL_RAW_DATA_PATH, sep=";")

        # Ambil 10 baris data terakhir sebagai contoh data baru
        # Dalam penggunaan nyata, ini adalah data historis hingga hari ini
        sample_new_data = df_all_raw.tail(10)
        logger.info(
            f"Menggunakan {len(sample_new_data)} baris terakhir dari '{ALL_RAW_DATA_PATH}' sebagai data input baru."
        )

        # 3. Inisialisasi recommender dan jalankan
        recommender = ManagerialRecommender(
            models_base_path=MODELS_FOLDER_PATH,
            base_dir=BASE_DIR
        )
        final_report = recommender.generate_recommendations(raw_data_df=sample_new_data)

        # 4. Tampilkan laporan
        print("\n\n")
        print(final_report)

    except FileNotFoundError:
        logger.error(
            f"File data sampel tidak ditemukan di '{ALL_RAW_DATA_PATH}'. Pastikan path benar."
        )
    except Exception as e:
        logger.error(f"Terjadi error pada saat eksekusi utama: {e}", exc_info=True)
