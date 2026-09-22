import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
import tensorflow as tf
from tcn import TCN
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import time

sys.path.append("1. Normalisasi")
try:
    from Feature.FeatureEngineeringPipeline import FeatureEngineeringPipeline
except ImportError:
    print("ERROR: File 'FeatureEngineeringPipeline.py' tidak ditemukan.")
    print(
        "Pastikan Anda telah menjalankan 'MainPipeline.py' dengan sukses terlebih dahulu."
    )

# Set page config
st.set_page_config(
    page_title="FCR Prediction App",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stNumberInput, .stFileUploader {
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Feature translations (same as original)
FEATURE_TRANSLATIONS = {
    "AGE": "Usia ayam (dalam hari)",
    "JUMLAH_AYAM_AKHIR": "Jumlah ayam yang masih hidup di akhir hari",
    "PERIODE": "Siklus pemeliharaan ayam",
    "MATI": "Jumlah ayam yang mati dalam sehari",
    "JML_DEPLESI": "Total ayam yang hilang per hari (mati atau dibuang)",
    "STD_CUM_DEPLESI_%": "Persentase rata-rata ayam yang hilang dari awal periode",
    "PAKAN_STD_CUM_GR/EK": "Standar jumlah pakan total per ayam (gram)",
    "DELTA_PAKAN_ZAK": "Selisih pakan yang diberikan dengan standar (zak)",
    "DELTA_PAKAN_CUM_GR/EKOR": "Selisih total pakan per ayam dibandingkan standar (gram)",
    "PAKAN_DG_RATIO": "Rasio pakan dibandingkan pertumbuhan harian ayam",
    "FCR_STD": "Standar efisiensi pakan",
    "FCR_ACT": "Efisiensi pakan aktual hari ini",
    "DELTA_FCR": "Selisih efisiensi pakan aktual dengan standar",
    "IP_STD": "Standar indeks performa ayam",
    "IP_ACT": "Indeks performa aktual ayam hari ini",
    "DELTA_ABW": "Selisih berat badan ayam aktual dengan standar",
    "DG_ACT": "Pertumbuhan harian ayam (penambahan berat, gram)",
    "FCR_ACT_lag_1": "Efisiensi pakan kemarin",
    "DG_ACT_lag_1": "Pertumbuhan harian ayam kemarin",
    "MATI_lag_1": "Jumlah ayam mati kemarin",
    "DELTA_PAKAN_GR/EKOR_lag_1": "Selisih pakan per ayam kemarin dibandingkan standar",
    "DELTA_DG_lag_1": "Selisih pertumbuhan harian kemarin dibandingkan standar",
    "FEED_EFFICIENCY_lag_1": "Efisiensi pakan invers kemarin (g berat/g pakan)",
    "DG_ACT_lag_2": "Pertumbuhan harian ayam dua hari lalu",
    "DG_ACT_lag_3": "Pertumbuhan harian ayam tiga hari lalu",
    "MATI_lag_2": "Jumlah ayam mati dua hari lalu",
    "MATI_lag_3": "Jumlah ayam mati tiga hari lalu",
    "DELTA_PAKAN_GR/EKOR_lag_2": "Selisih pakan per ayam dua hari lalu dibandingkan standar",
    "DELTA_PAKAN_GR/EKOR_lag_3": "Selisih pakan per ayam tiga hari lalu dibandingkan standar",
    "DELTA_DG_lag_2": "Selisih pertumbuhan harian dua hari lalu dibandingkan standar",
    "DELTA_DG_lag_3": "Selisih pertumbuhan harian tiga hari lalu dibandingkan standar",
    "FCR_ACT_lag_7": "Efisiensi pakan seminggu lalu",
    "DG_ACT_lag_7": "Pertumbuhan harian ayam seminggu lalu",
    "MATI_lag_7": "Jumlah ayam mati seminggu lalu",
    "JML_DEPLESI_lag_7": "Total ayam yang hilang seminggu lalu",
    "DELTA_PAKAN_GR/EKOR_lag_7": "Selisih pakan per ayam seminggu lalu dibandingkan standar",
    "DELTA_DG_lag_7": "Selisih pertumbuhan harian seminggu lalu dibandingkan standar",
    "FCR_ACT_lag_14": "Efisiensi pakan dua minggu lalu",
    "PAKAN_ACT_GR/EK_lag_14": "Jumlah pakan per ayam dua minggu lalu (gram)",
    "DG_ACT_lag_14": "Pertumbuhan harian ayam dua minggu lalu",
    "MATI_lag_14": "Jumlah ayam mati dua minggu lalu",
    "JML_DEPLESI_lag_14": "Total ayam yang hilang dua minggu lalu",
    "DELTA_PAKAN_GR/EKOR_lag_14": "Selisih pakan per ayam dua minggu lalu dibandingkan standar",
    "DELTA_DG_lag_14": "Selisih pertumbuhan harian dua minggu lalu dibandingkan standar",
    "PAKAN_ACT_GR/EK_rolling_std_3": "Ketidakstabilan jumlah pakan per ayam dalam 3 hari terakhir",
    "DG_ACT_rolling_mean_3": "Rata-rata pertumbuhan harian ayam dalam 3 hari terakhir",
    "DG_ACT_rolling_std_3": "Ketidakstabilan pertumbuhan harian dalam 3 hari terakhir",
    "DG_ACT_rolling_min_3": "Pertumbuhan harian paling kecil dalam 3 hari terakhir",
    "DG_ACT_rolling_max_3": "Pertumbuhan harian paling besar dalam 3 hari terakhir",
    "MATI_rolling_mean_3": "Rata-rata jumlah ayam mati dalam 3 hari terakhir",
    "MATI_rolling_std_3": "Ketidakstabilan jumlah ayam mati dalam 3 hari terakhir",
    "MATI_rolling_min_3": "Jumlah ayam mati paling sedikit dalam 3 hari terakhir",
    "MATI_rolling_max_3": "Jumlah ayam mati paling banyak dalam 3 hari terakhir",
    "JML_DEPLESI_rolling_std_3": "Ketidakstabilan total ayam yang hilang dalam 3 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_mean_3": "Rata-rata selisih pakan per ayam dalam 3 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_std_3": "Ketidakstabilan selisih pakan per ayam dalam 3 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_min_3": "Selisih pakan per ayam paling kecil dalam 3 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_max_3": "Selisih pakan per ayam paling besar dalam 3 hari terakhir",
    "DELTA_DG_rolling_mean_3": "Rata-rata selisih pertumbuhan harian dalam 3 hari terakhir",
    "DELTA_DG_rolling_min_3": "Selisih pertumbuhan harian paling kecil dalam 3 hari terakhir",
    "DELTA_DG_rolling_max_3": "Selisih pertumbuhan harian paling besar dalam 3 hari terakhir",
    "PAKAN_ACT_GR/EK_rolling_std_7": "Ketidakstabilan jumlah pakan per ayam dalam 7 hari terakhir",
    "DG_ACT_rolling_std_7": "Ketidakstabilan pertumbuhan harian dalam 7 hari terakhir",
    "DG_ACT_rolling_min_7": "Pertumbuhan harian paling kecil dalam 7 hari terakhir",
    "MATI_rolling_mean_7": "Rata-rata jumlah ayam mati dalam 7 hari terakhir",
    "MATI_rolling_std_7": "Ketidakstabilan jumlah ayam mati dalam 7 hari terakhir",
    "MATI_rolling_min_7": "Jumlah ayam mati paling sedikit dalam 7 hari terakhir",
    "MATI_rolling_max_7": "Jumlah ayam mati paling banyak dalam 7 hari terakhir",
    "JML_DEPLESI_rolling_std_7": "Ketidakstabilan total ayam yang hilang dalam 7 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_mean_7": "Rata-rata selisih pakan per ayam dalam 7 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_std_7": "Ketidakstabilan selisih pakan per ayam dalam 7 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_min_7": "Selisih pakan per ayam paling kecil dalam 7 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_max_7": "Selisih pakan per ayam paling besar dalam 7 hari terakhir",
    "DELTA_DG_rolling_mean_7": "Rata-rata selisih pertumbuhan harian dalam 7 hari terakhir",
    "DELTA_DG_rolling_std_7": "Ketidakstabilan selisih pertumbuhan harian dalam 7 hari terakhir",
    "DELTA_DG_rolling_min_7": "Selisih pertumbuhan harian paling kecil dalam 7 hari terakhir",
    "DELTA_DG_rolling_max_7": "Selisih pertumbuhan harian paling besar dalam 7 hari terakhir",
    "PAKAN_ACT_GR/EK_rolling_std_14": "Ketidakstabilan jumlah pakan per ayam dalam 14 hari terakhir",
    "DG_ACT_rolling_std_14": "Ketidakstabilan pertumbuhan harian dalam 14 hari terakhir",
    "DG_ACT_rolling_min_14": "Pertumbuhan harian paling kecil dalam 14 hari terakhir",
    "MATI_rolling_mean_14": "Rata-rata jumlah ayam mati dalam 14 hari terakhir",
    "MATI_rolling_std_14": "Ketidakstabilan jumlah ayam mati dalam 14 hari terakhir",
    "MATI_rolling_min_14": "Jumlah ayam mati paling sedikit dalam 14 hari terakhir",
    "MATI_rolling_max_14": "Jumlah ayam mati paling banyak dalam 14 hari terakhir",
    "JML_DEPLESI_rolling_std_14": "Ketidakstabilan total ayam yang hilang dalam 14 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_mean_14": "Rata-rata selisih pakan per ayam dalam 14 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_std_14": "Ketidakstabilan selisih pakan per ayam dalam 14 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_min_14": "Selisih pakan per ayam paling kecil dalam 14 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_rolling_max_14": "Selisih pakan per ayam paling besar dalam 14 hari terakhir",
    "DELTA_DG_rolling_mean_14": "Rata-rata selisih pertumbuhan harian dalam 14 hari terakhir",
    "DELTA_DG_rolling_std_14": "Ketidakstabilan selisih pertumbuhan harian dalam 14 hari terakhir",
    "DELTA_DG_rolling_min_14": "Selisih pertumbuhan harian paling kecil dalam 14 hari terakhir",
    "DELTA_DG_rolling_max_14": "Selisih pertumbuhan harian paling besar dalam 14 hari terakhir",
    "PAKAN_ACT_GR/EK_momentum_1": "Perubahan jumlah pakan per ayam dari kemarin ke hari ini",
    "PAKAN_ACT_GR/EK_acceleration": "Kecepatan perubahan jumlah pakan per ayam",
    "PAKAN_ACT_GR/EK_volatility_3": "Ketidakstabilan jumlah pakan per ayam dalam 3 hari terakhir",
    "PAKAN_ACT_GR/EK_trend_slope_3": "Arah perubahan jumlah pakan per ayam dalam 3 hari",
    "DG_ACT_momentum_1": "Perubahan pertumbuhan harian ayam dari kemarin ke hari ini",
    "DG_ACT_acceleration": "Kecepatan perubahan pertumbuhan harian ayam",
    "DG_ACT_volatility_3": "Ketidakstabilan pertumbuhan harian dalam 3 hari terakhir",
    "DG_ACT_trend_slope_3": "Arah perubahan pertumbuhan harian dalam 3 hari",
    "MATI_momentum_1": "Perubahan jumlah ayam mati dari kemarin ke hari ini",
    "MATI_acceleration": "Kecepatan perubahan jumlah ayam mati",
    "MATI_volatility_3": "Ketidakstabilan jumlah ayam mati dalam 3 hari terakhir",
    "MATI_trend_slope_3": "Arah perubahan jumlah ayam mati dalam 3 hari",
    "JML_DEPLESI_momentum_1": "Perubahan total ayam yang hilang dari kemarin ke hari ini",
    "JML_DEPLESI_acceleration": "Kecepatan perubahan total ayam yang hilang",
    "JML_DEPLESI_volatility_3": "Ketidakstabilan total ayam yang hilang dalam 3 hari terakhir",
    "ABW_ACT_acceleration": "Kecepatan perubahan berat badan ayam",
    "ABW_ACT_volatility_3": "Ketidakstabilan berat badan ayam dalam 3 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_momentum_1": "Perubahan selisih pakan per ayam dari kemarin ke hari ini",
    "DELTA_PAKAN_GR/EKOR_acceleration": "Kecepatan perubahan selisih pakan per ayam",
    "DELTA_PAKAN_GR/EKOR_volatility_3": "Ketidakstabilan selisih pakan per ayam dalam 3 hari terakhir",
    "DELTA_PAKAN_GR/EKOR_trend_slope_3": "Arah perubahan selisih pakan per ayam dalam 3 hari",
}


def asymmetric_objective(y_true, y_pred):
    """
    Fungsi loss kustom untuk XGBoost yang memberikan penalti lebih besar pada over-prediction.
    """
    residual = y_pred - y_true
    grad = np.where(residual > 0, 2.0 * 1.5 * residual, 4.0 * residual)
    hess = np.where(residual > 0, 2.0 * 1.5, 4.0)
    return grad, hess


# Helper classes (same as original, but modified for Streamlit display)
class EvaluationMetrics:
    def __init__(self):
        self.metrics_history = []

    def calculate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2_Score": r2,
            "MAPE": mape,
            "Mean_Residual": mean_residual,
            "Std_Residual": std_residual,
            "N_Samples": len(y_true),
        }
        return metrics

    def display_metrics(self, metrics, title="Model Evaluation Metrics"):
        st.subheader(title)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Samples", f"{metrics['N_Samples']}")
            st.metric("Mean Absolute Error", f"{metrics['MAE']:.6f}")
            st.metric("Mean Squared Error", f"{metrics['MSE']:.6f}")

        with col2:
            st.metric("Root Mean Squared Error", f"{metrics['RMSE']:.6f}")
            st.metric("R² Score", f"{metrics['R2_Score']:.6f}")
            st.metric("Mean Abs. Percentage Error", f"{metrics['MAPE']:.4f}%")

        with col3:
            st.metric("Mean Residual", f"{metrics['Mean_Residual']:.6f}")
            st.metric("Std Residual", f"{metrics['Std_Residual']:.6f}")

            if metrics["R2_Score"] >= 0.9:
                interpretation = "Excellent"
                color = "green"
            elif metrics["R2_Score"] >= 0.8:
                interpretation = "Very Good"
                color = "blue"
            elif metrics["R2_Score"] >= 0.7:
                interpretation = "Good"
                color = "orange"
            elif metrics["R2_Score"] >= 0.5:
                interpretation = "Moderate"
                color = "darkorange"
            else:
                interpretation = "Poor"
                color = "red"

            st.markdown(
                f"**Model Performance:** <span style='color:{color}'>{interpretation}</span> (R² = {metrics['R2_Score']:.4f})",
                unsafe_allow_html=True,
            )

    def create_evaluation_plots(self, y_true, y_pred):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color="blue", s=50)
        axes[0, 0].plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        axes[0, 0].set_xlabel("Actual FCR")
        axes[0, 0].set_ylabel("Predicted FCR")
        axes[0, 0].set_title("Actual vs Predicted FCR")
        axes[0, 0].grid(True, alpha=0.3)
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].annotate(
            f"R² = {r2:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Residuals vs Predicted
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color="green", s=50)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted FCR")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Predicted")
        axes[0, 1].grid(True, alpha=0.3)

        # Distribution of Residuals
        axes[1, 0].hist(
            residuals, bins=20, alpha=0.7, color="orange", edgecolor="black"
        )
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Residuals")
        axes[1, 0].axvline(
            np.mean(residuals),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(residuals):.4f}",
        )
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Distribution of Absolute Errors
        abs_errors = np.abs(residuals)
        axes[1, 1].hist(
            abs_errors, bins=20, alpha=0.7, color="purple", edgecolor="black"
        )
        axes[1, 1].set_xlabel("Absolute Error")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Absolute Errors")
        axes[1, 1].axvline(
            np.mean(abs_errors),
            color="red",
            linestyle="--",
            label=f"MAE: {np.mean(abs_errors):.4f}",
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


class DataNormalizer:
    def __init__(self, info_path: str, robust_scaler_path: str):
        with open(info_path, "r") as f:
            self.info = json.load(f)
        with open(robust_scaler_path, "rb") as f:
            self.robust_scaler = pickle.load(f)
        self.features_to_normalize = list(set(self.info["robust_features"]))
        self.feature_names_order = self.robust_scaler.feature_names_in_
        st.success("DataNormalizer berhasil dimuat.")
        st.info(
            f"Menggunakan satu RobustScaler untuk {len(self.features_to_normalize)} fitur."
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        st.info("Menerapkan normalisasi pada data...")
        if self.features_to_normalize:
            cols_to_transform = [
                col
                for col in self.feature_names_order
                if col in df_transformed.columns and col in self.features_to_normalize
            ]
            try:
                df_transformed[cols_to_transform] = self.robust_scaler.transform(
                    df_transformed[cols_to_transform]
                )
            except Exception as e:
                st.error(f"Error saat normalisasi: {str(e)}")
                try:
                    temp_df = pd.DataFrame()
                    for col in cols_to_transform:
                        if col in df_transformed.columns:
                            temp_df[col] = df_transformed[col]
                    transformed_values = self.robust_scaler.transform(temp_df)
                    for i, col in enumerate(cols_to_transform):
                        if col in df_transformed.columns:
                            df_transformed[col] = transformed_values[:, i]
                except Exception as e2:
                    st.error(
                        f"Gagal normalisasi dengan pendekatan alternatif: {str(e2)}"
                    )
                    st.warning("Mengembalikan data tanpa normalisasi")
        return df_transformed


class XAIExplainer:
    def __init__(self, feature_translations):
        self.feature_translations = feature_translations
        self.feature_thresholds = {
            "FCR_ACT_lag_1": {"high": 1.5, "low": 1.3},
            "PAKAN_DG_RATIO": {"high": 2.0, "low": 1.5},
            "DELTA_FCR": {"high": 0.2, "low": -0.2},
            "FEED_EFFICIENCY_lag_1": {"high": 0.77, "low": 0.67},
            "DELTA_PAKAN_ZAK": {"high": 0.5, "low": -0.5},
            "MATI_rolling_std_7": {"high": 10.0, "low": 2.0},
            "DELTA_DG_rolling_std_7": {"high": 5.0, "low": 1.0},
        }
        self.feature_units = {
            "FCR_ACT_lag_1": "FCR",
            "PAKAN_DG_RATIO": "rasio",
            "DELTA_FCR": "",
            "FEED_EFFICIENCY_lag_1": "g berat/g pakan",
            "DELTA_PAKAN_ZAK": "zak",
            "MATI_rolling_std_7": "ayam",
            "DELTA_DG_rolling_std_7": "gram",
        }

    def generate_explanation(
        self, feature_importance_dict, feature_values_dict, predicted_fcr, horizon
    ):
        """Validasi input sebelum membuat penjelasan"""
        if not feature_importance_dict or not feature_values_dict:
            st.error("Data feature importance atau feature values tidak valid")
            return {
                "predicted_fcr": predicted_fcr,
                "horizon": horizon,
                "explanations": ["Data tidak cukup untuk membuat penjelasan XAI"],
                "recommendations": ["Harap periksa input data dan coba lagi"],
                "top_features": [],
            }

        # Ubah dari 5 menjadi 10 fitur teratas
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:10]
        explanations = []
        recommendations = []

        for feature, importance in sorted_features:
            simple_name = self.feature_translations.get(feature, feature)
            value = feature_values_dict.get(feature, 0)
            unit = self.feature_units.get(feature, "")
            display_value = f"{value:.2f} {unit}".strip() if unit else f"{value:.2f}"

            if feature == "FCR_ACT_lag_1":
                if value > self.feature_thresholds["FCR_ACT_lag_1"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) terlalu tinggi, menunjukkan efisiensi pakan kemarin buruk."
                    )
                    recommendations.append(
                        "🚨 Tinjau jumlah pakan yang diberikan dan pastikan sesuai dengan pertumbuhan ayam."
                    )
                elif value < self.feature_thresholds["FCR_ACT_lag_1"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) sangat baik, menunjukkan efisiensi pakan kemarin optimal."
                    )
                    recommendations.append(
                        "✅ Pertahankan pola pemberian pakan saat ini."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, berkontribusi stabil pada prediksi."
                    )
                    recommendations.append(
                        "📊 Lanjutkan pemantauan efisiensi pakan harian."
                    )

            elif feature == "PAKAN_DG_RATIO":
                if value > self.feature_thresholds["PAKAN_DG_RATIO"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) terlalu tinggi, artinya pakan yang diberikan berlebihan dibandingkan pertumbuhan ayam."
                    )
                    recommendations.append(
                        "🔻 Kurangi pakan harian sebanyak 5-10 gram per ekor untuk menyeimbangkan pertumbuhan."
                    )
                elif value < self.feature_thresholds["PAKAN_DG_RATIO"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) sangat rendah, menunjukkan pakan cukup efisien."
                    )
                    recommendations.append(
                        "✅ Pertahankan jumlah pakan saat ini, tapi pastikan ayam mendapat nutrisi cukup."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, menunjukkan keseimbangan pakan dan pertumbuhan."
                    )
                    recommendations.append(
                        "📊 Lanjutkan pemberian pakan sesuai standar."
                    )

            elif feature == "DELTA_FCR":
                if value > self.feature_thresholds["DELTA_FCR"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan efisiensi pakan jauh di atas standar, yang meningkatkan FCR."
                    )
                    recommendations.append(
                        "⚠️ Tinjau ulang strategi pakan untuk mendekati standar efisiensi."
                    )
                elif value < self.feature_thresholds["DELTA_FCR"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan efisiensi pakan lebih baik dari standar."
                    )
                    recommendations.append("✅ Pertahankan strategi pakan saat ini.")
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) mendekati standar, berkontribusi stabil pada prediksi."
                    )
                    recommendations.append("📊 Lanjutkan pemantauan efisiensi pakan.")

            elif feature == "FEED_EFFICIENCY_lag_1":
                if value < self.feature_thresholds["FEED_EFFICIENCY_lag_1"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) terlalu rendah, menunjukkan efisiensi pakan kemarin buruk."
                    )
                    recommendations.append(
                        "🚨 Sesuaikan pakan agar lebih sesuai dengan pertumbuhan ayam."
                    )
                elif value > self.feature_thresholds["FEED_EFFICIENCY_lag_1"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) sangat baik, menunjukkan efisiensi pakan kemarin optimal."
                    )
                    recommendations.append("✅ Pertahankan pola pakan saat ini.")
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, berkontribusi stabil pada prediksi."
                    )
                    recommendations.append(
                        "📊 Lanjutkan pemantauan efisiensi pakan harian."
                    )

            elif feature == "DELTA_PAKAN_ZAK":
                if value > self.feature_thresholds["DELTA_PAKAN_ZAK"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan pakan berlebihan dibandingkan standar."
                    )
                    recommendations.append(
                        "🔻 Kurangi jumlah pakan sebanyak 0.5-1 zak per hari."
                    )
                elif value < self.feature_thresholds["DELTA_PAKAN_ZAK"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan pakan kurang dari standar."
                    )
                    recommendations.append(
                        "🔺 Tambah pakan secukupnya untuk memenuhi standar."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) sesuai standar, berkontribusi stabil pada prediksi."
                    )
                    recommendations.append(
                        "📊 Lanjutkan pemberian pakan sesuai standar."
                    )

            elif feature == "MATI_rolling_std_7":
                if value > self.feature_thresholds["MATI_rolling_std_7"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) tinggi, menunjukkan kematian ayam tidak stabil dalam 7 hari terakhir."
                    )
                    recommendations.append(
                        "🚨 Periksa kesehatan ayam dan kondisi kandang untuk mengurangi kematian."
                    )
                elif value < self.feature_thresholds["MATI_rolling_std_7"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) rendah, menunjukkan kematian ayam stabil."
                    )
                    recommendations.append(
                        "✅ Pertahankan kondisi kandang dan kesehatan ayam."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, menunjukkan stabilitas kematian ayam."
                    )
                    recommendations.append("📊 Lanjutkan pemantauan kesehatan ayam.")

            elif feature == "DELTA_DG_rolling_std_7":
                if value > self.feature_thresholds["DELTA_DG_rolling_std_7"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) tinggi, menunjukkan pertumbuhan ayam tidak stabil dalam 7 hari terakhir."
                    )
                    recommendations.append(
                        "⚠️ Periksa kualitas pakan dan kondisi kandang untuk menstabilkan pertumbuhan."
                    )
                elif value < self.feature_thresholds["DELTA_DG_rolling_std_7"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) rendah, menunjukkan pertumbuhan ayam stabil."
                    )
                    recommendations.append(
                        "✅ Pertahankan strategi pakan dan pemeliharaan saat ini."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, menunjukkan stabilitas pertumbuhan ayam."
                    )
                    recommendations.append("📊 Lanjutkan pemantauan pertumbuhan ayam.")

            # Tambahan untuk fitur lag_ dan rolling_
            elif "lag_" in feature:
                if "FCR" in feature.upper():
                    if value > 1.5:
                        explanations.append(
                            f"{simple_name} ({display_value}) menunjukkan efisiensi pakan historis buruk."
                        )
                        recommendations.append(
                            "🚨 Evaluasi pola pemberian pakan dalam periode tersebut."
                        )
                    else:
                        explanations.append(
                            f"{simple_name} ({display_value}) menunjukkan efisiensi pakan historis baik."
                        )
                        recommendations.append(
                            "✅ Pelajari pola pakan sukses dari periode tersebut."
                        )
                elif "DG" in feature.upper():
                    if value < 20:
                        explanations.append(
                            f"{simple_name} ({display_value}) menunjukkan pertumbuhan historis rendah."
                        )
                        recommendations.append(
                            "⚠️ Tingkatkan kualitas pakan untuk pertumbuhan optimal."
                        )
                    else:
                        explanations.append(
                            f"{simple_name} ({display_value}) menunjukkan pertumbuhan historis baik."
                        )
                        recommendations.append(
                            "✅ Pertahankan strategi pakan yang mendukung pertumbuhan."
                        )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) berkontribusi pada prediksi berdasarkan data historis."
                    )
                    recommendations.append(
                        "🔍 Monitor tren data ini untuk konsistensi jangka panjang."
                    )

            elif "rolling_" in feature:
                if "std" in feature:
                    if value > 5:
                        explanations.append(
                            f"{simple_name} ({display_value}) menunjukkan variabilitas tinggi dalam periode tersebut."
                        )
                        recommendations.append(
                            "⚠️ Stabilkan manajemen untuk mengurangi fluktuasi."
                        )
                    else:
                        explanations.append(
                            f"{simple_name} ({display_value}) menunjukkan stabilitas baik dalam periode tersebut."
                        )
                        recommendations.append(
                            "✅ Pertahankan konsistensi manajemen saat ini."
                        )
                elif "mean" in feature:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan rata-rata dalam periode tersebut."
                    )
                    recommendations.append(
                        "📊 Gunakan sebagai baseline untuk evaluasi performa."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) berkontribusi pada prediksi berdasarkan tren periode."
                    )
                    recommendations.append(
                        "🔍 Analisis tren ini untuk optimasi jangka panjang."
                    )

            else:
                # Untuk fitur umum lainnya
                if value > 100:  # Threshold umum untuk nilai tinggi
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan nilai tinggi yang mempengaruhi prediksi."
                    )
                    recommendations.append(
                        "⚠️ Evaluasi faktor ini untuk optimasi lebih lanjut."
                    )
                elif value < 0:  # Nilai negatif
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan nilai negatif yang perlu perhatian."
                    )
                    recommendations.append(
                        "🚨 Periksa dan perbaiki faktor yang menyebabkan nilai negatif."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) berkontribusi pada prediksi FCR."
                    )
                    recommendations.append(
                        "📊 Lanjutkan pemantauan data ini untuk konsistensi."
                    )

        return {
            "predicted_fcr": predicted_fcr,
            "horizon": horizon,
            "explanations": explanations,
            "recommendations": recommendations,
            "top_features": sorted_features,
        }

    def display_explanation(self, explanation_dict, st_container=None):
        """Menampilkan penjelasan XAI di Streamlit"""
        display = st_container if st_container else st

        display.subheader(
            f"📊 Penjelasan Prediksi FCR (Horizon +{explanation_dict['horizon']} Hari)"
        )

        # Tampilkan prediksi
        cols = display.columns(3)
        cols[0].metric("FCR Diprediksi", f"{explanation_dict['predicted_fcr']:.4f}")

        # Tab untuk penjelasan dan rekomendasi
        tab1, tab2 = display.tabs(["Faktor Penentu", "Rekomendasi Manajerial"])

        with tab1:
            display.markdown("### Faktor Utama yang Mempengaruhi Prediksi")
            for i, explanation in enumerate(explanation_dict["explanations"], 1):
                display.markdown(f"{i}. {explanation}")
                display.progress(min(0.9, 0.2 + i * 0.15))  # Visualisasi progress bar

        with tab2:
            display.markdown("### Rekomendasi untuk Tindakan Manajerial")
            for i, recommendation in enumerate(explanation_dict["recommendations"], 1):
                display.success(f"✅ {recommendation}")

        # Tampilkan feature importance plot
        display.markdown("---")
        self.create_simple_feature_importance_plot(
            {k: v for k, v in explanation_dict["top_features"]}
        )

    def create_simple_feature_importance_plot(self, feature_importance_dict):
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:5]
        feature_names = [
            self.feature_translations.get(f[0], f[0]) for f in sorted_features
        ]
        importance_values = [f[1] for f in sorted_features]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(feature_names)), importance_values, color="#4ECDC4")
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Pentingnya Fitur")
        ax.set_title(
            "Faktor Utama yang Memengaruhi Prediksi FCR", fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def show_xai_explanation(explanation_dict):
        """Template tampilan XAI yang responsif"""

        st.markdown(
            """
      <style>
      .xai-header {
            color: #2e86c1;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
      }
      .recommendation-card {
            background-color: #eaf2f8;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      </style>
      """,
            unsafe_allow_html=True,
        )

        # Header
        st.markdown(
            f"""
      <h2 class='xai-header'>
      🧠 Analisis Prediksi FCR (Horizon +{explanation_dict['horizon']} Hari)
      </h2>
      """,
            unsafe_allow_html=True,
        )

        # Predicted value
        st.metric(
            "**FCR Diprediksi**",
            f"{explanation_dict['predicted_fcr']:.4f}",
            help="Feed Conversion Ratio yang diprediksi oleh model",
        )

        # Main content in tabs
        tab1, tab2, tab3 = st.tabs(
            ["📌 Faktor Kunci", "💡 Rekomendasi", "📈 Visualisasi"]
        )

        with tab1:
            st.subheader("Faktor Penentu Utama")
            for i, (feature, importance) in enumerate(
                explanation_dict["top_features"], 1
            ):
                st.markdown(
                    f"""
                  <div style="margin-bottom: 15px;">
                  <b>{i}. {FEATURE_TRANSLATIONS.get(feature, feature)}</b>
                  <br>Pengaruh: <b>{importance:.2f}</b>
                  </div>
                  """,
                    unsafe_allow_html=True,
                )

        with tab2:
            st.subheader("Rekomendasi Manajerial")
            for i, recommendation in enumerate(explanation_dict["recommendations"], 1):
                st.markdown(
                    f"""
                  <div class="recommendation-card">
                  <b>✅ Rekomendasi {i}:</b> {recommendation}
                  </div>
                  """,
                    unsafe_allow_html=True,
                )

        with tab3:
            self.create_simple_feature_importance_plot(
                {k: v for k, v in explanation_dict["top_features"]}
            )
            st.caption("Visualisasi pengaruh fitur terhadap prediksi FCR")


class FeatureAnalyzer:
    def __init__(self):
        self.feature_names = []
        self.feature_importance = []
        self.feature_values = []
        self.temporal_features = []
        self.tabular_features = []

    def extract_feature_importance(self, xgb_model, feature_names):
        importance_dict = xgb_model.get_booster().get_score(importance_type="weight")
        feature_importance_mapped = {}
        for i, name in enumerate(feature_names):
            feature_key = f"f{i}"
            feature_importance_mapped[name] = importance_dict.get(feature_key, 0)
        return feature_importance_mapped

    def categorize_features(self, feature_names, n_tabular_features):
        tabular_features = feature_names[:n_tabular_features]
        temporal_features = feature_names[n_tabular_features:]
        return tabular_features, temporal_features

    def create_feature_importance_plot(
        self,
        feature_importance_dict,
        title="Top 50 Feature Importance untuk Prediksi FCR",
    ):
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:50]
        feature_names = [
            FEATURE_TRANSLATIONS.get(item[0], item[0]) for item in sorted_features
        ]
        importance_values = [item[1] for item in sorted_features]

        fig, ax = plt.subplots(figsize=(12, 10))
        bars = ax.barh(range(len(feature_names)), importance_values)
        colors = [
            "#FF6B6B" if name.startswith("temporal_") or "TCN" in name else "#4ECDC4"
            for name in feature_names
        ]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Feature Importance (Weight)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#FF6B6B", label="Fitur Temporal (TCN)"),
            Patch(facecolor="#4ECDC4", label="Fitur Tabular"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        return sorted_features

    def create_feature_distribution_plot(self, feature_values, feature_names):
        n_features_to_plot = min(12, len(feature_names))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        for i in range(n_features_to_plot):
            if i < len(feature_values[0]):
                values = [row[i] for row in feature_values]
                axes[i].hist(
                    values, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
                )
                axes[i].set_title(
                    f"{FEATURE_TRANSLATIONS.get(feature_names[i], feature_names[i])[:20]}...",
                    fontsize=10,
                )
                axes[i].set_xlabel("Nilai")
                axes[i].set_ylabel("Frekuensi")
                axes[i].grid(alpha=0.3)
        for i in range(n_features_to_plot, len(axes)):
            axes[i].set_visible(False)
        plt.suptitle(
            "Distribusi Nilai Fitur untuk Prediksi", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def create_feature_correlation_heatmap(self, feature_values, feature_names):
        df_features = pd.DataFrame(
            feature_values,
            columns=[FEATURE_TRANSLATIONS.get(name, name) for name in feature_names],
        )
        correlation_matrix = df_features.corr()
        n_features = min(15, len(feature_names))
        top_features = [
            FEATURE_TRANSLATIONS.get(name, name) for name in feature_names[:n_features]
        ]
        correlation_subset = correlation_matrix.loc[top_features, top_features]

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_subset, dtype=bool))
        sns.heatmap(
            correlation_subset,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            fmt=".2f",
            ax=ax,
        )
        plt.title("Korelasi Antar Fitur Penting", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def create_temporal_vs_tabular_comparison(
        self, feature_importance_dict, n_tabular_features
    ):
        tabular_importance = sum(
            [
                imp
                for i, (name, imp) in enumerate(feature_importance_dict.items())
                if i < n_tabular_features
            ]
        )
        temporal_importance = sum(
            [
                imp
                for i, (name, imp) in enumerate(feature_importance_dict.items())
                if i >= n_tabular_features
            ]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        labels = ["Fitur Tabular", "Fitur Temporal (TCN)"]
        values = [tabular_importance, temporal_importance]
        colors = ["#4ECDC4", "#FF6B6B"]

        ax1.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Kontribusi Fitur Temporal vs Tabular", fontweight="bold")

        ax2.bar(labels, values, color=colors, alpha=0.8)
        ax2.set_ylabel("Total Feature Importance")
        ax2.set_title("Perbandingan Total Importance", fontweight="bold")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        return tabular_importance, temporal_importance


class HybridPredictor:
    def __init__(self, models_dir: str, config_dir: str, sliding_window: int = None):
        st.info("Memuat semua model dan pipeline untuk prediksi...")

        # Load feature engineering pipeline
        fe_path = os.path.join(models_dir, "feature_engineering_pipeline.pkl")
        with open(fe_path, "rb") as f:
            self.feature_pipeline = pickle.load(f)

        # Load normalizer
        info_path = os.path.join(config_dir, "normalization_info.json")
        robust_path = os.path.join(models_dir, "robust_scaler.pkl")
        self.normalizer = DataNormalizer(info_path, robust_path)

        # Load TCN encoder
        tcn_path = os.path.join(models_dir, "tcn_encoder.keras")
        config_path = os.path.join(models_dir, "hybrid_model_config.json")
        self.tcn_encoder = tf.keras.models.load_model(
            tcn_path, custom_objects={"TCN": TCN}
        )

        # Load config
        with open(config_path, "r") as f:
            self.hybrid_config = json.load(f)

        self.sequence_length = (
            sliding_window if sliding_window else self.hybrid_config["sequence_length"]
        )
        self.horizons = [1, 3]
        self.xgb_models = {}

        # Load XGBoost models for each horizon
        for horizon in self.horizons:
            xgb_path = os.path.join(
                models_dir, f"xgboost_hybrid_model_horizon_{horizon}d.pkl"
            )
            with open(xgb_path, "rb") as f:
                self.xgb_models[f"horizon_{horizon}d"] = pickle.load(f)

        self.analyzer = FeatureAnalyzer()
        self.evaluator = EvaluationMetrics()
        self.explainer = XAIExplainer(FEATURE_TRANSLATIONS)

        st.success("Semua komponen prediksi berhasil dimuat.")
        st.info(f"Model mendukung prediksi untuk horizon: {self.horizons} hari")
        st.info(f"Menggunakan sliding window: {self.sequence_length} hari")

    def predict_single_sample(
        self, new_day_data: pd.DataFrame, historical_data: pd.DataFrame, horizon=None
    ):
        with st.spinner("Menggabungkan data..."):
            combined_data = pd.concat(
                [historical_data, new_day_data], ignore_index=True
            )
            combined_data = combined_data.sort_values(["PERIODE", "AGE"]).reset_index(
                drop=True
            )

        with st.spinner("Menjalankan rekayasa fitur..."):
            data_enhanced_full = self.feature_pipeline.transform(combined_data)
            data_enhanced = data_enhanced_full.tail(len(new_day_data))

        with st.spinner("Menormalisasi data..."):
            data_normalized = self.normalizer.transform(data_enhanced)
            historical_enhanced = self.feature_pipeline.transform(historical_data)
            historical_normalized = self.normalizer.transform(historical_enhanced)
            full_normalized_history = pd.concat(
                [historical_normalized, data_normalized], ignore_index=True
            )

            if len(full_normalized_history) < self.sequence_length:
                st.error(
                    f"Data tidak cukup untuk sekuens. Diperlukan {self.sequence_length} hari, tersedia {len(full_normalized_history)} hari."
                )
                return None

            last_sequence = full_normalized_history.tail(self.sequence_length)
            features_for_model = [
                col
                for col in last_sequence.columns
                if col not in ["PERIODE", "FCR_ACT", "TANGGAL"]
            ]
            last_sequence_values = last_sequence[features_for_model].values.reshape(
                1, self.sequence_length, -1
            )

        with st.spinner("Mengekstrak fitur temporal..."):
            temporal_features = self.tcn_encoder.predict(
                last_sequence_values, verbose=0
            )
            tabular_features = data_normalized[features_for_model].values.reshape(1, -1)
            hybrid_features = np.concatenate(
                [tabular_features, temporal_features], axis=1
            )

        predictions = {}
        if horizon is not None:
            if horizon not in self.horizons:
                st.error(
                    f"Horizon {horizon} tidak didukung. Horizon yang tersedia: {self.horizons}"
                )
                return None
            model_key = f"horizon_{horizon}d"
            predictions[horizon] = self.xgb_models[model_key].predict(hybrid_features)[
                0
            ]
            return predictions[horizon]
        else:
            for h in self.horizons:
                model_key = f"horizon_{h}d"
                predictions[h] = self.xgb_models[model_key].predict(hybrid_features)[0]
            return predictions

    def evaluate_test_data(
        self,
        test_data_path: str,
        horizon=None,
        max_rows=35,
    ):
        st.info("--- Memulai Evaluasi Model pada Data Test ---")
        if horizon is not None:
            st.info(f"Evaluasi untuk horizon: {horizon} hari")
            if horizon not in self.horizons:
                st.error(
                    f"Horizon {horizon} tidak didukung. Horizon yang tersedia: {self.horizons}"
                )
                return None
        else:
            st.info(f"Evaluasi untuk semua horizon: {self.horizons}")

        if not os.path.exists(test_data_path):
            st.error(f"File test data tidak ditemukan di {test_data_path}")
            return None

        with st.spinner("Memuat data test..."):
            df_test = pd.read_csv(test_data_path, sep=";")
            st.info(f"Data test dimuat: {len(df_test)} samples")

            if len(df_test) > max_rows:
                st.info(f"Membatasi data input ke {max_rows} baris teratas")
                df_test = df_test.head(max_rows)

            unique_combinations = (
                df_test[["PERIODE", "AGE"]]
                .drop_duplicates()
                .sort_values(["PERIODE", "AGE"])
            )

            predictions = []
            actual_values = []
            periode_values = []
            age_values = []
            valid_predictions = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, (_, row) in enumerate(unique_combinations.iterrows()):
                periode = row["PERIODE"]
                age = row["AGE"]

                try:
                    current_sample = df_test[
                        (df_test["PERIODE"] == periode) & (df_test["AGE"] == age)
                    ].copy()

                    if current_sample.empty or "FCR_ACT" not in current_sample.columns:
                        continue

                    min_history_days = max(3, self.sequence_length - 1)
                    historical_data = df_test[
                        (df_test["PERIODE"] == periode)
                        & (df_test["AGE"] < age)
                        & (df_test["AGE"] >= max(1, age - min_history_days))
                    ].copy()

                    if len(historical_data) < min_history_days:
                        continue

                    if horizon is not None:
                        target_age = age + horizon
                        target_sample = df_test[
                            (df_test["PERIODE"] == periode)
                            & (df_test["AGE"] == target_age)
                        ]
                        if (
                            target_sample.empty
                            or "FCR_ACT" not in target_sample.columns
                        ):
                            continue
                        actual_fcr = target_sample["FCR_ACT"].iloc[0]
                    else:
                        h = self.horizons[0]
                        target_age = age + h
                        target_sample = df_test[
                            (df_test["PERIODE"] == periode)
                            & (df_test["AGE"] == target_age)
                        ]
                        if (
                            target_sample.empty
                            or "FCR_ACT" not in target_sample.columns
                        ):
                            continue
                        actual_fcr = target_sample["FCR_ACT"].iloc[0]

                    predicted_fcr = self.predict_single_sample(
                        current_sample, historical_data, horizon
                    )

                    predictions.append(predicted_fcr)
                    actual_values.append(actual_fcr)
                    periode_values.append(periode)
                    age_values.append(target_age)
                    valid_predictions += 1

                    progress = (idx + 1) / len(unique_combinations)
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Memproses {idx + 1}/{len(unique_combinations)} sampel..."
                    )

                except Exception as e:
                    st.warning(
                        f"Error processing PERIODE {periode}, AGE {age}: {str(e)}"
                    )
                    continue

            st.success(
                f"Berhasil memproses {valid_predictions} prediksi dari {len(unique_combinations)} sampel"
            )

            if len(predictions) == 0:
                st.error(
                    "Tidak ada prediksi yang berhasil. Periksa data dan konfigurasi model."
                )
                return None

            y_true = np.array(actual_values)
            y_pred = np.array(predictions)

            st.info("Menghitung metrics evaluasi...")
            metrics = self.evaluator.calculate_metrics(y_true, y_pred)

            # Display metrics
            self.evaluator.display_metrics(metrics, "Test Data Evaluation Results")

            # Show evaluation plots
            st.subheader("Visualisasi Evaluasi Model")
            self.evaluator.create_evaluation_plots(y_true, y_pred)

            # Create results dataframe
            results_df = pd.DataFrame(
                {
                    "PERIODE": periode_values,
                    "AGE": age_values,
                    "Actual_FCR": y_true,
                    "Predicted_FCR": y_pred,
                    "Absolute_Error": np.abs(y_true - y_pred),
                    "Squared_Error": (y_true - y_pred) ** 2,
                    "Percentage_Error": np.abs((y_true - y_pred) / y_true) * 100,
                }
            )

            return metrics, y_true, y_pred, results_df

    def predict_with_analysis(
        self,
        new_day_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        horizon=None,
    ):
        st.info("--- Memulai Proses Prediksi dengan Analisis ---")
        if horizon is not None:
            st.info(f"Prediksi untuk horizon: {horizon} hari")
            if horizon not in self.horizons:
                st.error(
                    f"Horizon {horizon} tidak didukung. Horizon yang tersedia: {self.horizons}"
                )
                return None
        else:
            st.info(f"Prediksi untuk semua horizon: {self.horizons}")
            horizon = self.horizons[0]

        with st.spinner("Langkah A: Menjalankan rekayasa fitur..."):
            combined_data = pd.concat(
                [historical_data, new_day_data], ignore_index=True
            )
            combined_data = combined_data.sort_values(["PERIODE", "AGE"]).reset_index(
                drop=True
            )
            data_enhanced_full = self.feature_pipeline.transform(combined_data)
            data_enhanced = data_enhanced_full.tail(len(new_day_data))
            st.info(
                f"Rekayasa fitur selesai. Menghasilkan {data_enhanced.shape[1]} fitur."
            )

        with st.spinner("Langkah B: Menjalankan normalisasi..."):
            data_normalized = self.normalizer.transform(data_enhanced)
            historical_enhanced = self.feature_pipeline.transform(historical_data)
            historical_normalized = self.normalizer.transform(historical_enhanced)

        with st.spinner("Langkah C: Menjalankan prediksi model hibrida..."):
            full_normalized_history = pd.concat(
                [historical_normalized, data_normalized], ignore_index=True
            )

            if len(full_normalized_history) < self.sequence_length:
                st.error(
                    f"Data tidak cukup untuk sekuens. Diperlukan {self.sequence_length} hari, tersedia {len(full_normalized_history)} hari."
                )
                return None

            last_sequence = full_normalized_history.tail(self.sequence_length)
            features_for_model = [
                col
                for col in last_sequence.columns
                if col not in ["PERIODE", "FCR_ACT", "TANGGAL"]
            ]
            last_sequence_values = last_sequence[features_for_model].values.reshape(
                1, self.sequence_length, -1
            )
            temporal_features = self.tcn_encoder.predict(
                last_sequence_values, verbose=0
            )
            tabular_features = data_normalized[features_for_model].values.reshape(1, -1)
            hybrid_features = np.concatenate(
                [tabular_features, temporal_features], axis=1
            )
            feature_values_raw = dict(
                zip(features_for_model, data_enhanced[features_for_model].iloc[-1])
            )

            if horizon is not None:
                model_key = f"horizon_{horizon}d"
                final_prediction = self.xgb_models[model_key].predict(hybrid_features)
                predictions = {horizon: final_prediction[0]}
            else:
                predictions = {}
                for h in self.horizons:
                    model_key = f"horizon_{h}d"
                    predictions[h] = self.xgb_models[model_key].predict(
                        hybrid_features
                    )[0]

        with st.spinner("Langkah D: Menjalankan analisis fitur..."):
            n_tabular_features = len(features_for_model)
            n_temporal_features = temporal_features.shape[1]
            h_for_analysis = horizon if horizon is not None else self.horizons[0]
            model_key = f"horizon_{h_for_analysis}d"

            all_feature_names = features_for_model + [
                f"temporal_feature_{i+1}" for i in range(n_temporal_features)
            ]

            feature_importance_dict = self._analyze_features(
                hybrid_features,
                n_tabular_features,
                features_for_model,
                n_temporal_features,
                model_key,
                h_for_analysis,
            )

        with st.spinner("🧠 Membuat penjelasan XAI..."):
            explanation_dict = self.explainer.generate_explanation(
                feature_importance_dict,
                feature_values_raw,
                predictions[h_for_analysis],
                h_for_analysis,
            )

            # Display explanation
            self.explainer.display_explanation(explanation_dict)

            # Show simple feature importance
            st.subheader("Fitur Paling Berpengaruh pada Prediksi")
            self.explainer.create_simple_feature_importance_plot(
                feature_importance_dict
            )

            # Ekspander untuk detail teknis
            with st.expander("🔍 Detail Teknis untuk Analis"):
                st.write("Debug - Feature Importance:", feature_importance_dict)
                st.write("Debug - Feature Values:", feature_values_raw)

                col1, col2 = st.columns(2)
                with col1:
                    self.analyzer.create_feature_distribution_plot(
                        [hybrid_features[0]], all_feature_names
                    )
                with col2:
                    self.analyzer.create_feature_correlation_heatmap(
                        [hybrid_features[0]], all_feature_names
                    )

        return predictions

    def _analyze_features(
        self,
        hybrid_features,
        n_tabular_features,
        base_feature_names,
        n_temporal_features,
        model_key="horizon_1d",
        horizon=1,
    ):
        tabular_feature_names = base_feature_names[:n_tabular_features]
        temporal_feature_names = [
            f"temporal_feature_{i+1}" for i in range(n_temporal_features)
        ]
        all_feature_names = tabular_feature_names + temporal_feature_names

        feature_importance_dict = self.analyzer.extract_feature_importance(
            self.xgb_models[model_key], all_feature_names
        )

        st.subheader("Analisis Fitur")

        # Feature importance plot
        st.markdown("#### Feature Importance")
        top_features = self.analyzer.create_feature_importance_plot(
            feature_importance_dict,
            title=f"Feature Importance (Horizon {horizon} hari)",
        )

        # Feature distribution
        st.markdown("#### Distribusi Fitur")
        self.analyzer.create_feature_distribution_plot(
            [hybrid_features[0]], all_feature_names
        )

        # Temporal vs tabular comparison
        st.markdown("#### Perbandingan Fitur Temporal vs Tabular")
        tab_importance, temp_importance = (
            self.analyzer.create_temporal_vs_tabular_comparison(
                feature_importance_dict, n_tabular_features
            )
        )

        # Display summary
        st.subheader(f"Ringkasan Analisis Fitur (Horizon {horizon} Hari)")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Fitur Tabular", n_tabular_features)
            st.metric(
                "Kontribusi Fitur Tabular",
                f"{tab_importance:.2f} ({tab_importance/(tab_importance+temp_importance)*100:.1f}%)",
            )

        with col2:
            st.metric("Total Fitur Temporal", n_temporal_features)
            st.metric(
                "Kontribusi Fitur Temporal",
                f"{temp_importance:.2f} ({temp_importance/(tab_importance+temp_importance)*100:.1f}%)",
            )

        return feature_importance_dict


# Streamlit App
def main():
    st.title("🐔 Aplikasi Prediksi FCR Ayam Hybrid (TCN + XGBoost)")
    st.markdown(
        """
    Aplikasi ini memprediksi Feed Conversion Ratio (FCR) ayam menggunakan model hybrid yang menggabungkan:
    - **Temporal Convolutional Network (TCN)** untuk menangkap pola temporal
    - **XGBoost** untuk prediksi akhir berdasarkan fitur tabular dan temporal
    """
    )

    # Sidebar configuration
    st.sidebar.title("Konfigurasi Model")
    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, "5. Model", "models_terbaru")
    CONFIG_DIR = os.path.join(BASE_DIR, "2. Hasil Normalisasi", "Terbaru")

    # Check if model files exist
    required_files = [
        os.path.join(MODELS_DIR, "feature_engineering_pipeline.pkl"),
        os.path.join(CONFIG_DIR, "normalization_info.json"),
        os.path.join(MODELS_DIR, "robust_scaler.pkl"),
        os.path.join(MODELS_DIR, "tcn_encoder.keras"),
        os.path.join(MODELS_DIR, "hybrid_model_config.json"),
        os.path.join(MODELS_DIR, "xgboost_hybrid_model_horizon_1d.pkl"),
        os.path.join(MODELS_DIR, "xgboost_hybrid_model_horizon_3d.pkl"),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.sidebar.error("File model tidak lengkap. File berikut tidak ditemukan:")
        for f in missing_files:
            st.sidebar.write(f"- {f}")
        st.error(
            "Tidak dapat memuat model. Pastikan semua file model ada di direktori yang benar."
        )
        return

    # Sliding window input
    sliding_window = st.sidebar.number_input(
        "Ukuran Sliding Window (default: gunakan nilai dari model)",
        min_value=1,
        max_value=30,
        value=None,
        step=1,
        help="Biarkan kosong untuk menggunakan nilai default dari model",
    )

    # Operation mode selection
    mode = st.sidebar.radio(
        "Pilih Mode Operasi:",
        ["Prediksi Sampel Tunggal dengan Analisis", "Evaluasi pada Seluruh Data Test"],
        index=0,
    )

    # Horizon selection
    horizon_choice = st.sidebar.radio(
        "Pilih Horizon untuk Prediksi:",
        ["Horizon +1 hari", "Horizon +3 hari", "Kedua horizon"],
        index=0,
    )

    if horizon_choice == "Horizon +1 hari":
        horizon_to_evaluate = 1
    elif horizon_choice == "Horizon +3 hari":
        horizon_to_evaluate = 3
    else:
        horizon_to_evaluate = None

    # Initialize predictor
    try:
        predictor = HybridPredictor(
            models_dir=MODELS_DIR,
            config_dir=CONFIG_DIR,
            sliding_window=sliding_window if sliding_window else None,
        )
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.error(
            "Pastikan Anda telah menjalankan 'MainPipeline.py' dengan sukses terlebih dahulu."
        )
        return

    # Load test data
    test_data_path = os.path.join(
        BASE_DIR, "3. non-normalize", "NORMALISASI_PERIODE_14.csv"
    )

    if not os.path.exists(test_data_path):
        st.error(f"File data test tidak ditemukan di {test_data_path}")
        st.error("Pastikan file data test ada di direktori yang benar.")
        return

    if mode == "Evaluasi pada Seluruh Data Test":
        st.header("Evaluasi Model pada Data Test")

        max_rows = st.slider(
            "Jumlah Maksimal Baris Data untuk Evaluasi",
            min_value=10,
            max_value=100,
            value=35,
            step=5,
        )

        if st.button("Mulai Evaluasi"):
            with st.spinner("Menjalankan evaluasi..."):
                evaluation_results = predictor.evaluate_test_data(
                    test_data_path=test_data_path,
                    horizon=horizon_to_evaluate,
                    max_rows=max_rows,
                )

                if evaluation_results:
                    metrics, y_true, y_pred, results_df = evaluation_results

                    # Show detailed results
                    st.subheader("Hasil Prediksi Detail")
                    st.dataframe(
                        results_df.style.format(
                            {
                                "Actual_FCR": "{:.6f}",
                                "Predicted_FCR": "{:.6f}",
                                "Absolute_Error": "{:.6f}",
                                "Squared_Error": "{:.6f}",
                                "Percentage_Error": "{:.2f}%",
                            }
                        )
                    )

                    # Download button for results
                    csv = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Unduh Hasil Prediksi (CSV)",
                        data=csv,
                        file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

    else:  # Single sample prediction with analysis
        st.header("Prediksi Sampel Tunggal dengan Analisis")

        with st.spinner("Memuat data contoh..."):
            df_raw_sample = pd.read_csv(test_data_path, sep=";")

            if len(df_raw_sample) > 40:
                df_raw_sample = df_raw_sample.head(35)
                st.info("Data dibatasi ke 35 baris teratas")

            # Select sample to predict
            selected_periode = st.selectbox(
                "Pilih Periode", df_raw_sample["PERIODE"].unique()
            )

            selected_age = st.selectbox(
                "Pilih Hari (AGE) untuk Prediksi",
                sorted(
                    df_raw_sample[df_raw_sample["PERIODE"] == selected_periode][
                        "AGE"
                    ].unique()
                ),
            )

            new_data = df_raw_sample[
                (df_raw_sample["PERIODE"] == selected_periode)
                & (df_raw_sample["AGE"] == selected_age)
            ].copy()

            min_history_days = max(3, predictor.sequence_length - 1)
            historical_data = df_raw_sample[
                (df_raw_sample["PERIODE"] == selected_periode)
                & (df_raw_sample["AGE"] < selected_age)
                & (df_raw_sample["AGE"] >= max(1, selected_age - min_history_days))
            ].copy()

            st.info(
                f"Data baru untuk diprediksi: Periode {new_data['PERIODE'].iloc[0]}, Hari ke-{new_data['AGE'].iloc[0]}"
            )
            st.info(f"Data historis yang digunakan: {len(historical_data)} hari.")
            st.info(f"Sequence length yang diperlukan: {predictor.sequence_length}")

            if st.button("Mulai Prediksi dan Analisis"):
                if not new_data.empty and len(historical_data) > 0:
                    with st.spinner("Menjalankan prediksi dan analisis..."):
                        predicted_values = predictor.predict_with_analysis(
                            new_day_data=new_data,
                            historical_data=historical_data,
                            horizon=horizon_to_evaluate,
                        )

                        if predicted_values:
                            st.subheader("Hasil Prediksi")

                            if isinstance(predicted_values, dict):
                                for h, pred_value in predicted_values.items():
                                    current_age = new_data["AGE"].iloc[0]
                                    target_age = current_age + h
                                    periode = new_data["PERIODE"].iloc[0]

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("PERIODE", periode)
                                    with col2:
                                        st.metric("AGE saat ini", current_age)
                                    with col3:
                                        st.metric(
                                            f"Prediksi FCR (horizon +{h} hari)",
                                            f"{pred_value:.6f}",
                                        )

                                    target_sample = df_raw_sample[
                                        (df_raw_sample["PERIODE"] == periode)
                                        & (df_raw_sample["AGE"] == target_age)
                                    ]

                                    if (
                                        not target_sample.empty
                                        and "FCR_ACT" in target_sample.columns
                                    ):
                                        actual_h = target_sample["FCR_ACT"].iloc[0]

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric(
                                                "FCR Aktual",
                                                f"{actual_h:.6f}",
                                                delta=f"{abs(pred_value - actual_h):.6f} (Absolute Error)",
                                            )
                                        with col2:
                                            st.metric(
                                                "Relative Error",
                                                f"{abs(pred_value - actual_h)/actual_h*100:.4f}%",
                                            )
                            else:
                                h = horizon_to_evaluate
                                current_age = new_data["AGE"].iloc[0]
                                target_age = current_age + h
                                periode = new_data["PERIODE"].iloc[0]

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("PERIODE", periode)
                                with col2:
                                    st.metric("AGE saat ini", current_age)
                                with col3:
                                    st.metric(
                                        f"Prediksi FCR (horizon +{h} hari)",
                                        f"{predicted_values:.6f}",
                                    )

                                target_sample = df_raw_sample[
                                    (df_raw_sample["PERIODE"] == periode)
                                    & (df_raw_sample["AGE"] == target_age)
                                ]

                                if (
                                    not target_sample.empty
                                    and "FCR_ACT" in target_sample.columns
                                ):
                                    actual_h = target_sample["FCR_ACT"].iloc[0]

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            "FCR Aktual",
                                            f"{actual_h:.6f}",
                                            delta=f"{abs(predicted_values - actual_h):.6f} (Absolute Error)",
                                        )
                                    with col2:
                                        st.metric(
                                            "Relative Error",
                                            f"{abs(predicted_values - actual_h)/actual_h*100:.4f}%",
                                        )
                else:
                    st.error(
                        "Tidak dapat melakukan prediksi sampel tunggal karena data tidak mencukupi."
                    )


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    asymmetric_objective = asymmetric_objective

    # Set style untuk visualisasi
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    main()
