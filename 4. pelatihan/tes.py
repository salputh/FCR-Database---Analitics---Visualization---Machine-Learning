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
from scipy.stats import pearsonr, spearmanr  # Added for correlation analysis

# Impor kelas pipeline dari file terpisah
try:
    import sys

    sys.path.append("1. Normalisasi")
    from Feature.FeatureEngineeringPipeline import FeatureEngineeringPipeline
except ImportError:
    print("ERROR: File 'FeatureEngineeringPipeline.py' tidak ditemukan.")
    exit()

# Matikan pesan log TensorFlow yang tidak relevan
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# Set style untuk visualisasi
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Tabel terjemahan fitur ke bahasa peternak
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


class EvaluationMetrics:
    """
    Kelas untuk menghitung dan memvisualisasikan metrics evaluasi model
    """

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

    def print_metrics_summary(self, metrics, title="Model Evaluation Metrics"):
        print(f"\n{'='*60}")
        print(f"   {title}")
        print(f"{'='*60}")
        print(f"Number of Samples       : {metrics['N_Samples']}")
        print(f"Mean Absolute Error     : {metrics['MAE']:.6f}")
        print(f"Mean Squared Error      : {metrics['MSE']:.6f}")
        print(f"Root Mean Squared Error : {metrics['RMSE']:.6f}")
        print(f"R² Score               : {metrics['R2_Score']:.6f}")
        print(f"Mean Absolute Perc. Err : {metrics['MAPE']:.4f}%")
        print(f"Mean Residual          : {metrics['Mean_Residual']:.6f}")
        print(f"Std Residual           : {metrics['Std_Residual']:.6f}")
        print(f"{'='*60}\n")
        if metrics["R2_Score"] >= 0.9:
            interpretation = "Excellent"
        elif metrics["R2_Score"] >= 0.8:
            interpretation = "Very Good"
        elif metrics["R2_Score"] >= 0.7:
            interpretation = "Good"
        elif metrics["R2_Score"] >= 0.5:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"
        print(f"Model Performance: {interpretation} (R² = {metrics['R2_Score']:.4f})")

    def create_evaluation_plots(self, y_true, y_pred, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
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
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color="green", s=50)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted FCR")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Predicted")
        axes[0, 1].grid(True, alpha=0.3)
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
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def create_metrics_comparison_plot(self, metrics_dict, save_path=None):
        plot_metrics = {
            k: v
            for k, v in metrics_dict.items()
            if k not in ["N_Samples"] and isinstance(v, (int, float))
        }
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        main_metrics = ["MAE", "MSE", "RMSE"]
        main_values = [plot_metrics.get(m, 0) for m in main_metrics]
        bars1 = axes[0, 0].bar(
            main_metrics,
            main_values,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
            alpha=0.8,
        )
        axes[0, 0].set_ylabel("Error Value")
        axes[0, 0].set_title("Error Metrics")
        axes[0, 0].grid(True, alpha=0.3)
        for bar, value in zip(bars1, main_values):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )
        mape_value = plot_metrics.get("MAPE", 0)
        bars2 = axes[0, 1].bar(["MAPE"], [mape_value], color="#FFA07A", alpha=0.8)
        axes[0, 1].set_ylabel("MAPE (%)")
        axes[0, 1].set_title("Mean Absolute Percentage Error")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].text(
            0,
            mape_value + mape_value * 0.02,
            f"{mape_value:.2f}%",
            ha="center",
            va="bottom",
        )
        if mape_value <= 5:
            mape_interpretation = "Excellent"
            mape_color = "green"
        elif mape_value <= 10:
            mape_interpretation = "Very Good"
            mape_color = "blue"
        elif mape_value <= 20:
            mape_interpretation = "Good"
            mape_color = "orange"
        elif mape_value <= 30:
            mape_interpretation = "Acceptable"
            mape_color = "darkorange"
        else:
            mape_interpretation = "Poor"
            mape_color = "red"
        axes[0, 1].text(
            0,
            mape_value / 2,
            mape_interpretation,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=mape_color,
        )
        r2_score = plot_metrics.get("R2_Score", 0)
        bars3 = axes[1, 0].bar(["R² Score"], [r2_score], color="#96CEB4", alpha=0.8)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_ylabel("R² Score")
        axes[1, 0].set_title("Model Performance (R² Score)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].text(0, r2_score + 0.02, f"{r2_score:.4f}", ha="center", va="bottom")
        if r2_score >= 0.9:
            interpretation = "Excellent"
            color = "green"
        elif r2_score >= 0.8:
            interpretation = "Very Good"
            color = "blue"
        elif r2_score >= 0.7:
            interpretation = "Good"
            color = "orange"
        else:
            interpretation = "Needs Improvement"
            color = "red"
        axes[1, 0].text(
            0,
            r2_score / 2,
            interpretation,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        residual_metrics = ["Mean_Residual", "Std_Residual"]
        residual_values = [abs(plot_metrics.get(m, 0)) for m in residual_metrics]
        residual_labels = ["Mean Residual", "Std Residual"]
        bars4 = axes[1, 1].bar(
            residual_labels, residual_values, color=["#DDA0DD", "#98FB98"], alpha=0.8
        )
        axes[1, 1].set_ylabel("Residual Value")
        axes[1, 1].set_title("Residual Statistics")
        axes[1, 1].grid(True, alpha=0.3)
        for bar, value, original_value in zip(
            bars4, residual_values, [plot_metrics.get(m, 0) for m in residual_metrics]
        ):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{original_value:.4f}",
                ha="center",
                va="bottom",
            )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


class DataNormalizer:
    def __init__(self, info_path: str, robust_scaler_path: str):
        with open(info_path, "r") as f:
            self.info = json.load(f)
        with open(robust_scaler_path, "rb") as f:
            self.robust_scaler = pickle.load(f)
        self.features_to_normalize = list(set(self.info["robust_features"]))
        self.feature_names_order = self.robust_scaler.feature_names_in_
        print("DataNormalizer berhasil dimuat.")
        print(
            f"Menggunakan satu RobustScaler untuk {len(self.features_to_normalize)} fitur."
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        print("  - Menerapkan normalisasi pada data...")
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
                print(f"  Error saat normalisasi: {str(e)}")
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
                    print(
                        f"  Gagal normalisasi dengan pendekatan alternatif: {str(e2)}"
                    )
                    print("  Mengembalikan data tanpa normalisasi")
        return df_transformed


class XAIExplainer:
    """
    Kelas untuk memberikan penjelasan XAI berbasis feature importance dalam bahasa peternak
    dan rekomendasi tindakan manajerial.
    """

    def __init__(self, feature_translations):
        self.feature_translations = feature_translations
        self.feature_thresholds = {
            "FCR_ACT_lag_1": {"high": 1.5, "low": 1.3},  # Dalam satuan FCR
            "PAKAN_DG_RATIO": {"high": 2.0, "low": 1.5},  # Dalam rasio
            "DELTA_FCR": {"high": 0.2, "low": -0.2},  # Dalam selisih FCR
            "FEED_EFFICIENCY_lag_1": {
                "high": 0.77,
                "low": 0.67,
            },  # Dalam satuan g berat/g pakan
            "DELTA_PAKAN_ZAK": {"high": 0.5, "low": -0.5},  # Dalam zak
            "MATI_rolling_std_7": {"high": 10.0, "low": 2.0},  # Dalam jumlah ayam
            "DELTA_DG_rolling_std_7": {"high": 5.0, "low": 1.0},  # Dalam gram
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
        """
        Menghasilkan penjelasan XAI dan rekomendasi berdasarkan feature importance dengan nilai asli.
        """
        # Ambil 10 fitur teratas (bukan 5)
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )[
            :10
        ]  # Ubah dari 5 ke 10
        explanations = []
        recommendations = []

        for feature, importance in sorted_features:
            simple_name = self.feature_translations.get(feature, feature)
            value = feature_values_dict.get(feature, 0)
            unit = self.feature_units.get(feature, "")
            display_value = f"{value:.2f} {unit}".strip() if unit else f"{value:.2f}"

            # Aturan if-else untuk fitur utama dengan nilai asli
            if feature == "FCR_ACT_lag_1":
                if value > self.feature_thresholds["FCR_ACT_lag_1"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) terlalu tinggi, menunjukkan efisiensi pakan kemarin buruk."
                    )
                    recommendations.append(
                        "⚠️ PERLU PERBAIKAN: Tinjau jumlah pakan yang diberikan dan pastikan sesuai dengan pertumbuhan ayam."
                    )
                elif value < self.feature_thresholds["FCR_ACT_lag_1"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) sangat baik, menunjukkan efisiensi pakan kemarin optimal."
                    )
                    recommendations.append(
                        "✅ PERTAHANKAN: Pertahankan pola pemberian pakan saat ini."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, berkontribusi stabil pada prediksi."
                    )
                    recommendations.append(
                        "📊 MONITOR: Lanjutkan pemantauan efisiensi pakan harian."
                    )

            elif feature == "PAKAN_DG_RATIO":
                if value > self.feature_thresholds["PAKAN_DG_RATIO"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) terlalu tinggi, artinya pakan yang diberikan berlebihan dibandingkan pertumbuhan ayam."
                    )
                    recommendations.append(
                        "🔻 KURANGI: Kurangi pakan harian sebanyak 5-10 gram per ekor untuk menyeimbangkan pertumbuhan."
                    )
                elif value < self.feature_thresholds["PAKAN_DG_RATIO"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) sangat rendah, menunjukkan pakan cukup efisien."
                    )
                    recommendations.append(
                        "✅ BAIK: Pertahankan jumlah pakan saat ini, tapi pastikan ayam mendapat nutrisi cukup."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, menunjukkan keseimbangan pakan dan pertumbuhan."
                    )
                    recommendations.append(
                        "📊 NORMAL: Lanjutkan pemberian pakan sesuai standar."
                    )

            elif feature == "DELTA_FCR":
                if value > self.feature_thresholds["DELTA_FCR"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan efisiensi pakan jauh di atas standar, yang meningkatkan FCR."
                    )
                    recommendations.append(
                        "⚠️ TINJAU ULANG: Tinjau ulang strategi pakan untuk mendekati standar efisiensi."
                    )
                elif value < self.feature_thresholds["DELTA_FCR"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan efisiensi pakan lebih baik dari standar."
                    )
                    recommendations.append(
                        "✅ EXCELLENT: Pertahankan strategi pakan saat ini."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) mendekati standar, berkontribusi stabil pada prediksi."
                    )
                    recommendations.append(
                        "📊 MONITOR: Lanjutkan pemantauan efisiensi pakan."
                    )

            elif feature == "FEED_EFFICIENCY_lag_1":
                if value < self.feature_thresholds["FEED_EFFICIENCY_lag_1"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) terlalu rendah, menunjukkan efisiensi pakan kemarin buruk."
                    )
                    recommendations.append(
                        "🔧 PERBAIKI: Sesuaikan pakan agar lebih sesuai dengan pertumbuhan ayam."
                    )
                elif value > self.feature_thresholds["FEED_EFFICIENCY_lag_1"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) sangat baik, menunjukkan efisiensi pakan kemarin optimal."
                    )
                    recommendations.append(
                        "✅ PERTAHANKAN: Pertahankan pola pakan saat ini."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, berkontribusi stabil pada prediksi."
                    )
                    recommendations.append(
                        "📊 MONITOR: Lanjutkan pemantauan efisiensi pakan harian."
                    )

            elif feature == "DELTA_PAKAN_ZAK":
                if value > self.feature_thresholds["DELTA_PAKAN_ZAK"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan pakan berlebihan dibandingkan standar."
                    )
                    recommendations.append(
                        "🔻 KURANGI: Kurangi jumlah pakan sebanyak 0.5-1 zak per hari."
                    )
                elif value < self.feature_thresholds["DELTA_PAKAN_ZAK"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan pakan kurang dari standar."
                    )
                    recommendations.append(
                        "🔺 TAMBAH: Tambah pakan secukupnya untuk memenuhi standar."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) sesuai standar, berkontribusi stabil pada prediksi."
                    )
                    recommendations.append(
                        "📊 NORMAL: Lanjutkan pemberian pakan sesuai standar."
                    )

            elif feature == "MATI_rolling_std_7":
                if value > self.feature_thresholds["MATI_rolling_std_7"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) tinggi, menunjukkan kematian ayam tidak stabil dalam 7 hari terakhir."
                    )
                    recommendations.append(
                        "🚨 URGENT: Periksa kesehatan ayam dan kondisi kandang untuk mengurangi kematian."
                    )
                elif value < self.feature_thresholds["MATI_rolling_std_7"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) rendah, menunjukkan kematian ayam stabil."
                    )
                    recommendations.append(
                        "✅ BAIK: Pertahankan kondisi kandang dan kesehatan ayam."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, menunjukkan stabilitas kematian ayam."
                    )
                    recommendations.append(
                        "📊 MONITOR: Lanjutkan pemantauan kesehatan ayam."
                    )

            elif feature == "DELTA_DG_rolling_std_7":
                if value > self.feature_thresholds["DELTA_DG_rolling_std_7"]["high"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) tinggi, menunjukkan pertumbuhan ayam tidak stabil dalam 7 hari terakhir."
                    )
                    recommendations.append(
                        "⚠️ STABILKAN: Periksa kualitas pakan dan kondisi kandang untuk menstabilkan pertumbuhan."
                    )
                elif value < self.feature_thresholds["DELTA_DG_rolling_std_7"]["low"]:
                    explanations.append(
                        f"{simple_name} ({display_value}) rendah, menunjukkan pertumbuhan ayam stabil."
                    )
                    recommendations.append(
                        "✅ EXCELLENT: Pertahankan strategi pakan dan pemeliharaan saat ini."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) normal, menunjukkan stabilitas pertumbuhan ayam."
                    )
                    recommendations.append(
                        "📊 MONITOR: Lanjutkan pemantauan pertumbuhan ayam."
                    )

            # Tambahkan logika untuk fitur-fitur lain yang mungkin muncul di top 10
            elif "lag_" in feature:
                # Untuk fitur lag (data historis)
                if abs(value) > 1.0:  # Threshold umum untuk fitur lag
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan variasi signifikan dari periode sebelumnya."
                    )
                    recommendations.append(
                        "🔍 ANALISIS: Tinjau tren historis dan identifikasi pola yang menyebabkan variasi ini."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan konsistensi dengan periode sebelumnya."
                    )
                    recommendations.append(
                        "📊 KONSISTEN: Pertahankan pola manajemen saat ini."
                    )

            elif "rolling_" in feature:
                # Untuk fitur rolling (statistik bergulir)
                if "std" in feature and value > 5.0:  # Standar deviasi tinggi
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan ketidakstabilan tinggi."
                    )
                    recommendations.append(
                        "⚠️ STABILKAN: Identifikasi dan atasi faktor penyebab ketidakstabilan."
                    )
                elif "mean" in feature and value < 0:
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan tren menurun."
                    )
                    recommendations.append(
                        "🔺 TINGKATKAN: Implementasikan strategi untuk meningkatkan performa."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) berkontribusi pada prediksi FCR."
                    )
                    recommendations.append(
                        "📊 MONITOR: Lanjutkan pemantauan data ini untuk konsistensi."
                    )

            else:
                # Untuk fitur lainnya
                if abs(value) > 2.0:  # Nilai ekstrem
                    explanations.append(
                        f"{simple_name} ({display_value}) menunjukkan nilai yang ekstrem."
                    )
                    recommendations.append(
                        "🔍 INVESTIGASI: Periksa dan validasi data ini, mungkin perlu tindakan korektif."
                    )
                else:
                    explanations.append(
                        f"{simple_name} ({display_value}) berkontribusi pada prediksi FCR."
                    )
                    recommendations.append(
                        "📊 NORMAL: Lanjutkan pemantauan data ini untuk konsistensi."
                    )

        return {
            "predicted_fcr": predicted_fcr,
            "horizon": horizon,
            "explanations": explanations,
            "recommendations": recommendations,
            "top_features": sorted_features,
        }

    def create_simple_feature_importance_plot(
        self, feature_importance_dict, save_path=None
    ):
        """
        Membuat plot sederhana untuk menunjukkan 3-5 fitur terpenting dalam bahasa peternak.
        """
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:5]
        feature_names = [
            self.feature_translations.get(f[0], f[0]) for f in sorted_features
        ]
        importance_values = [f[1] for f in sorted_features]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(feature_names)), importance_values, color="#4ECDC4")
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel("Pentingnya Fitur")
        plt.title(
            "Faktor Utama yang Memengaruhi Prediksi FCR", fontsize=12, fontweight="bold"
        )
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_explanation_report(self, explanation_dict, save_path):
        """
        Menyimpan laporan penjelasan XAI dalam format teks.
        """
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(
                f"Penjelasan Prediksi FCR (Horizon +{explanation_dict['horizon']} Hari)\n"
            )
            f.write("=" * 60 + "\n")
            f.write(f"FCR Diprediksi: {explanation_dict['predicted_fcr']:.6f}\n")
            f.write("\nFaktor Utama yang Memengaruhi Prediksi:\n")
            for i, explanation in enumerate(explanation_dict["explanations"], 1):
                f.write(f"{i}. {explanation}\n")
            f.write("\nRekomendasi untuk Peternak:\n")
            for i, recommendation in enumerate(explanation_dict["recommendations"], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("=" * 60 + "\n")


class FeatureAnalyzer:
    """
    Kelas untuk analisis fitur dengan Pearson correlation yang komprehensif
    """

    def __init__(self):
        self.feature_names = []
        self.feature_importance = []
        self.feature_values = []
        self.temporal_features = []
        self.tabular_features = []
        self.correlation_results = {}

    def calculate_pearson_pvalues(self, df):
        """
        Menghitung p-values untuk korelasi Pearson antara semua pasangan fitur.
        """
        features = df.select_dtypes(include=[np.number]).columns
        n_features = len(features)
        p_values = np.ones((n_features, n_features))

        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    # Hapus NaN values untuk perhitungan yang akurat
                    data1 = df[features[i]].dropna()
                    data2 = df[features[j]].dropna()

                    # Ambil indeks yang sama untuk kedua series
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) > 2:  # Minimal 3 data points
                        _, p_val = pearsonr(data1[common_idx], data2[common_idx])
                        p_values[i, j] = p_val
                        p_values[j, i] = p_val
                    else:
                        p_values[i, j] = 1.0
                        p_values[j, i] = 1.0
                except:
                    p_values[i, j] = 1.0
                    p_values[j, i] = 1.0

        return pd.DataFrame(p_values, index=features, columns=features)

    def analyze_feature_correlations(
        self, feature_values, feature_names, target_values=None, save_path=None
    ):
        """
        Melakukan analisis korelasi Pearson yang komprehensif pada fitur.
        """
        print("\n=== MEMULAI ANALISIS KORELASI PEARSON FITUR ===")

        # 1. Buat DataFrame dari feature values
        if isinstance(feature_values, np.ndarray):
            df_features = pd.DataFrame(feature_values, columns=feature_names)
        else:
            df_features = feature_values.copy()

        # 2. Hitung matriks korelasi Pearson dan Spearman
        print("Menghitung matriks korelasi Pearson...")
        pearson_correlation_matrix = df_features.corr(method="pearson")

        print("Menghitung matriks korelasi Spearman...")
        spearman_correlation_matrix = df_features.corr(method="spearman")

        # 3. Hitung p-values untuk korelasi Pearson
        print("Menghitung p-values untuk korelasi Pearson...")
        pearson_pvalues = self.calculate_pearson_pvalues(df_features)

        # 4. Identifikasi korelasi yang signifikan (p < 0.05)
        significant_correlations = (pearson_pvalues < 0.05) & (
            pearson_correlation_matrix.abs() > 0.1
        )

        # 5. Kategorisasi fitur
        temporal_cols = [
            col
            for col in feature_names
            if any(
                keyword in col.lower()
                for keyword in [
                    "lag",
                    "rolling",
                    "momentum",
                    "acceleration",
                    "volatility",
                    "trend",
                ]
            )
        ]
        tabular_cols = [col for col in feature_names if col not in temporal_cols]

        # 6. Simpan hasil dalam dictionary
        analysis_results = {
            "pearson_matrix": pearson_correlation_matrix,
            "spearman_matrix": spearman_correlation_matrix,
            "p_values": pearson_pvalues,
            "significant_correlations": significant_correlations,
            "temporal_features": temporal_cols,
            "tabular_features": tabular_cols,
        }

        # 7. Analisis korelasi dengan target (jika tersedia)
        if target_values is not None:
            target_correlations = pd.Series(index=feature_names, dtype=float)
            target_pvalues = pd.Series(index=feature_names, dtype=float)

            for feature in feature_names:
                try:
                    # Hapus NaN values
                    feature_data = df_features[feature].dropna()
                    target_data = pd.Series(target_values).dropna()

                    # Ambil indeks yang sama
                    common_idx = feature_data.index.intersection(target_data.index)
                    if len(common_idx) > 2:
                        corr_val, p_val = pearsonr(
                            feature_data[common_idx], target_data[common_idx]
                        )
                        target_correlations[feature] = corr_val
                        target_pvalues[feature] = p_val
                    else:
                        target_correlations[feature] = np.nan
                        target_pvalues[feature] = 1.0
                except:
                    target_correlations[feature] = np.nan
                    target_pvalues[feature] = 1.0

            print("\n=== TOP 10 KORELASI FITUR DENGAN TARGET (PEARSON) ===")
            top_correlations = (
                target_correlations.abs().sort_values(ascending=False).head(10)
            )

            for i, (feature, corr_val) in enumerate(top_correlations.items()):
                p_val = target_pvalues[feature]
                significance = (
                    "***"
                    if p_val < 0.001
                    else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                )
                print(
                    f"  {i+1:2d}. {feature}: {corr_val:.4f} (p={p_val:.4f}) {significance}"
                )

            analysis_results["target_correlations"] = target_correlations
            analysis_results["target_pvalues"] = target_pvalues

        # 8. Statistik korelasi
        all_correlations = pearson_correlation_matrix.values[
            np.triu_indices_from(pearson_correlation_matrix.values, k=1)
        ]
        all_correlations = all_correlations[~np.isnan(all_correlations)]

        # Check if we have valid correlations before calculating statistics
        if len(all_correlations) > 0:
            correlation_stats = {
                "mean_correlation": float(np.mean(np.abs(all_correlations))),
                "std_correlation": float(np.std(all_correlations)),
                "min_correlation": float(np.min(all_correlations)),
                "max_correlation": float(np.max(all_correlations)),
                "strong_correlations_03": int(np.sum(np.abs(all_correlations) > 0.3)),
                "strong_correlations_05": int(np.sum(np.abs(all_correlations) > 0.5)),
                "strong_correlations_07": int(np.sum(np.abs(all_correlations) > 0.7)),
            }

            print("\n=== STATISTIK KORELASI PEARSON ===")
            print(f"  Mean |correlation|: {correlation_stats['mean_correlation']:.4f}")
            print(f"  Std correlation: {correlation_stats['std_correlation']:.4f}")
            print(f"  Min correlation: {correlation_stats['min_correlation']:.4f}")
            print(f"  Max correlation: {correlation_stats['max_correlation']:.4f}")
            print(
                f"  Korelasi |r| > 0.3: {correlation_stats['strong_correlations_03']}"
            )
            print(
                f"  Korelasi |r| > 0.5: {correlation_stats['strong_correlations_05']}"
            )
            print(
                f"  Korelasi |r| > 0.7: {correlation_stats['strong_correlations_07']}"
            )
        else:
            # Handle case when no valid correlations exist
            correlation_stats = {
                "mean_correlation": 0.0,
                "std_correlation": 0.0,
                "min_correlation": 0.0,
                "max_correlation": 0.0,
                "strong_correlations_03": 0,
                "strong_correlations_05": 0,
                "strong_correlations_07": 0,
            }

            print("\n=== STATISTIK KORELASI PEARSON ===")
            print("  Tidak ada korelasi valid yang dapat dihitung (semua nilai NaN)")
            print("  Ini mungkin terjadi karena:")
            print("    - Data memiliki variance yang sangat rendah")
            print("    - Terlalu banyak nilai yang hilang")
            print("    - Fitur memiliki nilai konstan")

        analysis_results["correlation_statistics"] = correlation_stats

        # 9. Simpan hasil analisis
        if save_path:
            # Simpan matriks korelasi
            pearson_correlation_matrix.to_csv(
                save_path.replace(".json", "_pearson_correlation_matrix.csv")
            )
            spearman_correlation_matrix.to_csv(
                save_path.replace(".json", "_spearman_correlation_matrix.csv")
            )
            pearson_pvalues.to_csv(save_path.replace(".json", "_pearson_pvalues.csv"))

            # Simpan analisis dalam JSON
            json_results = {
                "correlation_statistics": correlation_stats,
                "temporal_features_count": len(temporal_cols),
                "tabular_features_count": len(tabular_cols),
                "total_features": len(feature_names),
            }

            if target_values is not None:
                json_results["target_correlations"] = target_correlations.to_dict()
                json_results["target_pvalues"] = target_pvalues.to_dict()

            with open(
                save_path.replace(".json", "_correlation_analysis.json"), "w"
            ) as f:
                json.dump(json_results, f, indent=2)

            print(f"\nHasil analisis korelasi disimpan:")
            print(
                f"  - Pearson matrix: {save_path.replace('.json', '_pearson_correlation_matrix.csv')}"
            )
            print(
                f"  - Spearman matrix: {save_path.replace('.json', '_spearman_correlation_matrix.csv')}"
            )
            print(f"  - P-values: {save_path.replace('.json', '_pearson_pvalues.csv')}")
            print(
                f"  - Analysis JSON: {save_path.replace('.json', '_correlation_analysis.json')}"
            )

        return analysis_results

    def create_comprehensive_correlation_plots(self, analysis_results, save_path=None):
        """
        Membuat visualisasi korelasi yang komprehensif.
        """
        pearson_matrix = analysis_results["pearson_matrix"]
        spearman_matrix = analysis_results["spearman_matrix"]
        p_values = analysis_results["p_values"]
        significant_correlations = analysis_results["significant_correlations"]

        # 1. Heatmap korelasi Pearson lengkap
        plt.figure(figsize=(15, 12))

        # Pilih subset fitur untuk visualisasi yang lebih jelas
        n_features = min(20, len(pearson_matrix.columns))
        top_features = pearson_matrix.columns[:n_features]
        correlation_subset = pearson_matrix.loc[top_features, top_features]

        mask = np.triu(np.ones_like(correlation_subset, dtype=bool))
        sns.heatmap(
            correlation_subset,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            fmt=".2f",
        )
        plt.title(
            "Heatmap Korelasi Pearson - Top Features", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path.replace(".png", "_pearson_heatmap.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

        # 2. Analisis distribusi korelasi
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Histogram distribusi korelasi
        all_correlations = pearson_matrix.values[
            np.triu_indices_from(pearson_matrix.values, k=1)
        ]
        all_correlations = all_correlations[~np.isnan(all_correlations)]

        if len(all_correlations) > 0:
            axes[0, 0].hist(
                all_correlations, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
            )
            axes[0, 0].axvline(
                np.mean(all_correlations),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(all_correlations):.3f}",
            )
            axes[0, 0].set_xlabel("Korelasi Pearson")
            axes[0, 0].set_ylabel("Frekuensi")
            axes[0, 0].set_title("Distribusi Korelasi Pearson")
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "Tidak ada korelasi valid",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
            axes[0, 0].set_title("Distribusi Korelasi Pearson")

        # Box plot korelasi dengan target (jika ada)
        if "target_correlations" in analysis_results:
            target_corrs = analysis_results["target_correlations"]
            temporal_cols = analysis_results["temporal_features"]
            tabular_cols = analysis_results["tabular_features"]

            temporal_target_corrs = [
                abs(target_corrs[col])
                for col in temporal_cols
                if col in target_corrs.index and not np.isnan(target_corrs[col])
            ]
            tabular_target_corrs = [
                abs(target_corrs[col])
                for col in tabular_cols
                if col in target_corrs.index and not np.isnan(target_corrs[col])
            ]

            if temporal_target_corrs or tabular_target_corrs:
                box_data = [temporal_target_corrs, tabular_target_corrs]
                box_labels = ["Temporal", "Tabular"]

                axes[0, 1].boxplot(box_data, labels=box_labels)
                axes[0, 1].set_ylabel("|Korelasi Pearson dengan Target|")
                axes[0, 1].set_title("Korelasi Fitur dengan Target")
                axes[0, 1].grid(alpha=0.3)
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "Tidak ada korelasi valid dengan target",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("Korelasi dengan Target")
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Target tidak tersedia",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Korelasi dengan Target")

        # Scatter plot: Pearson vs Spearman
        pearson_vals = pearson_matrix.values[
            np.triu_indices_from(pearson_matrix.values, k=1)
        ]
        spearman_vals = spearman_matrix.values[
            np.triu_indices_from(spearman_matrix.values, k=1)
        ]

        valid_idx = ~(np.isnan(pearson_vals) | np.isnan(spearman_vals))
        if np.any(valid_idx):
            axes[1, 0].scatter(
                pearson_vals[valid_idx],
                spearman_vals[valid_idx],
                alpha=0.6,
                color="green",
            )
            axes[1, 0].plot([-1, 1], [-1, 1], "r--", alpha=0.8)
            axes[1, 0].set_xlabel("Korelasi Pearson")
            axes[1, 0].set_ylabel("Korelasi Spearman")
            axes[1, 0].set_title("Pearson vs Spearman Correlation")
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "Tidak ada korelasi valid",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Pearson vs Spearman Correlation")

        # Bar plot korelasi terkuat dengan target
        if "target_correlations" in analysis_results:
            target_corrs = analysis_results["target_correlations"]
            valid_corrs = target_corrs.dropna()

            if len(valid_corrs) > 0:
                top_corrs = valid_corrs.abs().sort_values(ascending=False).head(10)

                axes[1, 1].barh(
                    range(len(top_corrs)), top_corrs.values, color="orange", alpha=0.8
                )
                axes[1, 1].set_yticks(range(len(top_corrs)))
                axes[1, 1].set_yticklabels(
                    [
                        name[:15] + "..." if len(name) > 15 else name
                        for name in top_corrs.index
                    ]
                )
                axes[1, 1].set_xlabel("|Korelasi Pearson dengan Target|")
                axes[1, 1].set_title("Top 10 Fitur Berkorelasi dengan Target")
                axes[1, 1].grid(alpha=0.3)
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "Tidak ada korelasi valid dengan target",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Top Korelasi dengan Target")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Target tidak tersedia",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Top Korelasi dengan Target")

        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path.replace(".png", "_correlation_analysis.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

        print("\nVisualisasi korelasi Pearson selesai dibuat!")
        if save_path:
            print(f"Plot disimpan dengan prefix: {save_path.replace('.png', '')}")

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
        save_path=None,
        title="Top 50 Feature Importance untuk Prediksi FCR",
    ):
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:50]
        feature_names = [
            FEATURE_TRANSLATIONS.get(item[0], item[0]) for item in sorted_features
        ]
        importance_values = [item[1] for item in sorted_features]
        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(feature_names)), importance_values)
        colors = [
            "#FF6B6B" if name.startswith("temporal_") or "TCN" in name else "#4ECDC4"
            for name in feature_names
        ]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel("Feature Importance (Weight)")
        plt.title(title, fontsize=12, fontweight="bold")
        plt.grid(axis="x", alpha=0.3)
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#FF6B6B", label="Fitur Temporal (TCN)"),
            Patch(facecolor="#4ECDC4", label="Fitur Tabular"),
        ]
        plt.legend(handles=legend_elements, loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return sorted_features

    def create_feature_distribution_plot(
        self, feature_values, feature_names, save_path=None
    ):
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
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def create_feature_correlation_heatmap(
        self, feature_values, feature_names, save_path=None
    ):
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
        plt.figure(figsize=(12, 10))
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
        )
        plt.title("Korelasi Antar Fitur Penting", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def create_temporal_vs_tabular_comparison(
        self, feature_importance_dict, n_tabular_features, save_path=None
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
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return tabular_importance, temporal_importance


class HybridPredictor:
    def __init__(self, models_dir: str, config_dir: str, sliding_window: int = None):
        print("Memuat semua model dan pipeline untuk prediksi...")
        fe_path = os.path.join(models_dir, "feature_engineering_pipeline.pkl")
        with open(fe_path, "rb") as f:
            self.feature_pipeline = pickle.load(f)
        info_path = os.path.join(config_dir, "normalization_info.json")
        robust_path = os.path.join(models_dir, "robust_scaler.pkl")
        self.normalizer = DataNormalizer(info_path, robust_path)
        tcn_path = os.path.join(models_dir, "tcn_encoder.keras")
        config_path = os.path.join(models_dir, "hybrid_model_config.json")
        self.tcn_encoder = tf.keras.models.load_model(
            tcn_path, custom_objects={"TCN": TCN}
        )
        with open(config_path, "r") as f:
            self.hybrid_config = json.load(f)
        self.sequence_length = (
            sliding_window if sliding_window else self.hybrid_config["sequence_length"]
        )
        self.horizons = [1, 3]
        self.xgb_models = {}
        for horizon in self.horizons:
            xgb_path = os.path.join(
                models_dir, f"xgboost_hybrid_model_horizon_{horizon}d.pkl"
            )
            with open(xgb_path, "rb") as f:
                self.xgb_models[f"horizon_{horizon}d"] = pickle.load(f)
        self.analyzer = FeatureAnalyzer()
        self.evaluator = EvaluationMetrics()
        self.explainer = XAIExplainer(FEATURE_TRANSLATIONS)
        print("Semua komponen prediksi berhasil dimuat.\n")
        print(f"Model mendukung prediksi untuk horizon: {self.horizons} hari")
        print(f"Menggunakan sliding window: {self.sequence_length} hari")

    def predict_single_sample(
        self, new_day_data: pd.DataFrame, historical_data: pd.DataFrame, horizon=None
    ):
        combined_data = pd.concat([historical_data, new_day_data], ignore_index=True)
        combined_data = combined_data.sort_values(["PERIODE", "AGE"]).reset_index(
            drop=True
        )
        data_enhanced_full = self.feature_pipeline.transform(combined_data)
        data_enhanced = data_enhanced_full.tail(len(new_day_data))
        data_normalized = self.normalizer.transform(data_enhanced)
        historical_enhanced = self.feature_pipeline.transform(historical_data)
        historical_normalized = self.normalizer.transform(historical_enhanced)
        full_normalized_history = pd.concat(
            [historical_normalized, data_normalized], ignore_index=True
        )
        if len(full_normalized_history) < self.sequence_length:
            raise ValueError(
                f"Data tidak cukup untuk sekuens. Diperlukan {self.sequence_length} hari, tersedia {len(full_normalized_history)} hari."
            )
        last_sequence = full_normalized_history.tail(self.sequence_length)
        features_for_model = [
            col
            for col in last_sequence.columns
            if col not in ["PERIODE", "FCR_ACT", "TANGGAL"]
        ]
        last_sequence_values = last_sequence[features_for_model].values.reshape(
            1, self.sequence_length, -1
        )
        temporal_features = self.tcn_encoder.predict(last_sequence_values, verbose=0)
        tabular_features = data_normalized[features_for_model].values.reshape(1, -1)
        hybrid_features = np.concatenate([tabular_features, temporal_features], axis=1)
        predictions = {}
        if horizon is not None:
            if horizon not in self.horizons:
                raise ValueError(
                    f"Horizon {horizon} tidak didukung. Horizon yang tersedia: {self.horizons}"
                )
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
        create_visualizations=True,
        save_plots=True,
        horizon=None,
        max_rows=35,
    ):
        print("--- Memulai Evaluasi Model pada Data Test ---")
        if horizon is not None:
            print(f"Evaluasi untuk horizon: {horizon} hari")
            if horizon not in self.horizons:
                raise ValueError(
                    f"Horizon {horizon} tidak didukung. Horizon yang tersedia: {self.horizons}"
                )
        else:
            print(f"Evaluasi untuk semua horizon: {self.horizons}")
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"File test data tidak ditemukan di {test_data_path}"
            )
        df_test = pd.read_csv(test_data_path, sep=";")
        print(f"Data test dimuat: {len(df_test)} samples")
        if len(df_test) > max_rows:
            print(f"Membatasi data input ke {max_rows} baris teratas")
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
        print("Memproses prediksi untuk setiap sampel...")
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
                        (df_test["PERIODE"] == periode) & (df_test["AGE"] == target_age)
                    ]
                    if target_sample.empty or "FCR_ACT" not in target_sample.columns:
                        continue
                    actual_fcr = target_sample["FCR_ACT"].iloc[0]
                else:
                    h = self.horizons[0]
                    target_age = age + h
                    target_sample = df_test[
                        (df_test["PERIODE"] == periode) & (df_test["AGE"] == target_age)
                    ]
                    if target_sample.empty or "FCR_ACT" not in target_sample.columns:
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
                if (idx + 1) % 10 == 0:
                    print(
                        f"  Processed {idx + 1}/{len(unique_combinations)} samples..."
                    )
            except Exception as e:
                print(f"  Error processing PERIODE {periode}, AGE {age}: {str(e)}")
                continue
        print(
            f"Berhasil memproses {valid_predictions} prediksi dari {len(unique_combinations)} sampel"
        )
        if len(predictions) == 0:
            print(
                "Tidak ada prediksi yang berhasil. Periksa data dan konfigurasi model."
            )
            return None
        y_true = np.array(actual_values)
        y_pred = np.array(predictions)
        print("Menghitung metrics evaluasi...")
        metrics = self.evaluator.calculate_metrics(y_true, y_pred)
        self.evaluator.print_metrics_summary(metrics, "Test Data Evaluation Results")
        if create_visualizations:
            print("Membuat visualisasi evaluasi...")
            save_dir = "evaluation_plots"
            if save_plots:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                os.path.join(save_dir, f"evaluation_plots_{timestamp}.png")
                if save_plots
                else None
            )
            self.evaluator.create_evaluation_plots(y_true, y_pred, save_path)
            save_path = (
                os.path.join(save_dir, f"metrics_comparison_{timestamp}.png")
                if save_plots
                else None
            )
            self.evaluator.create_metrics_comparison_plot(metrics, save_path)
            if save_plots:
                print(f"Plot evaluasi disimpan di direktori: {save_dir}")
        if save_plots:
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
            results_path = os.path.join(save_dir, f"prediction_results_{timestamp}.csv")
            results_df.to_csv(results_path, index=False)
            print(f"Hasil prediksi detail disimpan di: {results_path}")
        return metrics, y_true, y_pred

    def predict_with_analysis(
        self,
        new_day_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        create_visualizations=True,
        save_plots=True,
        horizon=None,
    ):
        print("--- Memulai Proses Prediksi dengan Analisis ---")
        if horizon is not None:
            print(f"Prediksi untuk horizon: {horizon} hari")
            if horizon not in self.horizons:
                raise ValueError(
                    f"Horizon {horizon} tidak didukung. Horizon yang tersedia: {self.horizons}"
                )
        else:
            print(f"Prediksi untuk semua horizon: {self.horizons}")
            horizon = self.horizons[0]
        print("Langkah A: Menjalankan rekayasa fitur...")
        combined_data = pd.concat([historical_data, new_day_data], ignore_index=True)
        combined_data = combined_data.sort_values(["PERIODE", "AGE"]).reset_index(
            drop=True
        )
        data_enhanced_full = self.feature_pipeline.transform(combined_data)
        data_enhanced = data_enhanced_full.tail(len(new_day_data))
        print(
            f"  - Rekayasa fitur selesai. Menghasilkan {data_enhanced.shape[1]} fitur."
        )
        print("Langkah B: Menjalankan normalisasi...")
        data_normalized = self.normalizer.transform(data_enhanced)
        historical_enhanced = self.feature_pipeline.transform(historical_data)
        historical_normalized = self.normalizer.transform(historical_enhanced)
        print("Langkah C: Menjalankan prediksi model hibrida...")
        full_normalized_history = pd.concat(
            [historical_normalized, data_normalized], ignore_index=True
        )
        if len(full_normalized_history) < self.sequence_length:
            raise ValueError(
                f"Data tidak cukup untuk sekuens. Diperlukan {self.sequence_length} hari, tersedia {len(full_normalized_history)} hari."
            )
        last_sequence = full_normalized_history.tail(self.sequence_length)
        features_for_model = [
            col
            for col in last_sequence.columns
            if col not in ["PERIODE", "FCR_ACT", "TANGGAL"]
        ]
        last_sequence_values = last_sequence[features_for_model].values.reshape(
            1, self.sequence_length, -1
        )
        temporal_features = self.tcn_encoder.predict(last_sequence_values, verbose=0)
        tabular_features = data_normalized[features_for_model].values.reshape(1, -1)
        hybrid_features = np.concatenate([tabular_features, temporal_features], axis=1)
        feature_values_raw = dict(
            zip(features_for_model, data_enhanced[features_for_model].iloc[-1])
        )  # Simpan nilai asli
        if horizon is not None:
            model_key = f"horizon_{horizon}d"
            final_prediction = self.xgb_models[model_key].predict(hybrid_features)
            predictions = {horizon: final_prediction[0]}
        else:
            predictions = {}
            for h in self.horizons:
                model_key = f"horizon_{h}d"
                predictions[h] = self.xgb_models[model_key].predict(hybrid_features)[0]
        if create_visualizations:
            print("Langkah D: Menjalankan analisis fitur...")
            n_tabular_features = len(features_for_model)
            n_temporal_features = temporal_features.shape[1]
            h_for_analysis = horizon if horizon is not None else self.horizons[0]
            model_key = f"horizon_{h_for_analysis}d"

            # Analisis korelasi Pearson
            print("Langkah D1: Menjalankan analisis korelasi Pearson...")
            save_dir = "feature_analysis_plots"
            if save_plots:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                correlation_save_path = os.path.join(
                    save_dir, f"correlation_analysis_{timestamp}.json"
                )
            else:
                correlation_save_path = None

            # Gabungkan fitur tabular dan temporal untuk analisis korelasi
            all_feature_names = features_for_model + [
                f"temporal_feature_{i+1}" for i in range(n_temporal_features)
            ]
            combined_features = np.concatenate(
                [tabular_features, temporal_features], axis=1
            )

            correlation_results = self.analyzer.analyze_feature_correlations(
                combined_features, all_feature_names, save_path=correlation_save_path
            )

            # Buat visualisasi korelasi
            if save_plots:
                correlation_plot_path = os.path.join(
                    save_dir, f"correlation_plots_{timestamp}.png"
                )
            else:
                correlation_plot_path = None

            self.analyzer.create_comprehensive_correlation_plots(
                correlation_results, save_path=correlation_plot_path
            )

            feature_importance_dict = self._analyze_features(
                hybrid_features,
                n_tabular_features,
                features_for_model,
                n_temporal_features,
                save_plots,
                model_key,
                h_for_analysis,
            )
            explanation_dict = self.explainer.generate_explanation(
                feature_importance_dict,
                feature_values_raw,
                predictions[h_for_analysis],
                h_for_analysis,
            )
            print(f"\n{'='*50}")
            print(f"Penjelasan Prediksi FCR (Horizon +{h_for_analysis} Hari)")
            print(f"{'='*50}")
            print(f"FCR Diprediksi: {explanation_dict['predicted_fcr']:.6f}")
            print("\nFaktor Utama yang Memengaruhi Prediksi:")
            for i, explanation in enumerate(explanation_dict["explanations"], 1):
                print(f"{i}. {explanation}")
            print("\nRekomendasi untuk Peternak:")
            for i, recommendation in enumerate(explanation_dict["recommendations"], 1):
                print(f"{i}. {recommendation}")
            print(f"{'='*50}\n")
            if save_plots:
                save_dir = "feature_analysis_plots"
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    save_dir,
                    f"simple_feature_importance_horizon_{h_for_analysis}d_{timestamp}.png",
                )
                self.explainer.create_simple_feature_importance_plot(
                    feature_importance_dict, save_path
                )
                explanation_path = os.path.join(
                    save_dir,
                    f"xai_explanation_horizon_{h_for_analysis}d_{timestamp}.txt",
                )
                self.explainer.save_explanation_report(
                    explanation_dict, explanation_path
                )
                print(f"Laporan penjelasan XAI disimpan di: {explanation_path}")
        return predictions

    def _analyze_features(
        self,
        hybrid_features,
        n_tabular_features,
        base_feature_names,
        n_temporal_features,
        save_plots=True,
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
        save_dir = "feature_analysis_plots"
        if save_plots:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("  - Membuat plot feature importance...")
        save_path = (
            os.path.join(
                save_dir, f"feature_importance_horizon_{horizon}d_{timestamp}.png"
            )
            if save_plots
            else None
        )
        top_features = self.analyzer.create_feature_importance_plot(
            feature_importance_dict,
            save_path,
            title=f"Feature Importance (Horizon {horizon} hari)",
        )
        print("  - Membuat plot distribusi fitur...")
        save_path = (
            os.path.join(
                save_dir, f"feature_distribution_horizon_{horizon}d_{timestamp}.png"
            )
            if save_plots
            else None
        )
        self.analyzer.create_feature_distribution_plot(
            [hybrid_features[0]], all_feature_names, save_path
        )
        print("  - Membuat perbandingan fitur temporal vs tabular...")
        save_path = (
            os.path.join(
                save_dir, f"temporal_vs_tabular_horizon_{horizon}d_{timestamp}.png"
            )
            if save_plots
            else None
        )
        tab_importance, temp_importance = (
            self.analyzer.create_temporal_vs_tabular_comparison(
                feature_importance_dict, n_tabular_features, save_path
            )
        )
        print(f"\n{'='*50}")
        print(f"RINGKASAN ANALISIS FITUR (HORIZON {horizon} HARI)")
        print(f"{'='*50}")
        print(f"Total Fitur Tabular: {n_tabular_features}")
        print(f"Total Fitur Temporal: {n_temporal_features}")
        print(
            f"Kontribusi Fitur Tabular: {tab_importance:.2f} ({tab_importance/(tab_importance+temp_importance)*100:.1f}%)"
        )
        print(
            f"Kontribusi Fitur Temporal: {temp_importance:.2f} ({temp_importance/(tab_importance+temp_importance)*100:.1f}%)"
        )
        print(f"\nTop 50 Fitur Terpenting:")
        for i, (feature_name, importance) in enumerate(top_features[:50]):
            print(
                f"  {i+1}. {FEATURE_TRANSLATIONS.get(feature_name, feature_name)}: {importance:.1f}"
            )
        print(f"{'='*50}\n")
        if save_plots:
            print(f"Plot disimpan di direktori: {save_dir}")
            print(
                f"Analisis korelasi Pearson juga telah disimpan dengan visualisasi lengkap."
            )
        return feature_importance_dict


if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, "5. Model", "models_terbaru")
    CONFIG_DIR = os.path.join(BASE_DIR, "2. Hasil Normalisasi", "Terbaru")
    try:
        print("Masukkan ukuran sliding window (default: gunakan nilai dari model):")
        sliding_window_input = input()
        sliding_window = (
            int(sliding_window_input) if sliding_window_input.strip() else None
        )
        predictor = HybridPredictor(
            models_dir=MODELS_DIR, config_dir=CONFIG_DIR, sliding_window=sliding_window
        )
        print("Pilih mode operasi:")
        print("1. Prediksi sampel tunggal dengan analisis")
        print("2. Evaluasi pada seluruh data test")
        print("3. Keduanya")
        mode = int(input("Pilihan Anda (1/2/3): "))
        test_data_path = os.path.join(
            BASE_DIR, "3. non-normalize", "NORMALISASI_PERIODE_15.csv"
        )
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"File data test tidak ditemukan di {test_data_path}"
            )
        print("\nPilih horizon untuk prediksi:")
        print("1. Horizon +1 hari")
        print("2. Horizon +3 hari")
        print("3. Kedua horizon")
        horizon_choice = int(input("Pilihan Anda (1/2/3): "))
        if horizon_choice == 1:
            horizon_to_evaluate = 1
        elif horizon_choice == 2:
            horizon_to_evaluate = 3
        else:
            horizon_to_evaluate = None
        if mode in [2, 3]:
            print("\n" + "=" * 60)
            print("MODE: EVALUASI SELURUH DATA TEST")
            print("=" * 60)
            max_rows = 35
            print(f"Menggunakan maksimal {max_rows} baris data untuk evaluasi")
            print(
                "Evaluasi ini akan mencakup analisis korelasi Pearson yang komprehensif."
            )
            evaluation_results = predictor.evaluate_test_data(
                test_data_path=test_data_path,
                create_visualizations=True,
                save_plots=True,
                horizon=horizon_to_evaluate,
                max_rows=max_rows,
            )
        if mode in [1, 3]:
            print("\n" + "=" * 60)
            print("MODE: PREDIKSI SAMPEL TUNGGAL DENGAN ANALISIS")
            print("Termasuk analisis korelasi Pearson yang komprehensif")
            print("=" * 60)
            df_raw_sample = pd.read_csv(test_data_path, sep=";")
            if len(df_raw_sample) > 40:
                df_raw_sample = df_raw_sample.head(35)
                print("Data dibatasi ke 35 baris teratas")
            new_data = df_raw_sample[df_raw_sample["AGE"] == 5].copy()
            min_history_days = max(3, predictor.sequence_length - 1)
            historical_data = df_raw_sample[
                (df_raw_sample["AGE"] < 5)
                & (df_raw_sample["AGE"] >= max(1, 5 - min_history_days))
            ].copy()
            print(
                f"Data baru untuk diprediksi: Periode {new_data['PERIODE'].iloc[0]}, Hari ke-{new_data['AGE'].iloc[0]}"
            )
            print(f"Data historis yang digunakan: {len(historical_data)} hari.")
            print(f"Sequence length yang diperlukan: {predictor.sequence_length}")
            if not new_data.empty and len(historical_data) > 0:
                print(
                    "Memulai prediksi dengan analisis korelasi Pearson yang komprehensif..."
                )
                predicted_values = predictor.predict_with_analysis(
                    new_day_data=new_data,
                    historical_data=historical_data,
                    create_visualizations=True,
                    save_plots=True,
                    horizon=horizon_to_evaluate,
                )
                print(f"\n{'='*50}")
                print(f"   HASIL PREDIKSI FCR SAMPEL TUNGGAL   ")
                print(f"{'='*50}")
                if isinstance(predicted_values, dict):
                    for h, pred_value in predicted_values.items():
                        current_age = new_data["AGE"].iloc[0]
                        target_age = current_age + h
                        periode = new_data["PERIODE"].iloc[0]
                        print(f"   PERIODE: {periode}")
                        print(f"   AGE saat ini: {current_age}")
                        print(
                            f"   FCR Diprediksi untuk AGE {target_age} (horizon +{h} hari): {pred_value:.6f}"
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
                            print(
                                f"   FCR Aktual untuk AGE {target_age}: {actual_h:.6f}"
                            )
                            print(
                                f"   Absolute Error: {abs(pred_value - actual_h):.6f}"
                            )
                            print(
                                f"   Relative Error: {abs(pred_value - actual_h)/actual_h*100:.4f}%"
                            )
                        print("   ---")
                else:
                    h = horizon_to_evaluate
                    current_age = new_data["AGE"].iloc[0]
                    target_age = current_age + h
                    periode = new_data["PERIODE"].iloc[0]
                    print(f"   PERIODE: {periode}")
                    print(f"   AGE saat ini: {current_age}")
                    print(
                        f"   FCR Diprediksi untuk AGE {target_age} (horizon +{h} hari): {predicted_values:.6f}"
                    )
                    target_sample = df_raw_sample[
                        (df_raw_sample["PERIODE"] == periode)
                        & (df_raw_sample["AGE"] == target_age)
                    ]
                    if not target_sample.empty and "FCR_ACT" in target_sample.columns:
                        actual_h = target_sample["FCR_ACT"].iloc[0]
                        print(f"   FCR Aktual untuk AGE {target_age}: {actual_h:.6f}")
                        print(
                            f"   Absolute Error: {abs(predicted_values - actual_h):.6f}"
                        )
                        print(
                            f"   Relative Error: {abs(predicted_values - actual_h)/actual_h*100:.4f}%"
                        )
                print(f"{'='*50}\n")
                print(
                    "\n📊 CATATAN: Analisis korelasi Pearson telah disimpan dalam file terpisah."
                )
                print(
                    "   File ini berisi matriks korelasi, p-values, dan statistik korelasi lengkap."
                )
            else:
                print(
                    "\nTidak dapat melakukan prediksi sampel tunggal karena data tidak mencukupi."
                )
    except FileNotFoundError as e:
        print(f"\nERROR: Gagal memuat model atau file. {e}")
        print(
            "Pastikan Anda telah menjalankan 'MainPipeline.py' dengan sukses terlebih dahulu."
        )
    except Exception as e:
        import traceback

        print(f"\nTerjadi kesalahan tak terduga: {e}")
        traceback.print_exc()
