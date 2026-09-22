# PredictionScript_with_Visualization.py
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
from scipy.stats import pearsonr, spearmanr

# Impor kelas pipeline dari file terpisah.
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


def asymmetric_objective(y_true, y_pred):
    """
    Fungsi loss kustom untuk XGBoost yang memberikan penalti lebih besar pada over-prediction.
    """
    residual = y_pred - y_true
    # Beri penalti 0.5x lebih besar untuk over-prediction (residual > 0)
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
        """
        Menghitung berbagai metrics evaluasi
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Menghitung MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Menghitung metrics tambahan
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
        """
        Mencetak ringkasan metrics dengan format yang rapi
        """
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

        # Interpretasi R² Score
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
        """
        Membuat plot evaluasi komprehensif
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Actual vs Predicted Plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color="blue", s=50)
        axes[0, 0].plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        axes[0, 0].set_xlabel("Actual FCR")
        axes[0, 0].set_ylabel("Predicted FCR")
        axes[0, 0].set_title("Actual vs Predicted FCR")
        axes[0, 0].grid(True, alpha=0.3)

        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].annotate(
            f"R² = {r2:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # 2. Residuals Plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color="green", s=50)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted FCR")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Predicted")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuals Distribution
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

        # 4. Error Distribution (Absolute)
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
        """
        Membuat plot perbandingan metrics
        """
        # Prepare metrics for plotting (include MAPE now)
        plot_metrics = {
            k: v
            for k, v in metrics_dict.items()
            if k not in ["N_Samples"] and isinstance(v, (int, float))
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Bar plot for main error metrics
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

        # Add value labels on bars
        for bar, value in zip(bars1, main_values):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        # MAPE plot
        mape_value = plot_metrics.get("MAPE", 0)
        bars2 = axes[0, 1].bar(["MAPE"], [mape_value], color="#FFA07A", alpha=0.8)
        axes[0, 1].set_ylabel("MAPE (%)")
        axes[0, 1].set_title("Mean Absolute Percentage Error")
        axes[0, 1].grid(True, alpha=0.3)

        # Add value label for MAPE
        axes[0, 1].text(
            0,
            mape_value + mape_value * 0.02,
            f"{mape_value:.2f}%",
            ha="center",
            va="bottom",
        )

        # Add MAPE interpretation
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

        # R² Score plot
        r2_score = plot_metrics.get("R2_Score", 0)
        bars3 = axes[1, 0].bar(["R² Score"], [r2_score], color="#96CEB4", alpha=0.8)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_ylabel("R² Score")
        axes[1, 0].set_title("Model Performance (R² Score)")
        axes[1, 0].grid(True, alpha=0.3)

        # Add value label for R²
        axes[1, 0].text(0, r2_score + 0.02, f"{r2_score:.4f}", ha="center", va="bottom")

        # Add performance interpretation for R²
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

        # Residual metrics plot
        residual_metrics = ["Mean_Residual", "Std_Residual"]
        residual_values = [abs(plot_metrics.get(m, 0)) for m in residual_metrics]
        residual_labels = ["Mean Residual", "Std Residual"]

        bars4 = axes[1, 1].bar(
            residual_labels,
            residual_values,
            color=["#DDA0DD", "#98FB98"],
            alpha=0.8,
        )
        axes[1, 1].set_ylabel("Residual Value")
        axes[1, 1].set_title("Residual Statistics")
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels for residuals
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
    """
    Kelas untuk memuat model scaler dan menormalisasi data baru secara konsisten.
    Menggunakan satu RobustScaler untuk semua fitur.
    """

    def __init__(self, info_path: str, robust_scaler_path: str):
        with open(info_path, "r") as f:
            self.info = json.load(f)
        with open(robust_scaler_path, "rb") as f:
            self.robust_scaler = pickle.load(f)

        # Gabungkan semua fitur yang perlu dinormalisasi
        self.features_to_normalize = list(set(self.info["robust_features"]))
        # Simpan urutan fitur dari model scaler untuk memastikan konsistensi
        self.feature_names_order = (
            self.robust_scaler.feature_names_in_
        )  # Perbaikan disini
        print("DataNormalizer berhasil dimuat.")
        print(
            f"Menggunakan satu RobustScaler untuk {len(self.features_to_normalize)} fitur."
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        print("  - Menerapkan normalisasi pada data...")
        if self.features_to_normalize:
            # Filter kolom yang ada di dataframe dan di daftar fitur yang akan dinormalisasi
            cols_to_transform = [
                col
                for col in self.feature_names_order
                if col in df_transformed.columns and col in self.features_to_normalize
            ]

            # Pastikan urutan kolom sama dengan urutan saat model dilatih
            try:
                # Gunakan kolom dalam urutan yang sama dengan saat model dilatih
                df_transformed[cols_to_transform] = self.robust_scaler.transform(
                    df_transformed[cols_to_transform]
                )
            except Exception as e:
                print(f"  Error saat normalisasi: {str(e)}")
                # Jika gagal, coba dengan pendekatan yang lebih eksplisit
                try:
                    # Buat DataFrame sementara dengan kolom dalam urutan yang benar
                    temp_df = pd.DataFrame()
                    for col in cols_to_transform:
                        if col in df_transformed.columns:
                            temp_df[col] = df_transformed[col]

                    # Transform data dengan urutan kolom yang benar
                    transformed_values = self.robust_scaler.transform(temp_df)

                    # Masukkan kembali nilai yang sudah ditransformasi ke dataframe asli
                    for i, col in enumerate(cols_to_transform):
                        if col in df_transformed.columns:
                            df_transformed[col] = transformed_values[:, i]
                except Exception as e2:
                    print(
                        f"  Gagal normalisasi dengan pendekatan alternatif: {str(e2)}"
                    )
                    # Jika masih gagal, kembalikan data asli tanpa transformasi
                    print("  Mengembalikan data tanpa normalisasi")

        return df_transformed


class FeatureAnalyzer:
    """
    Kelas untuk menganalisis dan memvisualisasikan feature importance
    """

    def __init__(self):
        self.feature_names = []
        self.feature_importance = []
        self.feature_values = []
        self.temporal_features = []
        self.tabular_features = []

    def extract_feature_importance(self, xgb_model, feature_names):
        """Ekstrak feature importance dari model XGBoost"""
        importance_dict = xgb_model.get_booster().get_score(importance_type="weight")

        # Buat mapping dari feature index ke nama
        feature_importance_mapped = {}
        for i, name in enumerate(feature_names):
            feature_key = f"f{i}"
            if feature_key in importance_dict:
                feature_importance_mapped[name] = importance_dict[feature_key]
            else:
                feature_importance_mapped[name] = 0

        return feature_importance_mapped

    def categorize_features(self, feature_names, n_tabular_features):
        """Kategorikan fitur menjadi tabular dan temporal"""
        tabular_features = feature_names[:n_tabular_features]
        temporal_features = feature_names[n_tabular_features:]
        return tabular_features, temporal_features

    def create_feature_importance_plot(
        self,
        feature_importance_dict,
        save_path=None,
        title="Top 50 Feature Importance untuk Prediksi FCR",
    ):
        """Buat plot feature importance"""
        # Sort features by importance
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        # Ambil top 50 features
        top_features = sorted_features[:50]
        feature_names = [item[0] for item in top_features]
        importance_values = [item[1] for item in top_features]

        # Create plot
        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(feature_names)), importance_values)

        # Color bars based on feature type
        colors = []
        for name in feature_names:
            if name.startswith("temporal_") or "TCN" in name:
                colors.append("#FF6B6B")  # Red for temporal
            else:
                colors.append("#4ECDC4")  # Teal for tabular

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel("Feature Importance (Weight)")
        plt.title(
            "Top 50 Feature Importance untuk Prediksi FCR",
            fontsize=12,
            fontweight="bold",
        )
        plt.grid(axis="x", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#FF6B6B", label="Temporal Features (TCN)"),
            Patch(facecolor="#4ECDC4", label="Tabular Features"),
        ]
        plt.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return top_features

    def create_feature_distribution_plot(
        self, feature_values, feature_names, save_path=None
    ):
        """Buat plot distribusi nilai fitur"""
        # Pilih beberapa fitur penting untuk divisualisasikan
        n_features_to_plot = min(12, len(feature_names))

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for i in range(n_features_to_plot):
            if i < len(feature_values[0]):
                values = [row[i] for row in feature_values]
                axes[i].hist(
                    values, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
                )
                axes[i].set_title(f"{feature_names[i][:20]}...", fontsize=10)
                axes[i].set_xlabel("Nilai")
                axes[i].set_ylabel("Frekuensi")
                axes[i].grid(alpha=0.3)

        # Hide unused subplots
        for i in range(n_features_to_plot, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(
            "Distribusi Nilai Fitur untuk Prediksi", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def calculate_pearson_pvalues(self, df):
        """
        Menghitung p-values untuk korelasi Pearson.
        """
        n_features = len(df.columns)
        p_values = np.ones((n_features, n_features))

        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    col1, col2 = df.columns[i], df.columns[j]
                    valid_data = df[[col1, col2]].dropna()
                    if len(valid_data) > 2:
                        _, p_val = pearsonr(valid_data[col1], valid_data[col2])
                        p_values[i, j] = p_val
                        p_values[j, i] = p_val
                except:
                    p_values[i, j] = 1.0
                    p_values[j, i] = 1.0

        return pd.DataFrame(p_values, index=df.columns, columns=df.columns)

    def analyze_feature_correlations(
        self, feature_values, feature_names, target_values=None, save_path=None
    ):
        """
        Analisis korelasi fitur yang komprehensif dengan Pearson correlation.
        """
        print("\n=== MEMULAI ANALISIS KORELASI PEARSON FITUR ===")

        # Konversi ke DataFrame
        df_features = pd.DataFrame(feature_values, columns=feature_names)

        if target_values is not None:
            df_features["FCR_TARGET"] = target_values

        # 1. Hitung matriks korelasi Pearson
        print("Menghitung matriks korelasi Pearson...")
        pearson_correlation_matrix = df_features.corr(method="pearson")

        # 2. Hitung matriks korelasi Spearman untuk perbandingan
        print("Menghitung matriks korelasi Spearman...")
        spearman_correlation_matrix = df_features.corr(method="spearman")

        # 3. Hitung p-values untuk korelasi Pearson
        print("Menghitung p-values untuk korelasi Pearson...")
        pearson_pvalues = self.calculate_pearson_pvalues(df_features)

        # 4. Identifikasi korelasi yang signifikan
        significant_correlations = pearson_pvalues < 0.05

        # 5. Kategorisasi fitur
        temporal_cols = [
            col for col in feature_names if col.startswith("temporal_") or "TCN" in col
        ]
        tabular_cols = [col for col in feature_names if col not in temporal_cols]

        # 6. Analisis korelasi antar kategori
        analysis_results = {
            "pearson_matrix": pearson_correlation_matrix,
            "spearman_matrix": spearman_correlation_matrix,
            "p_values": pearson_pvalues,
            "significant_correlations": significant_correlations,
            "temporal_features": temporal_cols,
            "tabular_features": tabular_cols,
        }

        # 7. Analisis korelasi dengan target jika tersedia
        if target_values is not None:
            target_correlations = pearson_correlation_matrix["FCR_TARGET"].drop(
                "FCR_TARGET"
            )
            target_pvalues = pearson_pvalues["FCR_TARGET"].drop("FCR_TARGET")

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

        # Box plot korelasi dengan target (jika ada)
        if "target_correlations" in analysis_results:
            target_corrs = analysis_results["target_correlations"]
            temporal_cols = analysis_results["temporal_features"]
            tabular_cols = analysis_results["tabular_features"]

            temporal_target_corrs = [
                abs(target_corrs[col])
                for col in temporal_cols
                if col in target_corrs.index
            ]
            tabular_target_corrs = [
                abs(target_corrs[col])
                for col in tabular_cols
                if col in target_corrs.index
            ]

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
        axes[1, 0].scatter(
            pearson_vals[valid_idx], spearman_vals[valid_idx], alpha=0.6, color="green"
        )
        axes[1, 0].plot([-1, 1], [-1, 1], "r--", alpha=0.8)
        axes[1, 0].set_xlabel("Korelasi Pearson")
        axes[1, 0].set_ylabel("Korelasi Spearman")
        axes[1, 0].set_title("Pearson vs Spearman Correlation")
        axes[1, 0].grid(alpha=0.3)

        # Bar plot korelasi terkuat dengan target
        if "target_correlations" in analysis_results:
            target_corrs = analysis_results["target_correlations"]
            top_corrs = target_corrs.abs().sort_values(ascending=False).head(10)

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

        # 3. Heatmap korelasi signifikan saja
        if "target_correlations" in analysis_results:
            target_corrs = analysis_results["target_correlations"]
            target_pvals = analysis_results["target_pvalues"]

            # Filter korelasi yang signifikan
            significant_target_corrs = target_corrs[target_pvals < 0.05]

            if len(significant_target_corrs) > 0:
                plt.figure(figsize=(12, 8))

                # Buat DataFrame untuk heatmap
                sig_corr_df = pd.DataFrame(
                    {"Correlation": significant_target_corrs.values},
                    index=significant_target_corrs.index,
                )

                sns.heatmap(
                    sig_corr_df.T,
                    annot=True,
                    cmap="RdBu_r",
                    center=0,
                    cbar_kws={"label": "Korelasi Pearson (p < 0.05)"},
                    fmt=".3f",
                )
                plt.title(
                    "Korelasi Signifikan dengan Target (p < 0.05)",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.tight_layout()

                if save_path:
                    plt.savefig(
                        save_path.replace(".png", "_significant_correlations.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                plt.show()

        print("\nVisualisasi korelasi Pearson selesai dibuat!")
        if save_path:
            print(f"Plot disimpan dengan prefix: {save_path.replace('.png', '')}")

    def create_feature_correlation_heatmap(
        self, feature_values, feature_names, save_path=None
    ):
        """Buat heatmap korelasi antar fitur"""
        # Konversi ke DataFrame
        df_features = pd.DataFrame(feature_values, columns=feature_names)

        # Hitung korelasi
        correlation_matrix = df_features.corr()

        # Pilih fitur yang paling penting untuk heatmap
        n_features = min(15, len(feature_names))
        top_features = feature_names[:n_features]
        correlation_subset = correlation_matrix.loc[top_features, top_features]

        # Create heatmap
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
        """Buat perbandingan kontribusi fitur temporal vs tabular"""
        tabular_importance = 0
        temporal_importance = 0

        for i, (feature_name, importance) in enumerate(feature_importance_dict.items()):
            if i < n_tabular_features:
                tabular_importance += importance
            else:
                temporal_importance += importance

        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Pie chart
        labels = ["Tabular Features", "Temporal Features (TCN)"]
        values = [tabular_importance, temporal_importance]
        colors = ["#4ECDC4", "#FF6B6B"]

        ax1.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Kontribusi Fitur Temporal vs Tabular", fontweight="bold")

        # Bar chart
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
    """
    Mengelola seluruh pipeline prediksi dari data mentah hingga hasil akhir.
    Mendukung prediksi multi-horizon.
    """

    def __init__(self, models_dir: str, config_dir: str, sliding_window: int = None):
        print("Memuat semua model dan pipeline untuk prediksi...")

        # 1. Muat pipeline rekayasa fitur
        fe_path = os.path.join(models_dir, "feature_engineering_pipeline.pkl")
        with open(fe_path, "rb") as f:
            self.feature_pipeline = pickle.load(f)

        # 2. Inisialisasi normalizer
        info_path = os.path.join(config_dir, "normalization_info.json")
        robust_path = os.path.join(
            models_dir, "robust_scaler.pkl"
        )  # Gunakan hanya satu scaler
        self.normalizer = DataNormalizer(info_path, robust_path)

        # 3. Muat model TCN, XGBoost, dan config hibrida
        tcn_path = os.path.join(models_dir, "tcn_encoder.keras")
        config_path = os.path.join(models_dir, "hybrid_model_config.json")

        self.tcn_encoder = tf.keras.models.load_model(
            tcn_path, custom_objects={"TCN": TCN}
        )

        # Muat konfigurasi model hibrida
        with open(config_path, "r") as f:
            self.hybrid_config = json.load(f)

        # Gunakan sliding_window yang diberikan atau default dari model
        self.sequence_length = (
            sliding_window if sliding_window else self.hybrid_config["sequence_length"]
        )
        self.horizons = [1, 3]  # Tetapkan horizon ke +1 dan +3 hari

        # Muat model XGBoost untuk setiap horizon
        self.xgb_models = {}
        for horizon in self.horizons:
            xgb_path = os.path.join(
                models_dir, f"xgboost_hybrid_model_horizon_{horizon}d.pkl"
            )
            with open(xgb_path, "rb") as f:
                self.xgb_models[f"horizon_{horizon}d"] = pickle.load(f)

        # 4. Inisialisasi analyzer dan evaluator
        self.analyzer = FeatureAnalyzer()
        self.evaluator = EvaluationMetrics()

        print("Semua komponen prediksi berhasil dimuat.\n")
        print(f"Model mendukung prediksi untuk horizon: {self.horizons} hari")
        print(f"Menggunakan sliding window: {self.sequence_length} hari")

    def predict_single_sample(
        self, new_day_data: pd.DataFrame, historical_data: pd.DataFrame, horizon=None
    ):
        """
        Prediksi untuk satu sampel tanpa visualisasi (untuk batch prediction)

        Args:
            new_day_data: Data hari yang akan diprediksi
            historical_data: Data historis untuk konteks
            horizon: Horizon prediksi dalam hari (1, 3, dll). Jika None, semua horizon akan diprediksi

        Returns:
            Jika horizon=None: Dictionary berisi prediksi untuk semua horizon
            Jika horizon ditentukan: Nilai prediksi untuk horizon tersebut
        """
        # Langkah A: Rekayasa Fitur
        combined_data = pd.concat([historical_data, new_day_data], ignore_index=True)
        combined_data = combined_data.sort_values(["PERIODE", "AGE"]).reset_index(
            drop=True
        )

        data_enhanced_full = self.feature_pipeline.transform(combined_data)
        data_enhanced = data_enhanced_full.tail(len(new_day_data))

        # Langkah B: Normalisasi Fitur
        data_normalized = self.normalizer.transform(data_enhanced)
        historical_enhanced = self.feature_pipeline.transform(historical_data)
        historical_normalized = self.normalizer.transform(historical_enhanced)

        # Langkah C: Prediksi Hibrida
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

        # Ekstrak Fitur Temporal dengan TCN
        temporal_features = self.tcn_encoder.predict(last_sequence_values, verbose=0)

        # Ambil Fitur Tabular dari hari terakhir
        tabular_features = data_normalized[features_for_model].values.reshape(1, -1)

        # Gabungkan fitur dan lakukan prediksi
        hybrid_features = np.concatenate([tabular_features, temporal_features], axis=1)

        # Prediksi untuk semua horizon atau horizon tertentu
        predictions = {}

        if horizon is not None:
            # Prediksi untuk horizon tertentu
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
            # Prediksi untuk semua horizon
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
        max_rows=35,  # Batasi data input ke 35 baris
    ):
        """
        Evaluasi model pada seluruh dataset test

        Args:
            test_data_path: Path ke file data test
            create_visualizations: Apakah akan membuat visualisasi
            save_plots: Apakah akan menyimpan plot
            horizon: Horizon prediksi dalam hari (1, 3, dll). Jika None, semua horizon akan diprediksi
            max_rows: Jumlah maksimum baris data yang digunakan (default: 35)
        """
        print("--- Memulai Evaluasi Model pada Data Test ---")

        if horizon is not None:
            print(f"Evaluasi untuk horizon: {horizon} hari")
            if horizon not in self.horizons:
                raise ValueError(
                    f"Horizon {horizon} tidak didukung. Horizon yang tersedia: {self.horizons}"
                )
        else:
            print(f"Evaluasi untuk semua horizon: {self.horizons}")

        # Load test data
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"File test data tidak ditemukan di {test_data_path}"
            )

        df_test = pd.read_csv(test_data_path, sep=";")
        print(f"Data test dimuat: {len(df_test)} samples")

        # Batasi data input ke max_rows
        if len(df_test) > max_rows:
            print(f"Membatasi data input ke {max_rows} baris teratas")
            df_test = df_test.head(max_rows)

        # Get unique periods and ages for evaluation
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
                # Get current sample
                current_sample = df_test[
                    (df_test["PERIODE"] == periode) & (df_test["AGE"] == age)
                ].copy()

                if current_sample.empty or "FCR_ACT" not in current_sample.columns:
                    continue

                # Get historical data for this prediction
                min_history_days = max(3, self.sequence_length - 1)
                historical_data = df_test[
                    (df_test["PERIODE"] == periode)
                    & (df_test["AGE"] < age)
                    & (df_test["AGE"] >= max(1, age - min_history_days))
                ].copy()

                # Check if we have enough historical data
                if len(historical_data) < min_history_days:
                    continue

                # Untuk horizon +1 dan +3, kita perlu data aktual di hari berikutnya
                if horizon is not None:
                    target_age = age + horizon
                    target_sample = df_test[
                        (df_test["PERIODE"] == periode) & (df_test["AGE"] == target_age)
                    ]
                    if target_sample.empty or "FCR_ACT" not in target_sample.columns:
                        continue
                    actual_fcr = target_sample["FCR_ACT"].iloc[0]
                else:
                    # Jika prediksi untuk semua horizon, gunakan horizon pertama untuk evaluasi
                    h = self.horizons[0]
                    target_age = age + h
                    target_sample = df_test[
                        (df_test["PERIODE"] == periode) & (df_test["AGE"] == target_age)
                    ]
                    if target_sample.empty or "FCR_ACT" not in target_sample.columns:
                        continue
                    actual_fcr = target_sample["FCR_ACT"].iloc[0]

                # Make prediction
                predicted_fcr = self.predict_single_sample(
                    current_sample, historical_data, horizon
                )

                predictions.append(predicted_fcr)
                actual_values.append(actual_fcr)
                periode_values.append(periode)  # Simpan nilai PERIODE
                age_values.append(target_age)  # Simpan target_age, bukan age saat ini
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

        # Convert to numpy arrays
        y_true = np.array(actual_values)
        y_pred = np.array(predictions)

        # Calculate metrics
        print("Menghitung metrics evaluasi...")
        metrics = self.evaluator.calculate_metrics(y_true, y_pred)

        # Print metrics summary
        self.evaluator.print_metrics_summary(metrics, "Test Data Evaluation Results")

        # Create visualizations if requested
        if create_visualizations:
            print("Membuat visualisasi evaluasi...")

            save_dir = "evaluation_plots"
            if save_plots:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. Evaluation plots
            save_path = (
                os.path.join(save_dir, f"evaluation_plots_{timestamp}.png")
                if save_plots
                else None
            )
            self.evaluator.create_evaluation_plots(y_true, y_pred, save_path)

            # 2. Metrics comparison plot
            save_path = (
                os.path.join(save_dir, f"metrics_comparison_{timestamp}.png")
                if save_plots
                else None
            )
            self.evaluator.create_metrics_comparison_plot(metrics, save_path)

            if save_plots:
                print(f"Plot evaluasi disimpan di direktori: {save_dir}")

        # Save detailed results
        if save_plots:
            results_df = pd.DataFrame(
                {
                    "PERIODE": periode_values,
                    "AGE": age_values,
                    "Actual_FCR": y_true,
                    "Predicted_FCR": y_pred,
                    "Absolute_Error": np.abs(y_true - y_pred),
                    "Squared_Error": (y_true - y_pred)
                    ** 2,  # SE = (Actual - Predicted)^2
                    "Percentage_Error": np.abs((y_true - y_pred) / y_true) * 100,
                }
            )

            results_path = os.path.join(
                save_dir if save_plots else ".", f"prediction_results_{timestamp}.csv"
            )
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
        """
        Menjalankan prediksi dengan analisis fitur lengkap dan visualisasi.

        Args:
            new_day_data: Data hari yang akan diprediksi
            historical_data: Data historis untuk konteks
            create_visualizations: Apakah akan membuat visualisasi
            save_plots: Apakah akan menyimpan plot
            horizon: Horizon prediksi dalam hari (1, 3, dll). Jika None, semua horizon akan diprediksi
        """
        print("--- Memulai Proses Prediksi dengan Analisis ---")

        if horizon is not None:
            print(f"Prediksi untuk horizon: {horizon} hari")
            if horizon not in self.horizons:
                raise ValueError(
                    f"Horizon {horizon} tidak didukung. Horizon yang tersedia: {self.horizons}"
                )
        else:
            print(f"Prediksi untuk semua horizon: {self.horizons}")
            # Default ke horizon pertama untuk analisis
            horizon = self.horizons[0]

        # Langkah A: Rekayasa Fitur
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

        # Langkah B: Normalisasi Fitur
        print("Langkah B: Menjalankan normalisasi...")

        data_normalized = self.normalizer.transform(data_enhanced)
        historical_enhanced = self.feature_pipeline.transform(historical_data)
        historical_normalized = self.normalizer.transform(historical_enhanced)

        # Langkah C: Prediksi Hibrida
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

        # Ekstrak Fitur Temporal dengan TCN
        temporal_features = self.tcn_encoder.predict(last_sequence_values, verbose=0)

        # Ambil Fitur Tabular dari hari terakhir
        tabular_features = data_normalized[features_for_model].values.reshape(1, -1)

        # Gabungkan fitur dan lakukan prediksi
        hybrid_features = np.concatenate([tabular_features, temporal_features], axis=1)

        # Prediksi untuk semua horizon atau horizon tertentu
        if horizon is not None:
            # Gunakan model untuk horizon yang dipilih
            model_key = f"horizon_{horizon}d"
            final_prediction = self.xgb_models[model_key].predict(hybrid_features)
            predictions = {horizon: final_prediction[0]}
        else:
            # Prediksi untuk semua horizon
            predictions = {}
            for h in self.horizons:
                model_key = f"horizon_{h}d"
                predictions[h] = self.xgb_models[model_key].predict(hybrid_features)[0]

        # Analisis fitur
        if create_visualizations:
            print("Langkah D: Menjalankan analisis fitur...")
            n_tabular_features = len(features_for_model)
            n_temporal_features = temporal_features.shape[1]

            # Jika horizon tidak ditentukan, gunakan horizon pertama untuk analisis
            h_for_analysis = horizon if horizon is not None else self.horizons[0]
            model_key = f"horizon_{h_for_analysis}d"

            # Set up save directory
            save_dir = "feature_analysis_plots"
            if save_plots:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Langkah E: Analisis Korelasi Pearson (TAMBAHAN BARU)
            print("Langkah E: Menjalankan analisis korelasi Pearson...")

            # Buat nama fitur untuk hybrid features
            tabular_feature_names = features_for_model[:n_tabular_features]
            temporal_feature_names = [
                f"temporal_feature_{i+1}" for i in range(n_temporal_features)
            ]
            all_feature_names = tabular_feature_names + temporal_feature_names

            # Ekstrak nilai fitur untuk analisis korelasi
            feature_values = hybrid_features

            # Jika ada target values, gunakan untuk analisis
            target_values = None
            if "FCR_ACT" in new_day_data.columns:
                target_values = new_day_data["FCR_ACT"].values

            # Lakukan analisis korelasi
            save_correlation_path = None
            if save_plots:
                save_correlation_path = os.path.join(
                    save_dir, f"correlation_analysis_{timestamp}.json"
                )

            correlation_results = self.analyzer.analyze_feature_correlations(
                feature_values=feature_values,
                feature_names=all_feature_names,
                target_values=target_values,
                save_path=save_correlation_path,
            )

            # Buat visualisasi korelasi
            correlation_plot_path = None
            if save_plots:
                correlation_plot_path = os.path.join(
                    save_dir, f"correlation_plots_{timestamp}.png"
                )

            self.analyzer.create_comprehensive_correlation_plots(
                analysis_results=correlation_results, save_path=correlation_plot_path
            )

            self._analyze_features(
                hybrid_features,
                n_tabular_features,
                features_for_model,
                n_temporal_features,
                save_plots,
                model_key=model_key,
                horizon=h_for_analysis,
            )

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
        """Analisis feature importance dengan visualisasi"""

        # Buat nama fitur untuk hybrid features
        tabular_feature_names = base_feature_names[:n_tabular_features]
        temporal_feature_names = [
            f"temporal_feature_{i+1}" for i in range(n_temporal_features)
        ]
        all_feature_names = tabular_feature_names + temporal_feature_names

        # Extract feature importance dari model XGBoost untuk horizon tertentu
        feature_importance_dict = self.analyzer.extract_feature_importance(
            self.xgb_models[model_key], all_feature_names
        )

        # Set up save directory
        save_dir = "feature_analysis_plots"
        if save_plots:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Feature Importance Plot
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

        # 2. Feature Distribution Plot
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

        # 3. Temporal vs Tabular Comparison
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

        # Print summary
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
            print(f"  {i+1}. {feature_name}: {importance:.1f}")
        print(f"{'='*50}\n")

        if save_plots:
            print(f"Plot disimpan di direktori: {save_dir}")


if __name__ == "__main__":
    # =======================================================================
    # SIMULASI PENGGUNAAN DENGAN ANALISIS FITUR DAN EVALUASI
    # =======================================================================

    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, "5. Model", "models_terbaru")
    CONFIG_DIR = os.path.join(BASE_DIR, "2. Hasil Normalisasi", "Terbaru")

    try:
        # Pilih sliding window (opsional)
        print("Masukkan ukuran sliding window (default: gunakan nilai dari model):")
        sliding_window_input = input()
        sliding_window = (
            int(sliding_window_input) if sliding_window_input.strip() else None
        )

        # 1. Inisialisasi Predictor
        predictor = HybridPredictor(
            models_dir=MODELS_DIR, config_dir=CONFIG_DIR, sliding_window=sliding_window
        )

        # 2. Pilih mode operasi
        print("Pilih mode operasi:")
        print("1. Prediksi sampel tunggal dengan analisis")
        print("2. Evaluasi pada seluruh data test")
        print("3. Keduanya")

        mode = int(input("Pilihan Anda (1/2/3): "))

        # Path ke data test
        test_data_path = os.path.join(
            BASE_DIR, "3. non-normalize", "NORMALISASI_PERIODE_14.csv"
        )

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"File data test tidak ditemukan di {test_data_path}"
            )

        # Pilih horizon untuk evaluasi
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

            # Batasi jumlah baris data yang digunakan
            max_rows = 35
            print(f"Menggunakan maksimal {max_rows} baris data untuk evaluasi")

            # Evaluasi pada seluruh data test
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
            print("=" * 60)

            # Siapkan data untuk prediksi sampel tunggal
            df_raw_sample = pd.read_csv(test_data_path, sep=";")

            # Batasi data ke 35 baris jika lebih dari itu
            if len(df_raw_sample) > 35:
                df_raw_sample = df_raw_sample.head(35)
                print("Data dibatasi ke 35 baris teratas")

            # Ambil data hari ke-20
            new_data = df_raw_sample[df_raw_sample["AGE"] == 35].copy()

            # Ambil data historis
            min_history_days = max(3, predictor.sequence_length - 1)
            historical_data = df_raw_sample[
                (df_raw_sample["AGE"] < 35)
                & (df_raw_sample["AGE"] >= max(1, 35 - min_history_days))
            ].copy()

            print(
                f"Data baru untuk diprediksi: Periode {new_data['PERIODE'].iloc[0]}, Hari ke-{new_data['AGE'].iloc[0]}"
            )
            print(f"Data historis yang digunakan: {len(historical_data)} hari.")
            print(f"Sequence length yang diperlukan: {predictor.sequence_length}")

            if not new_data.empty and len(historical_data) > 0:
                # Lakukan prediksi dengan analisis untuk horizon yang dipilih
                predicted_values = predictor.predict_with_analysis(
                    new_day_data=new_data,
                    historical_data=historical_data,
                    create_visualizations=True,
                    save_plots=True,
                    horizon=horizon_to_evaluate,
                )

                # Tampilkan hasil dan bandingkan dengan nilai aktual jika ada
                print(f"\n{'='*50}")
                print(f"   HASIL PREDIKSI FCR SAMPEL TUNGGAL   ")
                print(f"{'='*50}")

                if isinstance(predicted_values, dict):
                    # Hasil untuk semua horizon
                    for h, pred_value in predicted_values.items():
                        # Tampilkan PERIODE dan AGE untuk prediksi
                        current_age = new_data["AGE"].iloc[0]
                        target_age = current_age + h
                        periode = new_data["PERIODE"].iloc[0]
                        print(f"   PERIODE: {periode}")
                        print(f"   AGE saat ini: {current_age}")
                        print(
                            f"   FCR Diprediksi untuk AGE {target_age} (horizon +{h} hari): {pred_value:.6f}"
                        )

                        # Jika ada data aktual untuk horizon ini
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
                    # Hasil untuk horizon tertentu
                    h = horizon_to_evaluate
                    current_age = new_data["AGE"].iloc[0]
                    target_age = current_age + h
                    periode = new_data["PERIODE"].iloc[0]
                    print(f"   PERIODE: {periode}")
                    print(f"   AGE saat ini: {current_age}")
                    print(
                        f"   FCR Diprediksi untuk AGE {target_age} (horizon +{h} hari): {predicted_values:.6f}"
                    )

                    # Jika ada data aktual untuk horizon ini
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
