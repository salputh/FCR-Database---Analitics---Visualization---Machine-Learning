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

# Impor kelas pipeline dari file terpisah.
try:
    import sys

    sys.path.append("6. Pelatihan2")
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
    grad = np.where(residual > 0, 2.0 * 0.5 * residual, 5.0 * residual)
    hess = np.where(residual > 0, 2.0 * 0.5, 5.0)
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
        # Prepare metrics for plotting (exclude non-numeric ones)
        plot_metrics = {
            k: v
            for k, v in metrics_dict.items()
            if k not in ["N_Samples", "MAPE"] and isinstance(v, (int, float))
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot for main metrics
        main_metrics = ["MAE", "MSE", "RMSE"]
        main_values = [plot_metrics.get(m, 0) for m in main_metrics]

        bars1 = ax1.bar(
            main_metrics,
            main_values,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
            alpha=0.8,
        )
        ax1.set_ylabel("Error Value")
        ax1.set_title("Error Metrics")
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, main_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        # R² Score plot
        r2_score = plot_metrics.get("R2_Score", 0)
        ax2.bar(["R² Score"], [r2_score], color="#96CEB4", alpha=0.8)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("R² Score")
        ax2.set_title("Model Performance (R² Score)")
        ax2.grid(True, alpha=0.3)

        # Add value label
        ax2.text(0, r2_score + 0.02, f"{r2_score:.4f}", ha="center", va="bottom")

        # Add performance interpretation
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

        ax2.text(
            0,
            r2_score / 2,
            interpretation,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


class DataNormalizer:
    """
    Kelas untuk memuat model scaler dan menormalisasi data baru secara konsisten
    dengan metode clipping + MinMaxScaler + seleksi fitur.
    """

    def __init__(
        self,
        info_path: str,
        minmax_scaler_path: str,
    ):
        with open(info_path, "r") as f:
            self.info = json.load(f)
        with open(minmax_scaler_path, "rb") as f:
            self.minmax_scaler = pickle.load(f)

        # Ambil informasi dari metadata normalisasi
        self.normalized_features = self.info.get("normalized_features", [])
        self.non_normalized_cols = self.info.get(
            "non_normalized", ["TANGGAL", "PERIODE", "FCR_ACT"]
        )
        self.clipping_info = self.info.get("clipping_info", {})
        self.feature_selection = self.info.get("feature_selection", {})
        self.final_columns = self.info.get("final_columns", [])

        # Tambahan: Ambil informasi original features untuk debugging
        self.original_features = self.info.get("feature_selection", {}).get(
            "original_features", []
        )
        self.normalization_method = self.info.get(
            "normalization_method", "clipping_minmax_with_feature_selection"
        )

        print("DataNormalizer berhasil dimuat.")
        print(f"  - Metode normalisasi: {self.normalization_method}")
        print(f"  - Fitur original: {len(self.original_features)}")
        print(f"  - Fitur yang akan dinormalisasi: {len(self.normalized_features)}")
        print(f"  - Fitur yang dikecualikan: {len(self.non_normalized_cols)}")
        if self.feature_selection:
            removed_count = self.feature_selection.get("low_variance_removal", {}).get(
                "removed_count", 0
            ) + self.feature_selection.get("high_correlation_removal", {}).get(
                "removed_count", 0
            )
            print(f"  - Fitur yang dihapus saat training: {removed_count}")
            print(
                f"    - Varians rendah: {self.feature_selection.get('low_variance_removal', {}).get('removed_count', 0)}"
            )
            print(
                f"    - Korelasi tinggi: {self.feature_selection.get('high_correlation_removal', {}).get('removed_count', 0)}"
            )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menerapkan transformasi yang sama seperti saat training:
        1. Clipping berdasarkan IQR
        2. MinMaxScaler
        3. Seleksi fitur (hapus kolom yang tidak ada di final_columns)
        """
        df_transformed = df.copy()
        print("  - Menerapkan normalisasi pada data...")
        print(f"    Input shape: {df_transformed.shape}")

        # 1. Identifikasi fitur yang perlu dinormalisasi
        features_to_normalize = [
            col for col in self.normalized_features if col in df_transformed.columns
        ]

        # Validasi: periksa fitur yang hilang
        missing_features = [
            col for col in self.normalized_features if col not in df_transformed.columns
        ]
        if missing_features:
            print(
                f"    PERINGATAN: {len(missing_features)} fitur tidak ditemukan dalam data:"
            )
            for feat in missing_features[:5]:  # Tampilkan 5 pertama
                print(f"      - {feat}")
            if len(missing_features) > 5:
                print(f"      ... dan {len(missing_features) - 5} fitur lainnya")

        if not features_to_normalize:
            print("    Tidak ada fitur yang perlu dinormalisasi.")
            # Tetap terapkan seleksi fitur meskipun tidak ada normalisasi
            if self.final_columns:
                available_final_cols = [
                    col for col in self.final_columns if col in df_transformed.columns
                ]
                df_transformed = df_transformed[available_final_cols]
            return df_transformed

        # 2. Isi nilai NaN dengan median (konsisten dengan training)
        print(f"    Mengisi nilai NaN untuk {len(features_to_normalize)} fitur...")
        features_data = df_transformed[features_to_normalize].copy()

        # Hitung dan tampilkan statistik NaN
        nan_counts = features_data.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            print(f"      Total nilai NaN ditemukan: {total_nans}")
            features_with_nans = nan_counts[nan_counts > 0]
            for feat, count in features_with_nans.head(3).items():
                print(f"        {feat}: {count} NaN")

        features_data = features_data.fillna(features_data.median())

        # 3. Terapkan clipping berdasarkan informasi dari training
        print(f"    Menerapkan clipping pada {len(features_to_normalize)} fitur...")
        total_clipped = 0
        for feature in features_to_normalize:
            if feature in self.clipping_info:
                clip_info = self.clipping_info[feature]
                lower_bound = clip_info["lower_bound"]
                upper_bound = clip_info["upper_bound"]

                # Lakukan clipping
                original_outliers = (
                    (features_data[feature] > upper_bound)
                    | (features_data[feature] < lower_bound)
                ).sum()

                features_data[feature] = features_data[feature].clip(
                    lower=lower_bound, upper=upper_bound
                )

                total_clipped += original_outliers
                if original_outliers > 0:
                    print(f"      {feature}: {original_outliers} pencilan di-clip")

        if total_clipped > 0:
            print(f"    Total pencilan yang di-clip: {total_clipped}")

        # 4. Terapkan MinMaxScaler
        print("    Menerapkan MinMaxScaler...")
        try:
            df_transformed[features_to_normalize] = self.minmax_scaler.transform(
                features_data
            )
        except Exception as e:
            print(f"    ERROR saat MinMaxScaler: {e}")
            print(f"    Shape features_data: {features_data.shape}")
            print(f"    Expected features: {len(self.normalized_features)}")
            raise

        # 5. Seleksi fitur - pastikan hanya kolom yang ada di final_columns yang dipertahankan
        if self.final_columns:
            print(
                f"    Menerapkan seleksi fitur (target: {len(self.final_columns)} kolom)..."
            )

            # Tambahkan kolom yang hilang dengan nilai 0 (konsisten dengan training)
            missing_final_cols = []
            for col in self.final_columns:
                if col not in df_transformed.columns:
                    df_transformed[col] = 0
                    missing_final_cols.append(col)

            if missing_final_cols:
                print(
                    f"      Menambahkan {len(missing_final_cols)} kolom yang hilang dengan nilai 0"
                )

            # Pilih hanya kolom yang ada di final_columns
            available_final_cols = [
                col for col in self.final_columns if col in df_transformed.columns
            ]

            original_cols = set(df.columns)
            df_transformed = df_transformed[available_final_cols]

            removed_cols = original_cols - set(available_final_cols)
            if removed_cols:
                print(f"    Fitur yang dihapus (seleksi fitur): {len(removed_cols)}")

        print(f"    Normalisasi selesai. Shape akhir: {df_transformed.shape}")
        return df_transformed

    def get_normalization_info(self):
        """
        Mengembalikan informasi detail tentang proses normalisasi.
        """
        return {
            "normalization_method": self.normalization_method,
            "original_features_count": len(self.original_features),
            "normalized_features_count": len(self.normalized_features),
            "normalized_features": self.normalized_features,
            "non_normalized_columns": self.non_normalized_cols,
            "clipping_applied": len(self.clipping_info),
            "feature_selection_applied": bool(self.feature_selection),
            "final_columns_count": len(self.final_columns),
            "removed_features_count": (
                (
                    self.feature_selection.get("low_variance_removal", {}).get(
                        "removed_count", 0
                    )
                    + self.feature_selection.get("high_correlation_removal", {}).get(
                        "removed_count", 0
                    )
                )
                if self.feature_selection
                else 0
            ),
        }

    def validate_data_compatibility(self, df: pd.DataFrame) -> dict:
        """
        Validasi kompatibilitas data dengan model yang telah dilatih.

        Returns:
            dict: Laporan validasi dengan informasi detail
        """
        validation_report = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        # 1. Periksa kolom yang diperlukan
        required_cols = set(self.original_features + self.non_normalized_cols)
        available_cols = set(df.columns)
        missing_cols = required_cols - available_cols
        extra_cols = available_cols - required_cols

        if missing_cols:
            validation_report["errors"].append(
                f"Kolom yang hilang: {list(missing_cols)[:5]}{'...' if len(missing_cols) > 5 else ''}"
            )
            validation_report["compatible"] = False

        if extra_cols:
            validation_report["warnings"].append(
                f"Kolom tambahan yang tidak digunakan: {len(extra_cols)} kolom"
            )

        # 2. Periksa tipe data
        for col in self.normalized_features:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_report["errors"].append(
                        f"Kolom {col} bukan tipe numerik: {df[col].dtype}"
                    )
                    validation_report["compatible"] = False

        # 3. Statistik data
        validation_report["statistics"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "required_columns_present": len(required_cols & available_cols),
            "required_columns_total": len(required_cols),
        }

        return validation_report

    def print_validation_report(self, validation_report: dict):
        """
        Mencetak laporan validasi dengan format yang rapi.
        """
        print("\n" + "=" * 60)
        print("   LAPORAN VALIDASI DATA")
        print("=" * 60)

        status = (
            "✅ KOMPATIBEL"
            if validation_report["compatible"]
            else "❌ TIDAK KOMPATIBEL"
        )
        print(f"Status: {status}")

        stats = validation_report["statistics"]
        print(f"\nStatistik Data:")
        print(f"  - Total baris: {stats['total_rows']}")
        print(f"  - Total kolom: {stats['total_columns']}")
        print(f"  - Kolom numerik: {stats['numeric_columns']}")
        print(
            f"  - Kolom yang diperlukan: {stats['required_columns_present']}/{stats['required_columns_total']}"
        )
        print(f"  - Nilai yang hilang: {stats['missing_values']}")

        if validation_report["errors"]:
            print(f"\n❌ Error ({len(validation_report['errors'])}):")
            for error in validation_report["errors"]:
                print(f"  - {error}")

        if validation_report["warnings"]:
            print(f"\n⚠️  Peringatan ({len(validation_report['warnings'])}):")
            for warning in validation_report["warnings"]:
                print(f"  - {warning}")

        print("=" * 60 + "\n")


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
        title="Top 20 Feature Importance untuk Prediksi FCR",
    ):
        """Buat plot feature importance"""
        # Sort features by importance
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        # Ambil top 20 features
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
            "Top 20 Feature Importance untuk Prediksi FCR",
            fontsize=14,
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
        minmax_path = os.path.join(models_dir, "minmax_scaler.pkl")
        self.normalizer = DataNormalizer(info_path, minmax_path)

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
                min_history_days = max(7, self.sequence_length - 1)
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

            save_dir = "6. Pelatihan2/evaluation_plots"
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
                    "Squared_Error": (y_true - y_pred) ** 2,
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
        save_dir = "6. Pelatihan2/feature_analysis_plots"
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
        print(f"\nTop 5 Fitur Terpenting:")
        for i, (feature_name, importance) in enumerate(top_features[:5]):
            print(f"  {i+1}. {feature_name}: {importance:.1f}")
        print(f"{'='*50}\n")

        if save_plots:
            print(f"Plot disimpan di direktori: {save_dir}")


if __name__ == "__main__":
    # =======================================================================
    # SIMULASI PENGGUNAAN DENGAN ANALISIS FITUR DAN EVALUASI
    # =======================================================================

    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, "6. Pelatihan2", "models")
    CONFIG_DIR = os.path.join(BASE_DIR, "6. Pelatihan2", "Normalisasi")

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
            new_data = df_raw_sample[df_raw_sample["AGE"] == 20].copy()

            # Ambil data historis
            min_history_days = max(3, predictor.sequence_length - 1)
            historical_data = df_raw_sample[
                (df_raw_sample["AGE"] < 20)
                & (df_raw_sample["AGE"] >= max(1, 20 - min_history_days))
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
