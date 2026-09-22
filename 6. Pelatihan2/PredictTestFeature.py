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


class DataNormalizer:
    """
    Kelas untuk memuat model scaler dan menormalisasi data baru secara konsisten.
    """

    def __init__(
        self, info_path: str, minmax_scaler_path: str, robust_scaler_path: str
    ):
        with open(info_path, "r") as f:
            self.info = json.load(f)
        with open(minmax_scaler_path, "rb") as f:
            self.minmax_scaler = pickle.load(f)
        with open(robust_scaler_path, "rb") as f:
            self.robust_scaler = pickle.load(f)

        self.minmax_features = self.info["minmax_features"]
        self.robust_features = self.info["robust_features"]
        print("DataNormalizer berhasil dimuat.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        print("  - Menerapkan normalisasi pada data...")
        if self.minmax_features:
            cols_to_transform = [
                col for col in self.minmax_features if col in df_transformed.columns
            ]
            df_transformed[cols_to_transform] = self.minmax_scaler.transform(
                df_transformed[cols_to_transform]
            )
        if self.robust_features:
            cols_to_transform = [
                col for col in self.robust_features if col in df_transformed.columns
            ]
            df_transformed[cols_to_transform] = self.robust_scaler.transform(
                df_transformed[cols_to_transform]
            )
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

    def create_feature_importance_plot(self, feature_importance_dict, save_path=None):
        """Buat plot feature importance"""
        # Sort features by importance
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        # Ambil top 20 features
        top_features = sorted_features[:20]
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
    """

    def __init__(self, models_dir: str, config_dir: str):
        print("Memuat semua model dan pipeline untuk prediksi...")

        # 1. Muat pipeline rekayasa fitur
        fe_path = os.path.join(models_dir, "feature_engineering_pipeline.pkl")
        with open(fe_path, "rb") as f:
            self.feature_pipeline = pickle.load(f)

        # 2. Inisialisasi normalizer
        info_path = os.path.join(config_dir, "normalization_info.json")
        minmax_path = os.path.join(models_dir, "minmax_scaler.pkl")
        robust_path = os.path.join(models_dir, "robust_scaler.pkl")
        self.normalizer = DataNormalizer(info_path, minmax_path, robust_path)

        # 3. Muat model TCN, XGBoost, dan config hibrida
        tcn_path = os.path.join(models_dir, "tcn_encoder.keras")
        xgb_path = os.path.join(models_dir, "xgboost_hybrid_model.pkl")
        config_path = os.path.join(models_dir, "hybrid_model_config.json")

        self.tcn_encoder = tf.keras.models.load_model(
            tcn_path, custom_objects={"TCN": TCN}
        )
        with open(xgb_path, "rb") as f:
            self.xgb_model = pickle.load(f)
        with open(config_path, "r") as f:
            self.hybrid_config = json.load(f)

        self.sequence_length = self.hybrid_config["sequence_length"]

        # 4. Inisialisasi analyzer
        self.analyzer = FeatureAnalyzer()

        print("Semua komponen prediksi berhasil dimuat.\n")

    def predict_with_analysis(
        self,
        new_day_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        create_visualizations=True,
        save_plots=True,
    ):
        """
        Menjalankan prediksi dengan analisis fitur lengkap dan visualisasi.
        """
        print("--- Memulai Proses Prediksi dengan Analisis ---")

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

        # C.2: Ekstrak Fitur Temporal dengan TCN
        print(
            f"  - Mengekstrak fitur temporal dari sekuens {self.sequence_length} hari..."
        )
        temporal_features = self.tcn_encoder.predict(last_sequence_values, verbose=0)

        # C.3: Ambil Fitur Tabular dari hari terakhir
        tabular_features = data_normalized[features_for_model].values.reshape(1, -1)

        # C.4: Gabungkan fitur dan lakukan prediksi
        hybrid_features = np.concatenate([tabular_features, temporal_features], axis=1)
        print(
            f"  - Menggabungkan fitur tabular ({tabular_features.shape[1]}) & temporal ({temporal_features.shape[1]}) -> Total {hybrid_features.shape[1]} fitur."
        )

        final_prediction = self.xgb_model.predict(hybrid_features)
        predicted_fcr = final_prediction[0]

        # Langkah D: Analisis Fitur
        if create_visualizations:
            print("Langkah D: Menganalisis feature importance...")
            self._analyze_features(
                hybrid_features,
                tabular_features.shape[1],
                features_for_model,
                temporal_features.shape[1],
                save_plots,
            )

        print("--- Prediksi Selesai ---")
        return predicted_fcr

    def _analyze_features(
        self,
        hybrid_features,
        n_tabular_features,
        base_feature_names,
        n_temporal_features,
        save_plots=True,
    ):
        """Analisis feature importance dengan visualisasi"""

        # Buat nama fitur untuk hybrid features
        tabular_feature_names = base_feature_names[:n_tabular_features]
        temporal_feature_names = [
            f"temporal_feature_{i+1}" for i in range(n_temporal_features)
        ]
        all_feature_names = tabular_feature_names + temporal_feature_names

        # Extract feature importance
        feature_importance_dict = self.analyzer.extract_feature_importance(
            self.xgb_model, all_feature_names
        )

        # Set up save directory
        save_dir = "feature_analysis_plots"
        if save_plots:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Feature Importance Plot
        print("  - Membuat plot feature importance...")
        save_path = (
            os.path.join(save_dir, f"feature_importance_{timestamp}.png")
            if save_plots
            else None
        )
        top_features = self.analyzer.create_feature_importance_plot(
            feature_importance_dict, save_path
        )

        # 2. Feature Distribution Plot
        print("  - Membuat plot distribusi fitur...")
        save_path = (
            os.path.join(save_dir, f"feature_distribution_{timestamp}.png")
            if save_plots
            else None
        )
        self.analyzer.create_feature_distribution_plot(
            [hybrid_features[0]], all_feature_names, save_path
        )

        # 3. Temporal vs Tabular Comparison
        print("  - Membuat perbandingan fitur temporal vs tabular...")
        save_path = (
            os.path.join(save_dir, f"temporal_vs_tabular_{timestamp}.png")
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
        print(f"RINGKASAN ANALISIS FITUR")
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
    # SIMULASI PENGGUNAAN DENGAN ANALISIS FITUR
    # =======================================================================

    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, "5. Model", "models_terbaru")
    CONFIG_DIR = os.path.join(BASE_DIR, "2. Hasil Normalisasi", "Terbaru")

    try:
        # 1. Inisialisasi Predictor
        predictor = HybridPredictor(models_dir=MODELS_DIR, config_dir=CONFIG_DIR)

        # 2. Siapkan Data Input
        print("Mensimulasikan data input baru...")
        raw_data_path = os.path.join(
            BASE_DIR, "3. non-normalize", "NORMALISASI_PERIODE_14.csv"
        )
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(
                f"File data mentah untuk simulasi tidak ditemukan di {raw_data_path}"
            )

        df_raw_sample = pd.read_csv(raw_data_path, sep=";")

        # Ambil data hari ke-n
        new_data = df_raw_sample[df_raw_sample["AGE"] == 21].copy()

        # Ambil data historis
        min_history_days = max(7, predictor.sequence_length - 1)
        historical_data = df_raw_sample[
            (df_raw_sample["AGE"] < 21)
            & (df_raw_sample["AGE"] >= max(1, 20 - min_history_days))
        ].copy()

        print(
            f"Data baru untuk diprediksi: Periode {new_data['PERIODE'].iloc[0]}, Hari ke-{new_data['AGE'].iloc[0]}"
        )
        print(f"Data historis yang digunakan: {len(historical_data)} hari.")
        print(f"Sequence length yang diperlukan: {predictor.sequence_length}")

        if new_data.empty:
            print("\nTidak dapat menemukan data untuk hari ke-8 dalam file sampel.")
        elif len(historical_data) == 0:
            print("\nTidak cukup data historis untuk melakukan prediksi.")
        else:
            # 3. Lakukan Prediksi dengan Analisis
            predicted_value = predictor.predict_with_analysis(
                new_day_data=new_data,
                historical_data=historical_data,
                create_visualizations=True,
                save_plots=True,
            )

            # 4. Tampilkan Hasil
            print(f"\n{'='*50}")
            print(f"   HASIL PREDIKSI FCR   ")
            print(f"{'='*50}")
            print(f"   FCR Diprediksi: {predicted_value:.4f}")
            print(f"{'='*50}\n")

    except FileNotFoundError as e:
        print(f"\nERROR: Gagal memuat model atau file. {e}")
        print(
            "Pastikan Anda telah menjalankan 'MainPipeline.py' dengan sukses terlebih dahulu."
        )
    except Exception as e:
        import traceback

        print(f"\nTerjadi kesalahan tak terduga: {e}")
        traceback.print_exc()
