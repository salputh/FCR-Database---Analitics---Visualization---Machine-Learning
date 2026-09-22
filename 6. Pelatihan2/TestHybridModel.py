import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tcn import TCN
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Tambahkan path untuk import FeatureEngineeringPipeline
sys.path.append("6. Pelatihan2")
from Feature.FeatureEngineeringPipeline import FeatureEngineeringPipeline


# Definisi fungsi asymmetric_objective untuk XGBoost
def asymmetric_objective(y_true, y_pred):
    """
    Custom loss function untuk XGBoost yang memberikan penalti lebih besar
    pada over-prediction (prediksi lebih tinggi dari aktual).
    """
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2 * residual, -5.0 * residual)
    hess = np.where(residual > 0, 2, 5.0)
    return grad, hess


class DataNormalizer:
    """
    Kelas untuk menormalisasi data baru menggunakan informasi normalisasi
    yang telah disimpan dari proses training.
    """

    def __init__(self, normalization_info_path: str, minmax_scaler_path: str):
        # Load informasi normalisasi
        with open(normalization_info_path, "r") as f:
            self.norm_info = json.load(f)

        # Load MinMaxScaler
        with open(minmax_scaler_path, "rb") as f:
            self.minmax_scaler = pickle.load(f)

        # Load RobustScaler jika ada
        robust_scaler_path = minmax_scaler_path.replace(
            "minmax_scaler.pkl", "robust_scaler.pkl"
        )
        if os.path.exists(robust_scaler_path):
            with open(robust_scaler_path, "rb") as f:
                self.robust_scaler = pickle.load(f)
        else:
            self.robust_scaler = None

        # Extract informasi penting - sesuaikan dengan struktur file yang ada
        self.minmax_features = self.norm_info.get("minmax_features", [])
        self.robust_features = self.norm_info.get("robust_features", [])
        self.non_normalized_cols = self.norm_info.get("non_normalized", [])

        # Gabungkan semua fitur yang dinormalisasi
        self.normalized_features = self.minmax_features + self.robust_features
        self.original_features = self.normalized_features + self.non_normalized_cols

        # Untuk final_columns, gunakan dari file jika ada
        self.final_columns = self.norm_info.get("final_columns", self.original_features)

        print(f"DataNormalizer diinisialisasi:")
        print(f"  - MinMax features: {len(self.minmax_features)}")
        print(f"  - Robust features: {len(self.robust_features)}")
        print(f"  - Total normalized features: {len(self.normalized_features)}")
        print(f"  - Non-normalized features: {len(self.non_normalized_cols)}")
        print(f"  - Final columns: {len(self.final_columns)}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menormalisasi data baru menggunakan informasi dari training.
        """
        print(f"\n  Memulai normalisasi data (shape: {df.shape})...")
        df_transformed = df.copy()

        # 1. Validasi kolom yang diperlukan
        required_cols = set(self.original_features)
        available_cols = set(df.columns)
        missing_cols = required_cols - available_cols

        if missing_cols:
            print(f"    WARNING: Kolom yang hilang: {len(missing_cols)} kolom")
            # Tambahkan kolom yang hilang dengan nilai 0
            for col in missing_cols:
                if col in self.final_columns:
                    df_transformed[col] = 0
                    print(f"      Menambahkan kolom '{col}' dengan nilai 0")

        # 2. Konversi tipe data untuk kolom numerik
        for col in self.normalized_features:
            if col in df_transformed.columns:
                df_transformed[col] = pd.to_numeric(
                    df_transformed[col], errors="coerce"
                )

        # 3. Handle NaN values
        if df_transformed[self.normalized_features].isnull().any().any():
            nan_count = df_transformed[self.normalized_features].isnull().sum().sum()
            print(f"    Menangani {nan_count} nilai NaN...")
            df_transformed[self.normalized_features] = df_transformed[
                self.normalized_features
            ].fillna(0)

        # 4. Terapkan MinMaxScaler untuk fitur minmax
        if self.minmax_features:
            minmax_cols_available = [
                col for col in self.minmax_features if col in df_transformed.columns
            ]
            if minmax_cols_available:
                print(
                    f"    Menerapkan MinMaxScaler pada {len(minmax_cols_available)} fitur..."
                )
                try:
                    # Pastikan urutan kolom sesuai dengan yang dilatih
                    minmax_data = df_transformed[minmax_cols_available]
                    df_transformed[minmax_cols_available] = (
                        self.minmax_scaler.transform(minmax_data)
                    )
                except Exception as e:
                    print(f"    ERROR saat MinMaxScaler: {e}")
                    print(
                        f"    Expected features: {self.minmax_scaler.feature_names_in_ if hasattr(self.minmax_scaler, 'feature_names_in_') else 'N/A'}"
                    )
                    print(f"    Available features: {minmax_cols_available}")
                    raise

        # 5. Terapkan RobustScaler untuk fitur robust
        if self.robust_features and self.robust_scaler:
            robust_cols_available = [
                col for col in self.robust_features if col in df_transformed.columns
            ]
            if robust_cols_available:
                print(
                    f"    Menerapkan RobustScaler pada {len(robust_cols_available)} fitur..."
                )
                try:
                    robust_data = df_transformed[robust_cols_available]
                    df_transformed[robust_cols_available] = (
                        self.robust_scaler.transform(robust_data)
                    )
                except Exception as e:
                    print(f"    ERROR saat RobustScaler: {e}")
                    raise

        # 6. Seleksi fitur - pastikan hanya kolom yang ada di final_columns yang dipertahankan
        if self.final_columns:
            print(
                f"    Menerapkan seleksi fitur (target: {len(self.final_columns)} kolom)..."
            )

            # Pastikan semua kolom final_columns ada
            for col in self.final_columns:
                if col not in df_transformed.columns:
                    df_transformed[col] = 0

            # Pilih hanya kolom yang ada di final_columns
            df_transformed = df_transformed[self.final_columns]

        print(f"    Normalisasi selesai. Shape akhir: {df_transformed.shape}")
        return df_transformed


class EvaluationMetrics:
    """
    Kelas untuk menghitung dan memvisualisasikan metrik evaluasi model.
    """

    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, y_true, y_pred, horizon=None):
        """
        Menghitung berbagai metrik evaluasi.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}

        if horizon:
            self.metrics[f"horizon_{horizon}d"] = metrics

        return metrics

    def create_evaluation_plots(self, y_true, y_pred, horizon=None, save_dir=None):
        """
        Membuat plot evaluasi komprehensif.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color="blue")
        axes[0, 0].plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        axes[0, 0].set_xlabel("Actual FCR")
        axes[0, 0].set_ylabel("Predicted FCR")
        axes[0, 0].set_title(
            f"Predicted vs Actual FCR (Horizon +{horizon} hari)"
            if horizon
            else "Predicted vs Actual FCR"
        )
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residual plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color="green")
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted FCR")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residual Plot")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution of residuals
        axes[1, 0].hist(
            residuals, bins=20, alpha=0.7, color="orange", edgecolor="black"
        )
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Residuals")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Time series plot (if applicable)
        axes[1, 1].plot(y_true, label="Actual", marker="o", markersize=4)
        axes[1, 1].plot(y_pred, label="Predicted", marker="s", markersize=4)
        axes[1, 1].set_xlabel("Sample Index")
        axes[1, 1].set_ylabel("FCR Value")
        axes[1, 1].set_title("Time Series Comparison")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            filename = (
                f"evaluation_plots_horizon_{horizon}d.png"
                if horizon
                else "evaluation_plots.png"
            )
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot evaluasi disimpan: {save_path}")

        plt.show()


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

        # 4. Inisialisasi evaluator
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
        max_rows=35,
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

        # Evaluasi untuk horizon yang ditentukan atau semua horizon
        horizons_to_evaluate = [horizon] if horizon is not None else self.horizons

        results = {}

        for h in horizons_to_evaluate:
            print(f"\nEvaluasi untuk horizon +{h} hari...")

            predictions = []
            actual_values = []
            periode_values = []
            age_values = []
            valid_predictions = 0

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

                    # Cari data aktual untuk horizon ini
                    target_age = age + h
                    target_sample = df_test[
                        (df_test["PERIODE"] == periode) & (df_test["AGE"] == target_age)
                    ]
                    if target_sample.empty or "FCR_ACT" not in target_sample.columns:
                        continue
                    actual_fcr = target_sample["FCR_ACT"].iloc[0]

                    # Make prediction
                    predicted_fcr = self.predict_single_sample(
                        new_day_data=current_sample,
                        historical_data=historical_data,
                        horizon=h,
                    )

                    predictions.append(predicted_fcr)
                    actual_values.append(actual_fcr)
                    periode_values.append(periode)
                    age_values.append(age)
                    valid_predictions += 1

                    if valid_predictions % 5 == 0:
                        print(f"  Processed {valid_predictions} valid predictions...")

                except Exception as e:
                    print(f"  Error processing sample {periode}-{age}: {e}")
                    continue

            if len(predictions) == 0:
                print(f"Tidak ada prediksi valid untuk horizon +{h} hari")
                continue

            # Convert to numpy arrays
            predictions = np.array(predictions)
            actual_values = np.array(actual_values)

            print(f"\nHasil evaluasi untuk horizon +{h} hari:")
            print(f"Total prediksi valid: {len(predictions)}")

            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(
                actual_values, predictions, horizon=h
            )

            print(f"Metrik Evaluasi:")
            print(f"  MAE: {metrics['MAE']:.6f}")
            print(f"  MSE: {metrics['MSE']:.6f}")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  R²: {metrics['R2']:.6f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")

            # Create visualizations
            if create_visualizations:
                save_dir = "evaluation_plots" if save_plots else None
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                self.evaluator.create_evaluation_plots(
                    actual_values, predictions, horizon=h, save_dir=save_dir
                )

            # Store results
            results[f"horizon_{h}d"] = {
                "predictions": predictions,
                "actual_values": actual_values,
                "metrics": metrics,
                "periode_values": periode_values,
                "age_values": age_values,
            }

        return results


if __name__ == "__main__":
    # =======================================================================
    # PENGUJIAN MODEL HIBRIDA TCN-XGBOOST UNTUK PREDIKSI FCR
    # =======================================================================

    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, "6. Pelatihan2", "models")
    CONFIG_DIR = os.path.join(BASE_DIR, "6. Pelatihan2", "Normalisasi")

    try:
        # 1. Inisialisasi Predictor
        print("Inisialisasi HybridPredictor...")
        predictor = HybridPredictor(models_dir=MODELS_DIR, config_dir=CONFIG_DIR)

        # 2. Path ke data test (data mentah)
        test_data_path = os.path.join(
            BASE_DIR, "3. non-normalize", "NORMALISASI_PERIODE_14.csv"
        )

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"File data test tidak ditemukan di {test_data_path}"
            )

        print(f"\nMenggunakan data test: {test_data_path}")

        # 3. Pilih mode evaluasi
        print("\nPilih mode evaluasi:")
        print("1. Evaluasi horizon +1 hari")
        print("2. Evaluasi horizon +3 hari")
        print("3. Evaluasi kedua horizon")

        try:
            choice = int(input("Pilihan Anda (1/2/3): "))
        except:
            choice = 3  # Default ke evaluasi kedua horizon

        if choice == 1:
            horizon_to_evaluate = 1
        elif choice == 2:
            horizon_to_evaluate = 3
        else:
            horizon_to_evaluate = None

        # 4. Evaluasi model
        print("\n" + "=" * 60)
        print("MEMULAI EVALUASI MODEL HIBRIDA TCN-XGBOOST")
        print("=" * 60)
        print(
            "Proses: Data Mentah → Ekstraksi Fitur → Clipping → Normalisasi → Prediksi"
        )
        print("=" * 60)

        # Batasi data untuk evaluasi
        max_rows = 35
        print(f"Menggunakan maksimal {max_rows} baris data untuk evaluasi")

        # Evaluasi pada data test
        evaluation_results = predictor.evaluate_test_data(
            test_data_path=test_data_path,
            create_visualizations=True,
            save_plots=True,
            horizon=horizon_to_evaluate,
            max_rows=max_rows,
        )

        # 5. Tampilkan ringkasan hasil
        print("\n" + "=" * 60)
        print("RINGKASAN HASIL EVALUASI")
        print("=" * 60)

        for horizon_key, results in evaluation_results.items():
            horizon_num = horizon_key.split("_")[1].replace("d", "")
            metrics = results["metrics"]
            n_predictions = len(results["predictions"])

            print(f"\nHorizon +{horizon_num} hari ({n_predictions} prediksi valid):")
            print(f"  MAE:  {metrics['MAE']:.6f}")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  R²:   {metrics['R2']:.6f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")

        print("\n" + "=" * 60)
        print("EVALUASI SELESAI")
        print("=" * 60)
        print("Plot evaluasi telah disimpan di direktori 'evaluation_plots'")

    except FileNotFoundError as e:
        print(f"\nERROR: Gagal memuat model atau file. {e}")
        print(
            "Pastikan Anda telah menjalankan 'FullTrainPipeline.py' dengan sukses terlebih dahulu."
        )
    except Exception as e:
        import traceback

        print(f"\nTerjadi kesalahan tak terduga: {e}")
        traceback.print_exc()
