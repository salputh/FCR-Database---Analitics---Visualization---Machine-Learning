    # PredictionScript.py
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
import tensorflow as tf
from tcn import TCN
import sys
sys.path.append('1. Normalisasi')
from Feature.FeatureEngineeringPipeline import FeatureEngineeringPipeline


# Matikan pesan log TensorFlow yang tidak relevan
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

class DataNormalizer:
    """
    Kelas untuk memuat model scaler dan menormalisasi data baru secara konsisten.
    """
    def __init__(self, info_path: str, minmax_scaler_path: str, robust_scaler_path: str):
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        with open(minmax_scaler_path, 'rb') as f:
            self.minmax_scaler = pickle.load(f)
        with open(robust_scaler_path, 'rb') as f:
            self.robust_scaler = pickle.load(f)
        
        self.minmax_features = self.info['minmax_features']
        self.robust_features = self.info['robust_features']
        print("DataNormalizer berhasil dimuat.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        print("  - Menerapkan normalisasi pada data...")
        if self.minmax_features:
            # Pastikan hanya kolom yang ada di df yang di-transform
            cols_to_transform = [col for col in self.minmax_features if col in df_transformed.columns]
            df_transformed[cols_to_transform] = self.minmax_scaler.transform(df_transformed[cols_to_transform])
        if self.robust_features:
            cols_to_transform = [col for col in self.robust_features if col in df_transformed.columns]
            df_transformed[cols_to_transform] = self.robust_scaler.transform(df_transformed[cols_to_transform])
        return df_transformed


class HybridPredictor:
    """
    Mengelola seluruh pipeline prediksi dari data mentah hingga hasil akhir.
    """
    def __init__(self, models_dir: str, config_dir: str):
        print("Memuat semua model dan pipeline untuk prediksi...")
        
        # 1. Muat pipeline rekayasa fitur
        fe_path = os.path.join(models_dir, 'feature_engineering_pipeline.pkl')
        with open(fe_path, 'rb') as f:
            self.feature_pipeline = pickle.load(f)
        
        # 2. Inisialisasi normalizer
        info_path = os.path.join(config_dir, 'normalization_info.json')
        minmax_path = os.path.join(models_dir, 'minmax_scaler.pkl')
        robust_path = os.path.join(models_dir, 'robust_scaler.pkl')
        self.normalizer = DataNormalizer(info_path, minmax_path, robust_path)

        # 3. Muat model TCN, XGBoost, dan config hibrida
        tcn_path = os.path.join(models_dir, 'tcn_encoder.keras')
        xgb_path = os.path.join(models_dir, 'xgboost_hybrid_model.pkl')
        config_path = os.path.join(models_dir, 'hybrid_model_config.json')
        
        self.tcn_encoder = tf.keras.models.load_model(
            tcn_path, 
            custom_objects={'TCN': TCN}
        )
        with open(xgb_path, 'rb') as f:
            self.xgb_model = pickle.load(f)
        with open(config_path, 'r') as f:
            self.hybrid_config = json.load(f)
        
        self.sequence_length = self.hybrid_config['sequence_length']
        print("Semua komponen prediksi berhasil dimuat.\n")

    def predict(self, new_day_data: pd.DataFrame, historical_data: pd.DataFrame):
        """
        Menjalankan seluruh pipeline untuk memprediksi FCR hari berikutnya.
        
        Args:
            new_day_data (pd.DataFrame): DataFrame berisi 1 baris data untuk hari ini.
            historical_data (pd.DataFrame): DataFrame berisi data historis (misal: 20 hari terakhir).
        """
        print("--- Memulai Proses Prediksi ---")
        
        # Langkah A: Rekayasa Fitur
        print("Langkah A: Menjalankan rekayasa fitur...")
        
        # PERBAIKAN: Gabungkan data terlebih dahulu, lalu transform
        # Pastikan data diurutkan berdasarkan waktu/urutan
        combined_data = pd.concat([historical_data, new_day_data], ignore_index=True)
        combined_data = combined_data.sort_values(['PERIODE', 'AGE']).reset_index(drop=True)
        
        # Transform seluruh data gabungan
        data_enhanced_full = self.feature_pipeline.transform(combined_data)
        
        # Ambil hanya baris terakhir (yang sesuai dengan new_day_data)
        data_enhanced = data_enhanced_full.tail(len(new_day_data))
        
        print(f"  - Rekayasa fitur selesai. Menghasilkan {data_enhanced.shape[1]} fitur.")
        
        # Langkah B: Normalisasi Fitur
        print("Langkah B: Menjalankan normalisasi...")
        
        # Normalisasi untuk data baru
        data_normalized = self.normalizer.transform(data_enhanced)
        
        # Normalisasi untuk data historis juga (diperlukan untuk sekuens TCN)
        historical_enhanced = self.feature_pipeline.transform(historical_data)
        historical_normalized = self.normalizer.transform(historical_enhanced)

        # Langkah C: Prediksi Hibrida
        print("Langkah C: Menjalankan prediksi model hibrida...")
        
        # C.1: Siapkan data sekuens untuk TCN
        # Gabungkan data historis yang sudah dinormalisasi dengan data baru yang sudah dinormalisasi
        full_normalized_history = pd.concat([historical_normalized, data_normalized], ignore_index=True)
        
        # Pastikan kita memiliki cukup data untuk sekuens
        if len(full_normalized_history) < self.sequence_length:
            raise ValueError(f"Data tidak cukup untuk sekuens. Diperlukan {self.sequence_length} hari, tersedia {len(full_normalized_history)} hari.")
        
        # Ambil sekuens terakhir
        last_sequence = full_normalized_history.tail(self.sequence_length)
        
        # Buang kolom non-fitur jika masih ada
        features_for_model = [col for col in last_sequence.columns if col not in ['PERIODE', 'FCR_ACT', 'TANGGAL']]
        last_sequence_values = last_sequence[features_for_model].values.reshape(1, self.sequence_length, -1)
        
        # C.2: Ekstrak Fitur Temporal dengan TCN
        print(f"  - Mengekstrak fitur temporal dari sekuens {self.sequence_length} hari...")
        temporal_features = self.tcn_encoder.predict(last_sequence_values, verbose=0)
        
        # C.3: Ambil Fitur Tabular dari hari terakhir
        tabular_features = data_normalized[features_for_model].values.reshape(1, -1)
        
        # C.4: Gabungkan fitur dan lakukan prediksi
        hybrid_features = np.concatenate([tabular_features, temporal_features], axis=1)
        print(f"  - Menggabungkan fitur tabular ({tabular_features.shape[1]}) & temporal ({temporal_features.shape[1]}) -> Total {hybrid_features.shape[1]} fitur.")
        
        final_prediction = self.xgb_model.predict(hybrid_features)
        
        # Karena model dilatih untuk memprediksi nilai FCR asli, tidak perlu de-normalisasi
        predicted_fcr = final_prediction[0]

        print("--- Prediksi Selesai ---")
        return predicted_fcr


if __name__ == '__main__':
    # =======================================================================
    # SIMULASI PENGGUNAAN DI APLIKASI UTAMA
    # =======================================================================
    
    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, '5. Model')
    CONFIG_DIR = os.path.join(BASE_DIR, '2. Hasil Normalisasi')

    try:
        # 1. Inisialisasi Predictor (lakukan sekali saat aplikasi/sistem dimulai)
        predictor = HybridPredictor(models_dir=MODELS_DIR, config_dir=CONFIG_DIR)

        # 2. Siapkan Data Input Baru (Simulasi)
        # Di aplikasi nyata, Anda akan mendapatkan data ini dari database atau input form
        print("Mensimulasikan data input baru...")
        # Kita ambil data dari salah satu periode sebagai contoh
        raw_data_path = os.path.join(BASE_DIR, '3. non-normalize', 'NORMALISASI_PERIODE_14.csv')
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"File data mentah untuk simulasi tidak ditemukan di {raw_data_path}")
            
        df_raw_sample = pd.read_csv(raw_data_path, sep=';')
        
        # Kita ingin memprediksi FCR untuk hari ke-25
        # Maka, kita butuh data hari ke-25 sebagai 'new_day_data'
        # dan data hari-hari sebelumnya sebagai 'historical_data'
        
        # Ambil data hari ke-8
        new_data = df_raw_sample[df_raw_sample['AGE'] == 8].copy()
        
        # Ambil data beberapa hari sebelum hari ke-8 sebagai histori
        # Pastikan kita punya cukup data untuk sequence_length
        min_history_days = max(7, predictor.sequence_length - 1)  # Minimal 7 hari atau sequence_length-1
        historical_data = df_raw_sample[
            (df_raw_sample['AGE'] < 8) & 
            (df_raw_sample['AGE'] >= max(1, 8 - min_history_days))
        ].copy()

        print(f"Data baru untuk diprediksi: Periode {new_data['PERIODE'].iloc[0]}, Hari ke-{new_data['AGE'].iloc[0]}")
        print(f"Data historis yang digunakan: {len(historical_data)} hari.")
        print(f"Sequence length yang diperlukan: {predictor.sequence_length}")
        
        if new_data.empty:
            print("\nTidak dapat menemukan data untuk hari ke-8 dalam file sampel.")
        elif len(historical_data) == 0:
            print("\nTidak cukup data historis untuk melakukan prediksi.")
        else:
            # 3. Lakukan Prediksi
            predicted_value = predictor.predict(
                new_day_data=new_data, 
                historical_data=historical_data
            )
        
            # 4. Tampilkan Hasil
            print(f"\n==============================================")
            print(f"   PREDIKSI FCR UNTUK HARI BERIKUTNYA   ")
            print(f"==============================================")
            print(f"   FCR Diprediksi: {predicted_value:.4f}")
            print(f"==============================================\n")

    except FileNotFoundError as e:
        print(f"\nERROR: Gagal memuat model atau file. {e}")
        print("Pastikan Anda telah menjalankan 'MainPipeline.py' dengan sukses terlebih dahulu.")
    except Exception as e:
        import traceback
        print(f"\nTerjadi kesalahan tak terduga: {e}")
        traceback.print_exc()