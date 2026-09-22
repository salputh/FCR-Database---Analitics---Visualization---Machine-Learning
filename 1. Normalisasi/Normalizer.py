# Normalizer.py
import pandas as pd
import pickle
import json
import os

class DataNormalizer:
    """
    Kelas untuk memuat model scaler yang telah dilatih dan menormalisasi data baru
    sesuai dengan fitur yang konsisten.
    """
    def __init__(self, info_path: str, minmax_scaler_path: str, robust_scaler_path: str):
        """
        Inisialisasi dengan memuat semua file yang diperlukan.
        
        Args:
            info_path (str): Path ke file 'normalization_info.json'.
            minmax_scaler_path (str): Path ke file 'minmax_scaler.pkl'.
            robust_scaler_path (str): Path ke file 'robust_scaler.pkl'.
        """
        if not all(os.path.exists(p) for p in [info_path, minmax_scaler_path, robust_scaler_path]):
            raise FileNotFoundError("Satu atau lebih file (info.json, .pkl) tidak ditemukan. Jalankan MainPipeline.py terlebih dahulu.")

        # Muat peta fitur dari JSON
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        # Muat model scaler dari file pickle
        with open(minmax_scaler_path, 'rb') as f:
            self.minmax_scaler = pickle.load(f)
        
        with open(robust_scaler_path, 'rb') as f:
            self.robust_scaler = pickle.load(f)
            
        # Ekstrak daftar fitur dari info
        self.minmax_features = self.info['minmax_features']
        self.robust_features = self.info['robust_features']
        self.final_columns_order = self.info['final_columns_order']
        print("DataNormalizer berhasil dimuat dan siap digunakan.")

    def transform(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Menerapkan transformasi normalisasi pada data baru.
        
        Args:
            new_data (pd.DataFrame): DataFrame baru yang akan dinormalisasi.
                                     Harus berisi semua kolom fitur yang diperlukan.
        
        Returns:
            pd.DataFrame: DataFrame yang telah dinormalisasi.
        """
        df_transformed = new_data.copy()
        
        # Periksa apakah semua fitur yang diperlukan ada
        required_features = self.minmax_features + self.robust_features
        missing_features = set(required_features) - set(df_transformed.columns)
        if missing_features:
            raise ValueError(f"Fitur berikut hilang dari data baru: {missing_features}")

        print("Menerapkan transformasi scaler...")
        
        # Terapkan transformasi HANYA dengan .transform() (bukan .fit_transform())
        if self.minmax_features:
            df_transformed[self.minmax_features] = self.minmax_scaler.transform(df_transformed[self.minmax_features])
        
        if self.robust_features:
            df_transformed[self.robust_features] = self.robust_scaler.transform(df_transformed[self.robust_features])
        
        print("Transformasi selesai.")
        
        # Pastikan urutan kolom sama seperti saat training
        return df_transformed[self.final_columns_order]

# Contoh Penggunaan 
if __name__ == '__main__':
    # Anggaplah kita punya data baru dalam bentuk DataFrame
    # Strukturnya harus sama dengan data sebelum normalisasi
    contoh_data_baru = pd.DataFrame({
        # Isi dengan beberapa contoh data, pastikan semua kolom yang diperlukan ada
        # Contoh sederhana:
        'AGE': [10, 11],
        'ABW_ACT': [250.5, 280.0],
        'DELTA_FCR': [0.05, -0.02],
        'PAKAN_ACT_GR/EK': [50.1, 52.3],
        'FCR_ACT_lag_1': [1.2, 1.25],
        # ... dan seterusnya untuk semua fitur lain yang ada di minmax_features & robust_features
        # Di aplikasi nyata, DataFrame ini akan memiliki semua kolom yang diperlukan.
    })
    
    # Untuk menjalankan contoh ini, Anda harus memiliki SEMUA kolom yang diperlukan.
    # Karena kita tidak punya semua kolom di contoh sederhana ini, kita akan simulasikan.
    print("Contoh ini hanya untuk demonstrasi alur kerja.")
    print("Di aplikasi nyata, Anda akan memuat DataNormalizer dan langsung memanggil .transform()")

    # Alur kerja di sistem utama Anda akan seperti ini:
    try:
        # 1. Tentukan path ke model dan info
        base_path = os.getcwd()
        info_path = os.path.join(base_path, 'Hasil Normalisasi', 'normalization_info.json')
        minmax_scaler_path = os.path.join(base_path, 'models', 'minmax_scaler.pkl')
        robust_scaler_path = os.path.join(base_path, 'models', 'robust_scaler.pkl')

        # 2. Inisialisasi normalizer (cukup lakukan sekali saat aplikasi dimulai)
        normalizer = DataNormalizer(
            info_path=info_path,
            minmax_scaler_path=minmax_scaler_path,
            robust_scaler_path=robust_scaler_path
        )

        # 3. Saat ada data baru masuk (misalnya dari input pengguna atau sensor)
        #    buat DataFrame dari data tersebut. Pastikan DataFrame ini sudah melalui
        #    proses rekayasa fitur yang sama untuk menghasilkan kolom-kolom yang diperlukan.
        #    (Untuk sekarang, kita lewati karena contoh_data_baru tidak lengkap)
        
        # data_baru_setelah_rekayasa_fitur = ...
        # data_ternormalisasi = normalizer.transform(data_baru_setelah_rekayasa_fitur)
        # print("\nContoh data setelah normalisasi:")
        # print(data_ternormalisasi.head())
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Pastikan Anda telah menjalankan 'MainPipeline.py' untuk menghasilkan file .pkl dan .json.")
    except ValueError as e:
        print(f"\nError saat transformasi: {e}")