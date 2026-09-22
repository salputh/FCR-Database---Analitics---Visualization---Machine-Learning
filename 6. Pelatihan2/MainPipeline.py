# -*- coding: utf-8 -*-
"""
Skrip pipeline lengkap untuk:
1. Menggabungkan data dari beberapa file CSV periode.
2. Melakukan rekayasa fitur (feature engineering) canggih pada data yang digabungkan.
3. Melakukan normalisasi fitur dengan metode campuran (MinMax dan Robust Scaler).

Alur Kerja:
- Langkah 1: Penggabungan data mentah.
- Langkah 2: Rekayasa fitur dari data gabungan.
- Langkah 3: Normalisasi fitur yang telah direkayasa.
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
import sys
sys.path.append('1. Normalisasi')
from Feature.FeatureEngineeringPipeline import FeatureEngineeringPipeline

# =============================================================================
# KONFIGURASI DAN SETUP AWAL
# =============================================================================

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mengabaikan peringatan (warnings) agar output lebih bersih
warnings.filterwarnings("ignore")

# Konfigurasi untuk rekayasa fitur
FEATURE_CONFIG = {
    "lag_periods": [1, 2, 3, 7, 14],
    "rolling_windows": [3, 7, 14],
    "health_weights": {
        "mati": 0.5,
        "afkir": 0.3,
        "deplesi": 0.2
    },
    "variance_threshold": 0.01,
    "correlation_threshold": 0.95,
    "decimal_places": 3
}


# =============================================================================
# LANGKAH 1: FUNGSI PENGGABUNGAN DATA
# =============================================================================

def merge_data_files(folder_path: str, output_file: str):
    """
    Membaca semua file CSV dari folder, menggabungkannya, dan menyimpannya.
    """
    print("="*80)
    print("MEMULAI LANGKAH 1: PENGGABUNGAN DATA")
    print("="*80)
    
    dataframes = []
    for i in range(1, 14):
        file_name = f'NORMALISASI_PERIODE_{i}.csv'
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            print(f"Membaca file: {file_name}")
            try:
                df = pd.read_csv(file_path, sep=';')
                if 'PERIODE' not in df.columns:
                    df['PERIODE'] = i
                else:
                    df['PERIODE'] = df['PERIODE'].fillna(i)
                df['PERIODE'] = df['PERIODE'].astype(int)
                dataframes.append(df)
            except Exception as e:
                print(f"  - ERROR membaca {file_name}: {str(e)}\n")
        else:
            print(f"File tidak ditemukan: {file_name}")

    if not dataframes:
        logger.error("GAGAL: Tidak ada file yang berhasil dibaca. Pipeline dihentikan.")
        return False

    combined_df = pd.concat(dataframes, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, sep=';', index=False)
    print(f"\nBERHASIL: Penggabungan data selesai. File disimpan sebagai: {output_file}\n")
    return True


# =============================================================================
# LANGKAH 2: FUNGSI REKAYASA FITUR (IMPLEMENTASI BARU)
# =============================================================================
def run_feature_engineering_step(input_file: str, output_data_path: str, output_pipeline_path: str, metadata_path: str, config: Dict) -> bool:
    """
    Menggunakan kelas FeatureEngineeringPipeline untuk memproses data dan menyimpan pipeline.
    """
    print("\n" + "="*80 + "\nMEMULAI LANGKAH 2: REKAYASA FITUR & PENYIMPANAN PIPELINE\n" + "="*80)
    if not os.path.exists(input_file):
        logger.critical(f"GAGAL: File input '{input_file}' tidak ditemukan.")
        return False
    
    df_raw = pd.read_csv(input_file, sep=';')
    logger.info(f"Data mentah dimuat untuk rekayasa fitur: {df_raw.shape}")

    # 1. Inisialisasi pipeline rekayasa fitur dengan konfigurasi
    feature_pipeline = FeatureEngineeringPipeline(config=config)

    # 2. Lakukan 'fit_transform' pada data mentah. Ini akan:
    #    a. "Melatih" pipeline (mempelajari kolom mana yang akan dihapus).
    #    b. Mentransformasi data menjadi data dengan fitur-fitur baru.
    #    c. Menyimpan metadata proses ke file JSON.
    logger.info("Menjalankan fit_transform pada FeatureEngineeringPipeline...")
    df_enhanced = feature_pipeline.fit_transform(df_raw, metadata_output_dir=metadata_path)

    # 3. Simpan data yang sudah direkayasa fiturnya (hasil dari pipeline)
    df_enhanced.to_csv(output_data_path, sep=';', index=False)
    logger.info(f"Data dengan fitur enhanced berhasil disimpan di: {output_data_path}")
    
    # 4. Simpan objek pipeline yang sudah di-'fit' ke file .pkl
    with open(output_pipeline_path, 'wb') as f:
        pickle.dump(feature_pipeline, f)
    logger.info(f"Model rekayasa fitur (pipeline) disimpan di: {output_pipeline_path}")
    
    return True

# =============================================================================
# LANGKAH 3: FUNGSI NORMALISASI FITUR
# =============================================================================

def normalize_features(
    input_file: str, 
    output_file: str, 
    info_file: str,
    minmax_scaler_path: str,
    robust_scaler_path: str
    ) -> bool:
    """
    Menerapkan strategi normalisasi metode campuran pada data.
    """
    print("\n" + "="*80)
    print("MEMULAI LANGKAH 3: NORMALISASI FITUR")
    print("="*80)

    if not os.path.exists(input_file):
        logger.error(f"File input '{input_file}' tidak ditemukan. Langkah 3 dihentikan.")
        return False
        
    df = pd.read_csv(input_file, delimiter=';')
    logger.info(f"Data untuk normalisasi dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    # Kolom yang tidak dinormalisasi (termasuk kolom target)
    non_normalized_cols = ['TANGGAL', 'PERIODE', 'FCR_ACT']

     # 1. Identifikasi semua kolom numerik yang tersedia
    all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

     # 2. Tentukan fitur mana yang akan dinormalisasi
    features_to_normalize = [col for col in all_numeric_cols if col not in non_normalized_cols]
    
    # Definisi fitur untuk setiap scaler
    minmax_features = []
    
    robust_features = []
    
     # Aturan untuk RobustScaler (jika nama mengandung keyword ini)
    robust_keywords = ['DELTA_', '_std_', 'momentum', 'acceleration', 'volatility']
    
    for feature in features_to_normalize:
        if any(keyword in feature for keyword in robust_keywords):
            robust_features.append(feature)
        else:
            minmax_features.append(feature)

    logger.info(f"Ditemukan {len(features_to_normalize)} fitur untuk dinormalisasi.")
    logger.info(f"Dialokasikan {len(minmax_features)} fitur ke MinMaxScaler.")
    logger.info(f"Dialokasikan {len(robust_features)} fitur ke RobustScaler.")

    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()

    df_normalized = df.copy()

    # 4. Terapkan normalisasi
    if minmax_features:
        minmax_data = df[minmax_features].fillna(df[minmax_features].median())
        df_normalized[minmax_features] = minmax_scaler.fit_transform(minmax_data)

        # Simpan model MinMaxScaler
        with open(minmax_scaler_path, 'wb') as f:
            pickle.dump(minmax_scaler, f)
        logger.info(f"Model MinMaxScaler disimpan ke: {minmax_scaler_path}")
    
    if robust_features:
        robust_data = df[robust_features].fillna(df[robust_features].median())
        df_normalized[robust_features] = robust_scaler.fit_transform(robust_data)

        # Simpan model RobustScaler
        with open(robust_scaler_path, 'wb') as f:
            pickle.dump(robust_scaler, f)
        logger.info(f"Model RobustScaler disimpan ke: {robust_scaler_path}")


    # Simpan hasil
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_normalized.to_csv(output_file, index=False, sep=';')
    
    # Simpan informasi normalisasi
    normalization_info = {
        'minmax_features': minmax_features,
        'robust_features': robust_features,
        'non_normalized': [col for col in non_normalized_cols if col in df_normalized.columns],
        'minmax_scaler_params': {'min_': minmax_scaler.min_.tolist(), 'scale_': minmax_scaler.scale_.tolist(), 'feature_names': minmax_features},
        'robust_scaler_params': {'center_': robust_scaler.center_.tolist(), 'scale_': robust_scaler.scale_.tolist(), 'feature_names': robust_features},
        'final_columns': df_normalized.columns.tolist()
    }
    with open(info_file, 'w') as f:
        json.dump(normalization_info, f, indent=2)
        
    print(f"\nBERHASIL: Normalisasi fitur selesai.")
    print(f" - Fitur dinormalisasi dengan MinMax Scaler: {len(minmax_features)}")
    print(f" - Fitur dinormalisasi dengan Robust Scaler: {len(robust_features)}")
    print(f" - File data ternormalisasi disimpan di: {output_file}")
    print(f" - Info normalisasi disimpan di: {info_file}")
    return True

# =============================================================================
# BLOK EKSEKUSI UTAMA
# =============================================================================
if __name__ == "__main__":
    base_folder_path = os.getcwd()
    
    # Definisi path
    data_input_folder = os.path.join(base_folder_path, '3. non-normalize')
    intermediate_data_folder = os.path.join(base_folder_path, '1. Normalisasi', 'Data')
    final_output_folder = os.path.join(base_folder_path, '2. Hasil Normalisasi')
    models_folder = os.path.join(base_folder_path, '5. Model')

    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(intermediate_data_folder, exist_ok=True)
    os.makedirs(final_output_folder, exist_ok=True)

    # Nama file untuk setiap tahap
    merged_file = os.path.join(intermediate_data_folder, 'File_Gabungan.csv')
    enhanced_file = os.path.join(intermediate_data_folder, 'File_Gabungan_Enhanced_Features.csv')
    normalized_file = os.path.join(final_output_folder, 'Data_Gabungan_Fitur_Rekayasa_Normalized.csv')
    
    # Path untuk menyimpan semua model (.pkl) dan info (.json)
    fe_pipeline_file = os.path.join(models_folder, 'feature_engineering_pipeline.pkl')
    minmax_scaler_file = os.path.join(models_folder, 'minmax_scaler.pkl')
    robust_scaler_file = os.path.join(models_folder, 'robust_scaler.pkl')
    feature_metadata_file = os.path.join(final_output_folder, 'feature_metadata.json')
    norm_info_file = os.path.join(final_output_folder, 'normalization_info.json')

    try:
        # Menjalankan pipeline secara berurutan
        # Langkah 1: Gabungkan data mentah
        if not merge_data_files(folder_path=data_input_folder, output_file=merged_file):
            raise Exception("Gagal pada langkah penggabungan data.")

        # Langkah 2: Lakukan rekayasa fitur dan simpan model pipeline-nya
        if not run_feature_engineering_step(
            input_file=merged_file, 
            output_data_path=enhanced_file, 
            output_pipeline_path=fe_pipeline_file,
            metadata_path=final_output_folder,
            config=FEATURE_CONFIG
        ):
            raise Exception("Gagal pada langkah rekayasa fitur.")

        # Langkah 3: Lakukan normalisasi dan simpan model scaler-nya
        if not normalize_features(
            input_file=enhanced_file, 
            output_file=normalized_file, 
            info_file=norm_info_file, 
            minmax_scaler_path=minmax_scaler_file, 
            robust_scaler_path=robust_scaler_file
        ):
            raise Exception("Gagal pada langkah normalisasi.")
            
        print("\n\n>>> SEMUA LANGKAH PIPELINE SELESAI DENGAN SUKSES! <<<")
        print("Data siap latih, model rekayasa fitur, dan model scaler telah dibuat.")
        print(f"Folder 'models' kini berisi: {os.listdir(models_folder)}")

    except Exception as e:
        logger.critical(f"Terjadi kesalahan fatal pada pipeline: {e}", exc_info=True)