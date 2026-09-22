# -*- coding: utf-8 -*-
"""
Skrip pipeline lengkap untuk:
1. Menggabungkan data dari beberapa file CSV periode.
2. Melakukan rekayasa fitur (feature engineering) canggih pada data yang digabungkan.
3. Melakukan normalisasi fitur dengan metode campuran (MinMax dan Robust Scaler).
4. Melatih model hibrida yang terdiri dari TCN dan XGBoost.

Alur Kerja:
- Langkah 1: Penggabungan data mentah.
- Langkah 2: Rekayasa fitur dari data gabungan.
- Langkah 3: Normalisasi fitur yang telah direkayasa.
- Langkah 4: Pelatihan model hibrida.

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
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error impor: {e}")
    print("Harap pasang library yang diperlukan: pip install tensorflow keras-tcn xgboost optuna scikit-learn")
    exit()

# Impor kelas pipeline dari file terpisah.
try:
    import sys
    sys.path.append('Normalisasi')
    from Feature.FeatureEngineeringPipeline import FeatureEngineeringPipeline
except ImportError:
    print("ERROR: File 'FeatureEngineeringPipeline.py' tidak ditemukan.")
    exit()


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
# LANGKAH 4: PELATIHAN MODEL HIBRIDA TCN-XGBOOST (IMPLEMENTASI BARU)
# =============================================================================
def train_tcn_xgboost_hybrid(
    data_file: str, 
    tcn_encoder_path: str,
    xgb_model_path: str, 
    hybrid_config_path: str
) -> bool:
    """
    Melatih seluruh pipeline model hibrida TCN-XGBoost.
    """
    print("\n" + "="*80)
    print("MEMULAI LANGKAH 4: PELATIHAN MODEL HIBRIDA TCN-XGBOOST")
    print("="*80)

    # --- Langkah 1: Persiapan Data (Dengan Sedikit Modifikasi) ---
    logger.info("Langkah 4.1: Memuat dan Mempersiapkan Data...")
    df = pd.read_csv(data_file, sep=';')
    
    # Target adalah FCR hari esok
    df['FCR_NEXT_DAY'] = df.groupby('PERIODE')['FCR_ACT'].shift(-1)
    df.dropna(subset=['FCR_NEXT_DAY'], inplace=True) # Hapus baris terakhir setiap periode

    TARGET = 'FCR_NEXT_DAY'
    # Buang kolom yang tidak diperlukan untuk training
    features_to_drop = ['TANGGAL', 'FCR_ACT'] 
    df = df.drop(columns=features_to_drop, errors='ignore')

    # Time-based Split (Train-Validation-Test)
    unique_periods = sorted(df['PERIODE'].unique())
    n_test = 1
    n_val = 1
    n_train = len(unique_periods) - n_test - n_val

    train_periods = unique_periods[:n_train]
    val_periods = unique_periods[n_train:n_train + n_val]
    test_periods = unique_periods[n_train + n_val:]
    
    train_df = df[df['PERIODE'].isin(train_periods)]
    val_df = df[df['PERIODE'].isin(val_periods)]
    test_df = df[df['PERIODE'].isin(test_periods)]

    features = [c for c in df.columns if c not in [TARGET, 'PERIODE']]
    X_train, y_train = train_df[features], train_df[TARGET]
    X_val, y_val = val_df[features], val_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]
    
    logger.info(f"Data dibagi -> Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # --- Langkah 2: Membangun Pengekstraksi Fitur TCN (TCN Encoder) ---
    logger.info("Langkah 4.2: Membangun & Melatih TCN Encoder...")
    
    # 2.1. Persiapan Data Sekuens
    def create_sequences(X, y, sequence_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X.iloc[i:(i + sequence_length)].values)
            y_seq.append(y.iloc[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)

    # 2.2. Optimisasi dan Pelatihan TCN Encoder
    def create_tcn_pretrain_model(trial, input_shape):
        nb_filters = trial.suggest_int('nb_filters', 24, 96, step=8)
        kernel_size = trial.suggest_categorical('kernel_size', [2, 3])
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        input_layer = Input(shape=input_shape)
        tcn_output = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=1, dilations=[1, 2, 4, 8],
                         use_skip_connections=True, dropout_rate=dropout_rate, activation='relu',
                         return_sequences=False)(input_layer)
        output_layer = Dense(1, name='regression_head')(tcn_output)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mae')
        return model

    def objective_tcn_pretrain(trial):
        sequence_length = trial.suggest_int('sequence_length', 7, 14)
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
        if len(X_train_seq) == 0 or len(X_val_seq) == 0: return float('inf')
        model = create_tcn_pretrain_model(trial, (sequence_length, X_train.shape[1]))
        model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
                  epochs=100, batch_size=64, callbacks=[EarlyStopping(patience=10)], verbose=0)
        return model.evaluate(X_val_seq, y_val_seq, verbose=0)

    study_tcn = optuna.create_study(direction='minimize')
    study_tcn.optimize(objective_tcn_pretrain, n_trials=40, timeout=1800) # Timeout 30 menit
    
    # Bangun TCN Encoder Final
    best_params_tcn = study_tcn.best_params
    final_sequence_length = best_params_tcn['sequence_length']
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    X_train_full_seq, y_train_full_seq = create_sequences(X_train_full, y_train_full, final_sequence_length)
    
    final_input_shape = (final_sequence_length, X_train.shape[1])
    final_tcn_pretrain_model = create_tcn_pretrain_model(optuna.trial.FixedTrial(best_params_tcn), final_input_shape)
    final_tcn_pretrain_model.fit(X_train_full_seq, y_train_full_seq, epochs=50, batch_size=64, verbose=1)
    
    # "Potong" lapisan Dense terakhir
    tcn_encoder = Model(inputs=final_tcn_pretrain_model.input, outputs=final_tcn_pretrain_model.layers[-2].output)
    tcn_encoder.save(tcn_encoder_path) 
    logger.info(f"TCN Encoder berhasil dilatih dan disimpan di: {tcn_encoder_path}")

    # --- Langkah 3: Membangun Regresor XGBoost ---
    logger.info("Langkah 4.3: Membangun & Melatih Regresor XGBoost...")

    # 3.1. Buat Set Data Final untuk XGBoost
    X_train_xgb_seq, y_train_xgb = create_sequences(X_train, y_train, final_sequence_length)
    X_val_xgb_seq, y_val_xgb = create_sequences(X_val, y_val, final_sequence_length)
    X_test_xgb_seq, y_test_xgb = create_sequences(X_test, y_test, final_sequence_length)

    temporal_features_train = tcn_encoder.predict(X_train_xgb_seq)
    temporal_features_val = tcn_encoder.predict(X_val_xgb_seq)
    temporal_features_test = tcn_encoder.predict(X_test_xgb_seq)

    tabular_features_train = X_train.iloc[final_sequence_length:].values
    tabular_features_val = X_val.iloc[final_sequence_length:].values
    tabular_features_test = X_test.iloc[final_sequence_length:].values
    
    X_train_hybrid = np.concatenate([tabular_features_train, temporal_features_train], axis=1)
    X_val_hybrid = np.concatenate([tabular_features_val, temporal_features_val], axis=1)
    X_test_hybrid = np.concatenate([tabular_features_test, temporal_features_test], axis=1)

    # 3.2. Optimisasi dan Pelatihan Final XGBoost
    def objective_xgb_hybrid(trial):
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 1500,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_hybrid, y_train_xgb, eval_set=[(X_val_hybrid, y_val_xgb)], verbose=False)
        return np.sqrt(mean_squared_error(y_val_xgb, model.predict(X_val_hybrid)))

    study_xgb_hybrid = optuna.create_study(direction='minimize')
    study_xgb_hybrid.optimize(objective_xgb_hybrid, n_trials=150, timeout=3600) # Timeout 1 jam

    best_params_hybrid = study_xgb_hybrid.best_params
    final_hybrid_model = xgb.XGBRegressor(n_estimators=1500, **best_params_hybrid)
    X_train_full_hybrid = np.concatenate([X_train_hybrid, X_val_hybrid])
    y_train_full_hybrid = np.concatenate([y_train_xgb, y_val_xgb])
    final_hybrid_model.fit(X_train_full_hybrid, y_train_full_hybrid)
    
    with open(xgb_model_path, 'wb') as f: pickle.dump(final_hybrid_model, f)
    logger.info(f"Model XGBoost Hybrid berhasil dilatih dan disimpan di: {xgb_model_path}")
    
    # Simpan konfigurasi penting
    hybrid_config = {'sequence_length': final_sequence_length, 'tcn_params': best_params_tcn, 'xgb_params': best_params_hybrid}
    with open(hybrid_config_path, 'w') as f: json.dump(hybrid_config, f, indent=4)
    logger.info(f"Konfigurasi model hibrida disimpan di: {hybrid_config_path}")

    # --- Langkah 4: Evaluasi Akhir Model Hibrida ---
    logger.info("Langkah 4.4: Evaluasi Akhir Model Hibrida...")
    test_preds_hybrid = final_hybrid_model.predict(X_test_hybrid)
    
    print("\n--- Performa Model Hibrida TCN-XGBoost di Test Set ---")
    print(f"MAE: {mean_absolute_error(y_test_xgb, test_preds_hybrid):.4f}")
    print(f"MSE: {mean_squared_error(y_test_xgb, test_preds_hybrid):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_xgb, test_preds_hybrid)):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test_xgb, test_preds_hybrid):.4f}")
    print(f"R-squared: {r2_score(y_test_xgb, test_preds_hybrid):.4f}")
    print("-" * 50)
    
    # Plot hasil
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_xgb, label='Aktual')
    plt.plot(test_preds_hybrid, label='Prediksi Hibrida', linestyle='--')
    plt.title('Evaluasi Model Hibrida TCN-XGBoost')
    plt.legend()
    plt.show()

    return True

# =============================================================================
# BLOK EKSEKUSI UTAMA
# =============================================================================
if __name__ == "__main__":
    base_folder_path = os.getcwd()
    
    # Definisi path
    data_input_folder = os.path.join(base_folder_path, 'non-normalize')
    intermediate_data_folder = os.path.join(base_folder_path, 'Normalisasi', 'Data')
    final_output_folder = os.path.join(base_folder_path, 'Hasil Normalisasi')
    models_folder = os.path.join(base_folder_path, 'models')

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

    # Path untuk model Hibrida
    tcn_encoder_model_path = os.path.join(models_folder, 'tcn_encoder.keras')
    xgb_hybrid_model_path = os.path.join(models_folder, 'xgboost_hybrid_model.pkl')
    hybrid_config_file_path = os.path.join(models_folder, 'hybrid_model_config.json')


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
            
          # Langkah 4: Pelatihan Model Hibrida
        if not train_tcn_xgboost_hybrid(
            data_file=normalized_file,
            tcn_encoder_path=tcn_encoder_model_path,
            xgb_model_path=xgb_hybrid_model_path,
            hybrid_config_path=hybrid_config_file_path
        ):
            raise Exception("Gagal pada langkah pelatihan model hibrida.")

        print("\n\n>>> SEMUA LANGKAH PIPELINE (TERMASUK PELATIHAN MODEL HIBRIDA) SELESAI DENGAN SUKSES! <<<")
        print("Model-model Anda siap untuk diimplementasikan.")
        print(f"Folder 'models' kini berisi: {os.listdir(models_folder)}")

    except Exception as e:
        logger.critical(f"Terjadi kesalahan fatal pada pipeline: {e}", exc_info=True)