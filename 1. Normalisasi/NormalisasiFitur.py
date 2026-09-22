import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

def analyze_and_normalize_data():
    """
    Implementasi strategi normalisasi metode campuran berdasarkan analisis karakteristik fitur
    sesuai dengan dokumen NewNormalisasi
    """
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('File_Gabungan_Enhanced_Features.csv', delimiter=';')
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Backup original data
    df_original = df.copy()
    
    # Identifikasi kolom yang tidak akan dinormalisasi
    non_numeric_cols = ['TANGGAL', 'PERIODE']
    
    # Definisi fitur berdasarkan analisis dalam dokumen NewNormalisasi
    
    # Fitur untuk MinMax Scaler (rentang jelas, distribusi seragam, tanpa outlier signifikan)
    minmax_features = [
        # Fitur numerik mentah
        'AGE', 'MATI', 'JML_DEPLESI', 'CUM_DEPLESI_%', 'STD_CUM_DEPLESI_%', 'PAKAN_STD_ZAK', 'PAKAN_STD_GR/EK', 'PAKAN_STD_CUM_GR/EK', 'PAKAN_ACT_ZAK', 'PAKAN_ACT_CUM_ZAK', 'PAKAN_ACT_GR/EK', 'PAKAN_CUM_GR/EK', 'ABW_STD', 'ABW_ACT', 'DG_STD', 'DG_ACT',
        'FCR_STD', 'FCR_ACT', 'IP_STD', 'IP_ACT',
        'SUHU_MIN', 'SUHU_MAX', 'KELEMBABAN',
        'JUMLAH_AYAM_AKHIR', 'JUMLAH_AYAM_AWAL',
        
        # Fitur lagged tanpa outlier ekstrem
        'FCR_ACT_lag_1', 'FCR_ACT_lag_2', 'FCR_ACT_lag_3', 'FCR_ACT_lag_7', 'FCR_ACT_lag_14',
        'PAKAN_ACT_GR/EK_lag_1', 'PAKAN_ACT_GR/EK_lag_2', 'PAKAN_ACT_GR/EK_lag_3', 
        'PAKAN_ACT_GR/EK_lag_7', 'PAKAN_ACT_GR/EK_lag_14',
        'DG_ACT_lag_1', 'DG_ACT_lag_2', 'DG_ACT_lag_3', 'DG_ACT_lag_7', 'DG_ACT_lag_14',
        'SUHU_MAX_lag_1', 'SUHU_MAX_lag_2', 'SUHU_MAX_lag_3', 'SUHU_MAX_lag_7', 'SUHU_MAX_lag_14',
        'SUHU_MIN_lag_1', 'SUHU_MIN_lag_2', 'SUHU_MIN_lag_3', 'SUHU_MIN_lag_7', 'SUHU_MIN_lag_14',
        'KELEMBABAN_lag_1', 'KELEMBABAN_lag_2', 'KELEMBABAN_lag_3', 'KELEMBABAN_lag_7', 'KELEMBABAN_lag_14',
        'MATI_lag_1', 'MATI_lag_2', 'MATI_lag_3', 'MATI_lag_7', 'MATI_lag_14',
        'JML_DEPLESI_lag_1', 'JML_DEPLESI_lag_2', 'JML_DEPLESI_lag_3', 'JML_DEPLESI_lag_7', 'JML_DEPLESI_lag_14',
        'ABW_ACT_lag_1', 'ABW_ACT_lag_2', 'ABW_ACT_lag_3', 'ABW_ACT_lag_7', 'ABW_ACT_lag_14',
        
        # Rolling mean features (relatif stabil)
        'FCR_ACT_rolling_mean_3', 'FCR_ACT_rolling_mean_7', 'FCR_ACT_rolling_mean_14',
        'PAKAN_ACT_GR/EK_rolling_mean_3', 'PAKAN_ACT_GR/EK_rolling_mean_7', 'PAKAN_ACT_GR/EK_rolling_mean_14',
        'DG_ACT_rolling_mean_3', 'DG_ACT_rolling_mean_7', 'DG_ACT_rolling_mean_14',
        'ABW_ACT_rolling_mean_3', 'ABW_ACT_rolling_mean_7', 'ABW_ACT_rolling_mean_14',
        'DELTA_FCR_rolling_mean_3', 'DELTA_FCR_rolling_mean_7', 'DELTA_FCR_rolling_mean_14',
        'DELTA_PAKAN_GR/EKOR_rolling_mean_3', 'DELTA_PAKAN_GR/EKOR_rolling_mean_7', 'DELTA_PAKAN_GR/EKOR_rolling_mean_14',
        'DELTA_DG_rolling_mean_3', 'DELTA_DG_rolling_mean_7', 'DELTA_DG_rolling_mean_14',
        
        # Rolling min/max features
        'FCR_ACT_rolling_min_3', 'FCR_ACT_rolling_max_3', 'FCR_ACT_rolling_min_7', 'FCR_ACT_rolling_max_7',
        'FCR_ACT_rolling_min_14', 'FCR_ACT_rolling_max_14',
        'PAKAN_ACT_GR/EK_rolling_min_3', 'PAKAN_ACT_GR/EK_rolling_max_3', 'PAKAN_ACT_GR/EK_rolling_min_7', 
        'PAKAN_ACT_GR/EK_rolling_max_7', 'PAKAN_ACT_GR/EK_rolling_min_14', 'PAKAN_ACT_GR/EK_rolling_max_14',
        'DG_ACT_rolling_min_3', 'DG_ACT_rolling_max_3', 'DG_ACT_rolling_min_7', 'DG_ACT_rolling_max_7',
        'DG_ACT_rolling_min_14', 'DG_ACT_rolling_max_14',
        'ABW_ACT_rolling_min_3', 'ABW_ACT_rolling_max_3', 'ABW_ACT_rolling_min_7', 'ABW_ACT_rolling_max_7',
        'ABW_ACT_rolling_min_14', 'ABW_ACT_rolling_max_14',
        'DELTA_FCR_rolling_min_3', 'DELTA_FCR_rolling_max_3', 'DELTA_FCR_rolling_min_7', 'DELTA_FCR_rolling_max_7',
        'DELTA_FCR_rolling_min_14', 'DELTA_FCR_rolling_max_14',
        'DELTA_PAKAN_GR/EKOR_rolling_min_3', 'DELTA_PAKAN_GR/EKOR_rolling_max_3', 
        'DELTA_PAKAN_GR/EKOR_rolling_min_7', 'DELTA_PAKAN_GR/EKOR_rolling_max_7',
        'DELTA_PAKAN_GR/EKOR_rolling_min_14', 'DELTA_PAKAN_GR/EKOR_rolling_max_14',
        'DELTA_DG_rolling_min_3', 'DELTA_DG_rolling_max_3', 'DELTA_DG_rolling_min_7', 'DELTA_DG_rolling_max_7',
        'DELTA_DG_rolling_min_14', 'DELTA_DG_rolling_max_14',
        
        # Domain-specific features
        'PAKAN_DG_RATIO', 'SUHU_KELEMBABAN_INTERACTION', 'STRESS_INDEX', 'FEED_EFFICIENCY',
        'AGE_SQUARED', 'AGE_CUBED', 'FCR_EFFICIENCY_SCORE', 'HEALTH_STATUS_SCORE', 'GROWTH_CONSISTENCY_INDEX'
    ]
    
    # Fitur untuk Robust Scaler (distribusi miring, outlier signifikan, terpusat di sekitar nol)
    robust_features = [
        # Semua fitur DELTA (terutama DELTA_FCR dengan outlier ekstrem)
        'DELTA_PAKAN_ZAK', 'DELTA_PAKAN_GR/EKOR', 'DELTA_PAKAN_CUM_GR/EKOK',
        'DELTA_ABW', 'DELTA_DG', 'DELTA_FCR',
        'DELTA_FCR_lag_1', 'DELTA_FCR_lag_2', 'DELTA_FCR_lag_3', 'DELTA_FCR_lag_7', 'DELTA_FCR_lag_14',
        'DELTA_PAKAN_GR/EKOR_lag_1', 'DELTA_PAKAN_GR/EKOR_lag_2', 'DELTA_PAKAN_GR/EKOR_lag_3',
        'DELTA_PAKAN_GR/EKOR_lag_7', 'DELTA_PAKAN_GR/EKOR_lag_14',
        'DELTA_DG_lag_1', 'DELTA_DG_lag_2', 'DELTA_DG_lag_3', 'DELTA_DG_lag_7', 'DELTA_DG_lag_14',
        
        # Semua fitur rolling standard deviation
        'FCR_ACT_rolling_std_3', 'FCR_ACT_rolling_std_7', 'FCR_ACT_rolling_std_14',
        'PAKAN_ACT_GR/EK_rolling_std_3', 'PAKAN_ACT_GR/EK_rolling_std_7', 'PAKAN_ACT_GR/EK_rolling_std_14',
        'DG_ACT_rolling_std_3', 'DG_ACT_rolling_std_7', 'DG_ACT_rolling_std_14',
        'ABW_ACT_rolling_std_3', 'ABW_ACT_rolling_std_7', 'ABW_ACT_rolling_std_14',
        'DELTA_FCR_rolling_std_3', 'DELTA_FCR_rolling_std_7', 'DELTA_FCR_rolling_std_14',
        'DELTA_PAKAN_GR/EKOR_rolling_std_3', 'DELTA_PAKAN_GR/EKOR_rolling_std_7', 'DELTA_PAKAN_GR/EKOR_rolling_std_14',
        'DELTA_DG_rolling_std_3', 'DELTA_DG_rolling_std_7', 'DELTA_DG_rolling_std_14',
        
        # Fitur momentum, akselerasi, dan tren (rentan outlier)

        # FCR features
        'FCR_ACT_momentum_1', 'FCR_ACT_acceleration', 'FCR_ACT_trend_slope_3', 'FCR_ACT_volatility_3',
        
        # Feed intake (PAKAN) features
        'PAKAN_ACT_GR/EK_momentum_1', 'PAKAN_ACT_GR/EK_acceleration', 
        'PAKAN_ACT_GR/EK_trend_slope_3', 'PAKAN_ACT_GR/EK_volatility_3',
        
        # Daily gain (DG) features
        'DG_ACT_momentum_1', 'DG_ACT_acceleration', 'DG_ACT_trend_slope_3', 'DG_ACT_volatility_3',
        
        # Delta FCR features
        'DELTA_FCR_momentum_1', 'DELTA_FCR_acceleration', 'DELTA_FCR_trend_slope_3', 'DELTA_FCR_volatility_3',
        
        # Delta feed intake features
        'DELTA_PAKAN_GR/EKOR_momentum_1', 'DELTA_PAKAN_GR/EKOR_acceleration',
        'DELTA_PAKAN_GR/EKOR_trend_slope_3', 'DELTA_PAKAN_GR/EKOR_volatility_3',
        
        # Delta daily gain features
        'DELTA_DG_momentum_1', 'DELTA_DG_acceleration', 'DELTA_DG_trend_slope_3', 'DELTA_DG_volatility_3',
        
        # Average body weight (ABW) features
        'ABW_ACT_momentum_1', 'ABW_ACT_acceleration', 'ABW_ACT_trend_slope_3', 'ABW_ACT_volatility_3'
    ]
    
    # Filter fitur yang benar-benar ada di dataset
    available_cols = df.columns.tolist()
    minmax_features = [col for col in minmax_features if col in available_cols]
    robust_features = [col for col in robust_features if col in available_cols]
    
    print(f"\nFitur untuk MinMax Scaler: {len(minmax_features)} kolom")
    print(f"Fitur untuk Robust Scaler: {len(robust_features)} kolom")
    
    # Cek overlap
    overlap = set(minmax_features) & set(robust_features)
    if overlap:
        print(f"\nPeringatan: Ada overlap fitur: {overlap}")
        # Hapus dari minmax_features jika ada overlap
        minmax_features = [col for col in minmax_features if col not in overlap]
    
    # Inisialisasi scalers
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    
    # Dictionary untuk menyimpan informasi normalisasi
    normalization_info = {
        'minmax_features': minmax_features,
        'robust_features': robust_features,
        'non_normalized': non_numeric_cols,
        'minmax_scaler_params': {},
        'robust_scaler_params': {},
        'data_statistics': {}
    }
    
    # Analisis statistik sebelum normalisasi
    print("\n=== ANALISIS STATISTIK SEBELUM NORMALISASI ===")
    
    def analyze_feature_stats(features, feature_type):
        if not features:
            return
        
        print(f"\n{feature_type} Features Analysis:")
        stats_data = []
        
        for col in features[:10]:  # Tampilkan 10 pertama untuk contoh
            if col in df.columns:
                data = df[col].dropna()
                stats = {
                    'feature': col,
                    'min': data.min(),
                    'max': data.max(),
                    'mean': data.mean(),
                    'std': data.std(),
                    'median': data.median(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'outliers_count': len(data[(data < data.quantile(0.25) - 1.5*(data.quantile(0.75) - data.quantile(0.25))) | 
                                             (data > data.quantile(0.75) + 1.5*(data.quantile(0.75) - data.quantile(0.25)))])
                }
                stats_data.append(stats)
                
                if col == 'DELTA_FCR':  # Analisis khusus untuk DELTA_FCR
                    print(f"\n*** ANALISIS KHUSUS DELTA_FCR ***")
                    print(f"Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
                    print(f"Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                    print(f"Outliers: {stats['outliers_count']} dari {len(data)} data")
        
        # Tampilkan ringkasan
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            print(f"\nRingkasan {feature_type}:")
            print(f"Range rata-rata: {stats_df['min'].mean():.3f} - {stats_df['max'].mean():.3f}")
            print(f"Total outliers: {stats_df['outliers_count'].sum()}")
    
    analyze_feature_stats(minmax_features, "MinMax Scaler")
    analyze_feature_stats(robust_features, "Robust Scaler")
    
    # Proses normalisasi
    print("\n=== PROSES NORMALISASI ===")
    df_normalized = df.copy()
    
    # Normalisasi dengan MinMax Scaler
    if minmax_features:
        print(f"\nMenerapkan MinMax Scaler pada {len(minmax_features)} fitur...")
        minmax_data = df[minmax_features].fillna(df[minmax_features].median())
        minmax_normalized = minmax_scaler.fit_transform(minmax_data)
        
        # Update dataframe
        for i, col in enumerate(minmax_features):
            df_normalized[col] = minmax_normalized[:, i]
        
        # Simpan parameter scaler
        normalization_info['minmax_scaler_params'] = {
            'data_min_': minmax_scaler.data_min_.tolist(),
            'data_max_': minmax_scaler.data_max_.tolist(),
            'data_range_': minmax_scaler.data_range_.tolist(),
            'feature_names': minmax_features
        }
        
        print(f"MinMax Scaler berhasil diterapkan. Range output: [0, 1]")
    
    # Normalisasi dengan Robust Scaler
    if robust_features:
        print(f"\nMenerapkan Robust Scaler pada {len(robust_features)} fitur...")
        robust_data = df[robust_features].fillna(df[robust_features].median())
        robust_normalized = robust_scaler.fit_transform(robust_data)
        
        # Update dataframe
        for i, col in enumerate(robust_features):
            df_normalized[col] = robust_normalized[:, i]
        
        # Simpan parameter scaler
        normalization_info['robust_scaler_params'] = {
            'center_': robust_scaler.center_.tolist(),
            'scale_': robust_scaler.scale_.tolist(),
            'feature_names': robust_features
        }
        
        print(f"Robust Scaler berhasil diterapkan. Menggunakan median dan IQR.")
    
    # Analisis hasil normalisasi
    print("\n=== ANALISIS HASIL NORMALISASI ===")
    
    def analyze_normalized_stats(features, feature_type):
        if not features:
            return
        
        print(f"\n{feature_type} - Hasil Normalisasi:")
        sample_features = features[:5]  # Ambil 5 contoh
        
        for col in sample_features:
            if col in df_normalized.columns:
                original = df[col].dropna()
                normalized = df_normalized[col].dropna()
                
                print(f"\n{col}:")
                print(f"  Original: min={original.min():.3f}, max={original.max():.3f}, std={original.std():.3f}")
                print(f"  Normalized: min={normalized.min():.3f}, max={normalized.max():.3f}, std={normalized.std():.3f}")
    
    analyze_normalized_stats(minmax_features, "MinMax Scaler")
    analyze_normalized_stats(robust_features, "Robust Scaler")
    
    # Simpan statistik data
    normalization_info['data_statistics'] = {
        'original_shape': df.shape,
        'normalized_shape': df_normalized.shape,
        'total_features_normalized': len(minmax_features) + len(robust_features),
        'minmax_count': len(minmax_features),
        'robust_count': len(robust_features),
        'non_normalized_count': len(non_numeric_cols)
    }
    
    # Buat folder 'Hasil Normalisasi' jika belum ada
    hasil_folder = 'Hasil Normalisasi'
    if not os.path.exists(hasil_folder):
        os.makedirs(hasil_folder)
        print(f"Folder '{hasil_folder}' berhasil dibuat.")
    
    # Simpan hasil
    output_file = os.path.join(hasil_folder, 'Data_Gabungan_Fitur_Rekayasa_Normalized.csv')
    df_normalized.to_csv(output_file, index=False, sep=';')
    print(f"\nData hasil normalisasi disimpan ke: {output_file}")
    
    # Simpan informasi normalisasi
    info_file = os.path.join(hasil_folder, 'normalization_info_metode_campuran.json')
    with open(info_file, 'w') as f:
        json.dump(normalization_info, f, indent=2)
    print(f"Informasi normalisasi disimpan ke: {info_file}")
    
    # Ringkasan akhir
    print("\n=== RINGKASAN NORMALISASI ===")
    print(f"Total fitur: {df.shape[1]}")
    print(f"Fitur dinormalisasi dengan MinMax Scaler: {len(minmax_features)}")
    print(f"Fitur dinormalisasi dengan Robust Scaler: {len(robust_features)}")
    print(f"Fitur tidak dinormalisasi: {len(non_numeric_cols)}")
    print(f"\nStrategi normalisasi metode campuran berhasil diterapkan!")
    print(f"- MinMax Scaler: untuk fitur dengan rentang jelas dan distribusi seragam")
    print(f"- Robust Scaler: untuk fitur dengan outlier dan distribusi miring")
    
    return df_normalized, normalization_info

if __name__ == "__main__":
    try:
        df_normalized, info = analyze_and_normalize_data()
        print("\nNormalisasi selesai dengan sukses!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()