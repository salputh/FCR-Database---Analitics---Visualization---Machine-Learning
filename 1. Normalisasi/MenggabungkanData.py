import pandas as pd
import os

# Path ke folder yang berisi file CSV
folder_path = '/Users/macmini/Documents/Skripsi/TCN-XGBOOST/non-normalize'
output_file = '/Users/macmini/Documents/Skripsi/TCN-XGBOOST/Normalisasi/Data/File_Gabungan.csv'

# List untuk menyimpan semua dataframe
dataframes = []

# Loop untuk membaca setiap file CSV (periode 1-13)
for i in range(1, 14):  # 1 sampai 13
    file_name = f'NORMALISASI_PERIODE_{i}.csv'
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.exists(file_path):
        print(f"Membaca file: {file_name}")
        try:
            # Baca CSV dengan separator semicolon (;)
            df = pd.read_csv(file_path, sep=';')
            
            # Debug: tampilkan info file
            print(f"  - Baris: {len(df)}, Kolom: {len(df.columns)}")
            print(f"  - Kolom tersedia: {list(df.columns)}")
            
            # Cek apakah kolom PERIODE ada
            if 'PERIODE' not in df.columns:
                print(f"  - PERINGATAN: Kolom PERIODE tidak ditemukan di {file_name}")
                print(f"  - Menambahkan kolom PERIODE dengan nilai {i}")
                df['PERIODE'] = i
            else:
                # Cek apakah ada nilai kosong di kolom PERIODE
                null_count = df['PERIODE'].isnull().sum()
                if null_count > 0:
                    print(f"  - PERINGATAN: {null_count} baris dengan PERIODE kosong di {file_name}")
                    print(f"  - Mengisi nilai kosong dengan {i}")
                    df['PERIODE'] = df['PERIODE'].fillna(i)
                
                # Cek apakah semua nilai PERIODE sama dengan nomor file
                unique_periods = df['PERIODE'].unique()
                print(f"  - Nilai PERIODE unik: {unique_periods}")
                if len(unique_periods) == 1 and unique_periods[0] != i:
                    print(f"  - PERINGATAN: PERIODE di file ({unique_periods[0]}) tidak sesuai nama file ({i})")
            
            # Pastikan PERIODE bertipe integer
            df['PERIODE'] = df['PERIODE'].astype(int)
            
            dataframes.append(df)
            print(f"  - Berhasil ditambahkan ke list\n")
            
        except Exception as e:
            print(f"  - ERROR membaca {file_name}: {str(e)}\n")
    else:
        print(f"File tidak ditemukan: {file_name}")

# Gabungkan semua dataframe
if dataframes:
    print("="*50)
    print(f"Menggabungkan {len(dataframes)} file...")
    
    # Concatenate semua dataframe, header hanya dari dataframe pertama
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Data gabungan - Baris: {len(combined_df)}, Kolom: {len(combined_df.columns)}")
    
    # Cek kolom PERIODE setelah penggabungan
    if 'PERIODE' in combined_df.columns:
        periode_stats = combined_df['PERIODE'].value_counts().sort_index()
        print(f"Distribusi PERIODE setelah penggabungan:")
        for periode, count in periode_stats.items():
            print(f"  Periode {periode}: {count} baris")
        
        # Cek apakah ada nilai kosong
        null_periode = combined_df['PERIODE'].isnull().sum()
        if null_periode > 0:
            print(f"PERINGATAN: {null_periode} baris dengan PERIODE kosong setelah penggabungan!")
    else:
        print("PERINGATAN: Kolom PERIODE tidak ada setelah penggabungan!")
    
    # Drop kolom AFKIR jika ada
    if 'AFKIR' in combined_df.columns:
        combined_df = combined_df.drop('AFKIR', axis=1)
        print("Kolom AFKIR berhasil dihapus dari data gabungan")
    else:
        print("Kolom AFKIR tidak ditemukan dalam data")
    
    # Pastikan folder output ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Simpan ke file CSV baru
    combined_df.to_csv(output_file, sep=';', index=False)
    
    print(f"\nBerhasil menggabungkan {len(dataframes)} file CSV")
    print(f"Total baris data: {len(combined_df)}")
    print(f"Total kolom setelah drop AFKIR: {len(combined_df.columns)}")
    print(f"File gabungan disimpan sebagai: {output_file}")
    
    # Tampilkan info singkat tentang data gabungan
    print(f"\nKolom yang tersedia: {list(combined_df.columns)}")
    
    if 'PERIODE' in combined_df.columns:
        unique_periods = sorted(combined_df['PERIODE'].unique())
        print(f"Periode yang tergabung: {unique_periods}")
        
        # Cek apakah ada periode yang hilang
        expected_periods = list(range(1, 14))
        missing_periods = set(expected_periods) - set(unique_periods)
        if missing_periods:
            print(f"PERINGATAN: Periode yang hilang: {sorted(missing_periods)}")
    else:
        print("PERINGATAN: Kolom PERIODE tidak tersedia!")
        
    # Tampilkan sample data
    print(f"\nSample 5 baris pertama:")
    print(combined_df.head())
    
else:
    print("Tidak ada file yang berhasil dibaca!")