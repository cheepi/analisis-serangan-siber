import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, pearsonr

# file sheet yang digunakan (dalam format url csv)
url = "https://docs.google.com/spreadsheets/d/1OMMIxN9VJ5RRyg2pBopFbxU8VnN1mNi_Rh8CeFxddew/gviz/tq?tqx=out:csv&sheet=count%20of%20attacks%20per%20day"
df_attacks_per_day = pd.read_csv(url)

# mengganti nama kolom untuk mempermudah akses
df_attacks_per_day.columns = ['Tanggal', 'jumlah', 'unused1', 'unused2']

# filtering baris yang hanya berisi data serangan harian (abaikan baris ringkasan)
df_attacks_per_day_cleaned = df_attacks_per_day[['Tanggal', 'jumlah']]
df_attacks_per_day_cleaned = df_attacks_per_day_cleaned[df_attacks_per_day_cleaned['jumlah'].notna()]

# konversi kolom jumlah ke format numerik dan kolom Tanggal ke format datetime
df_attacks_per_day_cleaned['jumlah'] = pd.to_numeric(df_attacks_per_day_cleaned['jumlah'], errors='coerce')
df_attacks_per_day_cleaned['Tanggal'] = pd.to_datetime(df_attacks_per_day_cleaned['Tanggal'])

# statistik deskriptif
statistik_deskriptif_jumlah = df_attacks_per_day_cleaned['jumlah'].describe()
print("Statistik Deskriptif untuk Jumlah Serangan Harian:")
print(statistik_deskriptif_jumlah)

# menambahkan kolom untuk hari dalam seminggu
df_attacks_per_day_cleaned['HariDalamMinggu'] = df_attacks_per_day_cleaned['Tanggal'].dt.dayofweek

# memisahkan data menjadi hari kerja (0-4) dan akhir pekan (5-6)
data_hari_kerja = df_attacks_per_day_cleaned[df_attacks_per_day_cleaned['HariDalamMinggu'] < 5]['jumlah']
data_akhir_pekan = df_attacks_per_day_cleaned[df_attacks_per_day_cleaned['HariDalamMinggu'] >= 5]['jumlah']

# melakukan uji-T antara hari kerja dan akhir pekan
t_stat, p_value = ttest_ind(data_hari_kerja, data_akhir_pekan, equal_var=False)
print("\nUji-T antara Hari Kerja dan Akhir Pekan:")
print(f"T-statistik: {t_stat}, P-value: {p_value}")

# ANOVA: membandingkan jumlah serangan di seluruh hari dalam seminggu
grup = [df_attacks_per_day_cleaned[df_attacks_per_day_cleaned['HariDalamMinggu'] == i]['jumlah']
        for i in range(7)]
anova_stat, anova_p_value = f_oneway(*grup)
print("\nANOVA untuk Jumlah Serangan di Seluruh Hari dalam Seminggu:")
print(f"F-statistik: {anova_stat}, P-value: {anova_p_value}")

# interval kepercayaan untuk weekdays vs weekend
z_score = 1.96  # interval kepercayaan 95%
n_hari_kerja = len(data_hari_kerja)
mean_hari_kerja = data_hari_kerja.mean()
std_hari_kerja = data_hari_kerja.std()
margin_error_hari_kerja = z_score * (std_hari_kerja / np.sqrt(n_hari_kerja))
ci_hari_kerja = (mean_hari_kerja - margin_error_hari_kerja, mean_hari_kerja + margin_error_hari_kerja)

n_akhir_pekan = len(data_akhir_pekan)
mean_akhir_pekan = data_akhir_pekan.mean()
std_akhir_pekan = data_akhir_pekan.std()
margin_error_akhir_pekan = z_score * (std_akhir_pekan / np.sqrt(n_akhir_pekan))
ci_akhir_pekan = (mean_akhir_pekan - margin_error_akhir_pekan, mean_akhir_pekan + margin_error_akhir_pekan)

print("\nInterval Kepercayaan untuk Jumlah Serangan:")
print(f"Hari Kerja: {ci_hari_kerja}")
print(f"Akhir Pekan: {ci_akhir_pekan}")
print(f"\n")

plt.figure(figsize=(12, 6))
plt.plot(df_attacks_per_day_cleaned.index, df_attacks_per_day_cleaned['jumlah'], label='Serangan Harian')
plt.axhline(y=df_attacks_per_day_cleaned['jumlah'].median(), color='r', linestyle='--', label=f'Median: {df_attacks_per_day_cleaned["jumlah"].median()}')  # Menambahkan garis median
plt.title("Serangan Harian Seiring Waktu dengan Garis Median")
plt.xlabel("Hari Sejak Awal Pengamatan")
plt.ylabel("Jumlah Serangan")
plt.legend()
plt.show()

# menghitung hari numerik
df_attacks_per_day_cleaned['HariNumerik'] = (df_attacks_per_day_cleaned['Tanggal'] - 
                                             df_attacks_per_day_cleaned['Tanggal'].min()).dt.days
x = df_attacks_per_day_cleaned['HariNumerik']
y = df_attacks_per_day_cleaned['jumlah']

plt.figure(figsize=(12, 6))
plt.plot(df_attacks_per_day_cleaned['Tanggal'], df_attacks_per_day_cleaned['jumlah'], label='Observasi', color='black')

# menyimpan nilai sementara untuk regresi terbaik (nilai r, derajatnya, dan koefisien dari derajat tersebut)
best_correlation = -1  
best_degree = 1        
best_koefisien = None  

# loop perhitungan regresi polinomial dan penentuan regresi terbaik berdasarkan nilai r
for derajat in range(1, 6):
    koefisien = np.polyfit(x, y, derajat)
    y_pred = np.polyval(koefisien, x)
    
    korelasi, p_value_korelasi = pearsonr(y, y_pred)
    
    print(f"Koefisien untuk Derajat {derajat}: {koefisien}")
    print(f"Korelasi untuk Derajat {derajat}: {korelasi}, P-value: {p_value_korelasi}\n")
    
    # menentukan r tertinggi
    if korelasi > best_correlation:
        best_correlation = korelasi
        best_degree = derajat
        best_koefisien = koefisien

    # plot untuk semua derajat
    plt.plot(df_attacks_per_day_cleaned['Tanggal'], y_pred, label=f'Garis Tren Derajat {derajat}')

plt.title("Regresi Linear dan Polinomial untuk Serangan Siber Harian Seiring Waktu")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Serangan")
plt.legend()
plt.show()

# prediksi 30 hari ke depan dengan regresi terbaik
hari_masa_depan = np.arange(x.max() + 1, x.max() + 31)
plt.figure(figsize=(12, 6))

prediksi_masa_depan = np.polyval(best_koefisien, hari_masa_depan)
plt.plot(hari_masa_depan, prediksi_masa_depan, label=f'Prediksi Derajat {best_degree} (Korelasi Terbaik)')

print(f"Prediksi Terbaik 30 Hari Ke Depan (Polinomial derajat {best_degree}): ")
print(prediksi_masa_depan)

plt.title("Prediksi 30 Hari Serangan Siber Berdasarkan Regresi Terbaik")
plt.xlabel("Hari Sejak Awal Pengamatan (Mulai dari Hari ke-1381 hingga Hari ke-1410)")
plt.ylabel("Jumlah Serangan")
plt.legend()
plt.show()
