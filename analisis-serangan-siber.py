import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, pearsonr

# load df (dalam csv)
url = "https://docs.google.com/spreadsheets/d/1OMMIxN9VJ5RRyg2pBopFbxU8VnN1mNi_Rh8CeFxddew/gviz/tq?tqx=out:csv&sheet=count%20of%20attacks%20per%20day"
df_attacks_per_day = pd.read_csv(url)

# rename kolom agar mudah diakses
df_attacks_per_day.columns = ['Tanggal', 'jumlah', 'unused1', 'unused2']

# ambil kolom penting + drop nan
df_attacks_per_day_cleaned = df_attacks_per_day[['Tanggal', 'jumlah']]
df_attacks_per_day_cleaned = df_attacks_per_day_cleaned[df_attacks_per_day_cleaned['jumlah'].notna()]

# konversi kolom jumlah ke format numerik dan kolom Tanggal ke format datetime
df_attacks_per_day_cleaned['jumlah'] = pd.to_numeric(df_attacks_per_day_cleaned['jumlah'], errors='coerce')
df_attacks_per_day_cleaned['Tanggal'] = pd.to_datetime(df_attacks_per_day_cleaned['Tanggal'])

# statistik deskriptif
statistik_deskriptif_jumlah = df_attacks_per_day_cleaned['jumlah'].describe()
print("Statistik Deskriptif untuk Jumlah Serangan Harian:")
print(statistik_deskriptif_jumlah)

# konversi kolom tanggal jadi angka hari (0=senin, 6=minggu)
df_attacks_per_day_cleaned['HariDalamMinggu'] = df_attacks_per_day_cleaned['Tanggal'].dt.dayofweek

data_hari_kerja = df_attacks_per_day_cleaned[df_attacks_per_day_cleaned['HariDalamMinggu'] < 5]['jumlah'] # senin hingga jumat (0-4)
data_akhir_pekan = df_attacks_per_day_cleaned[df_attacks_per_day_cleaned['HariDalamMinggu'] >= 5]['jumlah'] # sabtu, minggu (5-6)

# uji-T antara weekday dan weekend
t_stat, p_value = ttest_ind(data_hari_kerja, data_akhir_pekan, equal_var=False)
print("\nUji-T antara Hari Kerja dan Akhir Pekan:")
print(f"T-statistik: {t_stat}, P-value: {p_value}")

# ANOVA antar hari
grup = [df_attacks_per_day_cleaned[df_attacks_per_day_cleaned['HariDalamMinggu'] == i]['jumlah']
        for i in range(7)]
anova_stat, anova_p_value = f_oneway(*grup)
print("\nANOVA untuk Jumlah Serangan di Seluruh Hari dalam Seminggu:")
print(f"F-statistik: {anova_stat}, P-value: {anova_p_value}")

# interval kepercayaan untuk weekdays vs weekend
z_score = 1.96  # interval kepercayaan 95%

# define interval kepercayaan
def confidence_interval(data):
  n = len(data)
  mean = data.mean()
  std = data.std()
  margin = z_score * (std / np.sqrt(n))

  return (mean - margin, mean + margin)

ci_hari_kerja = confidence_interval(data_hari_kerja)
ci_akhir_pekan = confidence_interval(data_akhir_pekan)

print("\nInterval Kepercayaan untuk Jumlah Serangan:")
print(f"Hari Kerja: {ci_hari_kerja}")
print(f"Akhir Pekan: {ci_akhir_pekan}")
print(f"\n")

plt.figure(figsize=(12, 6))
plt.plot(df_attacks_per_day_cleaned['Tanggal'], df_attacks_per_day_cleaned['jumlah'], label='Serangan Harian')
plt.axhline(y=df_attacks_per_day_cleaned['jumlah'].median(), color='r', linestyle='--',
            label=f'Median: {df_attacks_per_day_cleaned["jumlah"].median():.2f}')
plt.title("Serangan Harian Seiring Waktu (dengan Garis Median)")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Serangan")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45) # supaya ga ngelebar di sumbu x
plt.tight_layout()
plt.show()

# konversi tanggal ke urutan hari numerik (0=hari paling awal)
df_attacks_per_day_cleaned['HariNumerik'] = (
    df_attacks_per_day_cleaned['Tanggal'] - df_attacks_per_day_cleaned['Tanggal'].min()
).dt.days

x = df_attacks_per_day_cleaned['HariNumerik'] #urutan hari secara kontinu
y = df_attacks_per_day_cleaned['jumlah']

# plot observasi aktual
plt.figure(figsize=(12, 6))
plt.plot(df_attacks_per_day_cleaned['Tanggal'], y, label='Observasi', color='black')

# inisialisasi variable utnuk store model dengan best performance
best_correlation = -1
best_degree = 1
best_koefisien = None

for derajat in range(1, 6):
  koefisien = np.polyfit(x, y, derajat)
  y_pred = np.polyval(koefisien, x)

  korelasi, p_val = pearsonr(y, y_pred)

  # adjusted r² biar gak bias derajat tinggi
  ss_res = np.sum((y - y_pred)**2)
  ss_tot = np.sum((y - np.mean(y))**2)
  r2 = 1 - (ss_res / ss_tot)
  adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - derajat - 1)

  print(f"Derajat {derajat}: Korelasi = {korelasi:.4f}, Adjusted R² = {adj_r2:.4f}")

  if korelasi > best_correlation:
      best_correlation = korelasi
      best_degree = derajat
      best_koefisien = koefisien

  plt.plot(df_attacks_per_day_cleaned['Tanggal'], y_pred, label=f'Derajat {derajat}')

# plot hasil
plt.title("Regresi Linear & Polinomial untuk Serangan Harian")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Serangan")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nModel terbaik: Derajat {best_degree} (Korelasi = {best_correlation:.4f})\n")

# prediksi 30 hari ke depan dengan regresi terbaik
# buat urutan hari numerik baru (lanjutan dari existing data terakhir)
hari_masa_depan = np.arange(x.max() + 1, x.max() + 31)

# prediksi jumlah serangan untuk 30 hari ke depan dengan regresi terbaik
prediksi_masa_depan = np.polyval(best_koefisien, hari_masa_depan)

# ubah urutan hari jadi format tanggal aktual agar sejajar dengan timeline data historis
tanggal_masa_depan = df_attacks_per_day_cleaned['Tanggal'].max() + pd.to_timedelta(hari_masa_depan - x.max(), unit='D')

plt.figure(figsize=(12, 6))
plt.plot(df_attacks_per_day_cleaned['Tanggal'], y, label='Data Historis', color='black')
plt.plot(tanggal_masa_depan, prediksi_masa_depan, label=f'Prediksi Derajat {best_degree}', color='blue')
plt.title("Prediksi 30 Hari Serangan Siber Berdasarkan Regresi Terbaik")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Serangan")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Prediksi 30 Hari ke Depan:")
print(prediksi_masa_depan)