# Analisis Serangan Siber Harian

ini disusun untuk memenuhi tugas mata kuliah Statistika (TKU212111). Analisis ini bertujuan untuk memahami pola dan tren serangan siber yang terjadi setiap hari, serta memprediksi potensi peningkatan aktivitas serangan di masa mendatang.

## Deskripsi Data

Data yang digunakan diambil dari [Kaggle: Cyber Security Attacks](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks/data?select=cybersecurity_attacks.csv). Data ini berisi informasi tentang serangan siber, yang kemudian difilter dan dihitung untuk mendapatkan jumlah serangan per hari.

## Tujuan 

1. Mengidentifikasi tren dan pola dalam serangan siber harian.

2. Mengetahui apakah terdapat perbedaan signifikan antara jumlah serangan pada hari kerja dan akhir pekan.

3. Memprediksi jumlah serangan di masa depan menggunakan model statistik dan regresi.

## Metode yang Digunakan

1. **Statistik Deskriptif**: Menghitung statistik dasar seperti mean, median, standar deviasi, dan lainnya untuk memahami distribusi serangan harian.
2. **Uji Hipotesis (Uji-T)**: Menganalisis perbedaan jumlah serangan antara hari kerja dan akhir pekan.
3. **ANOVA**: Membandingkan jumlah serangan di berbagai hari dalam seminggu.
4. **Regresi**: Menggunakan regresi linear dan polinomial untuk memodelkan dan memprediksi tren jumlah serangan siber di masa depan.
5. **Visualisasi**: Menampilkan grafik untuk melihat tren data dan prediksi masa depan.

## Modul yang Digunakan

- `pandas`: Untuk manipulasi data dan pengolahan dataset.
- `numpy`: Untuk operasi matematika dan perhitungan statistik.
- `matplotlib`: Untuk visualisasi grafik dan tren.
- `scipy`: Untuk uji hipotesis dan ANOVA.

## Sumber Data

Data yang digunakan diambil dari Kaggle, tersedia di [tautan ini](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks/data?select=cybersecurity_attacks.csv). 

Harap dicatat bahwa data ini telah diproses lebih lanjut untuk mendapatkan jumlah serangan per hari melalui query di google sheets.

