# Tugas Besar SDR

## Prediksi Harga Laptop Berdasarkan Spesifikasi

## Business Understanding
Dalam era teknologi yang terus berkembang, laptop menjadi salah satu perangkat yang sangat penting bagi individu maupun organisasi. Prediksi harga laptop dapat membantu konsumen untuk membuat keputusan pembelian yang lebih baik dan membantu produsen dalam strategi penetapan harga. Tujuan dari laporan ini adalah untuk mengembangkan model prediktif yang dapat memprediksi harga laptop berdasarkan beberapa fitur yang relevan.
### Problem Statements
- Konsumen sering kali kesulitan dalam membuat keputusan pembelian laptop karena kompleksitas pasar dan variasi harga yang luas.
- Produsen laptop menghadapi tantangan dalam menentukan harga yang tepat untuk produk mereka, yang dapat mempengaruhi daya saing dan profitabilitas mereka.
### Goals
- Membuat model prediktif yang dapat memprediksi harga laptop dengan akurasi yang tinggi berdasarkan fitur-fitur tertentu.
- Membantu konsumen dalam membuat keputusan pembelian yang lebih baik dengan menyediakan perkiraan harga yang dapat diandalkan.
- Membantu produsen dalam menetapkan harga yang kompetitif dan optimal untuk produk laptop mereka.
### Solution Statements
- Mengembangkan model machine learning yang dapat memprediksi harga laptop berdasarkan fitur-fitur seperti merek, spesifikasi teknis, ukuran layar, RAM, penyimpanan, dll.
- Menyediakan platform atau aplikasi yang mudah digunakan bagi konsumen untuk memperoleh perkiraan harga laptop berdasarkan preferensi dan kebutuhan mereka.
- Memberikan analisis harga pasar kepada produsen laptop untuk membantu mereka dalam menetapkan harga yang tepat dan kompetitif untuk produk mereka.
- 
## Data Understanding

Data yang digunakan dalam pembuatan model merupakan data sekunder. Data diambil dari Kaggle dengan nama dataset yaitu Laptop Price.

URL: https://www.kaggle.com/datasets/muhammetvarl/laptop-price

Berikut merupakan detail dari *dataset* yang digunakan untuk pembuatan model:
- Data set berupa CSV
  
Variabel variabel pada *dataset* :
- Perusahaan
- Produk
- Tipe
- Ukuran layar
- CPU
- RAM
- Memori
- CPU
- Sistem Operasi
- Berat
- Harga (Euro)
  
Untuk memahami data lebih lanjut, dilakukan Analisis Univariat dan Analisis Multivariat, serta Visualisasi Data

Analisis Univariat merupakan bentuk analisis data yang hanya merepresentasikan informasi yang terdapat pada satu variabel.  Jenis visualisasi ini umumnya digunakan untuk memberikan gambaran terkait distribusi sebuah variabel dalam suatu *dataset*. Sedangkan, Analisis Multivariat tmerupakan jenis analisis data yang terdapat dalam lebih dari dua variabel. Jenis visualisasi ini digunakan untuk merepresentasikan hubungan dan pola yang terdapat dalam multidimensional data. 

Selain melalui analisis, dilakukan juga Visualisasi Data. Memvisualisasikan data memberikan wawasan mendalam tentang perilaku berbagai fitur-fitur yang tersedia dalam *dataset*. 
Teknik visualisasi yang digunakan pada pembuatan model proyek ini adalah dengan menggunakan catplot yang digunakan untuk memplot distribusi data pada data kategori, pairplot yang digunakan untuk melakukan hubungan antar fitur dalam *dataset*, dan heatmap yang menampilkan korelasi antar fitur yang ada dalam *dataset* dalam bentuk matriks.

Berikut adalah hasil Exploratory Data Analysis (EDA), dimana Gambar 1 merupakan EDA Analisis Univariat dan Gambar 2 merupakan EDA Analisis Multivariat.

![image](https://github.com/dreas27/SDR/assets/164930154/3639f4b8-3abd-436c-b0a1-0cfbb51b1a07)

Gambar 1a. Analisis Univariat (Data Kategori)

![image](https://github.com/dreas27/SDR/assets/164930154/e85c7a32-6d4a-4748-9a88-0dfafa776854)

Gambar 1b. Analisis Univariat (Data Numerik)

Dari histogram "prices_euros", diperoleh beberapa informasi, antara lain:
- Peningkatan harga median rumah sebanding dengan penurunan jumlah sampel. Hal ini dapat terlihat jelas dari histogram "prices_euros" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
- Distribusi harga miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

![image](https://github.com/dreas27/SDR/assets/164930154/2245421d-af80-4974-a5fb-e0fd7f73d113)

Gambar 2a. Analisis Multivariat (Data Kategori)

![image](https://github.com/dreas27/SDR/assets/164930154/4290b78d-c970-4d15-8dcd-421bd71de89e)

Gambar 2b. Analisis Multivariat (Data Numerik)

![image](https://github.com/dreas27/SDR/assets/164930154/cab84463-b40e-46ba-80f7-a00afcc8dd61)

Gambar 2c. Analisis Multivariat (Correlation Matrix)

pada gambar 2a Dengan mengamati rata-rata 'price_euros' relatif terhadap fitur kategori di atas, diperoleh insight sebagai berikut:

- Pada fitur 'Company', rata-rata 'price_euros' cenderung bervariasi. Rentangnya berada antara 1000 hingga 1500 euro.
- Nilai 'Price_euros' tertinggi berada pada nilai 'company' yaitu 'LG' dan nilai 'Price_euros' terendah berada pada nilai 'company' yaitu 'vero'. Sehingga, fitur 'company' memiliki pengaruh yang signifikan terhadap rata-rata 'Price_euros'.
- Kesimpulan akhir, fitur kategori memiliki pengaruh terhadap 'Price_euros'.

pada gambar 2b Fungsi pairplot dari library seaborn menunjukkan relasi pasangan dalam dataset. Dari grafik, terlihat plot relasi masing-masing fitur numerik pada dataset. Pada pola sebaran data grafik pairplot sebelumnya, terlihat bahwa 'laptop_ID' memiliki korelasi dengan fitur 'Price_euros'. Sedangkan kedua fitur lainnya terlihat memiliki korelasi yang lemah karena sebarannya tidak membentuk pola

pada gambar 2c Jika diamati, fitur 'price_euros' memiliki skor korelasi yang cukup besar (0.08) dengan fitur target 'laptop_ID'. Artinya, fitur 'Price_euros' berkorelasi cukup tinggi dengan keempat fitur tersebut. Sementara itu, fitur lainnya memiliki korelasi negatif sehingga, fitur tersebut dapat di-drop.

## Data Preparation

Pada proses *Data Preparation* dilakukan kegiatan seperti *Data Gathering*, *Data Assessing*, dan *Data Cleaning*.
Pada proses *Data Gathering*, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan *datafram*e Pandas.
Untuk proses *Data Assessing*, berikut adalah beberapa pengecekan yang dilakukan:
- Duplicate data (data yang serupa dengan data lainnya)
- Missing value (data atau informasi yang "hilang" atau tidak tersedia)
- Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada)
 
Pada proses *Data Cleaning*, secara garis besar, terdapat tiga metode yang dapat digunakan antara lain seperti berikut:
- Dropping (metode yang dilakukan dengan cara menghapus sejumlah baris data)
- Imputation (metode yang dilakukan dengan cara mengganti nilai yang "hilang" atau tidak tersedia dengan nilai tertentu yang bisa berupa median atau mean dari data)
- Interpolation (metode menghasilkan titik-titik data baru dalam suatu jangkauan dari suatu data)

Beberapa pengamatan dalam satu set data kadang berada di luar lingkungan pengamatan lainnya. Pengamatan seperti itu disebut outlier.

Ada beberapa teknik untuk menangani outliers, antara lain:

- Hypothesis Testing
- Z-score method
- IQR Method
Pada kasus ini, Anda akan mendeteksi outliers dengan teknik visualisasi data (boxplot). Kemudian, Anda akan menangani outliers dengan teknik IQR method

$$ IQR = Q<sub>3</sub> - Q<sub>1</sub> $$



