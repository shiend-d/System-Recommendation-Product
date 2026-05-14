# 🛒 Sistem Rekomendasi Produk Tokopedia

Proyek UAS Sains Data — Sistem rekomendasi produk marketplace Tokopedia menggunakan **Content-Based Filtering (TF-IDF + Cosine Similarity)** dan **Hybrid Ranking**, divisualisasikan melalui dashboard interaktif **Streamlit**.

---

## 📋 Deskripsi Proyek

Pengguna e-commerce seringkali kesulitan menemukan produk yang relevan di antara ribuan produk yang tersedia. Proyek ini membangun sistem rekomendasi yang membantu pengguna menemukan produk serupa berdasarkan dua pendekatan:

| Pendekatan | Metode | Deskripsi |
|------------|--------|-----------|
| **Content-Based** | TF-IDF + Cosine Similarity | Merekomendasikan produk berdasarkan kemiripan konten (nama, kategori, lokasi toko) |
| **Hybrid Ranking** | Similarity + Popularity Score | Menggabungkan skor kemiripan konten dengan skor popularitas (rating & jumlah terjual) |

### Formula Hybrid Score

```
hybrid_score = 0.75 × cosine_similarity + 0.25 × popularity_norm
```

Dimana:

```
popularity_score = 0.7 × rating_average + 0.3 × log(1 + count_sold)
```

---

## 📊 Dataset

- **Sumber**: [Kaggle — Tokopedia Product and Review Dataset](https://www.kaggle.com/datasets/musabiam/tokopedia-product-and-review-dataset)
- **File**: `tokopedia_product_with_review.csv`
- **Ukuran**: 5.553 baris × 25 kolom (raw) → 5.410 baris × 11 kolom (cleaned)
- **Fitur utama**:

| Fitur | Deskripsi |
|-------|-----------|
| `product_id` | ID unik produk |
| `name` | Nama produk |
| `category` | Kategori produk |
| `price` | Harga produk |
| `rating_average` | Rating rata-rata (0–5) |
| `count_sold` | Jumlah terjual |
| `shop_location` | Lokasi toko |
| `gold_merchant` | Status Gold Merchant |
| `is_official` | Status Toko Resmi |

---

## 🖥️ Dashboard Streamlit

Dashboard interaktif terdiri dari **4 tab utama**:

### 📊 Tab 1 — EDA Dashboard
- Metric cards (total produk, rata-rata rating, median harga, jumlah kategori)
- Distribusi harga & rating (histogram interaktif)
- Top 15 kategori produk (bar chart horizontal)
- Scatter plot harga vs rating
- Heatmap korelasi fitur numerik

### 🔍 Tab 2 — Sistem Rekomendasi
- Input kata kunci produk via sidebar
- Hasil rekomendasi **side-by-side**: Content-Based vs Hybrid
- Bar chart skor similarity / hybrid score
- Ringkasan statistik (avg rating, total terjual, kategori dominan)

### ⚖️ Tab 3 — Perbandingan Model
- Evaluasi **Precision@10** pada 50 sampel acak
- Bar chart perbandingan rata-rata Precision@10
- Histogram distribusi Precision@10 kedua model
- Tabel perbandingan karakteristik
- Interpretasi hasil

### ℹ️ Tab 4 — Metodologi
- Penjelasan Feature Engineering
- Formula TF-IDF, Cosine Similarity, Hybrid Score
- Metrik evaluasi Precision@K
- Informasi dataset

---

## 🚀 Cara Menjalankan

### 1. Clone / Download Proyek

Pastikan file berikut ada dalam satu folder:
```
SA/
├── app.py                              # Streamlit dashboard
├── tokopedia_product_with_review.csv   # Dataset
├── Salinan dari codekita.ipynb         # Notebook analisis
└── README.md
```

### 2. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### 3. Jalankan Dashboard

```bash
streamlit run app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`.

> **Catatan**: Loading pertama membutuhkan waktu ~30–60 detik karena memproses dataset dan membangun model TF-IDF. Setelah itu, data di-cache dan akses berikutnya akan jauh lebih cepat.

---

## 📈 Hasil Evaluasi

Evaluasi menggunakan **Precision@10** (proxy: kesamaan kategori) pada 50 sampel acak:

| Model | Mean | Min | Max |
|-------|------|-----|-----|
| Content-Based (TF-IDF) | **~74.6%** | — | — |
| Hybrid Ranking | ~68.4% | — | — |

### Interpretasi

- **Content-Based** memiliki Precision@10 lebih tinggi karena murni mengandalkan kemiripan konten, sehingga cenderung merekomendasikan produk dari kategori yang sama persis.
- **Hybrid Ranking** menambahkan faktor popularitas yang dapat menyebabkan produk populer dari kategori berbeda masuk ke Top-10 — ini bukan berarti lebih buruk, melainkan menggunakan kriteria yang lebih beragam.
- Evaluasi ideal membutuhkan data interaksi pengguna (click-through rate, purchase history) yang tidak tersedia dalam dataset ini.

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python 3 |
| Analisis Data | Pandas, NumPy |
| Machine Learning | Scikit-learn (TF-IDF, Cosine Similarity) |
| Visualisasi | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Notebook | Google Colab / Jupyter |

---

## 👥 Tim Pengembang

Proyek UAS Mata Kuliah Sains Data — Semester 4

---

*Terakhir diperbarui: Mei 2026*
