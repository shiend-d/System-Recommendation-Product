import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ───
st.set_page_config(
    page_title="Sistem Rekomendasi Tokopedia",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #00b894, #00cec9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #636e72; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.8; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px; border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DATA LOADING & MODEL BUILDING (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Memuat dataset Tokopedia...")
def load_and_clean():
    df = pd.read_csv("tokopedia_product_with_review.csv")

    def to_num(x):
        if pd.isna(x): return np.nan
        return float(re.sub(r"[^0-9]", "", str(x)) or "nan")

    df["price_num"] = df["price"].apply(to_num)
    df["discounted_price_num"] = df["discounted_price"].apply(to_num)
    for c in ["count_sold","rating_average","stock","shop_id","warehouse_id","product_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["count_sold"] = df["count_sold"].fillna(0)

    for col in ["gold_merchant","is_official"]:
        df[f"{col}_text"] = df[col].fillna(False).astype(int).astype(str)

    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)

    unused = ['review_id','message','review_rating','review_time','review_timestamp',
              'review_response','review_like','bad_rating_reason','variant_name',
              'warehouse_id','shop_id','discounted_price','product_url','preorder',
              'is_official','is_topads','gold_merchant','price']
    df = df.drop(columns=[c for c in unused if c in df.columns])
    return df

@st.cache_resource(show_spinner="Membangun model TF-IDF & Cosine Similarity...")
def build_model(_df):
    df = _df.copy()
    df["combined_features"] = (
        df["name"].fillna("") + " " +
        df["category"].fillna("") + " " +
        df["shop_location"].fillna("") + " " +
        df["gold_merchant_text"].fillna("") + " " +
        df["is_official_text"].fillna("")
    ).str.lower()

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df.index, index=df["product_id"]).drop_duplicates()

    df["popularity_score"] = (
        df["rating_average"].fillna(0) * 0.7 +
        np.log1p(df["count_sold"].fillna(0)) * 0.3
    )
    return cosine_sim, indices, df


# ─── Recommendation Functions ───
def recommend_by_id(df, cosine_sim, indices, pid, top_n=10):
    if pid not in indices.index:
        return None
    idx = indices[pid]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_idx = [i for i,_ in scores]
    cols = [c for c in ["product_id","name","category","price_num","rating_average",
                        "count_sold","stock","shop_location"] if c in df.columns]
    result = df.loc[rec_idx, cols].copy()
    result["similarity"] = [s for _,s in scores]
    return result.sort_values("similarity", ascending=False)

def recommend_by_name(df, cosine_sim, indices, name, top_n=10):
    matches = df[df["name"].str.contains(name, case=False, na=False)]
    if matches.empty: return None, None
    pid = matches["product_id"].iloc[0]
    return recommend_by_id(df, cosine_sim, indices, pid, top_n), matches.iloc[0]

def hybrid_recommend(df, cosine_sim, indices, pid, top_n=10):
    if pid not in indices.index: return None
    idx = indices[pid]
    target_price = df.loc[idx, "price_num"]  # Harga produk target
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:50]
    rec_idx = [i for i,_ in scores]
    cols = [c for c in ["product_id","name","category","price_num","rating_average",
                        "count_sold","stock","shop_location","popularity_score"] if c in df.columns]
    base = df.loc[rec_idx, cols].copy().reset_index(drop=True)
    base["similarity"] = [s for _,s in scores]
    # Popularity normalization
    pmin, pmax = base["popularity_score"].min(), base["popularity_score"].max()
    base["popularity_norm"] = (base["popularity_score"]-pmin)/(pmax-pmin) if pmax>pmin else 0
    # Price Similarity
    max_price = df["price_num"].max()
    if max_price > 0 and not pd.isna(target_price):
        base["price_similarity"] = 1 - (abs(base["price_num"] - target_price) / max_price)
    else:
        base["price_similarity"] = 0
    # Hybrid Score: 60% Text + 20% Price + 20% Popularity
    base["hybrid_score"] = 0.60*base["similarity"] + 0.20*base["price_similarity"] + 0.20*base["popularity_norm"]
    return base.sort_values("hybrid_score", ascending=False).head(top_n)

def hybrid_by_name(df, cosine_sim, indices, name, top_n=10):
    matches = df[df["name"].str.contains(name, case=False, na=False)]
    if matches.empty: return None, None
    pid = matches["product_id"].iloc[0]
    return hybrid_recommend(df, cosine_sim, indices, pid, top_n), matches.iloc[0]

def eval_precision(df, cosine_sim, indices, sample_ids, k=10, use_hybrid=False):
    scores = []
    for pid in sample_ids:
        if pid not in indices.index: continue
        cat = df.loc[indices[pid], "category"]
        recs = hybrid_recommend(df,cosine_sim,indices,pid,k) if use_hybrid else recommend_by_id(df,cosine_sim,indices,pid,k)
        if recs is None: continue
        scores.append((recs["category"]==cat).sum()/k)
    return scores


# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════
def main():
    df_raw = load_and_clean()
    cosine_sim, indices, df = build_model(df_raw)

    # ─── Header ───
    st.markdown('<p class="main-header">🛒 Sistem Rekomendasi Produk Tokopedia</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Content-Based Filtering (TF-IDF) vs Hybrid Ranking — Dashboard Interaktif</p>', unsafe_allow_html=True)

    # ─── Sidebar ───
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Tokopedia.svg/320px-Tokopedia.svg.png", width=180)
        st.markdown("---")
        st.markdown("### ⚙️ Konfigurasi")
        search_query = st.text_input("🔍 Cari Produk (kata kunci)", value="mouse", placeholder="Contoh: mouse, lemari, sepatu...")
        top_n = st.slider("📊 Jumlah Rekomendasi", 3, 20, 5)
        st.markdown("---")
        st.markdown("### 📋 Info Dataset")
        st.metric("Total Produk", f"{len(df):,}")
        st.metric("Kategori", df["category"].nunique())
        st.metric("Lokasi Toko", df["shop_location"].nunique())

    # ─── Tabs ───
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA Dashboard", "🔍 Rekomendasi", "⚖️ Perbandingan Model", "ℹ️ Metodologi"])

    # ══════════════ TAB 1: EDA ══════════════
    with tab1:
        st.markdown("## 📊 Exploratory Data Analysis")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-card"><div class="metric-value">{:,}</div><div class="metric-label">Total Produk</div></div>'.format(len(df)), unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div class="metric-value">{:.1f}</div><div class="metric-label">Rata-rata Rating</div></div>'.format(df["rating_average"].mean()), unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card"><div class="metric-value">Rp {:,.0f}</div><div class="metric-label">Median Harga</div></div>'.format(df["price_num"].median()), unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-card"><div class="metric-value">{}</div><div class="metric-label">Kategori</div></div>'.format(df["category"].nunique()), unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="price_num", nbins=40, title="Distribusi Harga Produk",
                             labels={"price_num":"Harga (Rp)"}, color_discrete_sequence=["#00b894"])
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df.dropna(subset=["rating_average"]), x="rating_average", nbins=20,
                             title="Distribusi Rating Rata-rata",
                             labels={"rating_average":"Rating"}, color_discrete_sequence=["#fdcb6e"])
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            top_cats = df["category"].value_counts().head(15)
            fig = px.bar(x=top_cats.values, y=top_cats.index, orientation="h",
                        title="Top 15 Kategori Produk",
                        labels={"x":"Jumlah","y":"Kategori"}, color=top_cats.values,
                        color_continuous_scale="Teal")
            fig.update_layout(template="plotly_dark", height=450, showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.scatter(df.dropna(subset=["rating_average"]), x="price_num", y="rating_average",
                           opacity=0.3, title="Harga vs Rating",
                           labels={"price_num":"Harga (Rp)","rating_average":"Rating"},
                           color_discrete_sequence=["#e17055"])
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap Korelasi
        num_cols = ["price_num","rating_average","count_sold","stock"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Heatmap Korelasi Fitur Numerik",
                       color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════ TAB 2: REKOMENDASI ══════════════
    with tab2:
        st.markdown("## 🔍 Sistem Rekomendasi Interaktif")
        if not search_query:
            st.info("Masukkan kata kunci produk di sidebar untuk mendapatkan rekomendasi.")
        else:
            st.markdown(f"### Hasil untuk: **\"{search_query}\"**")
            col_cb, col_hy = st.columns(2)

            # Content-Based
            cb_result, cb_info = recommend_by_name(df, cosine_sim, indices, search_query, top_n)
            # Hybrid
            hy_result, hy_info = hybrid_by_name(df, cosine_sim, indices, search_query, top_n)

            if cb_result is None:
                st.error(f"Produk dengan kata kunci '{search_query}' tidak ditemukan.")
            else:
                # Info Produk Target
                st.markdown("#### 📦 Produk Target")
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.metric("Nama", cb_info["name"][:50])
                ic2.metric("Kategori", cb_info["category"])
                ic3.metric("Harga", f"Rp {cb_info['price_num']:,.0f}")
                ic4.metric("Rating", cb_info["rating_average"])

                st.markdown("---")

                with col_cb:
                    st.markdown("#### 🔵 Content-Based (TF-IDF)")
                    display_cb = cb_result[["name","category","price_num","rating_average","count_sold","similarity"]].reset_index(drop=True)
                    display_cb.index = display_cb.index + 1
                    st.dataframe(display_cb, use_container_width=True, height=300)

                    fig = px.bar(display_cb.head(top_n), x="similarity", y="name", orientation="h",
                               title="Skor Similarity", color="similarity",
                               color_continuous_scale="Blues")
                    fig.update_layout(template="plotly_dark", height=300, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

                with col_hy:
                    st.markdown("#### 🟠 Hybrid Ranking")
                    if hy_result is not None:
                        display_hy = hy_result[["name","category","price_num","rating_average","count_sold","hybrid_score"]].reset_index(drop=True)
                        display_hy.index = display_hy.index + 1
                        st.dataframe(display_hy, use_container_width=True, height=300)

                        fig = px.bar(display_hy.head(top_n), x="hybrid_score", y="name", orientation="h",
                                   title="Skor Hybrid", color="hybrid_score",
                                   color_continuous_scale="Oranges")
                        fig.update_layout(template="plotly_dark", height=300, yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

                # Ringkasan Statistik
                st.markdown("---")
                st.markdown("#### 📊 Ringkasan Statistik Rekomendasi")
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown("**Content-Based**")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Avg Rating", f"{cb_result['rating_average'].mean():.2f}")
                    sc2.metric("Total Terjual", f"{cb_result['count_sold'].sum():,.0f}")
                    sc3.metric("Kategori Dominan", cb_result["category"].value_counts().index[0] if len(cb_result)>0 else "-")
                with s2:
                    if hy_result is not None:
                        st.markdown("**Hybrid Ranking**")
                        sc4, sc5, sc6 = st.columns(3)
                        sc4.metric("Avg Rating", f"{hy_result['rating_average'].mean():.2f}")
                        sc5.metric("Total Terjual", f"{hy_result['count_sold'].sum():,.0f}")
                        sc6.metric("Kategori Dominan", hy_result["category"].value_counts().index[0] if len(hy_result)>0 else "-")

    # ══════════════ TAB 3: PERBANDINGAN ══════════════
    with tab3:
        st.markdown("## ⚖️ Perbandingan Content-Based vs Hybrid")

        with st.spinner("Mengevaluasi Precision@10 pada 50 sampel acak..."):
            random.seed(42)
            sample_pids = random.sample(list(df["product_id"].dropna().astype(int)), min(50, len(df)))
            cb_scores = eval_precision(df, cosine_sim, indices, sample_pids, k=10, use_hybrid=False)
            hy_scores = eval_precision(df, cosine_sim, indices, sample_pids, k=10, use_hybrid=True)

        m1, m2, m3 = st.columns(3)
        cb_mean = np.mean(cb_scores) if cb_scores else 0
        hy_mean = np.mean(hy_scores) if hy_scores else 0
        m1.metric("CB Precision@10", f"{cb_mean:.1%}")
        m2.metric("Hybrid Precision@10", f"{hy_mean:.1%}")
        m3.metric("Selisih", f"{(cb_mean-hy_mean):.1%}", delta=f"CB {'lebih baik' if cb_mean>hy_mean else 'lebih rendah'}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Content-Based", x=["Precision@10"], y=[cb_mean], marker_color="#0984e3"))
            fig.add_trace(go.Bar(name="Hybrid", x=["Precision@10"], y=[hy_mean], marker_color="#e17055"))
            fig.update_layout(title="Rata-rata Precision@10", template="plotly_dark", barmode="group", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Content-Based","Hybrid"))
            fig.add_trace(go.Histogram(x=cb_scores, nbinsx=10, marker_color="#0984e3", name="CB"), row=1, col=1)
            fig.add_trace(go.Histogram(x=hy_scores, nbinsx=10, marker_color="#e17055", name="Hybrid"), row=1, col=2)
            fig.update_layout(title="Distribusi Precision@10", template="plotly_dark", height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Tabel perbandingan
        st.markdown("### 📋 Perbandingan Karakteristik")
        comp_data = {
            "Aspek": ["Metode Scoring", "Formula", "Precision@10 (Mean)", "Kelebihan", "Kekurangan", "Cocok Untuk"],
            "Content-Based (TF-IDF)": [
                "Cosine Similarity murni",
                "cos(θ) = (A·B) / (‖A‖·‖B‖)",
                f"{cb_mean:.1%}",
                "Fokus relevansi konten tertinggi",
                "Mengabaikan popularitas produk",
                "Produk niche / spesifik"
            ],
            "Hybrid Ranking": [
                "60% Similarity + 20% Price + 20% Popularity",
                "0.60×sim + 0.20×price_sim + 0.20×pop_norm",
                f"{hy_mean:.1%}",
                "Mempertimbangkan harga, rating & penjualan",
                "Bobot banyak bisa menggeser relevansi teks",
                "Produk mainstream dengan harga relevan"
            ]
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.markdown("### 💡 Interpretasi")
        st.info("""
        **Mengapa Content-Based memiliki Precision@10 lebih tinggi?**

        Precision@10 di sini diukur berdasarkan *kesamaan kategori*. Karena Content-Based murni mengandalkan
        kemiripan konten (TF-IDF), ia cenderung merekomendasikan produk dari kategori yang **persis sama**.

        Hybrid Ranking menggabungkan tiga faktor: kemiripan teks (60%), kemiripan harga (20%), dan
        popularitas produk (20%). Faktor harga dan popularitas bisa menyebabkan produk dari kategori
        **berbeda** masuk ke Top-10 jika harganya mirip dan populer. Ini bukan berarti Hybrid
        lebih buruk — justru memberikan rekomendasi yang lebih **beragam dan realistis**.
        """)

    # ══════════════ TAB 4: METODOLOGI ══════════════
    with tab4:
        st.markdown("## ℹ️ Metodologi Sistem Rekomendasi")

        st.markdown("### 1. Feature Engineering")
        st.code("combined_features = name + category + shop_location + gold_merchant + is_official", language="python")

        st.markdown("### 2. TF-IDF Vectorization")
        st.latex(r"\text{TF-IDF}(t, d) = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}")
        st.markdown("Mengubah teks `combined_features` menjadi vektor numerik berbobot.")

        st.markdown("### 3. Cosine Similarity")
        st.latex(r"\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \times ||\mathbf{B}||}")
        st.markdown("Mengukur kemiripan antar produk berdasarkan sudut antara vektor TF-IDF.")

        st.markdown("### 4. Hybrid Score")
        st.latex(r"\text{hybrid\_score} = 0.60 \times \text{sim} + 0.20 \times \text{price\_sim} + 0.20 \times \text{pop\_norm}")
        st.markdown("Dimana:")
        st.latex(r"\text{price\_similarity} = 1 - \frac{|\text{harga\_produk} - \text{harga\_target}|}{\text{harga\_maksimum}}")
        st.latex(r"\text{popularity\_score} = 0.7 \times \text{rating} + 0.3 \times \log(1 + \text{count\_sold})")

        st.markdown("### 5. Evaluasi: Precision@K")
        st.latex(r"\text{Precision@K} = \frac{\text{jumlah rekomendasi dengan kategori sama}}{K}")
        st.markdown("Proxy metric menggunakan kesamaan kategori. Evaluasi ideal membutuhkan data interaksi pengguna.")

        st.markdown("### 📚 Dataset")
        st.markdown("""
        - **Sumber**: [Kaggle - Tokopedia Product and Review](https://www.kaggle.com/datasets/musabiam/tokopedia-product-and-review-dataset)
        - **Format**: CSV (`tokopedia_product_with_review.csv`)
        - **Ukuran**: 5.553 baris → 5.410 setelah cleaning
        - **Fitur utama**: name, category, price, rating_average, count_sold, shop_location, gold_merchant, is_official
        """)

if __name__ == "__main__":
    main()
