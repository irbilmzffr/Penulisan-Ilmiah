# ============================================================
# STREAMLIT APPLICATION
# Klasterisasi Sparepart Motor
# Bengkel Mandalika Motor
# ============================================================

import io
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as ReportLabImage
)

from kmeans_engine import run_analysis


# ============================================================
# KONSTANTA PERIODE DATA (BULAN)
# ============================================================

NAMA_BULAN = [
    "Januari", "Februari", "Maret", "April",
    "Mei", "Juni", "Juli", "Agustus",
    "September", "Oktober", "November", "Desember"
]


# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Klasterisasi Sparepart Motor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* =====================================================
GLOBAL
===================================================== */

.stApp {
background-color: #f6f8fb;
}

.block-container {
padding-top: 2rem;
padding-bottom: 3rem;
max-width: 1400px;
}


/* =====================================================
HEADER
===================================================== */

.hero {
background: linear-gradient(
135deg,
#111827 0%,
#1f2937 55%,
#374151 100%
);

padding: 2.5rem 2.8rem;
border-radius: 24px;
margin-bottom: 1.5rem;
box-shadow: 0 12px 35px rgba(0,0,0,0.10);
}

.hero-title {
color: white;
font-size: 2.25rem;
font-weight: 800;
margin-bottom: 0.4rem;
}

.hero-subtitle {
color: #d1d5db;
font-size: 1.05rem;
line-height: 1.6;
margin-bottom: 0;
}

.hero-badge {
display: inline-block;
background: rgba(255,255,255,0.12);
color: #e5e7eb;
padding: 0.4rem 0.8rem;
border-radius: 999px;
font-size: 0.8rem;
margin-bottom: 1rem;
border: 1px solid rgba(255,255,255,0.12);
}


/* =====================================================
SECTION
===================================================== */

.section-title {
font-size: 1.45rem;
font-weight: 750;
color: #111827;
margin-top: 1.8rem;
margin-bottom: 0.25rem;
}

.section-description {
color: #6b7280;
margin-bottom: 1rem;
}


/* =====================================================
METRIC CARD
===================================================== */

.metric-card {
background: white;
border: 1px solid #e5e7eb;
border-radius: 18px;
padding: 1.3rem 1.4rem;
box-shadow: 0 5px 18px rgba(0,0,0,0.04);
min-height: 125px;
}

.metric-label {
color: #6b7280;
font-size: 0.82rem;
font-weight: 600;
margin-bottom: 0.5rem;
}

.metric-value {
color: #111827;
font-size: 2rem;
font-weight: 800;
}

.metric-description {
color: #9ca3af;
font-size: 0.76rem;
margin-top: 0.25rem;
}


/* =====================================================
INFO CARD
===================================================== */

.info-card {
background: white;
border: 1px solid #e5e7eb;
border-radius: 18px;
padding: 1.4rem 1.5rem;
box-shadow: 0 5px 18px rgba(0,0,0,0.04);
margin-bottom: 1rem;
}

.info-card-title {
font-size: 1rem;
font-weight: 750;
color: #111827;
margin-bottom: 0.5rem;
}

.info-card-text {
color: #4b5563;
line-height: 1.65;
font-size: 0.92rem;
}


/* =====================================================
CATEGORY CARD
===================================================== */

.category-card {
background: white;
border: 1px solid #e5e7eb;
border-radius: 18px;
padding: 1.3rem;
box-shadow: 0 5px 18px rgba(0,0,0,0.04);
text-align: center;
}

.category-name {
font-size: 0.95rem;
font-weight: 700;
color: #374151;
}

.category-number {
font-size: 2rem;
font-weight: 800;
color: #111827;
margin: 0.3rem 0;
}

.category-caption {
font-size: 0.75rem;
color: #9ca3af;
}


/* =====================================================
UPLOAD BOX
===================================================== */

[data-testid="stFileUploader"] {
background: white;
border: 2px dashed #d1d5db;
border-radius: 18px;
padding: 0.5rem;
}


/* =====================================================
BUTTON
===================================================== */

.stButton > button {
border-radius: 12px;
font-weight: 700;
min-height: 45px;
}

.stDownloadButton > button {
border-radius: 12px;
font-weight: 700;
min-height: 45px;
}


/* =====================================================
TABLE
===================================================== */

[data-testid="stDataFrame"] {
border-radius: 14px;
overflow: hidden;
}


/* =====================================================
FOOTER
===================================================== */

.footer {
text-align: center;
color: #9ca3af;
font-size: 0.8rem;
padding-top: 2rem;
padding-bottom: 1rem;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 3. HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-badge">
DATA MINING • K-MEANS CLUSTERING
</div>

<div class="hero-title">
Klasterisasi Sparepart Motor
</div>

<p class="hero-subtitle">
Analisis tingkat penjualan sparepart pada
<b>Bengkel Mandalika Motor</b> menggunakan
algoritma K-Means sebagai pendukung keputusan
penyediaan stok.
</p>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 4. UPLOAD DATA
# ============================================================

st.markdown(
    '<div class="section-title">📂 Upload Data Penjualan</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-description">
Upload file CSV yang berisi kolom Nama Sparepart,
Total Penjualan, dan Stok Akhir.
</div>
""",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Pilih file CSV",
    type=["csv"],
    help=(
        "File harus memiliki kolom: "
        "Nama Barang, Total Penjualan, dan Stok Akhir."
    )
)


# ============================================================
# 5. JIKA BELUM ADA DATA
# ============================================================

if uploaded_file is None:

    st.info(
        "Silakan upload file CSV untuk memulai proses analisis."
    )

    st.markdown(
        """
<div class="info-card">

<div class="info-card-title">
📌 Alur Analisis
</div>

<div class="info-card-text">
Data Preparation →
Normalisasi → Penentuan K →
K-Means → Evaluasi →
Kategori Penjualan →
Informasi Klasterisasi
</div>

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="footer">
Penulisan Ilmiah • Informatika •
Bengkel Mandalika Motor
</div>
""",
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# 6. PREVIEW DATA
# ============================================================

try:

    preview_df = pd.read_csv(
        uploaded_file
    )

    uploaded_file.seek(0)

except Exception as e:

    st.error(
        f"File CSV tidak dapat dibaca: {e}"
    )

    st.stop()


st.markdown(
    '<div class="section-title">📄 Preview Data</div>',
    unsafe_allow_html=True
)

preview_col1, preview_col2, preview_col3 = st.columns(3)

with preview_col1:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
JUMLAH BARIS
</div>

<div class="metric-value">
{len(preview_df):,}
</div>

<div class="metric-description">
Data yang diunggah
</div>

</div>
""",
        unsafe_allow_html=True
    )


with preview_col2:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
JUMLAH KOLOM
</div>

<div class="metric-value">
{len(preview_df.columns)}
</div>

<div class="metric-description">
Atribut dalam dataset
</div>

</div>
""",
        unsafe_allow_html=True
    )


with preview_col3:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
NAMA SPAREPART
</div>

<div class="metric-value">
{preview_df["Nama Barang"].nunique()
if "Nama Barang" in preview_df.columns
else "-"}
</div>

<div class="metric-description">
Jenis sparepart unik
</div>

</div>
""",
        unsafe_allow_html=True
    )


with st.expander(
    "🔍 Lihat data yang diunggah"
):

    st.dataframe(
        preview_df,
        use_container_width=True,
        height=300
    )


# ============================================================
# 6. PERIODE DATA (BULAN & TAHUN)
# ============================================================
# Data CSV tidak memuat kolom tanggal, sehingga bulan data
# dipilih oleh pengguna. Pilihan ini dipakai pada judul laporan
# dan nama file unduhan agar setiap laporan sesuai bulannya.

st.markdown(
    '<div class="section-title">🗓️ Periode Data</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-description">
Pilih bulan dan tahun data penjualan yang diunggah.
Periode ini akan tercantum pada judul laporan.
</div>
""",
    unsafe_allow_html=True
)

_sekarang = datetime.now()

periode_col1, periode_col2 = st.columns(2)

with periode_col1:

    bulan_data = st.selectbox(
        "Bulan",
        NAMA_BULAN,
        index=_sekarang.month - 1,
        key="bulan_data"
    )

with periode_col2:

    tahun_data = st.selectbox(
        "Tahun",
        list(range(_sekarang.year - 5, _sekarang.year + 1)),
        index=5,
        key="tahun_data"
    )


# ============================================================
# 7. TOMBOL ANALISIS
# ============================================================

st.markdown("")

col_button1, col_button2, col_button3 = st.columns(
    [1, 1.2, 1]
)

with col_button2:

    mulai_analisis = st.button(
        "🚀  Mulai Analisis K-Means",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# 8. SESSION STATE
# ============================================================

if "hasil_analisis" not in st.session_state:

    st.session_state.hasil_analisis = None


# ============================================================
# 9. JALANKAN ANALISIS
# ============================================================

if mulai_analisis:

    uploaded_file.seek(0)

    progress = st.progress(
        0,
        text="Mempersiapkan analisis..."
    )

    try:

        progress.progress(
            15,
            text="Membaca dan memahami data..."
        )

        hasil = run_analysis(
            uploaded_file
        )

        progress.progress(
            100,
            text="Analisis selesai."
        )

        hasil["periode"] = f"{bulan_data} {tahun_data}"

        st.session_state.hasil_analisis = hasil

        st.success(
            "Analisis K-Means berhasil diselesaikan."
        )

    except Exception as e:

        progress.empty()

        st.error(
            f"Analisis gagal dilakukan: {e}"
        )

        st.stop()


# ============================================================
# 10. AMBIL HASIL
# ============================================================

hasil = st.session_state.hasil_analisis


if hasil is None:

    st.info(
        "Klik tombol **Mulai Analisis K-Means** "
        "untuk melihat hasil."
    )

    st.stop()


# ============================================================
# 11. DATA HASIL
# ============================================================

periode_laporan = hasil.get("periode", "")

periode_file = periode_laporan.replace(" ", "_")

K_FINAL = hasil["K_FINAL"]

k_elbow = hasil["k_elbow"]

k_silhouette = hasil["k_silhouette"]

k_dbi = hasil["k_dbi"]

evaluasi_final = hasil[
    "evaluasi_final"
]

silhouette_final = evaluasi_final[
    "Silhouette Score"
]

dbi_final = evaluasi_final[
    "Davies-Bouldin Index"
]

ringkasan_cluster = hasil[
    "ringkasan_cluster"
]

data_hasil = hasil[
    "data_hasil"
]

centroid_original = hasil[
    "centroid_original"
]

interpretasi_cluster = hasil[
    "interpretasi_cluster"
]


# ============================================================
# 12. HASIL UTAMA
# ============================================================

st.markdown(
    f'<div class="section-title">📊 Ringkasan Hasil Analisis '
    f'— Periode {periode_laporan}</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-description">
Hasil akhir proses clustering berdasarkan
data yang telah diolah.
</div>
""",
    unsafe_allow_html=True
)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
K FINAL
</div>

<div class="metric-value">
{K_FINAL}
</div>

<div class="metric-description">
Jumlah cluster yang digunakan
</div>

</div>
""",
        unsafe_allow_html=True
    )


with metric2:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
SILHOUETTE SCORE
</div>

<div class="metric-value">
{silhouette_final:.3f}
</div>

<div class="metric-description">
Semakin tinggi semakin baik
</div>

</div>
""",
        unsafe_allow_html=True
    )


with metric3:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
DAVIES-BOULDIN INDEX
</div>

<div class="metric-value">
{dbi_final:.3f}
</div>

<div class="metric-description">
Semakin rendah semakin baik
</div>

</div>
""",
        unsafe_allow_html=True
    )


with metric4:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
DATA DIANALISIS
</div>

<div class="metric-value">
{len(data_hasil):,}
</div>

<div class="metric-description">
Sparepart
</div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# 13. PENENTUAN K
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Penentuan Jumlah Cluster</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-description">
Perbandingan hasil Elbow Method, Silhouette Score,
dan Davies-Bouldin Index sebelum menentukan K final.
</div>
""",
    unsafe_allow_html=True
)


k_col1, k_col2, k_col3, k_col4 = st.columns(4)


with k_col1:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
ELBOW METHOD
</div>

<div class="metric-value">
K = {k_elbow}
</div>

<div class="metric-description">
Titik elbow terdeteksi
</div>

</div>
""",
        unsafe_allow_html=True
    )


with k_col2:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
SILHOUETTE
</div>

<div class="metric-value">
K = {k_silhouette}
</div>

<div class="metric-description">
Nilai Silhouette tertinggi
</div>

</div>
""",
        unsafe_allow_html=True
    )


with k_col3:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
DBI
</div>

<div class="metric-value">
K = {k_dbi}
</div>

<div class="metric-description">
Nilai DBI terendah
</div>

</div>
""",
        unsafe_allow_html=True
    )


with k_col4:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
K FINAL
</div>

<div class="metric-value">
K = {K_FINAL}
</div>

<div class="metric-description">
Digunakan untuk K-Means final
</div>

</div>
""",
        unsafe_allow_html=True
    )


st.markdown(
    f"""
<div class="info-card">

<div class="info-card-title">
💡 Dasar Pemilihan K Final
</div>

<div class="info-card-text">
{hasil["alasan_k"]}
K final yang digunakan dalam proses clustering
adalah <b>K = {K_FINAL}</b>.
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 14. VISUALISASI METODE PENENTUAN K
# ============================================================

st.markdown(
    '<div class="section-title">📈 Evaluasi Penentuan K</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(
    [
        "📉 Elbow Method",
        "📐 Silhouette Score",
        "🎯 Davies-Bouldin Index"
    ]
)


with tab1:

    st.pyplot(
        hasil["fig_elbow"],
        use_container_width=True
    )

    st.caption(
        f"K terbaik berdasarkan Elbow Method: "
        f"K = {k_elbow}"
    )


with tab2:

    st.pyplot(
        hasil["fig_silhouette"],
        use_container_width=True
    )

    st.caption(
        f"K terbaik berdasarkan Silhouette Score: "
        f"K = {k_silhouette}"
    )


with tab3:

    st.pyplot(
        hasil["fig_dbi"],
        use_container_width=True
    )

    st.caption(
        f"K terbaik berdasarkan DBI: "
        f"K = {k_dbi}"
    )


# ============================================================
# 15. TABEL PERBANDINGAN K
# ============================================================

with st.expander(
    "📋 Lihat seluruh hasil pengujian K = 2–10"
):

    evaluasi_k = hasil[
        "evaluasi_k"
    ].copy()

    evaluasi_k[
        "Inertia"
    ] = evaluasi_k[
        "Inertia"
    ].round(6)

    evaluasi_k[
        "Silhouette Score"
    ] = evaluasi_k[
        "Silhouette Score"
    ].round(6)

    evaluasi_k[
        "DBI"
    ] = evaluasi_k[
        "DBI"
    ].round(6)

    st.dataframe(
        evaluasi_k,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 16. HASIL CLUSTERING
# ============================================================

st.markdown(
    '<div class="section-title">🏷️ Hasil Klasterisasi</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-description">
Pengelompokan sparepart berdasarkan karakteristik
total penjualan dan stok akhir.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 17. CATEGORY CARDS
# ============================================================

kategori_data = (
    ringkasan_cluster
    .sort_values(
        "Rata_Rata_Penjualan"
    )
    .reset_index(drop=True)
)


category_columns = st.columns(
    len(kategori_data)
)


for column, (_, row) in zip(
    category_columns,
    kategori_data.iterrows()
):

    with column:

        kategori = row[
            "Kategori Penjualan"
        ]

        jumlah = int(
            row["Jumlah_Sparepart"]
        )

        rata_penjualan = row[
            "Rata_Rata_Penjualan"
        ]

        st.markdown(
            f"""
<div class="category-card">

<div class="category-name">
{kategori}
</div>

<div class="category-number">
{jumlah}
</div>

<div class="category-caption">
sparepart
</div>

<hr>

<div class="category-caption">
Rata-rata penjualan
</div>

<b>
{rata_penjualan:.2f} unit
</b>

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# 18. TABEL RINGKASAN
# ============================================================

st.markdown(
    "#### 📊 Ringkasan Setiap Cluster"
)

display_summary = ringkasan_cluster.copy()

display_summary = display_summary.rename(
    columns={
        "Cluster": "Cluster",
        "Kategori Penjualan": "Kategori",
        "Jumlah_Sparepart": "Jumlah Sparepart",
        "Rata_Rata_Penjualan": "Rata-rata Penjualan",
        "Rata_Rata_Stok_Akhir": "Rata-rata Stok Akhir",
        "Total_Penjualan": "Total Penjualan",
        "Total_Stok_Akhir": "Total Stok Akhir"
    }
)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 19. VISUALISASI HASIL CLUSTERING
# ============================================================

st.markdown(
    '<div class="section-title">🔬 Visualisasi Hasil Clustering</div>',
    unsafe_allow_html=True
)

st.pyplot(
    hasil["fig_clustering"],
    use_container_width=True
)


# ============================================================
# 20. CENTROID
# ============================================================

st.markdown(
    "#### 🎯 Karakteristik Centroid"
)

centroid_display = (
    centroid_original
    .reset_index()
    .copy()
)

centroid_display[
    "Kategori Penjualan"
] = centroid_display[
    "Cluster"
].map(
    hasil["cluster_to_category"]
)

centroid_display[
    "Total Penjualan"
] = centroid_display[
    "Total Penjualan"
].round(2)

centroid_display[
    "Stok Akhir"
] = centroid_display[
    "Stok Akhir"
].round(2)

st.dataframe(
    centroid_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 21. INTERPRETASI
# ============================================================

st.markdown(
    '<div class="section-title">💡 Interpretasi Hasil</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-description">
Informasi karakteristik masing-masing cluster
berdasarkan hasil pengelompokan.
</div>
""",
    unsafe_allow_html=True
)


for _, row in interpretasi_cluster.iterrows():

    cluster = row[
        "Cluster"
    ]

    kategori = row[
        "Kategori Penjualan"
    ]

    interpretasi = row[
        "Interpretasi"
    ]

    st.markdown(
        f"""
<div class="info-card">

<div class="info-card-title">
Cluster {cluster} — {kategori}
</div>

<div class="info-card-text">
{interpretasi}
</div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# 22. DATA HASIL PER SPAREPART
# ============================================================

st.markdown(
    '<div class="section-title">📋 Detail Hasil Clustering</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-description">
Setiap sparepart beserta hasil cluster dan
kategori tingkat penjualannya.
</div>
""",
    unsafe_allow_html=True
)


detail_columns = [
    "Nama Barang",
    "Total Penjualan",
    "Stok Akhir",
    "Cluster",
    "Kategori Penjualan"
]

detail_data = data_hasil[
    detail_columns
].copy()


st.dataframe(
    detail_data,
    use_container_width=True,
    hide_index=True,
    height=450
)


# ============================================================
# 23. DOWNLOAD CSV
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Download Hasil Analisis</div>',
    unsafe_allow_html=True
)

download_col1, download_col2, download_col3 = st.columns(3)


with download_col1:

    csv_hasil = data_hasil.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="⬇️ Hasil Clustering CSV",
        data=csv_hasil,
        file_name=f"hasil_clustering_{periode_file}.csv",
        mime="text/csv",
        use_container_width=True
    )


with download_col2:

    csv_ringkasan = (
        ringkasan_cluster.to_csv(
            index=False,
            encoding="utf-8-sig"
        )
    )

    st.download_button(
        label="⬇️ Ringkasan Cluster CSV",
        data=csv_ringkasan,
        file_name=f"ringkasan_cluster_{periode_file}.csv",
        mime="text/csv",
        use_container_width=True
    )


with download_col3:

    csv_centroid = (
        centroid_display.to_csv(
            index=False,
            encoding="utf-8-sig"
        )
    )

    st.download_button(
        label="⬇️ Centroid CSV",
        data=csv_centroid,
        file_name=f"centroid_cluster_{periode_file}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 24. GENERATE PDF
# ============================================================

def generate_pdf(hasil):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        spaceAfter=6
    )

    story = []

    # --------------------------------------------------------
    # JUDUL
    # --------------------------------------------------------

    periode_pdf = hasil.get("periode", "")

    story.append(
        Paragraph(
            "LAPORAN HASIL KLASTERISASI SPAREPART MOTOR"
            f"<br/>PERIODE {periode_pdf.upper()}",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Bengkel Mandalika Motor",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "Metode K-Means Clustering",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"Tanggal analisis: "
            f"{datetime.now().strftime('%d-%m-%Y %H:%M')}",
            subtitle_style
        )
    )


    # --------------------------------------------------------
    # RINGKASAN
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Ringkasan Analisis",
            heading_style
        )
    )

    summary_data = [
        ["Parameter", "Hasil"],

        [
            "Jumlah Data",
            f"{len(hasil['data_hasil'])}"
        ],

        [
            "K Elbow Method",
            f"{hasil['k_elbow']}"
        ],

        [
            "K Silhouette Score",
            f"{hasil['k_silhouette']}"
        ],

        [
            "K DBI",
            f"{hasil['k_dbi']}"
        ],

        [
            "K Final",
            f"{hasil['K_FINAL']}"
        ],

        [
            "Silhouette Score Final",
            f"{hasil['evaluasi_final']['Silhouette Score']:.6f}"
        ],

        [
            "Davies-Bouldin Index Final",
            f"{hasil['evaluasi_final']['Davies-Bouldin Index']:.6f}"
        ]
    ]

    table = Table(
        summary_data,
        colWidths=[
            8 * cm,
            8 * cm
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111827")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d1d5db")
            ),

            (
                "FONTNAME",
                (0, 1),
                (-1, -1),
                "Helvetica"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#f9fafb")
                ]
            )
        ])
    )

    story.append(table)


    # --------------------------------------------------------
    # DASAR PEMILIHAN K
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Penentuan Jumlah Cluster",
            heading_style
        )
    )

    story.append(
        Paragraph(
            (
                f"Berdasarkan hasil pengujian, "
                f"K terbaik berdasarkan Elbow Method "
                f"adalah {hasil['k_elbow']}, "
                f"Silhouette Score adalah "
                f"{hasil['k_silhouette']}, dan "
                f"Davies-Bouldin Index adalah "
                f"{hasil['k_dbi']}. "
                f"K final yang digunakan adalah "
                f"<b>{hasil['K_FINAL']}</b>."
            ),
            normal_style
        )
    )


    # --------------------------------------------------------
    # RINGKASAN CLUSTER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Ringkasan Cluster",
            heading_style
        )
    )

    summary_cluster = [
        [
            "Cluster",
            "Kategori",
            "Jumlah",
            "Rata-rata Penjualan",
            "Rata-rata Stok"
        ]
    ]

    for _, row in (
        hasil["ringkasan_cluster"].iterrows()
    ):

        summary_cluster.append([
            str(int(row["Cluster"])),

            str(
                row["Kategori Penjualan"]
            ),

            str(
                int(
                    row["Jumlah_Sparepart"]
                )
            ),

            f"{row['Rata_Rata_Penjualan']:.2f}",

            f"{row['Rata_Rata_Stok_Akhir']:.2f}"
        ])


    cluster_table = Table(
        summary_cluster,
        colWidths=[
            2 * cm,
            4 * cm,
            2.2 * cm,
            4 * cm,
            3.5 * cm
        ]
    )

    cluster_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111827")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d1d5db")
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#f9fafb")
                ]
            )
        ])
    )

    story.append(
        cluster_table
    )


    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Centroid Cluster",
            heading_style
        )
    )

    centroid_pdf = [
        [
            "Cluster",
            "Total Penjualan",
            "Stok Akhir",
            "Kategori"
        ]
    ]

    centroid_df = (
        hasil["centroid_original"]
        .reset_index()
    )

    for _, row in centroid_df.iterrows():

        cluster = int(
            row["Cluster"]
        )

        centroid_pdf.append([
            str(cluster),

            f"{row['Total Penjualan']:.2f}",

            f"{row['Stok Akhir']:.2f}",

            hasil[
                "cluster_to_category"
            ][cluster]
        ])


    centroid_table = Table(
        centroid_pdf,
        colWidths=[
            2.5 * cm,
            4 * cm,
            4 * cm,
            5 * cm
        ]
    )

    centroid_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111827")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d1d5db")
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(
        centroid_table
    )


    # --------------------------------------------------------
    # INTERPRETASI
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Interpretasi Hasil",
            heading_style
        )
    )

    for _, row in (
        hasil[
            "interpretasi_cluster"
        ].iterrows()
    ):

        story.append(
            Paragraph(
                (
                    f"<b>Cluster "
                    f"{int(row['Cluster'])} — "
                    f"{row['Kategori Penjualan']}</b>"
                ),
                normal_style
            )
        )

        story.append(
            Paragraph(
                row["Interpretasi"],
                normal_style
            )
        )


    # --------------------------------------------------------
    # DETAIL DATA
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "6. Detail Hasil Clustering Sparepart",
            heading_style
        )
    )

    detail = hasil[
        "data_hasil"
    ][
        [
            "Nama Barang",
            "Total Penjualan",
            "Stok Akhir",
            "Cluster",
            "Kategori Penjualan"
        ]
    ].copy()

    detail_pdf = [
        [
            "Nama Barang",
            "Penjualan",
            "Stok",
            "Cluster",
            "Kategori"
        ]
    ]

    for _, row in detail.iterrows():

        detail_pdf.append([
            str(row["Nama Barang"]),

            f"{row['Total Penjualan']:.0f}",

            f"{row['Stok Akhir']:.0f}",

            str(
                int(row["Cluster"])
            ),

            str(
                row["Kategori Penjualan"]
            )
        ])


    detail_table = Table(
        detail_pdf,
        colWidths=[
            7 * cm,
            2.2 * cm,
            2 * cm,
            2 * cm,
            4 * cm
        ],
        repeatRows=1
    )

    detail_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111827")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                colors.HexColor("#d1d5db")
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                6.5
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            )
        ])
    )

    story.append(
        detail_table
    )


    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    document.build(
        story
    )

    buffer.seek(0)

    return buffer


# ============================================================
# 25. TOMBOL DOWNLOAD PDF
# ============================================================

try:

    pdf_file = generate_pdf(
        hasil
    )

    st.download_button(
        label="📄  Download Laporan Lengkap PDF",
        data=pdf_file,
        file_name=f"laporan_klasterisasi_sparepart_{periode_file}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

except Exception as e:

    st.warning(
        f"Laporan PDF belum dapat dibuat: {e}"
    )


# ============================================================
# 26. FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

<b>Klasterisasi Sparepart Motor</b><br>

Algoritma K-Means •
CRISP-DM •
Bengkel Mandalika Motor

</div>
""",
    unsafe_allow_html=True
)