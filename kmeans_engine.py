# ============================================================
# K-MEANS ENGINE
# Penulisan Ilmiah
# Klasterisasi Sparepart Motor
# Bengkel Mandalika Motor
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. KONFIGURASI PENELITIAN
# ============================================================

KOLOM_NAMA = "Nama Barang"
KOLOM_PENJUALAN = "Total Penjualan"
KOLOM_STOK = "Stok Akhir"

K_MIN = 2
K_MAX = 10

RANDOM_STATE = 42
N_INIT = 20


# ============================================================
# 2. FUNGSI MEMBACA DATA
# ============================================================

def load_data(file):
    """
    Membaca file CSV yang diberikan.

    Parameter:
        file : path file atau file-like object

    Return:
        df : DataFrame
    """

    df = pd.read_csv(file)

    return df


# ============================================================
# 3. DATA UNDERSTANDING
# ============================================================

def data_understanding(df):
    """
    Melakukan pemeriksaan awal dataset.
    """

    informasi = {
        "nama_kolom": df.columns.tolist(),
        "jumlah_baris": df.shape[0],
        "jumlah_kolom": df.shape[1],
        "tipe_data": df.dtypes,
        "missing_value": df.isnull().sum(),
        "duplikasi_baris": int(df.duplicated().sum()),
        "statistik": df.describe(include="all")
    }

    return informasi


# ============================================================
# 4. VALIDASI KOLOM
# ============================================================

def validate_columns(df):
    """
    Memastikan kolom penelitian tersedia.
    """

    kolom_wajib = [
        KOLOM_NAMA,
        KOLOM_PENJUALAN,
        KOLOM_STOK
    ]

    kolom_tidak_ada = [
        kolom
        for kolom in kolom_wajib
        if kolom not in df.columns
    ]

    if kolom_tidak_ada:

        raise ValueError(
            "Kolom berikut tidak ditemukan: "
            + ", ".join(kolom_tidak_ada)
        )

    return True


# ============================================================
# 5. DATA SELECTION
# ============================================================

def data_selection(df):
    """
    Memilih atribut yang digunakan dalam penelitian.

    Variabel:
        - Nama Barang
        - Total Penjualan
        - Stok Akhir
    """

    data = df[
        [
            KOLOM_NAMA,
            KOLOM_PENJUALAN,
            KOLOM_STOK
        ]
    ].copy()

    return data


# ============================================================
# 6. KONVERSI VARIABEL NUMERIK
# ============================================================

def convert_numeric(data):
    """
    Mengubah Total Penjualan dan Stok Akhir
    menjadi tipe numerik.
    """

    data[KOLOM_PENJUALAN] = pd.to_numeric(
        data[KOLOM_PENJUALAN],
        errors="coerce"
    )

    data[KOLOM_STOK] = pd.to_numeric(
        data[KOLOM_STOK],
        errors="coerce"
    )

    return data


# ============================================================
# 7. DATA CLEANING
# ============================================================

def data_cleaning(data):
    """
    Membersihkan data dari:
        - Missing value
        - Nilai negatif
        - Duplikasi baris identik
    """

    jumlah_sebelum = len(data)

    # --------------------------------------------------------
    # Missing value
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            KOLOM_NAMA,
            KOLOM_PENJUALAN,
            KOLOM_STOK
        ]
    )

    # --------------------------------------------------------
    # Nilai negatif
    # --------------------------------------------------------

    data = data[
        (data[KOLOM_PENJUALAN] >= 0)
        &
        (data[KOLOM_STOK] >= 0)
    ]

    # --------------------------------------------------------
    # Duplikasi baris identik
    # --------------------------------------------------------

    data = data.drop_duplicates()

    jumlah_sesudah = len(data)

    cleaning_info = {
        "jumlah_sebelum": jumlah_sebelum,
        "jumlah_sesudah": jumlah_sesudah,
        "jumlah_dihapus": jumlah_sebelum - jumlah_sesudah,
        "missing_value": data.isnull().sum(),
        "duplikasi": int(data.duplicated().sum())
    }

    return data, cleaning_info


# ============================================================
# 8. CEK DUPLIKASI NAMA SPAREPART
# ============================================================

def check_duplicate_names(data):
    """
    Memeriksa apakah satu jenis sparepart
    muncul lebih dari satu kali.
    """

    jumlah_baris = len(data)

    jumlah_nama_unik = (
        data[KOLOM_NAMA].nunique()
    )

    hasil = {
        "jumlah_baris": jumlah_baris,
        "jumlah_nama_unik": jumlah_nama_unik,
        "semua_unik": (
            jumlah_baris == jumlah_nama_unik
        )
    }

    if not hasil["semua_unik"]:

        duplikat_nama = data[
            data[KOLOM_NAMA].duplicated(
                keep=False
            )
        ].sort_values(
            KOLOM_NAMA
        )

        hasil["data_duplikat"] = duplikat_nama

    else:

        hasil["data_duplikat"] = pd.DataFrame()

    return hasil


# ============================================================
# 9. STATISTIK VARIABEL
# ============================================================

def variable_statistics(data):
    """
    Statistik deskriptif variabel penelitian.
    """

    statistik = data[
        [
            KOLOM_PENJUALAN,
            KOLOM_STOK
        ]
    ].describe()

    return statistik


# ============================================================
# 10. VISUALISASI DATA AWAL
# ============================================================

def plot_data_awal(data):
    """
    Scatter plot data sebelum clustering.
    """

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        data[KOLOM_PENJUALAN],
        data[KOLOM_STOK]
    )

    ax.set_xlabel(
        "Total Penjualan"
    )

    ax.set_ylabel(
        "Stok Akhir"
    )

    ax.set_title(
        "Distribusi Sparepart Berdasarkan "
        "Total Penjualan dan Stok Akhir"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    return fig


# ============================================================
# 11. DETEKSI OUTLIER IQR
# ============================================================

def cek_outlier(data, kolom):
    """
    Mendeteksi outlier menggunakan metode IQR.
    """

    Q1 = data[kolom].quantile(0.25)

    Q3 = data[kolom].quantile(0.75)

    IQR = Q3 - Q1

    batas_bawah = (
        Q1 - 1.5 * IQR
    )

    batas_atas = (
        Q3 + 1.5 * IQR
    )

    outlier = data[
        (data[kolom] < batas_bawah)
        |
        (data[kolom] > batas_atas)
    ].copy()

    hasil = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "batas_bawah": batas_bawah,
        "batas_atas": batas_atas,
        "jumlah_outlier": len(outlier),
        "data_outlier": outlier
    }

    return hasil


def detect_outliers(data):
    """
    Deteksi outlier pada kedua variabel penelitian.
    """

    outlier_penjualan = cek_outlier(
        data,
        KOLOM_PENJUALAN
    )

    outlier_stok = cek_outlier(
        data,
        KOLOM_STOK
    )

    return {
        KOLOM_PENJUALAN: outlier_penjualan,
        KOLOM_STOK: outlier_stok
    }


# ============================================================
# 12. NORMALISASI MIN-MAX
# ============================================================

def normalize_data(data):
    """
    Normalisasi variabel penelitian menggunakan
    MinMaxScaler.
    """

    fitur = [
        KOLOM_PENJUALAN,
        KOLOM_STOK
    ]

    X = data[fitur].copy()

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=fitur,
        index=X.index
    )

    return X, X_scaled, scaler


# ============================================================
# 13. ELBOW METHOD
# ============================================================

def calculate_elbow(X_scaled):
    """
    Menghitung inertia untuk K = 2 sampai K = 10.
    """

    k_values = list(
        range(
            K_MIN,
            K_MAX + 1
        )
    )

    inertia_values = []

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )

        kmeans.fit(X_scaled)

        inertia_values.append(
            kmeans.inertia_
        )

    return (
        k_values,
        inertia_values
    )


# ============================================================
# 14. MENENTUKAN K DARI ELBOW
# ============================================================

def determine_elbow_k(
    k_values,
    inertia_values
):
    """
    Menentukan titik elbow menggunakan
    perubahan kedua inertia seperti notebook.
    """

    inertia_change = np.diff(
        inertia_values,
        n=2
    )

    elbow_index = (
        np.argmax(
            np.abs(inertia_change)
        )
        + 1
    )

    k_elbow = k_values[
        elbow_index
    ]

    return k_elbow


# ============================================================
# 15. PLOT ELBOW
# ============================================================

def plot_elbow(
    k_values,
    inertia_values
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        k_values,
        inertia_values,
        marker="o"
    )

    ax.set_xlabel(
        "Jumlah Cluster (K)"
    )

    ax.set_ylabel(
        "Inertia"
    )

    ax.set_title(
        "Elbow Method"
    )

    ax.set_xticks(
        k_values
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    return fig


# ============================================================
# 16. SILHOUETTE SCORE
# ============================================================

def calculate_silhouette(
    X_scaled,
    k_values
):
    """
    Menghitung Silhouette Score untuk setiap K.
    """

    silhouette_values = []

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )

        labels = kmeans.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels
        )

        silhouette_values.append(
            score
        )

    k_silhouette = k_values[
        np.argmax(
            silhouette_values
        )
    ]

    return (
        silhouette_values,
        k_silhouette
    )


# ============================================================
# 17. PLOT SILHOUETTE
# ============================================================

def plot_silhouette(
    k_values,
    silhouette_values
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        k_values,
        silhouette_values,
        marker="o"
    )

    ax.set_xlabel(
        "Jumlah Cluster (K)"
    )

    ax.set_ylabel(
        "Silhouette Score"
    )

    ax.set_title(
        "Silhouette Score untuk Berbagai Nilai K"
    )

    ax.set_xticks(
        k_values
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    return fig


# ============================================================
# 18. DAVIES-BOULDIN INDEX
# ============================================================

def calculate_dbi(
    X_scaled,
    k_values
):
    """
    Menghitung Davies-Bouldin Index untuk
    setiap K.
    """

    dbi_values = []

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )

        labels = kmeans.fit_predict(
            X_scaled
        )

        score = davies_bouldin_score(
            X_scaled,
            labels
        )

        dbi_values.append(
            score
        )

    k_dbi = k_values[
        np.argmin(
            dbi_values
        )
    ]

    return (
        dbi_values,
        k_dbi
    )


# ============================================================
# 19. PLOT DBI
# ============================================================

def plot_dbi(
    k_values,
    dbi_values
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        k_values,
        dbi_values,
        marker="o"
    )

    ax.set_xlabel(
        "Jumlah Cluster (K)"
    )

    ax.set_ylabel(
        "Davies-Bouldin Index"
    )

    ax.set_title(
        "Davies-Bouldin Index "
        "untuk Berbagai Nilai K"
    )

    ax.set_xticks(
        k_values
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    return fig


# ============================================================
# 20. RINGKASAN HASIL PENENTUAN K
# ============================================================

def create_k_evaluation_table(
    k_values,
    inertia_values,
    silhouette_values,
    dbi_values
):

    evaluasi_k = pd.DataFrame({
        "K": k_values,
        "Inertia": inertia_values,
        "Silhouette Score": silhouette_values,
        "DBI": dbi_values
    })

    return evaluasi_k


# ============================================================
# 21. PENENTUAN K FINAL
# ============================================================

def determine_final_k(
    k_elbow,
    k_silhouette,
    k_dbi
):
    """
    Menentukan K final menggunakan aturan voting mayoritas
    antara tiga metode evaluasi (Elbow, Silhouette, DBI),
    dengan Silhouette Score sebagai tie-breaker apabila
    ketiga metode memberikan hasil yang berbeda-beda.

    Urutan aturan:

    1. Jika Silhouette dan DBI memberikan K yang sama,
       K tersebut digunakan sebagai K final (kedua metode
       evaluasi cluster ini sepakat).

    2. Jika Silhouette dan DBI berbeda, tetapi salah satu
       dari keduanya sama dengan Elbow, maka K yang
       disepakati dua dari tiga metode tersebut yang
       dipakai sebagai K final.

    3. Jika ketiga metode memberikan K yang berbeda-beda,
       K final mengikuti Silhouette Score, karena Silhouette
       Score mempertimbangkan baik kekompakan (cohesion)
       maupun pemisahan antar cluster (separation) sehingga
       dianggap paling representatif sebagai tie-breaker.
    """

    if k_silhouette == k_dbi:

        K_FINAL = k_silhouette

        alasan_k = (
            "Silhouette Score dan "
            "Davies-Bouldin Index "
            "memberikan jumlah cluster "
            "yang sama."
        )

    elif k_elbow == k_silhouette:

        K_FINAL = k_elbow

        alasan_k = (
            "Silhouette Score dan Davies-Bouldin "
            "Index memberikan hasil yang berbeda, "
            "namun Metode Elbow dan Silhouette Score "
            "sepakat pada jumlah cluster yang sama "
            "sehingga digunakan sebagai K final."
        )

    elif k_elbow == k_dbi:

        K_FINAL = k_elbow

        alasan_k = (
            "Silhouette Score dan Davies-Bouldin "
            "Index memberikan hasil yang berbeda, "
            "namun Metode Elbow dan Davies-Bouldin "
            "Index sepakat pada jumlah cluster yang "
            "sama sehingga digunakan sebagai K final."
        )

    else:

        K_FINAL = k_silhouette

        alasan_k = (
            "Metode Elbow, Silhouette Score, dan "
            "Davies-Bouldin Index memberikan jumlah "
            "cluster yang berbeda-beda. K final "
            "ditentukan berdasarkan Silhouette Score "
            "karena mempertimbangkan kekompakan dan "
            "pemisahan antar cluster secara bersamaan."
        )

    return (
        K_FINAL,
        alasan_k
    )


# ============================================================
# 22. K-MEANS FINAL
# ============================================================

def train_final_kmeans(
    X_scaled,
    data,
    K_FINAL
):
    """
    Menjalankan K-Means final menggunakan K final.
    """

    if K_FINAL is None:

        raise ValueError(
            "K_FINAL belum dapat ditentukan. "
            "Silhouette Score dan DBI memberikan "
            "hasil K yang berbeda."
        )

    kmeans_final = KMeans(
        n_clusters=K_FINAL,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    cluster_labels = (
        kmeans_final.fit_predict(
            X_scaled
        )
    )

    data_hasil = data.copy()

    data_hasil["Cluster"] = (
        cluster_labels
    )

    return (
        kmeans_final,
        cluster_labels,
        data_hasil
    )


# ============================================================
# 23. CENTROID SETELAH NORMALISASI
# ============================================================

def get_centroid_scaled(
    kmeans_final
):

    centroid_scaled = pd.DataFrame(
        kmeans_final.cluster_centers_,
        columns=[
            KOLOM_PENJUALAN,
            KOLOM_STOK
        ]
    )

    centroid_scaled.index.name = (
        "Cluster"
    )

    return centroid_scaled


# ============================================================
# 24. CENTROID DALAM SKALA ASLI
# ============================================================

def get_centroid_original(
    kmeans_final,
    scaler
):

    centroid_original = pd.DataFrame(
        scaler.inverse_transform(
            kmeans_final.cluster_centers_
        ),
        columns=[
            KOLOM_PENJUALAN,
            KOLOM_STOK
        ]
    )

    centroid_original.index.name = (
        "Cluster"
    )

    return centroid_original


# ============================================================
# 25. EVALUASI MODEL FINAL
# ============================================================

def evaluate_final_model(
    X_scaled,
    cluster_labels,
    K_FINAL
):

    silhouette_final = silhouette_score(
        X_scaled,
        cluster_labels
    )

    dbi_final = davies_bouldin_score(
        X_scaled,
        cluster_labels
    )

    evaluasi_final = {
        "Jumlah Cluster": K_FINAL,
        "Silhouette Score": silhouette_final,
        "Davies-Bouldin Index": dbi_final
    }

    return evaluasi_final


# ============================================================
# 26. DISTRIBUSI CLUSTER
# ============================================================

def get_cluster_distribution(
    data_hasil
):

    jumlah_cluster = (
        data_hasil["Cluster"]
        .value_counts()
        .sort_index()
    )

    jumlah_cluster = (
        jumlah_cluster
        .to_frame(
            name="Jumlah Sparepart"
        )
    )

    return jumlah_cluster


# ============================================================
# 27. KATEGORI PENJUALAN OTOMATIS
# ============================================================

def create_categories(
    centroid_original,
    K_FINAL
):
    """
    Menentukan kategori berdasarkan urutan
    centroid Total Penjualan.

    K = 2:
        Rendah, Tinggi

    K = 3:
        Rendah, Sedang, Tinggi

    K = 4:
        Sangat Rendah, Rendah,
        Tinggi, Sangat Tinggi

    K = 5:
        Sangat Rendah, Rendah, Sedang,
        Tinggi, Sangat Tinggi

    K > 5:
        menggunakan Level Penjualan.
    """

    centroid_df = (
        centroid_original
        .reset_index()
        .copy()
    )

    centroid_sorted = (
        centroid_df
        .sort_values(
            by=KOLOM_PENJUALAN,
            ascending=True
        )
        .reset_index(drop=True)
    )

    kategori_berdasarkan_k = {

        2: [
            "Penjualan Rendah",
            "Penjualan Tinggi"
        ],

        3: [
            "Penjualan Rendah",
            "Penjualan Sedang",
            "Penjualan Tinggi"
        ],

        4: [
            "Penjualan Sangat Rendah",
            "Penjualan Rendah",
            "Penjualan Tinggi",
            "Penjualan Sangat Tinggi"
        ],

        5: [
            "Penjualan Sangat Rendah",
            "Penjualan Rendah",
            "Penjualan Sedang",
            "Penjualan Tinggi",
            "Penjualan Sangat Tinggi"
        ]
    }

    if K_FINAL in kategori_berdasarkan_k:

        kategori = (
            kategori_berdasarkan_k[K_FINAL]
        )

    else:

        kategori = [
            f"Penjualan Level {i + 1}"
            for i in range(K_FINAL)
        ]

    cluster_to_category = {}

    for i, row in centroid_sorted.iterrows():

        cluster = int(
            row["Cluster"]
        )

        cluster_to_category[
            cluster
        ] = kategori[i]

    return (
        cluster_to_category,
        centroid_sorted
    )


# ============================================================
# 28. RINGKASAN SETIAP CLUSTER
# ============================================================

def create_cluster_summary(
    data_hasil,
    cluster_to_category
):

    data_hasil = data_hasil.copy()

    data_hasil[
        "Kategori Penjualan"
    ] = (
        data_hasil["Cluster"]
        .map(cluster_to_category)
    )

    ringkasan_cluster = (
        data_hasil
        .groupby(
            [
                "Cluster",
                "Kategori Penjualan"
            ]
        )
        .agg(

            Jumlah_Sparepart=(
                KOLOM_NAMA,
                "count"
            ),

            Rata_Rata_Penjualan=(
                KOLOM_PENJUALAN,
                "mean"
            ),

            Rata_Rata_Stok_Akhir=(
                KOLOM_STOK,
                "mean"
            ),

            Total_Penjualan=(
                KOLOM_PENJUALAN,
                "sum"
            ),

            Total_Stok_Akhir=(
                KOLOM_STOK,
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Urutan berdasarkan tingkat penjualan
    # --------------------------------------------------------

    urutan_cluster = {}

    for i, cluster in enumerate(
        sorted(
            cluster_to_category,
            key=lambda x:
                data_hasil[
                    data_hasil["Cluster"] == x
                ][KOLOM_PENJUALAN].mean()
        )
    ):

        urutan_cluster[cluster] = i

    ringkasan_cluster[
        "Urutan"
    ] = (
        ringkasan_cluster["Cluster"]
        .map(urutan_cluster)
    )

    ringkasan_cluster = (
        ringkasan_cluster
        .sort_values("Urutan")
        .drop(columns="Urutan")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Pembulatan
    # --------------------------------------------------------

    ringkasan_cluster[
        [
            "Rata_Rata_Penjualan",
            "Rata_Rata_Stok_Akhir"
        ]
    ] = (
        ringkasan_cluster[
            [
                "Rata_Rata_Penjualan",
                "Rata_Rata_Stok_Akhir"
            ]
        ].round(2)
    )

    return (
        data_hasil,
        ringkasan_cluster
    )


# ============================================================
# 29. INTERPRETASI OTOMATIS
# ============================================================

def generate_interpretation(
    ringkasan_cluster
):
    """
    Menghasilkan interpretasi teks berdasarkan
    kategori penjualan dan karakteristik stok.
    """

    interpretasi_list = []

    for _, row in ringkasan_cluster.iterrows():

        cluster = int(
            row["Cluster"]
        )

        kategori = row[
            "Kategori Penjualan"
        ]

        jumlah = int(
            row["Jumlah_Sparepart"]
        )

        rata_penjualan = float(
            row["Rata_Rata_Penjualan"]
        )

        rata_stok = float(
            row["Rata_Rata_Stok_Akhir"]
        )

        total_penjualan = float(
            row["Total_Penjualan"]
        )

        total_stok = float(
            row["Total_Stok_Akhir"]
        )

        # ----------------------------------------------------
        # Karakteristik stok
        # ----------------------------------------------------

        if rata_stok < rata_penjualan:

            karakteristik_stok = (
                "memiliki rata-rata stok akhir "
                "yang relatif lebih rendah "
                "dibandingkan tingkat penjualannya"
            )

        elif rata_stok > rata_penjualan:

            karakteristik_stok = (
                "memiliki rata-rata stok akhir "
                "yang relatif lebih tinggi "
                "dibandingkan tingkat penjualannya"
            )

        else:

            karakteristik_stok = (
                "memiliki rata-rata stok akhir "
                "yang relatif seimbang dengan "
                "tingkat penjualannya"
            )

        # ----------------------------------------------------
        # Interpretasi kategori
        # ----------------------------------------------------

        if (
            "Tinggi" in kategori
            or "Tertinggi" in kategori
        ):

            interpretasi = (
                f"Cluster ini termasuk kelompok "
                f"{kategori.lower()} dengan rata-rata "
                f"penjualan sebesar "
                f"{rata_penjualan:.2f} unit. "
                f"Cluster {karakteristik_stok}. "
                f"Kelompok ini dapat menjadi perhatian "
                f"dalam penyediaan stok karena memiliki "
                f"tingkat penjualan yang relatif tinggi."
            )

        elif "Sedang" in kategori:

            interpretasi = (
                f"Cluster ini termasuk kelompok "
                f"{kategori.lower()} dengan rata-rata "
                f"penjualan sebesar "
                f"{rata_penjualan:.2f} unit. "
                f"Cluster {karakteristik_stok}. "
                f"Penyediaan stok dapat disesuaikan "
                f"dengan tingkat penjualan pada "
                f"kelompok ini."
            )

        elif (
            "Rendah" in kategori
            or "Terendah" in kategori
        ):

            interpretasi = (
                f"Cluster ini termasuk kelompok "
                f"{kategori.lower()} dengan rata-rata "
                f"penjualan sebesar "
                f"{rata_penjualan:.2f} unit. "
                f"Cluster {karakteristik_stok}. "
                f"Penyediaan stok dapat disesuaikan "
                f"dengan tingkat penjualan agar tidak "
                f"terjadi penumpukan stok."
            )

        else:

            interpretasi = (
                f"Cluster ini termasuk "
                f"{kategori.lower()} dengan rata-rata "
                f"penjualan sebesar "
                f"{rata_penjualan:.2f} unit dan rata-rata "
                f"stok akhir sebesar "
                f"{rata_stok:.2f} unit. "
                f"Karakteristik cluster menunjukkan "
                f"tingkat penjualan yang berbeda "
                f"dibandingkan cluster lainnya."
            )

        interpretasi_list.append({

            "Cluster": cluster,

            "Kategori Penjualan": kategori,

            "Jumlah Sparepart": jumlah,

            "Rata-rata Penjualan": rata_penjualan,

            "Rata-rata Stok Akhir": rata_stok,

            "Total Penjualan": total_penjualan,

            "Total Stok Akhir": total_stok,

            "Interpretasi": interpretasi
        })

    return pd.DataFrame(
        interpretasi_list
    )


# ============================================================
# 30. SCATTER PLOT HASIL CLUSTERING
# ============================================================

def plot_clustering_result(
    data_hasil,
    centroid_original,
    cluster_to_category
):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for cluster in sorted(
        data_hasil["Cluster"].unique()
    ):

        subset = data_hasil[
            data_hasil["Cluster"] == cluster
        ]

        kategori = cluster_to_category.get(
            cluster,
            f"Cluster {cluster}"
        )

        ax.scatter(
            subset[KOLOM_PENJUALAN],
            subset[KOLOM_STOK],
            label=(
                f"Cluster {cluster} - "
                f"{kategori}"
            ),
            s=60
        )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    ax.scatter(
        centroid_original[
            KOLOM_PENJUALAN
        ],

        centroid_original[
            KOLOM_STOK
        ],

        marker="X",

        s=250,

        label="Centroid"
    )

    ax.set_xlabel(
        "Total Penjualan"
    )

    ax.set_ylabel(
        "Stok Akhir"
    )

    ax.set_title(
        "Hasil Clustering Sparepart Motor "
        "Menggunakan K-Means"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    return fig


# ============================================================
# 31. EXPORT HASIL
# ============================================================

def export_results(
    data_hasil,
    ringkasan_cluster,
    centroid_original,
    cluster_to_category,
    output_dir="output"
):
    """
    Menyimpan hasil clustering ke CSV.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Hasil clustering
    # --------------------------------------------------------

    file_hasil = os.path.join(
        output_dir,
        "hasil_clustering.csv"
    )

    data_hasil.to_csv(
        file_hasil,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Ringkasan cluster
    # --------------------------------------------------------

    file_ringkasan = os.path.join(
        output_dir,
        "ringkasan_cluster.csv"
    )

    ringkasan_cluster.to_csv(
        file_ringkasan,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    centroid_export = (
        centroid_original
        .reset_index()
        .copy()
    )

    centroid_export[
        "Kategori Penjualan"
    ] = (
        centroid_export["Cluster"]
        .map(cluster_to_category)
    )

    file_centroid = os.path.join(
        output_dir,
        "centroid_cluster.csv"
    )

    centroid_export.to_csv(
        file_centroid,
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "hasil_clustering": file_hasil,
        "ringkasan_cluster": file_ringkasan,
        "centroid_cluster": file_centroid
    }


# ============================================================
# 32. FUNGSI UTAMA ANALISIS
# ============================================================

def run_analysis(
    file,
    export=False,
    output_dir="output"
):
    """
    Menjalankan seluruh proses analisis K-Means
    sesuai urutan notebook penelitian.

    Alur:

    Load Data
        ↓
    Data Understanding
        ↓
    Validasi Kolom
        ↓
    Data Selection
        ↓
    Konversi Numerik
        ↓
    Data Cleaning
        ↓
    Cek Duplikasi
        ↓
    Statistik
        ↓
    Visualisasi Awal
        ↓
    Deteksi Outlier
        ↓
    Normalisasi
        ↓
    Elbow
        ↓
    Silhouette
        ↓
    DBI
        ↓
    Penentuan K
        ↓
    K-Means Final
        ↓
    Centroid
        ↓
    Evaluasi Final
        ↓
    Distribusi Cluster
        ↓
    Kategori
        ↓
    Ringkasan
        ↓
    Visualisasi Hasil
        ↓
    Interpretasi
        ↓
    Export
    """

    # ========================================================
    # CELL 2 - LOAD DATA
    # ========================================================

    df = load_data(file)


    # ========================================================
    # CELL 3 - DATA UNDERSTANDING
    # ========================================================

    understanding = (
        data_understanding(df)
    )


    # ========================================================
    # CELL 4 - KONFIGURASI
    # ========================================================

    konfigurasi = {
        "Nama Barang": KOLOM_NAMA,
        "Total Penjualan": KOLOM_PENJUALAN,
        "Stok Akhir": KOLOM_STOK
    }


    # ========================================================
    # CELL 5 - VALIDASI KOLOM
    # ========================================================

    validate_columns(df)


    # ========================================================
    # CELL 6 - DATA SELECTION
    # ========================================================

    data = data_selection(df)


    # ========================================================
    # CELL 7 - KONVERSI NUMERIK
    # ========================================================

    data = convert_numeric(data)


    # ========================================================
    # CELL 8 - DATA CLEANING
    # ========================================================

    data, cleaning_info = (
        data_cleaning(data)
    )


    # ========================================================
    # CELL 9 - CEK DUPLIKASI NAMA
    # ========================================================

    duplicate_info = (
        check_duplicate_names(data)
    )


    # ========================================================
    # CELL 10 - STATISTIK
    # ========================================================

    statistik = (
        variable_statistics(data)
    )


    # ========================================================
    # CELL 11 - VISUALISASI AWAL
    # ========================================================

    fig_data_awal = (
        plot_data_awal(data)
    )


    # ========================================================
    # CELL 12 - OUTLIER
    # ========================================================

    outlier_info = (
        detect_outliers(data)
    )


    # ========================================================
    # CELL 13 - NORMALISASI
    # ========================================================

    X, X_scaled, scaler = (
        normalize_data(data)
    )


    # ========================================================
    # CELL 14 - ELBOW
    # ========================================================

    (
        k_values,
        inertia_values
    ) = calculate_elbow(
        X_scaled
    )

    k_elbow = determine_elbow_k(
        k_values,
        inertia_values
    )

    fig_elbow = plot_elbow(
        k_values,
        inertia_values
    )


    # ========================================================
    # CELL 15 - SILHOUETTE
    # ========================================================

    (
        silhouette_values,
        k_silhouette
    ) = calculate_silhouette(
        X_scaled,
        k_values
    )

    fig_silhouette = plot_silhouette(
        k_values,
        silhouette_values
    )


    # ========================================================
    # CELL 16 - DBI
    # ========================================================

    (
        dbi_values,
        k_dbi
    ) = calculate_dbi(
        X_scaled,
        k_values
    )

    fig_dbi = plot_dbi(
        k_values,
        dbi_values
    )


    # ========================================================
    # CELL 17 - TABEL EVALUASI K
    # ========================================================

    evaluasi_k = (
        create_k_evaluation_table(
            k_values,
            inertia_values,
            silhouette_values,
            dbi_values
        )
    )


    # ========================================================
    # CELL 18 - PENENTUAN K FINAL
    # ========================================================

    (
        K_FINAL,
        alasan_k
    ) = determine_final_k(
        k_elbow,
        k_silhouette,
        k_dbi
    )


    # ========================================================
    # K-MEANS FINAL
    # ========================================================

    (
        kmeans_final,
        cluster_labels,
        data_hasil
    ) = train_final_kmeans(
        X_scaled,
        data,
        K_FINAL
    )


    # ========================================================
    # CELL 19 - CENTROID NORMALISASI
    # ========================================================

    centroid_scaled = (
        get_centroid_scaled(
            kmeans_final
        )
    )


    # ========================================================
    # CELL 20 - CENTROID SKALA ASLI
    # ========================================================

    centroid_original = (
        get_centroid_original(
            kmeans_final,
            scaler
        )
    )


    # ========================================================
    # CELL 21 - EVALUASI FINAL
    # ========================================================

    evaluasi_final = (
        evaluate_final_model(
            X_scaled,
            cluster_labels,
            K_FINAL
        )
    )


    # ========================================================
    # CELL 22 - DISTRIBUSI CLUSTER
    # ========================================================

    jumlah_cluster = (
        get_cluster_distribution(
            data_hasil
        )
    )


    # ========================================================
    # CELL 23 - KATEGORI + RINGKASAN
    # ========================================================

    (
        cluster_to_category,
        centroid_sorted
    ) = create_categories(
        centroid_original,
        K_FINAL
    )

    (
        data_hasil,
        ringkasan_cluster
    ) = create_cluster_summary(
        data_hasil,
        cluster_to_category
    )


    # ========================================================
    # CELL 24 - VISUALISASI HASIL
    # ========================================================

    fig_clustering = (
        plot_clustering_result(
            data_hasil,
            centroid_original,
            cluster_to_category
        )
    )


    # ========================================================
    # CELL 25 - INTERPRETASI OTOMATIS
    # ========================================================

    interpretasi_cluster = (
        generate_interpretation(
            ringkasan_cluster
        )
    )


    # ========================================================
    # CELL 26 - EXPORT
    # ========================================================

    file_export = None

    if export:

        file_export = export_results(
            data_hasil,
            ringkasan_cluster,
            centroid_original,
            cluster_to_category,
            output_dir
        )


    # ========================================================
    # HASIL AKHIR
    # ========================================================

    hasil = {

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        "data_asli": df,

        "data_hasil": data_hasil,

        "jumlah_data": len(data_hasil),

        "konfigurasi": konfigurasi,

        "data_understanding": understanding,

        "cleaning_info": cleaning_info,

        "duplicate_info": duplicate_info,

        "statistik": statistik,

        "outlier_info": outlier_info,


        # ----------------------------------------------------
        # Normalisasi
        # ----------------------------------------------------

        "X": X,

        "X_scaled": X_scaled,

        "scaler": scaler,


        # ----------------------------------------------------
        # Penentuan K
        # ----------------------------------------------------

        "k_values": k_values,

        "inertia_values": inertia_values,

        "silhouette_values": silhouette_values,

        "dbi_values": dbi_values,

        "evaluasi_k": evaluasi_k,

        "k_elbow": k_elbow,

        "k_silhouette": k_silhouette,

        "k_dbi": k_dbi,

        "K_FINAL": K_FINAL,

        "alasan_k": alasan_k,


        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "kmeans_final": kmeans_final,

        "cluster_labels": cluster_labels,


        # ----------------------------------------------------
        # Centroid
        # ----------------------------------------------------

        "centroid_scaled": centroid_scaled,

        "centroid_original": centroid_original,

        "centroid_sorted": centroid_sorted,


        # ----------------------------------------------------
        # Evaluasi
        # ----------------------------------------------------

        "evaluasi_final": evaluasi_final,


        # ----------------------------------------------------
        # Cluster
        # ----------------------------------------------------

        "jumlah_cluster": jumlah_cluster,

        "cluster_to_category": cluster_to_category,

        "ringkasan_cluster": ringkasan_cluster,

        "interpretasi_cluster": interpretasi_cluster,


        # ----------------------------------------------------
        # Visualisasi
        # ----------------------------------------------------

        "fig_data_awal": fig_data_awal,

        "fig_elbow": fig_elbow,

        "fig_silhouette": fig_silhouette,

        "fig_dbi": fig_dbi,

        "fig_clustering": fig_clustering,


        # ----------------------------------------------------
        # Export
        # ----------------------------------------------------

        "file_export": file_export
    }


    return hasil