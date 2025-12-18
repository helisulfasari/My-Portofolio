import streamlit as st
import pandas as pd
from src.data_loader import load_telco_data
from src.processing import preprocess_telco
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧹 Telco Customer Churn — Data Preparation")
df_raw = load_telco_data()
# ============================
# ✅ Missing Value Sebelum Cleaning
# ============================

st.subheader("📌 Missing Value Sebelum Cleaning")

missing_counts = df_raw.isnull().sum()
missing_percent = (missing_counts / len(df_raw)) * 100

missing_table = pd.DataFrame({
    "Attribute": missing_counts.index,
    "Missing Values": missing_counts.values,
    "Percentage (%)": missing_percent.round(2)
})

st.dataframe(missing_table)

st.markdown("""
### 📝 Catatan:
- Hampir seluruh kolom pada dataset ini memiliki missing value kecuali pada **Total charge dan churn** .
- Missing value dapat menyebabkan bias dan mengganggu proses training model.
- Oleh karena itu, kita akan melakukan penanganan sebagai berikut:
  - **Kolom kategori** → diisi dengan *mode* (nilai yang paling sering muncul).
  - **Kolom numerik** → diisi dengan *mean* (rata-rata).
  - **customerID** → dihapus karena tidak relevan dengan prediksi churn.
""")
st.markdown("---")

# ============================
# ✅ Cleaning Data
# ============================

st.subheader("✅ Cleaning Data")
df_clean = preprocess_telco(df_raw.copy())

st.write("""
Langkah-langkah yang dilakukan:
- Menghapus kolom **customerID**
- Mengganti berbagai bentuk missing value (NA, Unknown, Null, dsb)
- Mengisi missing value kategori dengan **mode**
- Mengisi missing value numerik dengan **mean**
- Mengonversi kolom **TotalCharges** menjadi numerik
""")

# ============================
# ✅ Dataset Setelah Cleaning
# ============================

st.subheader("📁 Dataset Setelah Cleaning")
st.dataframe(df_clean.head())

# ============================
# ✅ Missing Value Setelah Cleaning
# ============================

st.subheader("📉 Missing Value Setelah Cleaning")
st.dataframe(df_clean.isnull().sum())
st.markdown("""
### 📝 Catatan:
Bisa kita lihat dari table diatas nilai mising value 0 yang artinya sudah tidak ditemukan 
missing value didataset. Missing value sudah berhasi di handling dengan baik.
""")
st.subheader("📊 Ringkasan Duplikasi & Unique Value per Kolom")

summary = []

for col in df_raw.columns:
    dup_count = df_raw[col].duplicated().sum()
    unique_vals = df_raw[col].dropna().unique()
    unique_count = len(unique_vals)

    # tampilkan max 5 unique values
    if unique_count <= 5:
        unique_preview = ", ".join([str(val) for val in unique_vals])
    else:
        unique_preview = f"{unique_count} unique values"

    summary.append({
        "Attribute": col,
        "Duplicated Values": dup_count,
        "Unique Count": unique_count,
        "Unique Preview": unique_preview
    })

summary_df = pd.DataFrame(summary)
st.dataframe(summary_df)

st.markdown("""
### 📝 Catatan:
- Duplikasi per kolom **tidak dianggap masalah**, karena setiap kolom memang memiliki kategori yang terbatas.
- Yang kita cek adalah **duplikasi baris**, karena itu dapat menyebabkan bias dalam model.
""")

st.markdown("---")

st.subheader("📦 Visualisasi Outlier")

fig, ax = plt.subplots(1, 3, figsize=(14, 5))  # lebih lebar dan tinggi

sns.boxplot(y=df_raw['Tenure'], ax=ax[0])
ax[0].set_title("Tenure", fontsize=12)
ax[0].tick_params(labelsize=10)

sns.boxplot(y=df_raw['MonthlyCharges'], ax=ax[1])
ax[1].set_title("Monthly Charges", fontsize=12)
ax[1].tick_params(labelsize=10)

sns.boxplot(y=df_raw['TotalCharges'], ax=ax[2])
ax[2].set_title("Total Charges", fontsize=12)
ax[2].tick_params(labelsize=10)

plt.tight_layout(pad=2.0)  # beri jarak antar plot

st.pyplot(fig)

st.markdown("""
### 📝 Catatan:
- Outlier pada kolom **TotalCharges** tidak dihapus karena secara bisnis nilai tersebut **masih wajar**.
- Pelanggan dengan tagihan tinggi biasanya memiliki:
  - masa langganan panjang,
  - layanan tambahan lebih banyak,
  - atau paket premium.
- Outlier seperti ini **bukan kesalahan data**, tetapi justru memberikan insight penting.
- Oleh karena itu, outlier **tidak dihapus**, Kolom **TotalCharges** merupakan hasil perkalian antara **Tenure** dan **MonthlyCharges**.
- Karena informasi ini sudah tercakup dalam dua kolom lainnya, maka **TotalCharges tidak dianalisis lebih lanjut**.
- Kolom ini akan tetap disimpan untuk referensi, tetapi **tidak digunakan sebagai fitur utama dalam modeling**.
""")
st.markdown("---")
st.subheader("📊 Distribusi Kategorikal")

categorical_cols = [
    'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
]

for col in categorical_cols:
    st.write(f"### {col}")

    fig, ax = plt.subplots(figsize=(6, 3))

    sns.countplot(
        data=df_clean,
        x=col,
        ax=ax,
        palette="Set2"   # warna modern
    )

    ax.set_title(f"Distribusi {col}", fontsize=11)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel("Jumlah", fontsize=10)

    ax.tick_params(axis='x', labelsize=9, rotation=20)
    ax.tick_params(axis='y', labelsize=9)

    # grid halus
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # label jumlah
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width()/2,
            p.get_height(),
            f'{int(p.get_height())}',
            ha='center', va='bottom',
            fontsize=9
        )

    st.pyplot(fig)
    

st.markdown("""   
### 📝 Insight Distribusi Kategorikal
- **Gender & Partner**: Distribusi pelanggan berdasarkan gender dan status partner relatif seimbang, sehingga tidak menunjukkan bias awal terhadap churn.
- **Dependents**: Mayoritas pelanggan tidak memiliki tanggungan, menunjukkan dominasi segmen pelanggan yang lebih mandiri.
- **Contract**: Mayoritas pelanggan menggunakan kontrak bulanan, kelompok ini biasanya paling rentan churn.
- **Paperless Billing**: Sebagian besar pelanggan menggunakan metode penagihan elektronik, menunjukkan adopsi digital yang cukup tinggi. 
- **PaymentMethod**: Electronic Check mendominasi dan sering dikaitkan dengan churn lebih tinggi.
- **SeniorCitizen**: Proporsi pelanggan lansia kecil, sehingga efeknya terhadap churn mungkin terbatas.
- **Churn**: Dataset imbalance — pelanggan churn lebih sedikit, perlu penanganan saat modeling.
""")
st.markdown("---")
st.subheader("📈 Distribusi Numerik")

numeric_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']

for col in numeric_cols:
    st.write(f"### {col}")

    fig, ax = plt.subplots(figsize=(6, 3))

    sns.histplot(
        df_clean[col],
        kde=True,
        ax=ax,
        color="#4C72B0"   # warna biru modern
    )

    ax.set_title(f"Distribusi {col}", fontsize=11)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel("Frekuensi", fontsize=10)

    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    st.pyplot(fig)
    

st.markdown("""  
### 📝 Insight Distribusi Numerik
- **Tenure**: Banyak pelanggan baru. Pelanggan baru cenderung churn lebih tinggi.
- **MonthlyCharges**: Variasi harga cukup lebar, menunjukkan banyaknya paket layanan.
- **TotalCharges**: Polanya mengikuti Tenure × MonthlyCharges, sehingga tidak dianalisis lebih lanjut.
""")

st.markdown("---")

import phik
from phik.report import plot_correlation_matrix

st.subheader("🔗 Korelasi Fitur (Phik) – Numerik & Kategorikal")

# hitung phik matrix
phik_matrix = df_clean.phik_matrix()

fig = plt.figure(figsize=(12, 10))
plot_correlation_matrix(
    phik_matrix.values,
    x_labels=phik_matrix.columns,
    y_labels=phik_matrix.index,
    vmin=0, vmax=1,
    color_map="YlGnBu"
)
plt.title("Phik Correlation Matrix", fontsize=14)

st.pyplot(plt)

st.markdown("""
### 🧠 Insight Korelasi (Phik)

#### **🔍 Potensi Multikolinearitas**
- **TotalCharges vs MonthlyCharges** → korelasi kuat (0.76).  
  TotalCharges dipengaruhi langsung oleh MonthlyCharges.
- **TotalCharges vs Tenure** → korelasi sangat kuat (0.84).  
  TotalCharges = Tenure × MonthlyCharges, sehingga fitur ini redundant.
- **Contract vs Tenure** → korelasi moderate (0.67).  
  Pelanggan dengan kontrak panjang cenderung memiliki tenure lebih lama.
- **Contract vs TotalCharges** → korelasi moderate (0.51).  
  Kontrak jangka panjang menghasilkan total biaya yang lebih besar.
- **Dependents vs Partner** → korelasi moderate (0.65).  
  Pelanggan yang memiliki partner cenderung juga memiliki dependents.

#### **🎯 Korelasi terhadap Target (Churn)**
- **PaymentMethod vs Churn** → korelasi rendah (0.45).  
  Electronic Check sering muncul pada pelanggan yang churn.
- **Tenure vs Churn** → korelasi rendah (0.47).  
  Pelanggan baru (tenure rendah) lebih rentan churn.
- **Gender vs Churn** → tidak memiliki korelasi berarti.  
  Gender bukan faktor penentu churn.

#### ✅ **Kesimpulan**
- **TotalCharges** sebaiknya dipertimbangkan untuk di-drop karena multikolinearitas tinggi.  
- Fitur yang paling relevan untuk churn: **Contract, Tenure, PaymentMethod, MonthlyCharges**.  
- Fitur dengan pengaruh kecil: **Gender, SeniorCitizen**.
""")
st.markdown("---")
st.subheader("📐 Variance Inflation Factor (VIF)")

vif_list = [
    'SeniorCitizen','Partner','Dependents','Tenure',
    'MonthlyCharges','TotalCharges','Contract','PaymentMethod',
    'PaperlessBilling','Gender'
]

# dummy encoding untuk VIF
df_vif = pd.get_dummies(df_clean[vif_list], drop_first=True).astype(int)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = df_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(df_vif.values, i)
    for i in range(df_vif.shape[1])
]

st.dataframe(vif_data)
st.markdown("""
### 🧮 Kenapa Kita Menghitung VIF?

Sebelum membuat model prediksi churn, kita perlu memastikan bahwa setiap fitur
yang kita gunakan benar-benar memberikan informasi baru. Jika ada dua fitur
yang isinya mirip atau saling mewakili, model bisa menjadi tidak stabil.
Kondisi ini disebut **multikolinearitas**.

**Variance Inflation Factor (VIF)** membantu kita mendeteksi fitur mana yang
terlalu mirip satu sama lain. Semakin tinggi nilai VIF, semakin besar risiko
fitur tersebut tidak memberikan informasi baru.

### 🎯 Kenapa Menggunakan Dummy Encoding?

Sebagian fitur kita berbentuk kategori (misalnya: Contract, PaymentMethod).
Model statistik seperti VIF hanya bisa membaca angka, bukan teks.  
Supaya tidak mengubah dataset asli, kita membuat **salinan khusus** yang sudah
diubah menjadi angka menggunakan *dummy encoding*.

Dengan cara ini:
- dataset asli tetap bersih untuk analisis lain  
- kita bisa menghitung VIF tanpa risiko data leakage  
- hasil VIF tetap akurat dan aman digunakan  

### ✅ Tujuan Akhir

Dengan VIF, kita bisa:
- menemukan fitur yang redundant  
- menentukan fitur mana yang perlu di-drop  
- memastikan model nanti lebih stabil dan akurat  
""")
st.markdown("---")
def plot_churn_rate(df, col, ax):
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()

    sns.barplot(data=churn_rate, x=col, y='Churn', ax=ax, palette="Set2")

    # label persentase
    for index, row in churn_rate.iterrows():
        ax.text(
            index,
            row['Churn'] + 0.5,
            f"{row['Churn']:.1f}%",
            ha='center',
            fontsize=9
        )

    ax.set_title(f"Churn Rate by {col}", fontsize=11)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel(col)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
st.subheader("📉 Churn Rate per Segmen")

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

cols_to_plot = ['Gender', 'Contract', 'PaymentMethod', 'SeniorCitizen', 'Partner']

for i, col in enumerate(cols_to_plot):
    plot_churn_rate(df_clean, col, axes[i])

plt.tight_layout()
st.pyplot(fig)
df_clean['tenure_bin'] = pd.cut(
    df_clean['Tenure'],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=['0–12','13–24','25–36','37–48','49–60','61–72'],
    include_lowest=True
)

fig, ax = plt.subplots(figsize=(6, 4))
plot_churn_rate(df_clean, 'tenure_bin', ax)
st.pyplot(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(data=df_clean, x='Churn', y='MonthlyCharges', ax=ax[0], palette="Set2")
ax[0].set_title("Monthly Charges vs Churn")
ax[0].grid(axis='y', linestyle='--', alpha=0.3)

sns.boxplot(data=df_clean, x='Churn', y='TotalCharges', ax=ax[1], palette="Set2")
ax[1].set_title("Total Charges vs Churn")
ax[1].grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
st.markdown("""
### 🎯 Kenapa Kita Menganalisis Churn per Segmen?

Sebelum membuat model prediksi churn, kita perlu memahami pola perilaku pelanggan
di setiap segmen. Tujuannya agar model tidak bias dan tetap mencerminkan kondisi
bisnis yang sebenarnya.

Dengan melihat churn rate berdasarkan kategori seperti kontrak, metode pembayaran,
tenure, dan lainnya, kita bisa:

- mengetahui segmen mana yang paling rentan churn  
- memahami faktor perilaku yang tidak terlihat dari korelasi saja  
- menghindari pemilihan fitur yang salah  
- memastikan model tidak hanya akurat secara statistik, tetapi juga relevan secara bisnis  

Analisis ini menjadi jembatan antara EDA dan pemilihan fitur untuk modeling.
""")
st.markdown("""
### 🧠 Insight Churn per Segmen

- **Gender** → Laki-laki dan perempuan memiliki peluang churn yang hampir sama.  
  Gender bukan faktor penentu churn.

- **Contract** → Pelanggan dengan **kontrak bulanan** memiliki churn rate tertinggi.  
  Kontrak jangka panjang jauh lebih stabil.

- **Payment Method** → Metode **Electronic Check** memiliki churn rate paling tinggi.  
  Metode pembayaran manual cenderung lebih berisiko.

- **Monthly Charges** → Pelanggan dengan tagihan bulanan **di atas $60** lebih mudah churn.  
  Harga tinggi menjadi faktor risiko.

- **Total Charges** → Pelanggan churn cenderung memiliki total tagihan rendah,  
  karena mereka biasanya baru berlangganan (tenure pendek).  
  Ini sejalan dengan hubungan TotalCharges = MonthlyCharges × Tenure.

- **Tenure** → Pelanggan baru (0–12 bulan) memiliki churn rate tertinggi.  
  Semakin lama pelanggan bertahan, semakin kecil risiko churn.

- **Senior Citizen** → Pelanggan lansia memiliki churn rate lebih tinggi dibanding non-lansia.

- **Partner** → Pelanggan tanpa partner lebih mudah churn.  
  Ini bisa menunjukkan stabilitas finansial atau preferensi layanan.
""")
st.markdown("---")
st.header("🧩 Feature Selection")

st.markdown("""
Pada tahap ini kita memilih fitur mana yang akan digunakan untuk proses modeling.
Keputusan ini didasarkan pada hasil EDA sebelumnya seperti korelasi, Phik, VIF,
dan analisis churn per segmen.

Tujuannya adalah:
- menghindari multikolinearitas,
- menghapus fitur yang tidak informatif,
- menjaga model tetap stabil dan mudah dijelaskan.
""")

st.subheader("📌 Fitur yang Dihapus & Alasannya")

st.markdown("""
### 1. **TotalCharges**
- Sangat berkorelasi dengan Tenure dan MonthlyCharges.
- VIF tinggi.
- Redundant karena TotalCharges = Tenure × MonthlyCharges.

### 2. **Gender**
- Tidak memiliki korelasi dengan churn.
- Churn rate laki-laki dan perempuan hampir sama.
- Tidak menambah informasi untuk model.

### 3. **tenure_bin** (jika dibuat)
- Hanya untuk visualisasi EDA.
- Tidak digunakan untuk modeling karena Tenure asli lebih informatif.

### 4. **SeniorCitizen** (opsional)
- Korelasi rendah.
- Pengaruh kecil terhadap churn.
- Bisa dipertahankan jika ingin insight bisnis, tapi tidak wajib untuk model.
""")
st.subheader("✅ Fitur Akhir yang Digunakan untuk Modeling")

drop_cols = ['TotalCharges', 'Gender', 'tenure_bin']  # senior citizen opsional

df_model = df_clean.drop(columns=drop_cols)

st.write(df_model.head())
st.success(f"Jumlah fitur akhir: {df_model.shape[1]}")
drop_cols = ['TotalCharges', 'Gender', 'tenure_bin', 'SeniorCitizen']
st.markdown("""
### 🎯 Kesimpulan Feature Selection
Dengan menghapus fitur yang redundant atau tidak informatif, kita memastikan bahwa
model nantinya:
- lebih stabil,
- tidak bias,
- lebih mudah dijelaskan,
- dan memiliki performa yang lebih baik.

Selanjutnya kita akan masuk ke tahap **Model Preparation** untuk melakukan
train-test split, encoding, dan scaling sebelum membangun model prediksi churn.
""")
st.session_state["df_model"] = df_model
df_model.to_pickle("data/df_model.pkl")

# Navigasi
col1, col2 = st.columns([1,1])

with col1:
    st.page_link("pages/03-telco_overview.py", label="⬅️ Previous", icon="📊")

with col2:
    st.page_link("pages/05-Model.py", label="Next ➡️", icon="📊")