import streamlit as st
import pandas as pd
st.markdown("""
## 📊 Insight Bisnis dari Model XGBoost

Model XGBoost menunjukkan performa yang kuat dalam mendeteksi pelanggan yang berpotensi churn:

- **Recall 0.731** → Model berhasil menangkap 73% pelanggan yang benar-benar churn.
- **ROC AUC 0.819** → Model memiliki kemampuan yang baik dalam membedakan pelanggan churn vs tidak churn.

---

## 🔍 Fitur-Fitur yang Paling Mempengaruhi Churn

Berdasarkan feature importance dari XGBoost, berikut adalah faktor-faktor utama yang mendorong churn:

| **Fitur**                       | **Importance** | **Insight Bisnis** |
|--------------------------------|----------------|---------------------|
| Contract_Month-to-month        | 0.4157         | Pelanggan dengan kontrak bulanan sangat berisiko churn. Mereka cenderung tidak terikat dan mudah berpindah. |
| Contract_Two year              | 0.1511         | Kontrak panjang cenderung menurunkan risiko churn. |
| PaymentMethod_Electronic check | 0.0481         | Metode ini sering dikaitkan dengan churn — mungkin karena fleksibel atau tidak otomatis. |
| MonthlyCharges                 | 0.0411         | Semakin mahal tagihan, semakin besar kemungkinan pelanggan churn. |
| Tenure                         | 0.0400         | Pelanggan yang sudah lama cenderung lebih loyal. |

---

## 🎯 Rekomendasi Strategi Retensi

Berdasarkan insight di atas, berikut adalah rekomendasi yang dapat diterapkan oleh tim bisnis:

### 1. Targetkan Pelanggan Kontrak Bulanan
- Buat program loyalitas atau diskon untuk pelanggan dengan kontrak bulanan.
- Tawarkan upgrade ke kontrak tahunan dengan benefit tambahan.

### 2. Segmentasi Berdasarkan Metode Pembayaran
- Identifikasi pelanggan dengan metode **Electronic Check** dan berikan edukasi atau penawaran khusus.
- Dorong penggunaan metode pembayaran otomatis untuk meningkatkan retensi.

### 3. Intervensi Berdasarkan Tagihan Bulanan
- Untuk pelanggan dengan tagihan tinggi, berikan opsi paket yang lebih fleksibel atau diskon loyalitas.

### 4. Fokus pada Pelanggan Baru
- Pelanggan dengan tenure rendah lebih berisiko churn.
- Buat onboarding yang lebih baik dan komunikasi intensif di 3 bulan pertama.

### 5. Gunakan Model untuk Prediksi Churn Secara Berkala
- Integrasikan model ke sistem CRM untuk memantau pelanggan berisiko.
- Lakukan evaluasi bulanan dan tindak lanjut otomatis.

---

Halaman ini dirancang untuk membantu tim bisnis memahami faktor-faktor utama churn dan mengambil keputusan strategis berbasis data.  
Model XGBoost memberikan landasan yang kuat untuk intervensi retensi yang lebih tepat sasaran.
""")

st.page_link("pages/05-Model.py", label="⬅️ Previous", icon="📊")