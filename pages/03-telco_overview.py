import streamlit as st
import pandas as pd
from src.data_loader import load_telco_data

st.title("📊 Telco Customer Churn — Overview")

st.write("""
Project ini bertujuan untuk memahami faktor-faktor yang menyebabkan pelanggan melakukan **churn** 
(berhenti berlangganan) dan bagaimana model machine learning dapat membantu mendeteksi pelanggan berisiko.
""")

# Load dataset
df = load_telco_data()

st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

st.subheader("ℹ️ Informasi Dataset")
st.write(f"Jumlah baris: {df.shape[0]}")
st.write(f"Jumlah kolom: {df.shape[1]}")

st.write("""
Dataset ini berisi informasi pelanggan seperti:
- Tenure (lama berlangganan)
- Contract type (type langganan, bulanan/tahunan)
- Monthly charges (tagihan per bulan)
- Internet service (paket internet yang diikuti)
- Payment method (matode pembayaran)
- Status churn (Yes/No)
""")

# Navigasi
col1, col2 = st.columns([1,1])

with col1:
    st.page_link("pages/01-about.py", label="⬅️ Previous", icon="📊")

with col2:
    st.page_link("pages/04-telco_preparation.py", label="Next ➡️", icon="📊")
