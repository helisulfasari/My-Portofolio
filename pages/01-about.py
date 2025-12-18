import streamlit as st

st.title("📌 Tentang Saya")

st.write(
    """
    Halo! Nama saya **Helis**, dari Banyuwangi, Indonesia.  
    Saya sedang fokus mendalami ** Bidang data dengan Python sebagai bahasanya**, terutama:
    - Fundamental Python
    - Python for data analyst (pandas, numpy)
    - Visualisasi with python (matplotlib, seaborn)
    - Algoritma (recursion, divide & conquer)
    - Struktur project yang rapi dan profesional
    - Git, GitHub dan streamlit 
    """
)

st.subheader("🎯 Tujuan Belajar")
st.write(
    """
    Saya ingin membangun fondasi kuat dalam pemrograman Python dan data,
    dimulai dari mengerjakan assignment - assignment yang bisa membantu saya memahami konsep secara praktis.
    Mencari insight - insight yang mungkin ditemukan dalam dataset untuk mempertajam konsep analisa.
    Portfolio ini adalah bagian dari perjalanan belajar saya.
    """
)

st.subheader("📚 Perjalanan Belajar")
st.write(
    """
    - Mulai belajar Python dari dasar  
    - Membangun project kecil untuk memahami konsep  
    - Belajar menggunakan Streamlit  
    - Membuat struktur folder profesional (src, pages, assets, data)  
    - Menyiapkan project untuk GitHub dan deployment  
    """
)

st.subheader("🧰 Tools & Teknologi")
st.write(
    """
    - VS Code  
    - Python Virtual Environment  
    - Streamlit  
    - Git & GitHub  
    - Pandas, Numpy  
    - Matplotlib, seaborn
    """
)

st.info("Terimakasih sudah mengunjungi laman ini!")
col1, col2 = st.columns([1, 1])

with col1:
    st.page_link("app.py", label="⬅️ Previous", icon="🏠")

with col2:
    st.page_link("pages/03-telco_overview.py", label="Next ➡️", icon="📊")
