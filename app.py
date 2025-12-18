import streamlit as st
from PIL import Image

#konfigurasi halaman
st.set_page_config(page_title="My Portfolio",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state = 'expanded')
# --- HEADER ---
st.title(" 👋 Selamat Datang di Portofolio Saya")
st.write ("Halo, saya **Helis**, seorang Data Analyst dan Data Science enthusiast yang sedang mempelajari bagaimana membuat system data end to end yang baik")
# --- FOTO PROFIL ---
col1, col2 = st.columns([1, 3])

with col1:
    try:
        img = Image.open("assets/personal.PNG")  # ganti dengan foto kamu kalau ada
        st.image(img, width=200)
    except:
        st.info("assets/batu.PNG")
with col2:
    st.subheader("Tentang Portfolio Ini")
    st.write(
        """
        Portfolio ini dibuat menggunakan **Python**,**GitHub** dan **Streamlit**.
        Kamu bisa menemukan:
        - Halaman About (profil lengkap)
        - Modul python
        - Visualisasi 
        - Pembahasan temuan - temuan pada dataset
        """
    )
# --- SKILLS ---
st.subheader("🛠️ Skill Utama")
st.write(
    """
    - Python (Fundamental)
    - Streamlit (Web App)
    - Data Analysis (Pandas, Numpy,SQL)
    - Visualization (Matplotlib,Seaborn)
    - Git & GitHub
    - Problem Solving & Algorithmic Thinking
    """
)
# --- CTA ---
st.success("Silakan pilih **Next Page** untuk buka halaman berikutnya, atau pilih  menu disidebar.")

st.page_link("pages/01-about.py", label="➡️ Next Page", icon="📄")

#st.image("assets/batu.png", caption="Foto Profil", width=200)