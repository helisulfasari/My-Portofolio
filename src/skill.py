import streamlit as st
import altair as alt
import pandas as pd

@st.cache_data
def get_skills_data():
    return {
        'Skills': ['🐍 Python', '🗄️ SQL', '🌐 Streamlit', '🔧 Git'],
        'Proficiency': [0.75, 0.80, 0.5, 0.5]
    }

data = get_skills_data()
df = pd.DataFrame(data)

st.title("📊 Skill & Kemampuan Saya")

chart = alt.Chart(df).mark_bar().encode(
    x='Skills',
    y='Proficiency'
)

st.altair_chart(chart, use_container_width=True)

st.write("Berikut adalah progres kemahiran saya dalam beberapa skill utama:")
