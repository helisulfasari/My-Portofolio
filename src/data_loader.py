import pandas as pd
import streamlit as st

@st.cache_data
def load_telco_data():
    df = pd.read_csv("data/churn.csv")
    return df
