import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


st.header("📊 Baseline Model – Logistic Regression")

st.markdown("""
Baseline model digunakan sebagai titik awal untuk menilai apakah model utama
(XGBoost) memberikan peningkatan performa yang signifikan. Logistic
Regression dipilih sebagai baseline karena sederhana, cepat, dan mudah
diinterpretasikan.
""")


df_model = pd.read_pickle("data/df_model.pkl")

#encode churn
df_model['Churn'] = df_model['Churn'].map({'No': 0, 'Yes': 1})

#splitting feature and target
from sklearn.model_selection import train_test_split

X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Encoding
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)


X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

#scalling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

#train besline logreg
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
st.markdown("---")

st.header("📊 Evaluation Baseline – Logistic Regression")
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
) 

y_pred = logreg.predict(X_test_scaled)
y_prob = logreg.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

st.subheader("📈 Baseline Logistic Regression Performance")
st.write(f"**Accuracy:** {accuracy:.3f}")
st.write(f"**Precision:** {precision:.3f}")
st.write(f"**Recall:** {recall:.3f}")
st.write(f"**ROC AUC:** {roc_auc:.3f}")

st.markdown("""
Pada model baseline, kita akan berfokus pada nilai **Recall**, dimana recall memiliki arti jumlah pelanggan yang benar -  benar churn
yang berhasil ditangkap oleh model adalah 0.504 artinya model menemukan pelanggan sebesar 50% yang akan churn.
""")
           
from sklearn.metrics import classification_report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.subheader("📋 Classification Report")
st.dataframe(report_df.style.format("{:.2f}"))

#Visualisasi ROC curve
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
RocCurveDisplay.from_estimator(logreg, X_test_scaled, y_test, ax=ax)
plt.title("ROC Curve – Logistic Regression")
st.pyplot(fig)

st.markdown("""
### 🎯 Kesimpulan Baseline Model
Model Logistic Regression memberikan gambaran awal performa prediksi churn.
Model utama (XGBoost) nantinya akan dibandingkan dengan baseline ini
untuk melihat apakah terjadi peningkatan performa yang signifikan.
""")
st.markdown("---")
#Modeling XGBOOST
#penentuan type kategorical dan numerical
categorical = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical = X_train.select_dtypes(include=['int64','float64']).columns.tolist()

#Preprocessing (OneHot + Scaling)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

#XGBoost model
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=5174/1872,  # imbalance handling
    eval_metric='logloss'
)
#pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("preprocess", prep),
    ("model", xgb_model)
])

#train model
pipeline.fit(X_train, y_train)

#prediction
y_pred_xgb = pipeline.predict(X_test)
y_proba_xgb = pipeline.predict_proba(X_test)[:, 1]

#evaluasi
st.header("🚀 Model Utama – XGBoost")

st.subheader("📈 Performance XGBoost")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred_xgb):.3f}")
st.write(f"**Precision:** {precision_score(y_test, y_pred_xgb):.3f}")
st.write(f"**Recall:** {recall_score(y_test, y_pred_xgb):.3f}")
st.write(f"**F1 Score:** {f1_score(y_test, y_pred_xgb):.3f}")
st.write(f"**ROC AUC:** {roc_auc_score(y_test, y_proba_xgb):.3f}")

st.markdown("""
## 🧠 Penjelasan Metrik Evaluasi XGBoost

### ✅ Accuracy (Akurasi)
Mengukur seberapa banyak prediksi model yang benar secara keseluruhan.
Namun untuk kasus churn yang datanya tidak seimbang, akurasi **bukan metrik utama**.

### ✅ Precision (Ketepatan)
Dari semua pelanggan yang diprediksi **churn**, berapa banyak yang benar-benar churn.
Precision 0.50 berarti masih ada pelanggan yang salah ter-flag (false positive), tetapi ini masih wajar untuk kasus churn.

### ✅ Recall (Sensitivitas)
Mengukur seberapa banyak pelanggan yang benar-benar churn berhasil ditangkap oleh model.
Recall 0.73 berarti model berhasil menemukan 73% pelanggan yang benar-benar akan churn.
Ini sangat penting untuk strategi retensi.

### ✅ F1 Score
Gabungan antara precision dan recall.
Nilai 0.59 menunjukkan model cukup seimbang, dengan fokus lebih kuat pada recall.

### ✅ ROC AUC
Mengukur kemampuan model membedakan pelanggan churn vs tidak churn.
Nilai 0.81 menunjukkan model **kuat dan stabil**, karena berada di atas 0.80.

---

### 🎯 Kesimpulan Singkat
Model XGBoost memiliki performa yang baik untuk mendeteksi pelanggan yang berpotensi churn.
Recall yang tinggi dan ROC AUC di atas 0.80 menunjukkan bahwa model efektif digunakan untuk strategi retensi pelanggan.
""")


#clasification report
report_dict_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
report_df_xgb = pd.DataFrame(report_dict_xgb).transpose()

st.subheader("📋 Classification Report – XGBoost")
st.dataframe(report_df_xgb.style.format("{:.2f}"))

#ROC curve
fig, ax = plt.subplots(figsize=(6, 4))
RocCurveDisplay.from_predictions(y_test, y_proba_xgb, ax=ax)
plt.title("ROC Curve – XGBoost")
st.pyplot(fig)

#Feature importance
import numpy as np

# ambil nama fitur setelah preprocessing
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
ohe_features = ohe.get_feature_names_out(categorical)
all_features = np.concatenate([ohe_features, numerical])

importances = pipeline.named_steps["model"].feature_importances_

feat_imp = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

st.subheader("🔥 Feature Importance – XGBoost")
st.dataframe(feat_imp.head(20))


#insight bisnis
top_features = feat_imp.head(10)

st.subheader("🔥 Insight Bisnis dari Fitur Terpenting")

st.markdown("""
## 🔍 Apa Arti Feature Importance?

Tabel di atas menunjukkan fitur-fitur yang paling berpengaruh dalam menentukan apakah pelanggan akan churn.

Semakin tinggi nilai **importance**, semakin besar kontribusi fitur tersebut dalam keputusan model.

### 📌 Insight Utama:
- **Contract_Month-to-month** adalah fitur paling dominan — pelanggan dengan kontrak bulanan lebih berisiko churn.
- **PaymentMethod_Electronic check** juga cukup berpengaruh — bisa jadi sinyal pelanggan yang fleksibel atau tidak terikat.
- **MonthlyCharges** dan **Tenure** menunjukkan bahwa harga dan loyalitas memengaruhi keputusan pelanggan.
- Fitur seperti **Dependents** dan **SeniorCitizen** membantu memahami segmen pelanggan yang lebih stabil.
""")


st.markdown("""
### 🧩 Interpretasi Awal
- Insight ini dapat digunakan untuk strategi retensi, seperti:
  - meningkatkan kualitas layanan pada segmen tertentu,
  - menawarkan paket yang lebih kompetitif,
  - atau memperbaiki pengalaman pelanggan pada titik-titik kritis.
""")

#comparasi XGboost dengan baseline model (Logreg)
comparison_df = pd.DataFrame({
    "Model": ["Baseline (LogReg)", "XGBoost"],
    "Accuracy": [accuracy, accuracy_score(y_test, y_pred_xgb)],
    "Precision": [precision, precision_score(y_test, y_pred_xgb)],
    "Recall": [recall, recall_score(y_test, y_pred_xgb)],
    "F1 Score": [f1_score(y_test, y_pred), f1_score(y_test, y_pred_xgb)],
    "ROC AUC": [roc_auc, roc_auc_score(y_test, y_proba_xgb)]
})


st.subheader("📊 Perbandingan Baseline vs XGBoost")
numeric_cols = comparison_df.select_dtypes(include=['float64', 'int64']).columns
st.dataframe(comparison_df.style.format("{:.3f}", subset=numeric_cols))

st.markdown("""
### 📌 Hasil Comparasi
- Recall naik dari 50 → 73% = model jauh lebih baik dalam menangkap pelanggan yang akan churn
- ROC AUC tetap tinggi = model tetap stabil dan kuat
- Precision turun sedikit = masih wajar untuk churn
### 🎯 Kesimpulan
- Model XGBoost sudah cukup kuat untuk dipakai dalam memprediksi churn
""")

# Navigasi

col1, col2 = st.columns([1,1])

with col1:
    st.page_link("pages/04-telco_preparation.py", label="⬅️ Previous", icon="📊")

with col2:
    st.page_link("pages/06 - insight and recomendation.py", label="Next ➡️", icon="📊")