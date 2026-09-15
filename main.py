import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load components
model = pickle.load(open('xgboost_model.pkl', 'rb'))
encoders = pickle.load(open('label_encoder.pkl', 'rb'))
scaler_fitur = pickle.load(open('scaler_fitur.pkl', 'rb'))
target_scaler = pickle.load(open('target_scaler.pkl', 'rb'))
rentang = pickle.load(open('rentang_fitur.pkl', 'rb'))

st.set_page_config(page_title="Estimasi Harga Rumah", layout="wide")
st.title("🏠 Aplikasi Prediksi Harga Rumah")
st.markdown("--- ")

# --- Input UI ---
with st.sidebar:
    st.header("Parameter Input")
    in_city = st.selectbox("Kota", encoders['city'].classes_)
    in_zip = st.selectbox("Kode Pos (Statezip)", encoders['statezip'].classes_)

col1, col2 = st.columns(2)

with col1:
    in_bed = st.slider("Bedrooms", int(rentang['bedrooms']['min']), int(rentang['bedrooms']['max']), 3)
    in_bath = st.slider("Bathrooms", float(rentang['bathrooms']['min']), float(rentang['bathrooms']['max']), 2.0, 0.25)
    in_flr = st.slider("Floors", float(rentang['floors']['min']), float(rentang['floors']['max']), 1.0, 0.5)

with col2:
    in_sqft = st.slider("Sqft Living", int(rentang['sqft_living']['min']), int(rentang['sqft_living']['max']), 2000)
    in_abv = st.slider("Sqft Above", int(rentang['sqft_above']['min']), int(rentang['sqft_above']['max']), 1500)

# --- Prediksi ---
if st.button("Prediksi Sekarang"):
    enc_city = encoders['city'].transform([in_city])[0]
    enc_zip = encoders['statezip'].transform([in_zip])[0]

    X_input = pd.DataFrame({
        'bedrooms': [in_bed],
        'bathrooms': [in_bath],
        'sqft_living': [in_sqft],
        'floors': [in_flr],
        'sqft_above': [in_abv],
        'city': [enc_city],
        'statezip': [enc_zip]
    })

    X_scaled = scaler_fitur.transform(X_input)
    pred_skala = model.predict(X_scaled)
    harga_asli = target_scaler.inverse_transform(pred_skala.reshape(-1, 1))

    st.balloons()
    st.success(f"### Hasil Estimasi Harga: ${harga_asli[0][0]:,.2f}")
