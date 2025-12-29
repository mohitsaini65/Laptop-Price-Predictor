import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- Load trained pipeline & data ----------------
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# ---------------- Page Config ----------------
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("💻 Laptop Price Predictor")

# ---------------- User Inputs (MATCHING TRAINING COLUMNS) ----------------

company = st.selectbox("Company", df['Company'].unique())
type_name = st.selectbox("Laptop Type", df['TypeName'].unique())
ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
weight = st.number_input("Weight (kg)", min_value=0.5, step=0.1)

touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])

screen_size = st.slider("Screen Size (inches)", 10.0, 18.0, 13.0)

resolution = st.selectbox(
    "Screen Resolution",
    ['1920x1080','1366x768','1600x900','3840x2160',
     '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)

cpu = st.selectbox("CPU Brand", df['Cpu brand'].unique())
cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, step=0.1)

hdd = st.selectbox("HDD (GB)", sorted(df['HDD'].unique()))
ssd = st.selectbox("SSD (GB)", sorted(df['SSD'].unique()))

gpu = st.selectbox("GPU Brand", df['Gpu brand'].unique())
os = st.selectbox("Operating System", df['os'].unique())

# ---------------- Prediction ----------------
if st.button("Predict Price"):

    # Binary conversions
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    # PPI calculation (EXACTLY like notebook logic)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    # Create DataFrame in SAME ORDER as training
    query = pd.DataFrame(
        [[company, type_name, ram, weight,
          touchscreen, ips, ppi,
          cpu, cpu_freq, hdd, ssd, gpu, os]],
        columns=[
            'Company',
            'TypeName',
            'Ram',
            'Weight',
            'Touchscreen',
            'Ips',
            'ppi',
            'Cpu brand',
            'Cpu_Frequency_GHz',
            'HDD',
            'SSD',
            'Gpu brand',
            'os'
        ]
    )

    # Prediction (inverse of log)
    predicted_price = np.exp(pipe.predict(query)[0])

    st.success(f"💰 Estimated Laptop Price: ₹ {int(predicted_price)}")

