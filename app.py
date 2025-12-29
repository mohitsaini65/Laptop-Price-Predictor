import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load model and dataframe
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

st.set_page_config(page_title="Laptop Price Predictor")
st.title("💻 Laptop Price Predictor")

company = st.selectbox("Brand", df["Company"].unique())
type_name = st.selectbox("Type", df["TypeName"].unique())
ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()))
weight = st.number_input("Weight (kg)", step=0.1)

touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])

screen_size = st.slider("Screen Size (inches)", 10.0, 18.0, 13.0)

resolution = st.selectbox(
    "Screen Resolution",
    [
        "1920x1080", "1366x768", "1600x900", "3840x2160",
        "3200x1800", "2880x1800", "2560x1600",
        "2560x1440", "2304x1440"
    ]
)

cpu_brand = st.selectbox("CPU Brand", df["Cpu brand"].unique())
cpu_freq = st.number_input("CPU Frequency (GHz)", step=0.1)

hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2000])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])

gpu = st.selectbox("GPU Brand", df["Gpu brand"].unique())
os = st.selectbox("Operating System", df["os"].unique())

if st.button("Predict Price"):
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    input_df = pd.DataFrame(
        [[
            company, type_name, ram, weight,
            touchscreen, ips, ppi,
            cpu_brand, cpu_freq,
            hdd, ssd, gpu, os
        ]],
        columns=df.drop("Price", axis=1).columns
    )

    price = int(np.exp(pipe.predict(input_df)[0]))
    st.success(f"💰 Estimated Price: ₹ {price}")

