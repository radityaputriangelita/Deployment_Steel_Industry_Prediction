import streamlit as st
import pandas as pd
import joblib
import datetime

# Load model
model = joblib.load('model_rf.joblib')

st.title('Prediksi Load Type Industri Baja')

# Buat dua kolom
col1, col2 = st.columns(2)

with col1:
    usage_kwh = st.number_input("Usage (kWh)", min_value=0.0)
    lagging_kvarh = st.number_input("Lagging kVarh", min_value=0.0)
    leading_kvarh = st.number_input("Leading kVarh", min_value=0.0)
    co2 = st.number_input("CO2 (tCO2)", min_value=0.0)

with col2:
    lagging_pf = st.number_input("Lagging Power Factor", min_value=0.0)
    leading_pf = st.number_input("Leading Power Factor", min_value=0.0)
    # Pilih tanggal dan waktu di kolom kedua
    selected_date = st.date_input("Pilih Tanggal", value=datetime.date.today())
    selected_time = st.time_input("Pilih Waktu (untuk NSM)", value=datetime.time(0, 0, 0))

# Hitung NSM dan flags setelah input kolom
nsm = selected_time.hour * 3600 + selected_time.minute * 60 + selected_time.second
day_of_week_idx = selected_date.weekday()  # 0=Monday ... 6=Sunday
week_status_flag = 1 if day_of_week_idx < 5 else 0  # 1=Weekday, 0=Weekend

input_data = pd.DataFrame([[
    usage_kwh, lagging_kvarh, leading_kvarh, co2,
    lagging_pf, leading_pf, nsm,
    week_status_flag, day_of_week_idx
]], columns=[
    'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor',
    'NSM', 'WeekStatus', 'Day_of_week'
])

if st.button('Prediksi'):
    prediction = model.predict(input_data)
    int_prediction = int(prediction[0])
    if int_prediction == 0:
        result = "Light Load"
    elif int_prediction == 1:
        result = "Maximum Load"
    elif int_prediction == 2:
        result = "Medium Load"
    st.success(f'Prediksi Load Type: {result}')