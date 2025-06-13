import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_rf.joblib')

st.title('Prediksi Load Type Industri Baja')

usage_kwh = st.number_input("Usage (kWh)", min_value=0.0)
lagging_kvarh = st.number_input("Lagging kVarh", min_value=0.0)
leading_kvarh = st.number_input("Leading kVarh", min_value=0.0)
co2 = st.number_input("CO2 (tCO2)", min_value=0.0)
lagging_pf = st.number_input("Lagging Power Factor", min_value=0.0)
leading_pf = st.number_input("Leading Power Factor", min_value=0.0)
nsm = st.number_input("NSM", min_value=0)
week_status = st.selectbox("Week Status", ['Weekday', 'Weekend'])
day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Manual encode sesuai training
week_status_map = {'Weekday': 1, 'Weekend': 0}
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

input_data = pd.DataFrame([[
    usage_kwh, lagging_kvarh, leading_kvarh, co2,
    lagging_pf, leading_pf, nsm,
    week_status_map[week_status], day_map[day_of_week]
]], columns=[
    'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor',
    'NSM', 'WeekStatus', 'Day_of_week'
])

if st.button('Prediksi'):
    prediction = model.predict(input_data)
    st.success(f'Prediksi Load Type (encoded): {int(prediction[0])}')