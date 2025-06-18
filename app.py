import streamlit as st
import pandas as pd
import joblib
import datetime
from streamlit_option_menu import option_menu


# Load model
model = joblib.load('model_rf.joblib')

# Sidebar menu


with st.sidebar:
    st.markdown(
        """
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
        <div style='padding: 10px; border-radius: 10px; display: flex; align-items: center; gap: 8px;'>
            <i class="bi bi-lightning-charge-fill" style="font-size: 1.5rem; color: #f5b000;"></i>
            <h4 style='font-size: 1.5rem; margin: 0;'>EnergyLoad Classifier</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = option_menu(
        menu_title="",
        options=["Dashboard", "Predict Load Type"],
        icons=["house-heart-fill", "bi-clipboard2-data-fill"],
        menu_icon="bi-lightning-charge-fill",
        default_index=0,
    )
    with st.expander("ℹ️ Kategori Klasifikasi"):
        st.markdown("- **Light Load**: Konsumsi energi rendah, mencerminkan aktivitas industri yang minim.")
        st.markdown("- **Medium Load**: Konsumsi energi sedang, menunjukkan aktivitas operasional yang stabil.")
        st.markdown("- **Maximum Load**: Konsumsi energi tinggi, biasanya terjadi saat beban produksi puncak.")





# Halaman Dashboard
if page == "Dashboard":
    st.markdown(
        """
        <div style="
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 8px solid #8FC6F2;
        ">
            <h1 style='color: black; font-size: 2.5rem; margin: 0;'>Welcome to EnergyLoad Classifier</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Overview")
    
    st.image("images/pendahuluan.png", caption="Ilustrasi Industri", use_container_width=True)
    st.markdown(
    """
    <div style='text-align: justify'>
    Dalam upaya <b>efisiensi energi</b>, dibutuhkan pemahaman konsumsi energi di berbagai sektor, seperti industri. Model klasifikasi ini dikembangkan untuk <b>mengelompokkan status konsumsi energi</b>, serta <b>memantau dan mengevaluasi penggunaan sumber daya</b> secara efisien. Model ini mendukung <b>perencanaan dan pengambilan keputusan</b> dalam <b>optimalisasi energi industri</b> melalui <b>classification</b> sesuai karakteristik penggunaan.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
            """"""
            )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Source")
        st.markdown(
        """
        <div style='text-align: justify'>
        Data dikumpulkan dari sebuah industi  area Daewoo steel Co. Ltd Korea Selatan yang memproduksi gulungan, pelat baja dan besi. Data yang digunakan merupakan hasil pencatatan sepanjang tahun 2018,  1 Januari hingga 31 Desember interval setiap 15 menit. Data konsumsi energy dibagi menjadi  tiga kategori yaitu Light, Medium, dan Maximum berdasarkan beban energinya.        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("### Modeling")
        st.markdown(
        """
        <div style='text-align: justify'>
        Proses modeling dilakukan dengan mengklasifikasikan penggunaan energi ke dalam tiga kategori menggunakan algoritma Random Forest. Algoritma ini merupakan gabungan dari beberapa decision tree yang digabungkan untuk meningkatkan akurasi prediksi dan mengurangi risiko overfitting.
        </div>
        """, unsafe_allow_html=True)


    st.markdown(
            """"""
            )

    st.markdown("### Model Evaluation")
    st.markdown("""
        <div style="color: black;display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 10px;">
            <div style="background-color: #8fc6f2; padding: 15px 20px; border-radius: 8px; text-align: center;">
                <div style="font-weight: bold;">Accuracy</div>
                <div>0.901</div>
            </div>
            <div style="background-color: #8fc6f2; padding: 15px 20px; border-radius: 8px; text-align: center;">
                <div style="font-weight: bold;">Precision</div>
                <div>0.873</div>
            </div>
            <div style="background-color: #8fc6f2; padding: 15px 20px; border-radius: 8px; text-align: center;">
                <div style="font-weight: bold;">Recall</div>
                <div>0.882</div>
            </div>
        </div>
        <p style="margin-top: 20px; font-size: 15px;">
            Lihat visualisasi confusion matrix selengkapnya di 
            <a href="https://colab.research.google.com/drive/1UIJ80qT9Pow9bq26xUly44tTeepKONA1#scrollTo=cbD1TGHTazO-" target="_blank" style="color: #007BFF; text-decoration: none;">
                Google Colab
            </a>
        </p>
    """, unsafe_allow_html=True)


    
# Halaman Prediksi Load Type
elif page == "Predict Load Type":
    st.markdown(
        """
        <div style="
            background-color: #ffffff;
            padding: 0.25rem;
            border-radius: 1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 8px solid #8FC6F2;
        ">
            <h1 style='color: black; font-size: 1.25rem; margin: 0;'>Clasify Your Energy Consumption</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Input data
    col1, col2 = st.columns(2)

    with col1:
        usage_kwh = st.number_input("Usage (kWh)", min_value=0.0)
        lagging_kvarh = st.number_input("Lagging kVarh", min_value=0.0)
        leading_kvarh = st.number_input("Leading kVarh", min_value=0.0)
        co2 = st.number_input("CO2 (tCO2)", min_value=0.0)

    with col2:
        lagging_pf = st.number_input("Lagging Power Factor", min_value=0.0)
        leading_pf = st.number_input("Leading Power Factor", min_value=0.0)
        selected_date = st.date_input("Pilih Tanggal", value=datetime.date.today())
        selected_time = st.time_input("Pilih Waktu", value=datetime.time(0, 0, 0))

    # Hitung fitur waktu
    nsm = selected_time.hour * 3600 + selected_time.minute * 60 + selected_time.second
    day_of_week_idx = selected_date.weekday()
    week_status_flag = 1 if day_of_week_idx < 5 else 0

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
        label = int(prediction[0])

        if label == 0:
            result = "Light Load"
            color = "green"
            advice = "✅ Kondisi energi masih aman, Anda dapat melanjutkan kegiatan seperti biasa."
        elif label == 2:
            result = "Medium Load"
            color = "orange"
            advice = "⚠️ Perhatikan penggunaan energi, Lakukan pemantauan agar tidak meningkat ke level maksimum."
        elif label == 1:
            result = "Maximum Load"
            color = "red"
            advice = "🚨 Kurangi penggunaan energi segera untuk menghindari pemborosan dan potensi kerusakan sistem."
        else:
            result = "Unknown"
            color = "gray"
            advice = "Data tidak dikenali. Silakan cek kembali input."

        st.markdown(
            f"<div style='padding: 10px; background-color: #{'d4edda' if color=='green' else 'fff3cd' if color=='orange' else 'f8d7da'}; border-left: 5px solid {color}; border-radius: 5px;'>"
            f"<h5 style='color: {color};'>Prediksi Load Type: {result}</h5>"
            f"<p style= 'color: black'>{advice}</p>"
            "</div>",
            unsafe_allow_html=True
        )
