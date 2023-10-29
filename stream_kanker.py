import pickle
import streamlit as st

# Membaca model
kanker_model = pickle.load(open('kanker_model.sav', 'rb'))

# Judul website
st.title('Data Mining Prediksi Diagnosis Kanker Payudara')

#membagi kolom
col1, col2 = st.columns(2)

with col1 :
   radius_mean = st.number_input ('input nilai radius_mean')

with col2 :
   texture_mean = st.number_input ('input nilai texture_mean')

with col1 :
   perimeter_mean = st.number_input ('input nilai perimeter_mean')

with col2 :
   area_mean = st.number_input ('input nilai area_mean')

with col1 :
   smoothness_mean = st.number_input ('input nilai smoothness_mean')

with col2 :
   compactness_mean = st.number_input ('input nilai compactness_mean')

with col1 :
   concavity_mean = st.number_input ('input nilai concavity_mean')

with col2 :
   concavepoints_mean = st.number_input ('input nilai concave points_mean')

# Code untuk prediksi
kanker_diagnosis = ''

# Membuat tombol untuk prediksi
if st.button('Test Prediksi Diagnosis Kanker Payudara'):
    kanker_diagnosis = kanker_model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concavepoints_mean]])
    if (kanker_diagnosis[0] == 1):
       kanker_diagnosis = 'Pasien terkena kanker jinak'
    else:
       kanker_diagnosis = 'Pasien terkena kanker ganas'
st.success(kanker_diagnosis)
