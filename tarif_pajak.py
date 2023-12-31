import pickle
import streamlit as st

model = pickle.load(open('Tarif_Pajak_diBoston.sav', 'rb'))

st.title('Estimasi Harga Pajak Rumah Di Boston')

CRIM = st.number_input('Masukan tingkat kejahatan')
ZN = st.number_input('Masukan nilai proposi lahan perumahan')
INDUS = st.number_input('Masukan nilai proporsi hektar bisnis ')
NOX = st.number_input('Masukan konsentrasi oksida nitrat')
RM = st.number_input('Masukan jumlah rata-rata kamar')
AGE = st.number_input('Masukan unit yang ditempati')
DIS = st.number_input('Masukan jarak ke lima pusat pekerjaan di Boston')
PTRATIO = st.number_input('Masukan rasio murid guru menurut kota')

predict = ''

if st.button('Estimasi Harga Pajak Rumah Di Boston'):
    # Masukkan data input ke dalam bentuk list
    input_data = [[CRIM, ZN, INDUS, NOX, RM, AGE, DIS, PTRATIO]]
    # Lakukan prediksi dengan model
    predict = model.predict(input_data)
    st.write('Estimasi Harga Pajak Rumah Di Boston dalam USD :', predict)
    st.write ('Estimasi Harga Pajak Rumah Di Boston dalam IDR (Juta) : ', predict*19000)
