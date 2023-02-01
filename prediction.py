import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import numpy as np
import joblib
import json


# Load Data yang sudah di save
with open('model_linreg.pkl', 'rb') as file_1:
  model_linreg = joblib.load(file_1)

with open('model_scaler.pkl', 'rb') as file_2:
  model_scaler = joblib.load(file_2)

with open('model_encoder.pkl', 'rb') as file_3:
  model_encoder = joblib.load(file_3)

with open('list_num_cols.txt', 'r') as file_4:
  df_num_columns = json.load(file_4)

with open('list_cat_cols.txt', 'r') as file_5:
  df_cat_columns = json.load(file_5)


# Pengisian Kolum di streamlit
with st.form(key='form_parameters'):
    Sex = st.selectbox('Sex', {'F','M','I'}, index=1)	
    Length = st.number_input('Length')
    Diameter = st.number_input('Diameter')
    Height = st.number_input('Height')
    Weight = st.number_input('Weight')
    Shucked_Weight = st.number_input('Shucked Weight')
    Viscera_Weight = st.number_input('Viscera Weight')
    Shell_Weight = st.number_input('Shell Weight')

    submitted = st.form_submit_button('Predict')


data_inf = {
    'Sex': Sex,
    'Length' : Length,
    'Diameter' : Diameter,
    'Height' : Height,
    'Weight' : Weight,
    'Shucked Weight' : Shucked_Weight,
    'Viscera Weight' : Viscera_Weight,
    'Shell Weight' : Shell_Weight
}

data_inf = pd.DataFrame([data_inf])
st.dataframe(data_inf)

if submitted:
    data_inf_num = data_inf[df_num_columns]
    data_inf_cat = data_inf[df_cat_columns]

    df_inf_num_scaled = model_scaler.fit_transform(data_inf_num)
    df_inf_cat_encoded = model_encoder.fit_transform(data_inf_cat)

    data_inf_final = np.concatenate([df_inf_num_scaled, df_inf_cat_encoded], axis=1)

    y_pred_inf = model_linreg.predict(data_inf_final)

    st.write('# Umur : ', str(int(y_pred_inf)))