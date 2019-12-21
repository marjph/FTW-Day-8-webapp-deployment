import pandas as pd
import streamlit as st
import joblib 
import numpy as np

st.title('Sales Forecasting')

st.write('We demonstrate how we can forecast advertising sales')
data = pd.read_csv("data/advertising_regression.csv")

data

st.sidebar.subheader ('Advertising Costs')

TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150)

Radio = st.sidebar.slider('Radio Advertising Cost', 0, 50, 25)

Newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 250, 125)

st.subheader ('Radio Advertising Cost Distribution')

hist_values = np.histogram(data.radio, bins=300, range=(0,300))[0]

st.bar_chart(hist_values)

st.subheader('TV Ad Cost Distribution')

hist_values = np.histogram(data.TV, bins=300, range=(0,300))[0]

st.bar_chart(hist_values)

st.subheader('Newspaper Advertising Cost Distribution')

hist_values = np.histogram(data.newspaper, bins=300, range=(0,300))[0]

st.bar_chart(hist_values)

saved_model = joblib.load('advertising_model.sav')

predicted_sales = saved_model.predict([[TV, Radio, Newspaper]])[0]

st.write(f"Predicted sales is {int(predicted_sales*1000)} dollars.")

