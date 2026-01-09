import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Price Predictor")

with open('datasets/df.pkl','rb') as file:
    df = pickle.load(file)
with open('datasets/pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

st.title("GuruGaon House Price Prediction")
st.header("Enter your Input : ")

def func(value):
    if value == "Yes":
        return 1
    else:
        return 0

property_type = st.selectbox("Poperty Type : ",options=['Flat','House'])
bedroom = float(st.selectbox("No of Bedrooms",sorted(df['bedRoom'].unique().tolist())))
bathroom = float(st.selectbox("No of Bathrooms",df['bathroom'].unique().tolist()))
balcony = st.selectbox("No of Balconies",sorted(df['balcony'].unique().tolist()))
study = st.selectbox("Study Room",options=['Yes','No'])
servant = st.selectbox("Servant Room",options=['Yes','No'])
pooja = st.selectbox("Pooja Room",options=['Yes','No'])
sector = st.selectbox("Sector",sorted(df.sector.unique().tolist()))
area = st.number_input("Enter Area in Sq ft ")
floor = st.selectbox("Floors",df.floor.unique().tolist())
age = st.selectbox("Age",df.age.unique().tolist())
Feature_score = st.selectbox("Feature Score",df.feature_score.unique().tolist())
Furnish_score = st.selectbox("Furnish Score",df.furnish_score.unique().tolist())

if st.button("Predict"):
    data = [[area,property_type.lower(),bedroom,bathroom,balcony,sector,func(study),func(servant),func(pooja),floor,age,Feature_score,Furnish_score]]
    columns = list(df.columns.values)

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)
    st.write(f"Price of the property is {np.expm1(pipeline.predict(one_df))}")
