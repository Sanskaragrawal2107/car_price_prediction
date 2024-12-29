import streamlit as st
import joblib
import pandas as pd
import numpy as np
model_pipeline = joblib.load("price_model_pipeline.pkl")

df = pd.read_csv("cleaned_car_data.csv")
st.title("Welcome to Car Price Predictor")

if "company" not in df.columns or "name" not in df.columns:
    st.error("The CSV must contain company and name columns")
else:
   
    com = ['select company'] + list(df["company"].unique())
    selected_company = st.selectbox("Select company:", options=com)

    filtered_models = ['select model'] + list(df[df["company"] == selected_company]['name'].unique())
    selected_model = st.selectbox("Select model:", options=filtered_models)

    fuel = df['fuel_type'].unique()
    selected_fuel = st.selectbox("Select fuel", options=fuel, index=2)

    kms = st.text_input("Enter number of kilometers traveled:", placeholder="Kilometers traveled")

    year = ['select year'] + sorted(df[df['company']==selected_company]['year'].unique(),reverse=True)
    selected_year = st.selectbox("Select year of purchase:", options=year)

if st.button("Predict"):
   input_data=pd.DataFrame({
       "company":[selected_company],
       "name":[selected_model],
       "year":[selected_year],
       "kms_driven":[kms],
       "fuel_type":[selected_fuel]
            })
   prediction=model_pipeline.predict(input_data)
   def format_price_to_words(prediction):
    lakhs = prediction // 100000
    thousands = (prediction % 100000) // 1000
    formatted_price = f"{lakhs} lakh" if lakhs else ""
    if thousands:
        formatted_price += f" {thousands} thousand"
    return formatted_price.strip()
   price_in_words = format_price_to_words(int(prediction))
   st.markdown(
                 f"<div style='background-color: #4CAF50; padding: 10px; border-radius: 5px;'>"
                f"<h3 style='color: white; text-align: center;'>The Predicted Price is: {price_in_words}</h3>"
                f"</div>", unsafe_allow_html=True)

