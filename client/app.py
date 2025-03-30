import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()

st.title("My beautiful App")

# Ajout d'un champ de texte
user_input = st.text_input("Enter some text")

button_clicked = st.button("Click me")

if button_clicked:
    if user_input:
        response = requests.get(f"http://server:8000/add/{user_input}")
        if response.status_code == 200:
            st.write("Request successful")
        else:
            st.write("Request failed")
    else:
        st.write("Please enter some text")


row_metrics = st.columns(2)
liste_fruits = requests.get("http://server:8000/list").json()
df = pd.DataFrame(liste_fruits["results"])
with row_metrics[0]:
    
    st.write(df)

with row_metrics[1]:
    fig = px.bar(df )
    st.plotly_chart(fig)


st.write("Formulaire d'entrée de données")
X = iris.data
sepal_length = st.slider("Sepal length", 0.0, max_value= max(X[:,0]), value=1.0)
sepal_width = st.slider("Sepal width", 0.0, max_value= max(X[:,1]), value=1.0)
petal_length = st.slider("Petal length", 0.0, max_value= max(X[:,2]), value=1.0)
petal_width = st.slider("Petal width", 0.0, max_value= max(X[:,3]), value=1.0)

if st.button("Predict"):
    response = requests.post("http://server:8000/predict", json={
        "sepal_length": sepal_length,
          "sepal_width": sepal_width, 
          "petal_length": petal_length, 
          "petal_width": petal_width})
    
    if response.status_code == 200:
        pred_class = response.json()["prediction"]
        st.write(f"Prediction: {pred_class}")
    else:
        st.write("Request failed")
