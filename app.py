import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from PIL import Image
import xgboost as xgb

st.title("MACHINE LEARNING PROJECT TWO MODELS")
st.header("MBD- Section 1- Judit Llorens")

screen = st.selectbox('Select Analysis:', ['Iris', 'Customers'])

if screen == 'Iris':

    st.title("Flower Data Analysis Using ANN's")

    sepal_length = st.slider("Select the value for Sepal Length:", 0.0, 8.0, 0.1)
    sepal_width = st.slider("Select the value for Sepal Width:", 0.0, 5.0, 0.1)
    petal_length = st.slider("Select the value for Petal Length:", 0.0, 7.0, 0.1)
    petal_width = st.slider("Select the value for Petal Width:", 0.0, 3.0, 0.1)


    model_iris = keras.models.load_model("model_iris.keras")
    pred_iris = np.argmax(model_iris.predict([[sepal_length, sepal_width, petal_length, petal_width]]))

    flowers_imgs = [Image.open('images/setosa.jpeg'), Image.open('images/versicolor.jpg'), Image.open('images/virginica.jpeg')]

    predicted_flower = flowers_imgs[pred_iris]

    categories = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    hibiscus_emoji = "\U0001F33A"
    st.header(categories[pred_iris] + hibiscus_emoji)

    st.image(predicted_flower)

else:

    st.title("Customers Data Analysis Using XGBoost")

    region = st.selectbox('Choose the region:', ['Lisbon', 'Oporto', 'Others'])

    region_id = 2
    if region == 'Lisbon': region_id = 3
    if region == 'Oporto': region_id = 1

    fresh = st.number_input('Fresh:', format='%i', step = 1)
    milk = st.number_input('Milk:', format='%i', step = 1)
    grocery = st.number_input('Grocery:', format='%i', step = 1)
    frozen = st.number_input('Frozen:', format='%i', step = 1)
    detergents_paper = st.number_input('Detergents paper:', format='%i', step = 1)
    delicassen = st.number_input('Delicassen:', format='%i', step = 1)


    model_customers = joblib.load('XGBoost.pkl')
    df = pd.DataFrame([[fresh, milk, grocery, frozen, detergents_paper, delicassen, region_id==1, region_id==2, region_id==3]],
                      columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen', 'Region_1', 'Region_2', 'Region_3'])
    pred_channel = model_customers.predict(df)[0]

    department_store_emoji = "\U0001F3EC"
    st.subheader("Predicted Channel:" + department_store_emoji)
    predicted_channel = 'Retail Channel'
    if pred_channel == 1:
        predicted_channel = 'Horeca Channel'

    st.header(predicted_channel)