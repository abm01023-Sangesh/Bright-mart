import streamlit as st
import joblib
import numpy as np

# Load the trained model
lr = joblib.load('linear.sav')

st.title('Sales Prediction using Linear Regression')
st.write('Enter the advertising budgets for TV, Radio, and Newspaper to predict sales.')

# Input fields for features
tv = st.slider('TV Advertising Budget ($)', 0.0, 300.0, 100.0)
radio = st.slider('Radio Advertising Budget ($)', 0.0, 50.0, 20.0)
newspaper = st.slider('Newspaper Advertising Budget ($)', 0.0, 120.0, 10.0)

# Predict button
if st.button('Predict Sales'):
    # Create a numpy array from the input values
    features = np.array([[tv, radio, newspaper]])
    
    # Make prediction
    prediction = lr.predict(features)[0]
    
    st.success(f'Predicted Sales: {prediction:.2f}')
