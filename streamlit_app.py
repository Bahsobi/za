import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# App title and description
st.title('💼 Startup Profit Predictor')
st.info('Predict the **Profit** based on startup data using Multiple Linear Regression.')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/Bahsobi/sii_project/main/50_Startups%20(1).csv')

# Show data
with st.expander('📄 Data Overview'):
    st.write(df)

# Prepare features and target
X_raw = df.drop('Profit', axis=1)
y = df['Profit']

# Encode categorical variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['State'])], remainder='passthrough')
X_encoded = ct.fit_transform(X_raw)

# Train model
regressor = LinearRegression()
regressor.fit(X_encoded, y)

# Initialize an empty list for storing predictions
if 'predictions_list' not in st.session_state:
    st.session_state.predictions_list = []

# Sidebar input
with st.sidebar:
    st.header('🚀 Enter Startup Details')

    state = st.selectbox('State', df['State'].unique())
    rnd_spend = st.number_input('R&D Spend', min_value=0.0, format="%.2f")
    
    
admin = st.number_input('Administration', min_value=0.0, max_value=300000.0, value=0.0, step=1000.0)
    
    
    marketing = st.number_input('Marketing Spend', min_value=0.0, format="%.2f")

    input_data = pd.DataFrame([[state, rnd_spend, admin, marketing]],
                              columns=['State', 'R&D Spend', 'Administration', 'Marketing Spend'])

    input_encoded = ct.transform(input_data)

# Prediction
prediction = regressor.predict(input_encoded)

# Append the new prediction to the session state list
st.session_state.predictions_list.append(prediction[0])

# Display result
st.subheader('📈 Predicted Profit')
st.success(f"💰 ${prediction[0]:,.2f}")

# Show summary stats of predicted profits
with st.expander("📊 Predicted Profit Summary"):
    # Show summary statistics (mean, min, max, etc.) for the predictions
    prediction_df = pd.DataFrame(st.session_state.predictions_list, columns=["Predicted Profit"])
    st.write(prediction_df.describe())

# Show summary stats of numerical columns
with st.expander("📊 Numeric Data Summary"):
    st.write(df.describe())
