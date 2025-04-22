import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import seaborn as sns
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

# Sidebar input
with st.sidebar:
    st.header('🚀 Enter Startup Details')

    state = st.selectbox('State', df['State'].unique())
    rnd_spend = st.number_input('R&D Spend', min_value=0.0, format="%.2f")
    admin = st.number_input('Administration', min_value=0.0, format="%.2f")
    marketing = st.number_input('Marketing Spend', min_value=0.0, format="%.2f")

    input_data = pd.DataFrame([[state, rnd_spend, admin, marketing]],
                              columns=['State', 'R&D Spend', 'Administration', 'Marketing Spend'])

    input_encoded = ct.transform(input_data)

# Prediction
prediction = regressor.predict(input_encoded)

# Display result
st.subheader('📈 Predicted ProfitZZ')
st.success(f"💰 ${prediction[0]:,.2f}")

# Correlation heatmap
with st.expander("📊 Feature Correlation Heatmap"):
    st.write("Correlations between numerical features:")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
