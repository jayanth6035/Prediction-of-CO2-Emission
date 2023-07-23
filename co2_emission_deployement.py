#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('co2_emissions (1).csv', sep=';')
data.drop_duplicates(inplace=True)
data.reset_index(inplace=True, drop=True)

# Remove the outliers
df_num_features = data.select_dtypes(include=np.number)
Q1 = df_num_features.quantile(0.25)
Q3 = df_num_features.quantile(0.75)
IQR = Q3 - Q1
outlier = pd.DataFrame((df_num_features < (Q1 - 1.5 * IQR)) | (df_num_features > (Q3 + 1.5 * IQR)))
outlier_indices = {}

for i in outlier.columns:
    outliers = outlier[outlier[i] == True].index.tolist()
    outlier_indices[i] = outliers

# Replacing the outliers with mean in each column
for column in outlier_indices.keys():
    mean_value = df_num_features[column].mean()
    df_num_features.loc[outlier_indices[column], column] = mean_value
    
data.drop_duplicates(inplace=True)
data.reset_index(inplace=True, drop=True)

data_ft = pd.get_dummies(data['fuel_type'], prefix='Fuel')
data_trans = pd.get_dummies(data["transmission"])
df = [data, data_ft, data_trans]
data = pd.concat(df, axis=1)
data.drop(['fuel_type'], inplace=True, axis=1)
data.drop(['transmission'], inplace=True, axis=1)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['make'] = label_encoder.fit_transform(data['make'])
data['model'] = label_encoder.fit_transform(data['model'])
data['vehicle_class'] = label_encoder.fit_transform(data['vehicle_class'])
data.drop(['Fuel_D','Fuel_N','Fuel_X','AS'], inplace=True, axis=1)

# Split the dataset into X and y
X = data.drop(['co2_emissions'], axis=1)
y = data['co2_emissions']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest regression model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Streamlit app
st.title('CO2 Emissions Prediction')
st.write('This app predicts CO2 emissions based on the input features.')

# Sidebar inputs
st.sidebar.header('Input Features')
make = st.sidebar.slider('Make', min_value=0, max_value=41)
model_slider = st.sidebar.slider('Model', min_value=0, max_value=2052)
vehicle_class = st.sidebar.slider('Vehicle_Class', min_value=0, max_value=15)
engine_size = st.sidebar.slider('Engine_Size', min_value=0.9, max_value=6.2)
cylinders = st.sidebar.slider('Cylinders', min_value=3, max_value=16)
fuel_consumption_city = st.sidebar.slider('Fuel Consumption City (l/100km)', min_value=4.2, max_value=21.5)
fuel_consumption_hwy = st.sidebar.slider('Fuel Consumption Hwy (l/100km)', min_value=4.0, max_value=14.5)
fuel_consumption_comb_l100km = st.sidebar.slider('Fuel Consumption Comb (l/100km)', min_value=4.1, max_value=18.4)
fuel_consumption_comb_mpg = st.sidebar.slider('Fuel Consumption Comb (mpg)', min_value=11, max_value=69)
fuel_E = st.sidebar.slider('Fuel Type E', min_value=0, max_value=1)
fuel_Z = st.sidebar.slider('Fuel Type Z', min_value=0, max_value=1)
A = st.sidebar.slider('A', min_value=0, max_value=1)
AM = st.sidebar.slider('AM', min_value=0, max_value=1)
AV = st.sidebar.slider('AV', min_value=0, max_value=1)
M = st.sidebar.slider('M', min_value=0, max_value=1)


# Predict button
predict_button = st.sidebar.button('Predict')

# Check if the Predict button is clicked
if predict_button:
    # Create a dataframe with the selected input features
    input_data = pd.DataFrame({
        'make': [make],
        'model': [model_slider],
        'vehicle_class': [vehicle_class],
        'engine_size': [engine_size],
        'cylinders': [cylinders],
        'fuel_consumption_city': [fuel_consumption_city],
        'fuel_consumption_hwy': [fuel_consumption_hwy],
        'fuel_consumption_comb(l/100km)': [fuel_consumption_comb_l100km],
        'fuel_consumption_comb(mpg)': [fuel_consumption_comb_mpg],
        'Fuel_E': [fuel_E],  
        'Fuel_Z': [fuel_Z],
        'A': [A],
        'AM': [AM],
        'AV': [AV],
        'M': [M]
    })

    # Standardize the input data using the same scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict CO2 emission
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    st.subheader('CO2 Emission Prediction')
    st.write(f'The predicted CO2 emission is {prediction[0]:.2f}')

    # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.subheader('Model Evaluation')
    st.write(f'RMSE: {rmse:.2f}')
    st.write(f'R2 Score: {r2:.2f}')

