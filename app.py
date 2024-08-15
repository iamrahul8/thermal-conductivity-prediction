import streamlit as st
st.write('# Thermal Coductivity of Bentonite prediction')
# prompt: not use bentonite.csv use inout auto

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#load data
dataset = pd.read_csv('bentonite.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to take input and predict
def predict_thermal_conductivity_updated(bulk_density, porosity, saturation):
    input_data = [bulk_density, porosity, saturation]
    if bulk_density == 0 and porosity == 0 and saturation == 0:
      return 0
    predicted_thermal_conductivity = model.predict([input_data])
    return predicted_thermal_conductivity[0]
import streamlit as st

# Example usage
dry_density = st.number_input('Dry density') # g/cm^3
porosity = st.number_input('Porosity')# fraction
saturation = st.number_input('Saturation') # fraction
predicted_thermal_conductivity = predict_thermal_conductivity_updated(dry_density, porosity, saturation)
print("Predicted Thermal Conductivity:", predicted_thermal_conductivity)
st.write('# Predicted Thermal Conductivity is ',predicted_thermal_conductivity)
st.write("---")

st.subheader("Under the guidance of Prof. Pawan Kishor Sah")
