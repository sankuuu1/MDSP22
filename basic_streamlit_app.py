
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Title
st.title("Basic UTS & Elongation Predictor")
st.write("Enter layer thickness and infill pattern to predict UTS and elongation.")

# Load Data
stress_strain_df = pd.read_excel("Stress_Strain_Pattern_Thickness.xlsx")
uts_elong_df = pd.read_excel("UTS_Elongation.xlsx", skiprows=2, usecols="B:E")
uts_elong_df.columns = ['Layer_Thickness_mm', 'Pattern', 'UTS_MPa', 'Elongation_percent']
uts_elong_df.dropna(inplace=True)

# Extract features from stress-strain curves
patterns = ['Cubic', 'Gyroid', 'Hexagonal']
layer_thicknesses = [0.12, 0.20, 0.28, 0.36]
extra_data = []

for i, thickness in enumerate(layer_thicknesses):
    for j, pattern in enumerate(patterns):
        stress_col = stress_strain_df.columns[i * 6 + j * 2]
        strain_col = stress_strain_df.columns[i * 6 + j * 2 + 1]
        temp = stress_strain_df[[stress_col, strain_col]].dropna()
        temp.columns = ['Stress', 'Strain']
        if not temp.empty:
            uts = temp['Stress'].max()
            elong = temp['Strain'].max() * 100
            extra_data.append({
                'Layer_Thickness_mm': thickness,
                'Pattern': pattern,
                'UTS_MPa': uts,
                'Elongation_percent': elong
            })

extra_df = pd.DataFrame(extra_data)
data = pd.concat([uts_elong_df, extra_df], ignore_index=True)

# Manually encode categorical pattern values
encoder = OneHotEncoder(sparse=False)
pattern_encoded = encoder.fit_transform(data[['Pattern']])
pattern_df = pd.DataFrame(pattern_encoded, columns=encoder.get_feature_names_out(['Pattern']))
X = pd.concat([data[['Layer_Thickness_mm']].reset_index(drop=True), pattern_df], axis=1)

# Scale the thickness
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled['Layer_Thickness_mm'] = scaler.fit_transform(X[['Layer_Thickness_mm']])

# Targets
y_uts = data['UTS_MPa'].values
y_elong = data['Elongation_percent'].values

# Train two separate models
model_uts = RandomForestRegressor(random_state=0)
model_elong = RandomForestRegressor(random_state=0)
model_uts.fit(X_scaled, y_uts)
model_elong.fit(X_scaled, y_elong)

# User input
user_thickness = st.number_input("Enter Layer Thickness (mm)", min_value=0.1, max_value=1.0, step=0.01, value=0.24)
user_pattern = st.selectbox("Select Infill Pattern", ['Cubic', 'Gyroid', 'Hexagonal'])

# Encode user input
user_pattern_encoded = encoder.transform([[user_pattern]])
user_input_df = pd.DataFrame(user_pattern_encoded, columns=encoder.get_feature_names_out(['Pattern']))
user_input_df.insert(0, 'Layer_Thickness_mm', user_thickness)
user_input_df['Layer_Thickness_mm'] = scaler.transform(user_input_df[['Layer_Thickness_mm']])

# Predict
if st.button("Predict"):
    pred_uts = model_uts.predict(user_input_df)[0]
    pred_elong = model_elong.predict(user_input_df)[0]
    st.success(f"Predicted UTS: {pred_uts:.2f} MPa")
    st.success(f"Predicted Elongation: {pred_elong:.2f} %")

    # Plot dark-themed stress-strain graph
    strain_vals = np.linspace(0, pred_elong / 100, 300)
    stress_vals = np.piecewise(strain_vals,
        [strain_vals < 0.02, strain_vals < 0.06, strain_vals >= 0.06],
        [lambda x: (pred_uts / 0.02) * x,
         lambda x: -200 * (x - 0.02)**2 + pred_uts,
         lambda x: -300 * (x - 0.06) + pred_uts * 0.9])
    stress_vals = np.maximum(stress_vals, 0)

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.plot(strain_vals * 100, stress_vals, color='cyan', linewidth=2)
    ax.set_xlabel("Strain (%)", color='white')
    ax.set_ylabel("Stress (MPa)", color='white')
    ax.set_title("Stress-Strain Curve", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
