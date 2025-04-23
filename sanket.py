import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor

# Set page config
st.set_page_config(page_title="UTS & Elongation Predictor", layout="centered")

# Title
st.title("UTS & Elongation Predictor ")
st.write("This app predicts **Ultimate Tensile Strength (UTS)** and **Elongation** based on layer thickness and infill pattern.")

# Step 1: Load data
uts_elong_df = pd.read_excel("UTS_Elongation.xlsx")
stress_strain_df = pd.read_excel("MDSP Dataset.xlsx")

# Rename columns for clarity
uts_elong_df.columns = ['Layer_Thickness_mm', 'Pattern', 'UTS_MPa', 'Elongation_percent']
uts_elong_df = uts_elong_df.dropna()

# Step 2: Extract extra features from stress-strain data
patterns = ['Cubic', 'Gyroid', 'Hexagonal', 'Tri Hexagon']
layer_thicknesses = [0.12, 0.20, 0.28]
extra_data = []

for thickness in layer_thicknesses:
    for pattern in patterns:
        # Filter data for specific thickness and pattern
        temp_df = stress_strain_df[(stress_strain_df['Thickness'] == thickness) & (stress_strain_df['Pattern'] == pattern)]
        if not temp_df.empty:
            uts = temp_df['Stress'].max()
            elong = temp_df['Strain'].max() * 100  # Convert to percentage
            extra_data.append({
                'Layer_Thickness_mm': thickness,
                'Pattern': pattern,
                'UTS_MPa': uts,
                'Elongation_percent': elong
            })

extra_df = pd.DataFrame(extra_data)
data = pd.concat([uts_elong_df, extra_df], ignore_index=True)

# Step 3: Prepare data for model
X = data[['Layer_Thickness_mm', 'Pattern']]
y = data[['UTS_MPa', 'Elongation_percent']]

# Step 4: Preprocessing
preprocess = ColumnTransformer([
    ('pattern', OneHotEncoder(), ['Pattern']),
    ('scale', StandardScaler(), ['Layer_Thickness_mm'])
])

# Step 5: Create and train model
model = Pipeline([
    ('prep', preprocess),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=0)))
])
model.fit(X, y)

# Step 6: User input
st.subheader("Enter Input Parameters")
layer_input = st.number_input("Layer Thickness (mm)", min_value=0.1, max_value=1.0, step=0.01, value=0.24)
pattern_input = st.selectbox("Infill Pattern", ['Cubic', 'Gyroid', 'Hexagonal', 'Tri Hexagon'])

# Step 7: Predict
if st.button("Predict"):
    input_df = pd.DataFrame([{'Layer_Thickness_mm': layer_input, 'Pattern': pattern_input}])
    pred = model.predict(input_df)[0]

    st.success(f"🔩 Predicted UTS: **{pred[0]:.2f} MPa**")
    st.success(f"📈 Predicted Elongation: **{pred[1]:.2f}%**")

    # Step 8: Plot stress-strain (DARK THEME)
    strain_vals = np.linspace(0, pred[1] / 100, 300)
    stress_vals = np.piecewise(strain_vals,
        [strain_vals < 0.02, strain_vals < 0.06, strain_vals >= 0.06],
        [lambda x: (pred[0] / 0.02) * x,
         lambda x: -200 * (x - 0.02)**2 + pred[0],
         lambda x: -300 * (x - 0.06) + pred[0] * 0.9])
    stress_vals = np.maximum(stress_vals, 0)

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.plot(strain_vals * 100, stress_vals, color='cyan', linewidth=2, label='Stress-Strain')
    ax.set_xlabel("Strain (%)", fontsize=12, color='white')
    ax.set_ylabel("Stress (MPa)", fontsize=12, color='white')
    ax.set_title("Predicted Stress-Strain Curve", fontsize=14, color='white')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', edgecolor='white', fontsize=10)
    st.pyplot(fig)