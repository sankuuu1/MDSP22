
# Dark-themed Stress-Strain Curve
strain_vals = np.linspace(0, pred[1] / 100, 300)
stress_vals = np.piecewise(strain_vals,
    [strain_vals < 0.02, strain_vals < 0.06, strain_vals >= 0.06],
    [lambda x: (pred[0] / 0.02) * x,
     lambda x: -200 * (x - 0.02)**2 + pred[0],
     lambda x: -300 * (x - 0.06) + pred[0] * 0.9])
stress_vals = np.maximum(stress_vals, 0)

plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.plot(strain_vals * 100, stress_vals, label='Stress-Strain', color='cyan', linewidth=2)
ax.set_xlabel("Strain (%)", fontsize=12, color='white')
ax.set_ylabel("Stress (MPa)", fontsize=12, color='white')
ax.set_title("Predicted Stress-Strain Curve", fontsize=14, color='white')
ax.grid(True, linestyle='--', alpha=0.3)
ax.tick_params(colors='white')
ax.legend(facecolor='black', edgecolor='white', fontsize=10)
st.pyplot(fig)
