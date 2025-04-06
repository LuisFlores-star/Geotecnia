"""Módulo de elasticidad del macizo rocoso_2025"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import ipywidgets as widgets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Ecuación Hoek-Diederichs
def Erm_Ei(GSI, D):
    return 0.02 + (1 - D / 2) / (1 + np.exp((60 + 15 * D - GSI) / 11))

# Función para calcular resultados
def calcular_resultados(RMR89, D, sci, model=None):
    # Calcular GSI automáticamente
    GSI = RMR89 - 5
    if GSI <= 0 or RMR89 <= 23:
        display(Markdown("**GSI debe ser mayor que 0 y RMR > 23 para aplicar esta fórmula.**"))
        return

    # Calcular módulo de roca intacta Ei
    Ei = 500 * sci  # relación empírica

    # Calcular relación Erm / Ei
    Erm_div_Ei = Erm_Ei(GSI, D)

    # Calcular módulo del macizo rocoso Erm
    Erm = Erm_div_Ei * Ei

    # Mostrar resultados
    display(Markdown(f"""
    ## 📊 Resultados Derivados

    - **RMR89:** `{RMR89:.2f}`
    - **GSI:** `{GSI:.2f}`
    - **D (Factor de daño):** `{D:.2f}`
    - **Resistencia de roca intacta (σci):** `{sci:.2f} MPa`

    ---
    - **Módulo de roca intacta (Ei):** `{Ei:.2f} MPa`
    - **Relación Erm / Ei:** `{Erm_div_Ei:.4f}`
    - **Módulo del macizo rocoso (Erm):** `{Erm:.2f} MPa`
    """))

    if model:
        # Predecir Erm usando el modelo de ML
        X_input = np.array([[RMR89, GSI, D, sci]])  # Características de entrada
        Erm_pred = model.predict(X_input)  # Predicción
        display(Markdown(f"### Predicción de Erm usando Machine Learning: `{Erm_pred[0]:.2f} MPa`"))

    # Graficar la curva Erm/Ei vs GSI para los valores de D
    GSI_range = np.linspace(10, 100, 500)

    # Graficar los valores de D predefinidos (0, 0.5, 1) con colores estándar
    colors = ['blue', 'green', 'purple']  # Colores estándar para las curvas de D predefinidos
    for i, D_fixed in enumerate([0, 0.5, 1]):
        Erm_Ei_values = Erm_Ei(GSI_range, D_fixed)
        plt.plot(GSI_range, Erm_Ei_values, label=f"D = {D_fixed}", color=colors[i])

    # Graficar el valor de D elegido (rojo, punteado y resaltado)
    Erm_Ei_values = Erm_Ei(GSI_range, D)
    plt.plot(GSI_range, Erm_Ei_values, label=f"D = {D}", linestyle='--', linewidth=3, color='red')

    # Configuración de la gráfica
    plt.xlabel("GSI")
    plt.ylabel("Erm / Ei")
    plt.title("Curva Hoek-Diederichs: Erm/Ei vs GSI")
    plt.grid(True)
    plt.legend()
    plt.show()

# Generación de datos ficticios para entrenamiento del modelo
np.random.seed(42)
n_samples = 1000
RMR89_data = np.random.uniform(25, 85, n_samples)
GSI_data = RMR89_data - 5  # GSI calculado automáticamente
D_data = np.random.uniform(0, 1, n_samples)
sci_data = np.random.uniform(50, 150, n_samples)

# Calcular 'Erm' como variable dependiente (con la ecuación Hoek-Diederichs)
Erm_data = []
for i in range(n_samples):
    GSI = GSI_data[i]
    D = D_data[i]
    sci = sci_data[i]
    Ei = 500 * sci
    Erm_div_Ei = Erm_Ei(GSI, D)
    Erm_data.append(Erm_div_Ei * Ei)

# Convertir en arrays de numpy para entrenar el modelo
X = np.column_stack([RMR89_data, GSI_data, D_data, sci_data])
y = np.array(Erm_data)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Machine Learning (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Mostrar las métricas de evaluación
display(Markdown(f"""
### 📊 Métricas del Modelo:

- **Error Cuadrático Medio (MSE):** `{mse:.2f}`
- **Coeficiente de Determinación (R²):** `{r2:.2f}`
- **Error Absoluto Medio (MAE):** `{mae:.2f}`
"""))

# Mostrar las gráficas por separado
# 1. Gráfico de Predicción vs Real
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Línea de igualdad
plt.title("Gráfico de Predicción vs Real")
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.grid(True)
plt.show()

# 2. Gráfico de Residuales
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Gráfico de Residuales")
plt.xlabel("Valores Predichos")
plt.ylabel("Residuos (Predicción - Real)")
plt.grid(True)
plt.show()

# Widgets
RMR_input = widgets.FloatText(value=60.0, description='RMR89:')
D_input = widgets.FloatSlider(value=0.5, min=0, max=1.0, step=0.05, description='D:')
sci_input = widgets.FloatText(value=100.0, description='σci (MPa):')
boton = widgets.Button(description="Calcular y Graficar")

def on_button_clicked(b):
    calcular_resultados(RMR_input.value, D_input.value, sci_input.value, model)

boton.on_click(on_button_clicked)

# Mostrar interfaz
display(Markdown("## Ingrese los datos para calcular y graficar parámetros del macizo rocoso"))
display(RMR_input, D_input, sci_input, boton)
