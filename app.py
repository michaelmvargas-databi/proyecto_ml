import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Datos de ejemplo (simulados)
# -----------------------------
# Creamos un dataset ficticio para entrenar el modelo
data = {
    "edad": [25, 40, 35, 50, 23, 45, 60, 30, 28, 55],
    "salario": [500, 1500, 1200, 2000, 400, 1800, 2500, 1000, 800, 2200],
    "prestamo": [200, 500, 400, 1000, 300, 700, 1200, 350, 250, 900],
    "apto": [1, 1, 1, 1, 0, 1, 0, 1, 0, 0]  # 1 = apto, 0 = no apto
}

df = pd.DataFrame(data)

# -----------------------------
# Entrenamiento del modelo
# -----------------------------
X = df[["edad", "salario", "prestamo"]]
y = df["apto"]

# Dividimos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo con hiperparámetros definidos
model = RandomForestClassifier(
    n_estimators=100,       # número de árboles
    max_depth=5,            # profundidad máxima
    random_state=42
)

model.fit(X_train, y_train)

# Evaluación inicial
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------------
# Interfaz con Streamlit
# -----------------------------
st.title("Evaluador de Crédito 💳")

st.write("Ingrese los datos del cliente para evaluar si es apto para crédito.")

# Inputs del usuario
edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
salario = st.number_input("Monto salario", min_value=0, value=1000)
prestamo = st.number_input("Monto préstamo", min_value=0, value=500)

# Botón de predicción
if st.button("Evaluar"):
    entrada = pd.DataFrame([[edad, salario, prestamo]], columns=["edad", "salario", "prestamo"])
    resultado = model.predict(entrada)[0]

    if resultado == 1:
        st.success("✅ El cliente es apto para crédito.")
    else:
        st.error("❌ El cliente NO es apto para crédito.")

# Mostrar precisión del modelo
st.write(f"Precisión del modelo en datos de prueba: {acc:.2f}")

  


    
 
