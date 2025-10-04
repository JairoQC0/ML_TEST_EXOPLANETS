# predict.py
import pandas as pd
import joblib

# 1. Cargar modelo entrenado
model = joblib.load("model.pkl")
print("✅ Modelo cargado correctamente")

# 2. Definir features
features = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad"
]

# 3. Pedir datos al usuario
print("\nIngresa los valores para el planeta candidato:")
valores = []
for f in features:
    valor = float(input(f"{f}: "))
    valores.append(valor)

# 4. Crear DataFrame con los valores ingresados
ejemplo = pd.DataFrame([dict(zip(features, valores))])

# 5. Hacer predicción
pred = model.predict(ejemplo)
print(f"\n🔮 Predicción para el ejemplo: {pred[0]}")
