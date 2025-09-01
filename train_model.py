# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Cargar dataset
df = pd.read_csv("data/cumulative.csv")
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# 2. Definir las features importantes
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

# 3. Filtrar solo CONFIRMED y FALSE POSITIVE
df_filtered = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
X = df_filtered[features]
y = df_filtered["koi_disposition"]

# 4. Separar en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluar modelo
y_pred = model.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 7. Guardar modelo entrenado
joblib.dump(model, "model.pkl")
print("\n✅ Modelo guardado como model.pkl")
