# api.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# --- Inicialización y Carga ---

# 1. Cargar el modelo entrenado
try:
    # Asegúrate de que este path sea correcto (está en el mismo directorio)
    MODEL = joblib.load("model.pkl")
    print("✅ Modelo cargado correctamente.")
except FileNotFoundError:
    print("❌ ERROR: No se encontró 'model.pkl'. ¿Entrenaste el modelo?")
    MODEL = None
except Exception as e:
    print(f"❌ ERROR al cargar el modelo: {e}")
    MODEL = None

# 2. Definir las features (debe coincidir con train_model.py)
FEATURES = [
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

# 3. Inicializar la aplicación Flask
app = Flask(__name__)


# --- Rutas de la API ---

@app.route('/')
def home():
    """Ruta de bienvenida o chequeo de salud."""
    return jsonify({
        "status": "ok",
        "message": "API de Predicción de Exoplanetas está funcionando.",
        "model_loaded": MODEL is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Ruta para recibir datos JSON y retornar una predicción.
    El JSON esperado debe ser un diccionario con las claves de FEATURES.
    """
    if MODEL is None:
        return jsonify({"error": "Modelo no cargado. Revisa los logs del servidor."}), 500

    # 1. Obtener los datos del request
    data = request.get_json(force=True)
    
    # 2. Validación y preparación de datos
    try:
        # Crea un diccionario solo con las features necesarias y en el orden correcto
        # Esto es crucial para la consistencia
        input_data = {
            feature: data[feature] for feature in FEATURES
        }
        
        # Convierte el diccionario a un DataFrame de pandas
        df_input = pd.DataFrame([input_data], columns=FEATURES)

    except KeyError as e:
        return jsonify({
            "error": "Datos de entrada incompletos o incorrectos.",
            "missing_key": str(e),
            "expected_keys": FEATURES
        }), 400 # 400 Bad Request

    # 3. Realizar la predicción
    try:
        # La predicción devuelve un array, tomamos el primer (y único) elemento
        prediction = MODEL.predict(df_input)[0]
        
        # Opcional: obtener las probabilidades (útil para confianza)
        probabilities = MODEL.predict_proba(df_input)[0]
        class_labels = MODEL.classes_
        
        # Crear un diccionario de probabilidades para el JSON
        prob_dict = dict(zip(class_labels, probabilities.tolist()))
        
        # 4. Retornar la respuesta
        return jsonify({
            "prediction": prediction,
            "probabilities": prob_dict,
            "status": "success"
        })
        
    except Exception as e:
        # En caso de un error inesperado durante la predicción
        print(f"Error durante la predicción: {e}")
        return jsonify({
            "error": "Error interno del servidor durante la predicción.",
            "details": str(e)
        }), 500 # 500 Internal Server Error


# --- Ejecución (Solo para desarrollo) ---

if __name__ == '__main__':
    # Usar '0.0.0.0' para que sea accesible externamente (útil en contenedores)
    app.run(debug=True, host='0.0.0.0', port=5000)