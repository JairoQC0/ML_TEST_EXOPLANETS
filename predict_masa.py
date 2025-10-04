import joblib
import pandas as pd
import numpy as np
import sys 
import os # Para verificar la existencia del archivo

# --- Configuración ---
# El nombre del archivo DEBE coincidir con el del script de entrenamiento optimizado
MODEL_FILENAME = 'modelo_masa_exoplaneta_ALTO_R2.pkl'

# Características que le pediremos al usuario (pl_rade_cubed se calcula internamente)
FEATURE_PROMPTS = {
    'pl_orbper': 'Período Orbital (días)',
    'pl_orbsmax': 'Semieje mayor orbital (AU)',
    'pl_rade': 'Radio del planeta (Radios Terrestres)', 
    'st_teff': 'Temperatura efectiva de la estrella (K)',
    'st_rad': 'Radio de la estrella (Radios Solares)',
    'st_mass': 'Masa de la estrella (Masas Solares)',
    'pl_eqt': 'Temperatura de equilibrio del planeta (K)',
    'pl_insol': 'Irradiancia estelar del planeta (Flujos Terrestres)'
}

def obtener_datos_usuario(features_to_ask):
    """
    Solicita al usuario que ingrese los datos para las características base.
    """
    print("\n--- INGRESO DE DATOS DEL EXOPLANETA ---")
    data_input = {}
    for feature_name, prompt in features_to_ask.items():
        while True:
            try:
                # Se pide el dato y se convierte a punto flotante (float)
                valor = float(input(f"Ingrese {prompt}: "))
                # Validación básica para valores físicos positivos
                if feature_name in ['pl_orbper', 'pl_rade', 'st_mass'] and valor <= 0:
                     print("¡Advertencia! Los valores físicos clave deben ser positivos. Intente de nuevo.")
                     continue
                data_input[feature_name] = valor 
                break
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")
    return data_input

def predecir_masa_exoplaneta_interactivo():
    """
    Carga el modelo entrenado, pide los datos, calcula la nueva característica,
    y realiza la predicción con la transformación inversa.
    """
    print("--- INICIANDO ALGORITMO DE PREDICCIÓN ---")

    # 1. Cargar el modelo y sus activos
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: No se encontró el archivo del modelo: '{MODEL_FILENAME}'.")
        print("Por favor, asegúrese de que el script de entrenamiento se haya ejecutado con éxito.")
        sys.exit(1)

    try:
        model_assets = joblib.load(MODEL_FILENAME)
        model = model_assets['model']
        imputer = model_assets['imputer']
        FEATURE_COLUMNS = model_assets['features'] 
        print(f"Modelo cargado correctamente ({type(model).__name__}).")
    except Exception as e:
        print(f"Error al cargar el modelo o los activos: {e}")
        sys.exit(1)

    # 2. Obtener datos del usuario (solo las 8 características básicas)
    # Solo pedimos los datos que no son 'pl_rade_cubed'
    features_to_ask = {k: v for k, v in FEATURE_PROMPTS.items()}
    new_data_dict_single_value = obtener_datos_usuario(features_to_ask)
    
    # 3. Ingeniería de Características: Calcular pl_rade_cubed
    radio = new_data_dict_single_value['pl_rade']
    new_data_dict_single_value['pl_rade_cubed'] = radio ** 3
    
    # 4. Preparación de los datos de entrada
    
    # Convertir el diccionario a DataFrame, asegurando el orden de las columnas del modelo
    try:
        new_data_df = pd.DataFrame([new_data_dict_single_value], columns=FEATURE_COLUMNS)
    except ValueError as e:
        print(f"Error al construir el DataFrame de entrada. Asegúrese de que todas las características esperadas por el modelo están presentes.")
        print(f"Características esperadas: {FEATURE_COLUMNS}")
        sys.exit(1)

    # Aplicar el imputer (preprocesamiento)
    new_X_imputed = imputer.transform(new_data_df)
    
     # 5. Predicción y Transformación Inversa
    
    # El modelo predice np.log(masa)
    prediction_log = model.predict(new_X_imputed)[0]
    
    # CAMBIO CRUCIAL: Revertir la transformación logarítmica para obtener la masa real
    # Fórmula: e^(y_predicha) - 1e-6
    prediction_masa_real = np.exp(prediction_log) - 1e-6 
    
    # Aseguramos que la masa no sea negativa (aunque es raro)
    prediction_masa_real = np.maximum(0, prediction_masa_real)
    # 6. Mostrar Resultado
    print("\n" + "="*70)
    print("                 RESULTADO DE LA PREDICCIÓN CON MODELO OPTIMIZADO")
    print("-" * 70)
    print(f"  Masa Predicha: {prediction_masa_real:.3f} Masas Terrestres (M_Tierra)")
    print("="*70 + "\n")

# Ejecutar la función de predicción
if __name__ == "__main__":
    predecir_masa_exoplaneta_interactivo()