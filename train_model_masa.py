import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import joblib
import numpy as np
import xgboost as xgb 

# --- Configuración ---
FILE_PATH = 'data/K2.csv' 
TARGET_COLUMN = 'pl_bmasse' 
MODEL_FILENAME = 'modelo_masa_exoplaneta_ALTO_R2.pkl'

# Características de entrada (se añade la característica ingenierizada)
FEATURE_COLUMNS = [
    'pl_orbper',
    'pl_orbsmax',
    'pl_rade',
    'pl_rade_cubed',
    'st_teff',
    'st_rad',
    'st_mass',
    'pl_eqt',
    'pl_insol'
]

def sanear_nombres_columnas(df):
    """Limpia los nombres de las columnas para evitar errores con XGBoost."""
    # Reemplaza caracteres problemáticos con un guion bajo
    df.columns = df.columns.str.replace(r'[\[\]<>]', '_', regex=True)
    # También es buena práctica reemplazar espacios por guiones bajos
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
    return df

def entrenar_modelo_alto_r2():
    print("--- INICIANDO ENTRENAMIENTO DEL MODELO DE ALTO RENDIMIENTO ---")

    try:
        df = pd.read_csv(FILE_PATH, skiprows=1) 
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return

    # APLICACIÓN DE LA CORRECCIÓN DEL ERROR XGBOOST
    df = sanear_nombres_columnas(df)

    # 1. Filtro de Calidad y Limpieza
    # Asegúrate de que las columnas de error también estén saneadas si las usas, aunque en este caso no las usamos en el filtrado
    df_cleaned = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Filtrar exoplanetas con un error de masa muy alto
    # Tuvimos que cambiar 'pl_bmasseerr1' y otras por si tienen caracteres problemáticos
    # Asumo que las columnas no contenían corchetes en los nombres originales, si no, usa las columnas saneadas
    df_cleaned = df_cleaned[
        (df_cleaned[TARGET_COLUMN] > 0) & 
        (df_cleaned['pl_bmasseerr1'].abs() < 500)
    ].copy()
    
    # 2. Ingeniería de Características
    # Añadir Radio al cubo (pl_rade^3)
    df_cleaned.loc[:, 'pl_rade_cubed'] = df_cleaned['pl_rade'] ** 3
    
    # Variables
    X = df_cleaned[[c for c in FEATURE_COLUMNS if c in df_cleaned.columns]] # Asegura que las columnas existan
    
    # Aplicar transformación logarítmica a la variable objetivo (y)
    y = np.log(df_cleaned[TARGET_COLUMN] + 1e-6) 

    # Imputación
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns) # Usamos X.columns (saneadas)

    # 3. División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    # 4. Entrenamiento del modelo (XGBoost)
    print("Entrenando XGBoost Regressor (Gradient Boosting)...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Entrenamiento con XGBoost completado.")

    # 5. Evaluación del modelo
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nCoeficiente de Determinación (R²): {r2:.4f}")

    # 6. Guardar el nuevo modelo y el imputer
    joblib.dump({
        'model': model,
        'imputer': imputer,
        'features': X.columns.tolist() # Guardamos la lista de columnas saneadas
    }, MODEL_FILENAME)
    print(f"\nModelo, Imputer y Features guardados como: {MODEL_FILENAME}")
    print("--- FIN DEL ENTRENAMIENTO ---")

# Ejecutar el entrenamiento
entrenar_modelo_alto_r2()