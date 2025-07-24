import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

print("Iniciando generación de predicciones...")

# Cargar datos
print("Cargando datos...")
df = pd.read_csv('data/smoking.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Datos de entrenamiento: {df.shape}")
print(f"Datos de test: {test_df.shape}")

# Preprocesar datos
print("Preprocesando datos...")
df_processed = df.drop(columns=['ID', 'gender', 'oral', 'tartar'])
test_processed = test_df.drop(columns=['id'])

# Tratar valores de ceguera
df_processed['eyesight(left)'] = df_processed['eyesight(left)'].replace(9.9, 0)
df_processed['eyesight(right)'] = df_processed['eyesight(right)'].replace(9.9, 0)
test_processed['eyesight(left)'] = test_processed['eyesight(left)'].replace(9.9, 0)
test_processed['eyesight(right)'] = test_processed['eyesight(right)'].replace(9.9, 0)

# Aplicar transformación logarítmica
def apply_log_transform(df, target_col=None):
    df_transformed = df.copy()
    features = list(df.columns)
    if target_col and target_col in features:
        features.remove(target_col)
    
    for col in features:
        df_transformed[col] = df_transformed[col].apply(lambda x: np.log1p(x))
    
    return df_transformed

df_transformed = apply_log_transform(df_processed, 'smoking')
test_transformed = apply_log_transform(test_processed)

# Preparar datos
X = df_transformed.drop(columns=['smoking'])
y = df_transformed['smoking']

print("Entrenando modelo XGBoost...")
# Entrenar modelo
xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 10,
    'min_child_weight': 25,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': -1
}

# Dividir para validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar modelo
model = XGBClassifier(**xgb_params)
model.fit(X_train, y_train)

# Validar modelo
val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"AUC en validación: {val_auc:.4f}")

# Entrenar modelo final con todos los datos
print("Entrenando modelo final...")
final_model = XGBClassifier(**xgb_params)
final_model.fit(X, y)

# Generar predicciones
print("Generando predicciones...")
test_predictions = final_model.predict_proba(test_transformed)[:, 1]
test_predictions_binary = (test_predictions > 0.5).astype(int)

# Crear archivo de predicciones
predictions_df = pd.DataFrame({
    'id': test_df['id'],
    'smoking_probability': test_predictions,
    'smoking_prediction': test_predictions_binary
})

# Guardar predicciones
predictions_df.to_csv('predicciones_finales.csv', index=False)
print(f"Predicciones guardadas en 'predicciones_finales.csv'")
print(f"Total de predicciones: {len(predictions_df)}")
print(f"Fumadores predichos: {np.sum(test_predictions_binary)}")
print(f"No fumadores predichos: {np.sum(test_predictions_binary == 0)}")

# Guardar modelo
with open('modelo_final.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Modelo guardado en 'modelo_final.pkl'")

print("\nPrimeras 10 predicciones:")
print(predictions_df.head(10))

print("\nProceso completado exitosamente!")
