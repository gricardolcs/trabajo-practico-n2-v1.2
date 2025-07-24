"""
Script para generar predicciones basado en el modelo entrenado
Basado en train.py pero enfocado en generar el archivo de predicciones
"""
import pickle
import pandas as pd
import numpy as np

# Configuración
SEED = 42

print("=== GENERANDO PREDICCIONES FINALES ===")

# Leer datos
print("1. Cargando datos...")
df_test = pd.read_csv("data/test.csv")
print(f"   - Datos de test: {df_test.shape}")

# Conservar IDs para el archivo final
test_ids = df_test['id'].copy()

# Preprocesamiento (igual que en train.py)
print("2. Preprocesando datos...")
df_test = df_test.drop(columns=['id'])

# Cambiar valores de ceguera a 0
df_test['eyesight(left)'] = df_test['eyesight(left)'].replace(to_replace=9.9, value=0)
df_test['eyesight(right)'] = df_test['eyesight(right)'].replace(to_replace=9.9, value=0)

# Aplicar transformación logarítmica
def apply_log_transform(df):
    features = list(df.columns)
    for col in features:
        df[col] = df[col].apply(lambda x: np.log1p(x))
    return df

df_test = apply_log_transform(df_test)
print(f"   - Datos preprocesados: {df_test.shape}")

# Cargar modelo
print("3. Cargando modelo entrenado...")
with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)
print("   - Modelo cargado exitosamente")

# Generar predicciones
print("4. Generando predicciones...")
predictions_prob = model.predict_proba(df_test)[:, 1]
predictions_binary = (predictions_prob > 0.5).astype(int)

print(f"   - Total de predicciones: {len(predictions_prob)}")
print(f"   - Fumadores predichos: {np.sum(predictions_binary == 1)}")
print(f"   - No fumadores predichos: {np.sum(predictions_binary == 0)}")
print(f"   - Probabilidad promedio: {np.mean(predictions_prob):.4f}")

# Crear archivo de predicciones con ID y etiquetas
print("5. Creando archivo de predicciones...")
predictions_df = pd.DataFrame({
    'id': test_ids,
    'smoking_probability': predictions_prob,
    'smoking_prediction': predictions_binary
})

# Guardar archivo
output_filename = 'predicciones_finales.csv'
predictions_df.to_csv(output_filename, index=False)

print(f"6. Archivo guardado: {output_filename}")
print("7. Primeras 15 predicciones:")
print(predictions_df.head(15).to_string(index=False))

print("\n=== RESUMEN ===")
print(f"✅ Archivo generado: {output_filename}")
print(f"✅ Total de registros: {len(predictions_df)}")
print(f"✅ Columnas: id, smoking_probability, smoking_prediction")
print("✅ Formato requerido: ID y etiqueta ✓")
print("✅ Proceso completado exitosamente")
