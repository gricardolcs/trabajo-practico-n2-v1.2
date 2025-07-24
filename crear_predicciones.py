#!/usr/bin/env python3
"""
Script simple para generar predicciones usando el modelo existente
"""

import pickle
import pandas as pd
import numpy as np

print("=" * 50)
print("GENERANDO PREDICCIONES CON MODELO EXISTENTE")
print("=" * 50)

try:
    # Cargar modelo existente
    print("1. Cargando modelo existente...")
    with open('model.bin', 'rb') as f:
        model = pickle.load(f)
    print("   ✅ Modelo cargado exitosamente")
    
    # Cargar datos de test
    print("\n2. Cargando datos de test...")
    test_df = pd.read_csv('data/test.csv')
    print(f"   ✅ Datos de test cargados: {test_df.shape}")
    
    # Preprocesar datos de test (igual que en train.py)
    print("\n3. Preprocesando datos...")
    
    # Eliminar columna id pero conservarla para el archivo final
    test_ids = test_df['id'].copy()
    test_processed = test_df.drop(columns=['id'])
    
    # Tratar valores de ceguera (9.9) como 0
    test_processed['eyesight(left)'] = test_processed['eyesight(left)'].replace(9.9, 0)
    test_processed['eyesight(right)'] = test_processed['eyesight(right)'].replace(9.9, 0)
    
    # Aplicar transformación logarítmica
    for col in test_processed.columns:
        test_processed[col] = test_processed[col].apply(lambda x: np.log1p(x))
    
    print(f"   ✅ Datos preprocesados: {test_processed.shape}")
    
    # Generar predicciones
    print("\n4. Generando predicciones...")
    predictions_prob = model.predict_proba(test_processed)[:, 1]
    predictions_binary = (predictions_prob > 0.5).astype(int)
    
    print(f"   ✅ Predicciones generadas: {len(predictions_prob)}")
    print(f"   - Fumadores predichos: {np.sum(predictions_binary == 1)}")
    print(f"   - No fumadores predichos: {np.sum(predictions_binary == 0)}")
    print(f"   - Probabilidad promedio: {np.mean(predictions_prob):.4f}")
    
    # Crear DataFrame con predicciones
    print("\n5. Creando archivo de predicciones...")
    predictions_df = pd.DataFrame({
        'id': test_ids,
        'smoking_probability': predictions_prob,
        'smoking_prediction': predictions_binary
    })
    
    # Guardar archivo
    output_file = 'predicciones_finales.csv'
    predictions_df.to_csv(output_file, index=False)
    print(f"   ✅ Archivo guardado: {output_file}")
    
    # Mostrar primeras predicciones
    print("\n6. Primeras 10 predicciones:")
    print(predictions_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 50)
    print("RESUMEN")
    print("=" * 50)
    print(f"✅ Total de predicciones: {len(predictions_df)}")
    print(f"✅ Archivo generado: {output_file}")
    print(f"✅ Columnas: id, smoking_probability, smoking_prediction")
    print(f"✅ Modelo utilizado: XGBoost (desde model.bin)")
    print("✅ Proceso completado exitosamente")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
