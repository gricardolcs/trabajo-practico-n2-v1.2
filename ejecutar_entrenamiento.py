#!/usr/bin/env python3
"""
Script para generar entrenamiento, validación y predicciones finales
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuración
SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("ENTRENAMIENTO Y VALIDACIÓN DE MODELOS")
print("=" * 60)

# 1. CARGA DE DATOS
print("\n1. Cargando datos...")
df = pd.read_csv('data/smoking.csv')
test_df = pd.read_csv('data/test.csv')

print(f"   - Datos de entrenamiento: {df.shape}")
print(f"   - Datos de test: {test_df.shape}")
print(f"   - Distribución de fumadores: {df['smoking'].value_counts().to_dict()}")

# 2. PREPROCESAMIENTO
print("\n2. Preprocesando datos...")

# Eliminar columnas no necesarias
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

print(f"   - Dataset procesado: {df_transformed.shape}")
print(f"   - Características: {list(df_transformed.drop(columns=['smoking']).columns)}")

# 3. PREPARACIÓN DE DATOS
print("\n3. Preparando datos para entrenamiento...")
X = df_transformed.drop(columns=['smoking'])
y = df_transformed['smoking']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"   - Entrenamiento: {X_train.shape}")
print(f"   - Validación: {X_val.shape}")

# 4. FUNCIÓN DE VALIDACIÓN CRUZADA
def fit_model_with_skf(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"      Fold {fold}/{n_splits}...", end='\r')
        
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict_proba(X_val_fold)[:, 1]
        
        auc = roc_auc_score(y_val_fold, y_pred)
        scores.append(auc)
    
    print(' ' * 30, end='\r')
    return np.mean(scores), np.std(scores)

# 5. ENTRENAMIENTO DE MODELOS
print("\n4. Entrenando modelos...")

# 5.1 Regresión Logística
print("\n   4.1 Regresión Logística")
lr_model = LogisticRegression(random_state=SEED, max_iter=1000)
lr_score, lr_std = fit_model_with_skf(X, y, lr_model)
lr_model.fit(X_train, y_train)
lr_pred_val = lr_model.predict_proba(X_val)[:, 1]
lr_auc_val = roc_auc_score(y_val, lr_pred_val)

print(f"       AUC CV: {lr_score:.4f} ± {lr_std:.4f}")
print(f"       AUC Val: {lr_auc_val:.4f}")

# 5.2 Random Forest
print("\n   4.2 Random Forest")
rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf_score, rf_std = fit_model_with_skf(X, y, rf_model)
rf_model.fit(X_train, y_train)
rf_pred_val = rf_model.predict_proba(X_val)[:, 1]
rf_auc_val = roc_auc_score(y_val, rf_pred_val)

print(f"       AUC CV: {rf_score:.4f} ± {rf_std:.4f}")
print(f"       AUC Val: {rf_auc_val:.4f}")

# 5.3 XGBoost
print("\n   4.3 XGBoost")
xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 10,
    'min_child_weight': 25,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': SEED,
    'verbosity': 0,
    'n_jobs': -1
}

xgb_model = XGBClassifier(**xgb_params)
xgb_score, xgb_std = fit_model_with_skf(X, y, xgb_model)
xgb_model.fit(X_train, y_train)
xgb_pred_val = xgb_model.predict_proba(X_val)[:, 1]
xgb_auc_val = roc_auc_score(y_val, xgb_pred_val)

print(f"       AUC CV: {xgb_score:.4f} ± {xgb_std:.4f}")
print(f"       AUC Val: {xgb_auc_val:.4f}")

# 6. COMPARACIÓN DE MODELOS
print("\n5. Comparación de modelos:")
results = {
    'Modelo': ['Regresión Logística', 'Random Forest', 'XGBoost'],
    'AUC_CV': [f"{lr_score:.4f} ± {lr_std:.4f}", 
               f"{rf_score:.4f} ± {rf_std:.4f}", 
               f"{xgb_score:.4f} ± {xgb_std:.4f}"],
    'AUC_Val': [f"{lr_auc_val:.4f}", f"{rf_auc_val:.4f}", f"{xgb_auc_val:.4f}"]
}

results_df = pd.DataFrame(results)
print("\n" + "=" * 50)
print("RESUMEN DE RESULTADOS")
print("=" * 50)
print(results_df.to_string(index=False))

# Determinar mejor modelo
scores = [lr_score, rf_score, xgb_score]
best_idx = np.argmax(scores)
best_model_name = results['Modelo'][best_idx]
print(f"\n🏆 Mejor modelo: {best_model_name}")

# 7. ENTRENAMIENTO DEL MODELO FINAL
print("\n6. Entrenando modelo final...")
final_model = XGBClassifier(**xgb_params)
final_model.fit(X, y)

# 8. ANÁLISIS DEL MEJOR MODELO
print("\n7. Análisis detallado del modelo XGBoost:")
xgb_pred_binary = (xgb_pred_val > 0.5).astype(int)
accuracy = accuracy_score(y_val, xgb_pred_binary)
print(f"   - Accuracy: {accuracy:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_val, xgb_pred_binary)
print(f"   - Matriz de confusión:")
print(f"     TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"     FN: {cm[1,0]}, TP: {cm[1,1]}")

# Top 5 características más importantes
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   - Top 5 características más importantes:")
for i, row in feature_importance.head(5).iterrows():
    print(f"     {row['feature']}: {row['importance']:.4f}")

# 9. GENERACIÓN DE PREDICCIONES
print("\n8. Generando predicciones...")
test_predictions = final_model.predict_proba(test_transformed)[:, 1]
test_predictions_binary = (test_predictions > 0.5).astype(int)

# Crear archivo de predicciones con ID y etiquetas
predictions_df = pd.DataFrame({
    'id': test_df['id'],
    'smoking_probability': test_predictions,
    'smoking_prediction': test_predictions_binary
})

# Guardar predicciones
predictions_df.to_csv('predicciones_finales.csv', index=False)

print(f"   - Predicciones generadas: {len(predictions_df)}")
print(f"   - No fumadores predichos: {np.sum(test_predictions_binary == 0)}")
print(f"   - Fumadores predichos: {np.sum(test_predictions_binary == 1)}")
print(f"   - Archivo guardado: predicciones_finales.csv")

# 10. GUARDAR MODELO
print("\n9. Guardando modelo...")
with open('modelo_final.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Guardar información del modelo
model_info = {
    'model_type': 'XGBoost',
    'parameters': xgb_params,
    'cv_score': float(xgb_score),
    'cv_std': float(xgb_std),
    'validation_auc': float(xgb_auc_val),
    'features': list(X.columns),
    'training_samples': len(X),
    'accuracy': float(accuracy)
}

import json
with open('modelo_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"   - Modelo guardado: modelo_final.pkl")
print(f"   - Info guardada: modelo_info.json")

# 11. RESUMEN FINAL
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"✅ Dataset procesado: {X.shape[0]} registros, {X.shape[1]} características")
print(f"✅ Modelos evaluados: 3 (Logística, Random Forest, XGBoost)")
print(f"✅ Mejor modelo: XGBoost (AUC: {xgb_score:.4f})")
print(f"✅ Archivo de predicciones: predicciones_finales.csv ({len(predictions_df)} filas)")
print(f"✅ Modelo guardado: modelo_final.pkl")
print(f"✅ Entrenamiento completado exitosamente")
print("\nArchivos generados:")
print("- predicciones_finales.csv: ID y etiquetas de predicción")
print("- modelo_final.pkl: Modelo entrenado")
print("- modelo_info.json: Información del modelo")

print(f"\nPrimeras 10 predicciones:")
print(predictions_df.head(10).to_string(index=False))
