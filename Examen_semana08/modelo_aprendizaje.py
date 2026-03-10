import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Configuración estética
random_state = 42
plt.rc('font', family='serif', size=12)

# CARGA DE DATOS
data = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")

# Variables categóricas (Identificadores)
cat_cols = ['anio', 'unidad']

# Variables numéricas (Indicadores financieros)
num_cols = [
    'ingresos_totales', 'gastos_personal', 'liquidez', 'dias_efectivo', 
    'cfo', 'participacion_regalias', 
    'participacion_servicios', 'participacion_matriculas', 'hhi_fuentes', 
    'endeudamiento', 'tendencia_ingresos', 'gp_ratio'
]

target = 'label'


# Eliminar filas donde el objetivo sea nulo
data = data.dropna(subset=[target])

# Convertir columnas numéricas por seguridad (manejo de posibles strings)
for col in num_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# MATRIZ DE CORRELACIÓN
# Calculamos correlaciones entre variables numéricas y el target
corr = data[num_cols + [target]].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={'label': 'Correlación'})
plt.title('Matriz de Correlación de Variables Numéricas y Objetivo')
plt.show()

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler()) # Normalización vital para finanzas
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, cat_cols),
    ('num', numeric_transformer, num_cols)
])

X = data[cat_cols + num_cols]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# DEFINICIÓN DEL MODELO (Random Forest Clasificador)
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=random_state))
])

# Ajuste de hiperparámetros (Optimizado para análisis financiero)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_leaf': [1, 5]
}

# ENTRENAMIENTO CON VALIDACIÓN CRUZADA
rf_search = GridSearchCV(pipeline_rf, cv=5, param_grid=param_grid, n_jobs=-1)
rf_search.fit(X_train, y_train)

best_model = rf_search.best_estimator_

# EVALUACIÓN
print(f"Mejores Parámetros: {rf_search.best_params_}")

def evaluar_modelo(X, y, nombre):
    preds = best_model.predict(X)
    print(f"--- Reporte {nombre} ---")
    print(f'Accuracy: {accuracy_score(y, preds):.4f}')
    print(f'Precision: {precision_score(y, preds):.4f}')
    print(f'Recall: {recall_score(y, preds):.4f}')
    print(f'F1-Score: {f1_score(y, preds):.4f}')
    print('\n')

evaluar_modelo(X_train, y_train, "Entrenamiento")
evaluar_modelo(X_test, y_test, "Pruebas (Test)")

# VISUALIZACIÓN DE RESULTADOS - MATRIZ DE CONFUSIÓN
y_pred = best_model.predict(X_test)

# Crear matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar matriz de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión en números absolutos
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Clase 0', 'Clase 1'],
            yticklabels=['Clase 0', 'Clase 1'],
            cbar_kws={'label': 'Cantidad'}, ax=axes[0])
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Valor Real')
axes[0].set_title('Matriz de Confusión - Valores Absolutos')

# Matriz de confusión normalizada (porcentajes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
            xticklabels=['Clase 0', 'Clase 1'],
            yticklabels=['Clase 0', 'Clase 1'],
            cbar_kws={'label': 'Porcentaje'}, ax=axes[1])
axes[1].set_xlabel('Predicción')
axes[1].set_ylabel('Valor Real')
axes[1].set_title('Matriz de Confusión - Porcentajes')

plt.tight_layout()
plt.show()

# Imprimir reporte de clasificación detallado
print("\n=== REPORTE DE CLASIFICACIÓN DETALLADO ===")
print(classification_report(y_test, y_pred, target_names=['Clase 0', 'Clase 1']))