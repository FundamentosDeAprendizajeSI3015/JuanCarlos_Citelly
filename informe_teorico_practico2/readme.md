# 🛒 Proyecto de Machine Learning - E-commerce Churn Analysis

## 📌 Descripción

Este proyecto tiene como objetivo analizar el comportamiento de clientes en un dataset de e-commerce, utilizando técnicas de aprendizaje automático supervisado y no supervisado para predecir el churn (abandono de clientes) y explorar patrones en los datos.

---

## 📊 Dataset

El dataset contiene información relevante de clientes, incluyendo:

- Customer ID
- Purchase Date
- Product Category
- Product Price
- Quantity
- Total Purchase Amount
- Payment Method
- Age / Customer Age
- Gender
- Returns
- Churn (variable objetivo)

---

## ⚙️ Preprocesamiento

Se realizaron las siguientes etapas:

- Eliminación de columnas irrelevantes (`Customer ID`, `Customer Name`)
- Conversión de fechas a variables numéricas (mes, día de la semana)
- Manejo de valores faltantes con imputación
- Escalado de variables numéricas
- Codificación de variables categóricas (OneHotEncoder)

---

## 🤖 Modelos Supervisados

Se entrenaron los siguientes modelos:

- Regresión Logística
- Árbol de Decisión
- Regresión Lineal

### 📈 Métricas utilizadas

- Accuracy
- Precision
- Recall
- F1-Score

---

## ⚠️ Resultados principales

| Modelo | Accuracy | Precision | Recall | F1 |
|--------|--------|----------|--------|----|
| Logistic Regression | ~0.80 | 0.00 | 0.00 | 0.00 |
| Decision Tree | ~0.80 | 0.00 | 0.00 | 0.00 |
| Linear Regression | ~0.80 | 0.00 | 0.00 | 0.00 |

### 🔎 Interpretación

- El modelo predice principalmente la clase mayoritaria
- Existe un fuerte **desbalance de clases**
- El accuracy es alto pero **engañoso**

---

## 🔄 Reevaluación con ruido (30%)

Se introdujo ruido en el 30% de las etiquetas para simular errores en los datos.

### 📉 Resultados

| Modelo | Accuracy |
|--------|--------|
| Logistic Regression | ~0.62 |
| Decision Tree | ~0.62 |
| Linear Regression | ~0.62 |

### 🔎 Conclusión

- El rendimiento disminuye significativamente
- Los modelos son sensibles a la calidad de los datos

---

## 🧠 Aprendizaje No Supervisado

Se aplicaron técnicas de clustering:

- K-Means
- DBSCAN
- Fuzzy C-Means

### 📊 Observaciones

- No se identificaron clusters bien definidos
- Silhouette Score bajo (~0.09)
- Los datos no presentan una estructura clara

---

## 📉 Visualización

Se utilizó PCA para reducir la dimensionalidad a 2 componentes y visualizar los clusters generados.

---

## ⚠️ Conclusiones Generales

- El dataset presenta **desbalance de clases**
- Los modelos no logran identificar correctamente el churn
- El accuracy no es una métrica suficiente
- No existen agrupaciones naturales en los datos
- La calidad de los datos impacta directamente el rendimiento

---

## 🚀 Mejoras Futuras

- Aplicar técnicas de balanceo (SMOTE, undersampling)
- Probar modelos más robustos (Random Forest, XGBoost)
- Ajuste de hiperparámetros
- Selección de características

---

## 🛠️ Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Scikit-fuzzy

---

## 📂 Ejecución

```bash
pip install -r requirements.txt
python main.py