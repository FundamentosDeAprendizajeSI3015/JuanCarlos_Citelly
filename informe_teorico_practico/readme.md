
---

```markdown
# Predicción de Retención de Clientes en E-commerce

## Descripción general
Este proyecto implementa un modelo de aprendizaje automático cuyo objetivo es
predecir si un cliente de un e-commerce continuará realizando compras o si ha
dejado de comprar.

La variable objetivo es **compra**, una variable binaria construida a partir de
la variable **Churn**:
- `compra = 1`: el cliente continúa comprando
- `compra = 0`: el cliente ha dejado de comprar

Este problema corresponde a un aprendizaje supervisado de clasificación binaria.

---

## Objetivo del modelo
El modelo busca apoyar el análisis de retención de clientes, permitiendo:
- Identificar clientes con mayor probabilidad de seguir comprando
- Analizar el abandono de clientes (churn)
- Apoyar estrategias de fidelización y marketing
- Reducir la pérdida de clientes en plataformas de e-commerce

---

## Dataset
Se utiliza el archivo:

```

ecommerce_customer_data_large.csv

```

El dataset contiene información demográfica y transaccional de clientes.

---

## Variables

### Variable objetivo
- **compra**: indica si el cliente continúa comprando (1) o no (0)

### Variables de entrada
- Gender  
- Product Category  
- Product Price  
- Quantity  
- Total Purchase Amount  
- Payment Method  
- Returns  

### Variables eliminadas
- Customer ID  
- Customer Name  
- Purchase Date  
- Age  
- Churn  

Estas variables fueron eliminadas por no aportar información relevante
o por introducir ruido en el modelo.

---

## Procesamiento de datos
- Limpieza de valores nulos
- Creación de la variable objetivo **compra**
- Eliminación de columnas no relevantes
- Codificación de variables categóricas mediante **One-Hot Encoding**
- Normalización de datos usando **Min-Max Scaler**
- Análisis exploratorio mediante correlación de Pearson y cálculo de la moda

---

## Entrenamiento del modelo
- División del dataset en entrenamiento (75%) y prueba (25%)
- Estratificación por la variable objetivo
- Uso de **Random Forest Classifier** con balanceo de clases o tambien regresion logistica

---

## Evaluación
El modelo se evalúa utilizando:
- Accuracy
- Matriz de confusión
- Reporte de clasificación (precisión, recall y F1-score)

---

## Conclusión
Este proyecto presenta un pipeline completo de aprendizaje automático aplicado
a un problema real de retención de clientes, integrando preprocesamiento,
entrenamiento y evaluación de un modelo de clasificación.
```

---
