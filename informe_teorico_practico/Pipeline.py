import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42

# Definicion del problema----------------------------------------------
'''
¿Que se quiere predecir?
Se busca predecir el valor de la variable objetivo compra, 
la cual indica si un cliente continuará realizando compras (1) 
o si ha dejado de comprar (0). Esta variable se construye a partir 
de la información de churn del cliente, donde un cliente que no ha 
abandonado se considera como comprador activo.

Este problema se enmarca dentro de un aprendizaje supervisado de 
clasificación binaria.

Impacto esperado:
* Permite identificar clientes con mayor probabilidad de continuar comprando.
* Facilita el análisis de retención de clientes en un entorno de e-commerce.
* Apoya la toma de decisiones en estrategias de fidelización y marketing.
* Contribuye a reducir la pérdida de clientes mediante acciones preventivas.

Variables involucradas:

Variable objetivo:
- compra:
  Variable binaria que indica si un cliente continúa comprando (1)
  o si ha dejado de hacerlo (0). 

Variables de entrada (features):

- Gender:
  Género del cliente, codificado mediante variables dummy.

- Product Category:
  Categoría del producto adquirido, codificada mediante variables dummy.

- Product Price:
  Precio del producto adquirido.

- Quantity:
  Cantidad de productos comprados.

- Total Purchase Amount:
  Monto total gastado en la transacción.

- Payment Method:
  Método de pago utilizado, codificado mediante variables dummy.

- Returns:
  Indica si el cliente realizó devoluciones en compras previas.

Las variables identificadoras, textuales y temporales fueron excluidas
del modelo debido a que no aportan información predictiva relevante
y pueden introducir ruido en el proceso de aprendizaje.

'''


# 2. Recolección de datos-------------------------------------------
df = pd.read_csv("ecommerce_customer_data_large.csv")
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())

# 3. Procesamiento de datos---------------------------------------

print("\nInformación del dataset:")
print(df.info())

# Limpieza
print("\nValores nulos por columna:")
print(df.isnull().sum())
df['Returns'] = df['Returns'].fillna(0)  # Limpieza de datos, coloca en cada fila los valores faltantes
df = df.dropna() # Limpieza de datos, elimina filas con valores nulos
df["compra"] = 1 - df["Churn"]  
df = df.drop(columns=[
    "Churn",           
    "Customer ID",
    "Customer Name",
    "Purchase Date",
    "Age"
])  # Limpieza de datos, elimina columnas que generan ruido           
X = df.drop(columns=["compra"])  
y = df["compra"]  

# Codificación de variables
X = pd.get_dummies(X, drop_first=True) # One-Hot Encoding
print("\nNumero de columnas con One-Hot Encoding:", X.shape[1])

df_pearson = X.copy()
df_pearson['target_compra'] = y
correlaciones = df_pearson.corr(method='pearson') # Correlación
print("\n--- Correlación de Pearson con la variable 'compra' ---")
print(correlaciones['target_compra'].sort_values(ascending=False))
moda_matematica_compra = df.mode() # Moda de cada columna
print("\nModa de las compras:")
print(moda_matematica_compra)

# Visualización de relaciones entre variables con la objetivo, para continuar con el codigo cierre la ventana del gráfico
print("Generando gráfico de relaciones, no olvidar cerrar la ventana del gráfico para continuar con el código")
columnas_interes = ['Customer Age', 'Total Purchase Amount', 'Quantity', 'Product Price', 'Returns']
sns.pairplot(df.sample(1000), vars=columnas_interes, hue='compra', palette='husl', diag_kind='kde')
plt.title("Matriz de Relaciones entre Variables")
plt.show()

# Normalizacion (Use Min-Max Scaler)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Entrenamiento de modelo------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

# model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 5. Evaluación del modelo----------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== RESULTADOS ===")
print(f"Accuracy: {accuracy:.3f}")
print("\nMatriz de confusión:") # Muy importante para ver aciertos y errores
print(cm)
print("\nReporte de clasificación:")
# classification_report muestra para cada clase precisión, recall, F1-score, support
print(classification_report(y_test, y_pred))