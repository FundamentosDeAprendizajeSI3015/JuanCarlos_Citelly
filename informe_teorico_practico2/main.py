import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN

# Fuzzy C-Means
import skfuzzy as fuzz

df = pd.read_csv("ecommerce.csv")

df = df.drop(columns=["Customer ID", "Customer Name"], errors='ignore')

if "Customer Age" in df.columns and "Age" in df.columns:
    df = df.drop(columns=["Customer Age"])

df["Purchase Date"] = pd.to_datetime(df["Purchase Date"])
df["month"] = df["Purchase Date"].dt.month
df["day_of_week"] = df["Purchase Date"].dt.dayofweek

df = df.drop(columns=["Purchase Date"])

X = df.drop("Churn", axis=1)
y = df["Churn"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Pipeline numérico
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Pipeline categórico
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocesador completo
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# modelos supervisados Dataset original
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Linear Regression": LinearRegression()
}

results_original = {}

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if name == "Linear Regression":
        y_pred = (y_pred > 0.5).astype(int)
    
    results_original[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

print("\n=== RESULTADOS DATASET ORIGINAL ===")
for k, v in results_original.items():
    print(k, v)

# clustering no supervisado


output_dir = "graficas"
os.makedirs(output_dir, exist_ok=True)


X_processed = preprocessor.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_processed)

# dbscan
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_processed)

# Fuzzy c means
X_np = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
X_np = X_np.T

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_np, c=3, m=2, error=0.005, maxiter=100
)

clusters_fuzzy = np.argmax(u, axis=0)

# ============================================
# FUNCIÓN PARA GUARDAR GRÁFICAS
# ============================================
def guardar_grafica(X, labels, titulo, filename):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(titulo)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# ============================================
# GENERAR GRÁFICAS
# ============================================

guardar_grafica(X_pca, clusters_kmeans, "K-Means Clustering", "kmeans.png")
guardar_grafica(X_pca, clusters_dbscan, "DBSCAN Clustering", "dbscan.png")
guardar_grafica(X_pca, clusters_fuzzy, "Fuzzy C-Means Clustering", "fuzzy.png")

# Evaluación clustering
try:
    print("\nSilhouette Score KMeans:", silhouette_score(X_processed, clusters_kmeans))
except:
    print("\nNo se pudo calcular Silhouette Score")


# REETIQUETADO (30% DE RUIDO)

y_noisy = y.copy()

n = int(0.3 * len(y))
indices = np.random.choice(len(y), n, replace=False)

y_noisy.iloc[indices] = 1 - y_noisy.iloc[indices]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy, test_size=0.2, random_state=42
)

results_noisy = {}

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)


    if name == "Linear Regression":
        y_pred = (y_pred > 0.5).astype(int)
    
    results_noisy[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

print("\n=== RESULTADOS CON RUIDO (30%) ===")
for k, v in results_noisy.items():
    print(k, v)


# Comparacion ffinal
print("\n=== COMPARACIÓN FINAL ===")

for model in models.keys():
    print(f"\nModelo: {model}")
    print("Original:", results_original[model])
    print("Con ruido:", results_noisy[model])
    print("-" * 40)