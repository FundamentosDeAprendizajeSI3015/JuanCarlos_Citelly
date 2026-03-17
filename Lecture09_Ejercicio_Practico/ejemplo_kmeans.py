import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

random_state = 42
plt.rc('font', family='serif', size=12)

def procesar_dataset(nombre_archivo):

    print(f"\n===== {nombre_archivo} =====")

    df = pd.read_csv(nombre_archivo, sep=",", encoding="latin-1")

    print(df.head())
    print(df.info())

    df_numeric = df.select_dtypes(include=[np.number])
    data = df_numeric.values

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, np.arange(data.shape[1])),
        ],
    )

    data_processed = preprocessor.fit_transform(data)

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_processed)

    clu_kmeans = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clustering", KMeans(n_clusters=2, random_state=random_state))
    ])

    clu_kmeans.fit(data)

    fig, ax = plt.subplots()
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c=clu_kmeans["clustering"].labels_)
    ax.set_title("K-Means (K=2)")
    fig.set_size_inches(6, 5)

    inert = []
    k_range = list(range(1, 11))

    for k in k_range:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("clustering", KMeans(n_clusters=k, random_state=random_state))
        ])
        model.fit(data)
        inert.append(model["clustering"].inertia_)

    fig, ax = plt.subplots()
    ax.plot(k_range, inert, marker='o')
    ax.set_title("Método del codo")
    ax.set_xlabel("K")
    ax.set_ylabel("Inercia")
    fig.set_size_inches(6, 5)

    clu_kmeans = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clustering", KMeans(n_clusters=4, random_state=random_state))
    ])

    clu_kmeans.fit(data)
    print(f'K=4 Inercia: {clu_kmeans["clustering"].inertia_}')

    fig, ax = plt.subplots()
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c=clu_kmeans["clustering"].labels_)
    ax.set_title("K-Means (K=4)")
    fig.set_size_inches(6, 5)

    clu_dbscan = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clustering", DBSCAN(eps=2.0, min_samples=5))
    ])

    clu_dbscan.fit(data)

    labels = clu_dbscan["clustering"].labels_
    print("DBSCAN clusters:", np.unique(labels, return_counts=True))

    fig, ax = plt.subplots()
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
    ax.set_title("DBSCAN")
    fig.set_size_inches(6, 5)


procesar_dataset("dataset_sintetico_FIRE_UdeA_realista.csv")
procesar_dataset("dataset_sintetico_FIRE_UdeA.csv")

plt.show()