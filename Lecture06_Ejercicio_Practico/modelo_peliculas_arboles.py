import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    PolynomialFeatures,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer # CAMBIO: Importación necesaria para manejar valores nulos (NaN)

random_state = 42

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.rc('font', family='serif', size=12)

data = pd.read_csv("movies.csv")

# LIMPIEZA INICIAL: Quitar espacios y saltos de línea (\n) de todo el dataset
data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

# --- VOTES: Quitar comas (ej: 1,234 -> 1234)
if 'VOTES' in data.columns:
    data['VOTES'] = data['VOTES'].astype(str).str.replace(',', '', regex=False)
    data['VOTES'] = pd.to_numeric(data['VOTES'], errors='coerce')

# --- Gross: Quitar '$' y 'M' (ej: $0.01M -> 0.01)
if 'Gross' in data.columns:
    data['Gross'] = data['Gross'].astype(str).str.replace('$', '', regex=False).str.replace('M', '', regex=False)
    data['Gross'] = pd.to_numeric(data['Gross'], errors='coerce')

cols_num_directas = ['RATING', 'RunTime']
for col in cols_num_directas:
    data[col] = pd.to_numeric(data[col], errors='coerce')


print("Estadísticas después de la limpieza:")
print(data.describe()) 

data.hist(bins=50, figsize=(20, 15))

cat_cols = ['GENRE', 'STARS'] 
num_cols = ['VOTES', 'RunTime']

data = data.dropna(subset=['RATING']) 

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), # CAMBIO: Manejo de nulos en texto
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]
)
encoder = categorical_transformer.fit(data.loc[:,cat_cols])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())
])

X_train, X_test, y_train, y_test = train_test_split(
    data[cat_cols + num_cols], 
    data['RATING'], 
    test_size=0.9, # originalmente 0.2
    random_state=random_state
)

#Definimos nuestro Pipeline de pre-procesamiento
preprocessor = ColumnTransformer(
    transformers = [
       ('cat', categorical_transformer, cat_cols),
       ('num', numeric_transformer, num_cols) 
      ])

#Definimos nuestro regresor
rf_base = RandomForestRegressor(random_state=random_state)

pipeline_rf = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', rf_base),
                            ])


param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': list(range(6, 10)),
    'regressor__min_samples_leaf': [10, 300] # Ajuste de valores para que el modelo sea más sensible
}

# Definamos nuestros modelo mediante GridSearchCV:
rf = GridSearchCV(pipeline_rf, cv=3, param_grid=param_grid)
rf.fit(X_train, y_train)
print(rf.best_params_)


print("Train set")
for name, model in [('Random Forest', rf.best_estimator_)]:
    print(f"Model: {name}")
    print(f'R^2: {model.score(X_train, y_train)}')
    print(f'MAE: {mean_absolute_error(y_train, model.predict(X_train))}')
    print('\n')

print("Test set")
for name, model in [('Random Forest', rf.best_estimator_)]:
    print(f"Model: {name}")
    print(f'R^2: {model.score(X_test, y_test)}')
    print(f'MAE: {mean_absolute_error(y_test, model.predict(X_test))}')
    print('\n')
y_pred = rf.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Línea de perfección
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Matriz de Dispersión: Real vs Predicción')
plt.show()