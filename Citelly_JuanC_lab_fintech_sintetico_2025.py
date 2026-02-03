
# =============================================================
# LAB FINTECH (SINTÉTICO 2025) — PREPROCESAMIENTO Y EDA
# Datos de entrada fijos para evitar errores de ruta/nombre.
# -------------------------------------------------------------
# Este script está listo para ejecutarse sin argumentos:
#   python lab_fintech_sintetico_2025.py
# 
# Archivos esperados en el mismo directorio:
#   - fintech_top_sintetico_2025.csv
#   - fintech_top_sintetico_dictionary.json
# Salidas (por defecto):
#   ./data_output_finanzas_sintetico/
#       ├─ fintech_train.parquet
#       ├─ fintech_test.parquet
#       ├─ processed_schema.json
#       └─ features_columns.txt
# =============================================================

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Constantes de la práctica
# ---------------------------
DATA_CSV = 'fintech_top_sintetico_2025.csv'
DATA_DICT = 'fintech_top_sintetico_dictionary.json'
OUTDIR = Path('./data_output_finanzas_sintetico')
SPLIT_DATE = '2025-09-01'  # partición temporal por defecto

# Columnas esperadas por diseño del dataset sintético
DATE_COL = 'Month'
ID_COLS = ['Company']
CAT_COLS = ['Country', 'Region', 'Segment', 'Subsegment', 'IsPublic', 'Ticker']
NUM_COLS = [
    'Users_M','NewUsers_K','TPV_USD_B','TakeRate_pct','Revenue_USD_M',
    'ARPU_USD','Churn_pct','Marketing_Spend_USD_M','CAC_USD','CAC_Total_USD_M',
    'Close_USD','Private_Valuation_USD_B'
]
PRICE_COLS = ['Close_USD']  # para calcular retornos opcionales

# ---------------------------
# 0) Carga de diccionario
# ---------------------------
print("\n=== 0) Cargando diccionario de datos ===")
dict_path = Path(DATA_DICT)
if not dict_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_DICT}. Asegúrate de tener el archivo en la misma carpeta.")

with open(dict_path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)
print("Descripción:", data_dict.get('description', '(sin descripción)'))
print("Periodo:", data_dict.get('period', '(desconocido)'))

# ---------------------------
# 1) Carga del CSV
# ---------------------------
print("\n=== 1) Cargando CSV sintético ===")
csv_path = Path(DATA_CSV)
if not csv_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_CSV}. Asegúrate de tener el archivo en la misma carpeta.")

df = pd.read_csv(csv_path)
print("Shape:", df.shape)

# Parseo de fecha y orden temporal
if DATE_COL not in df.columns:
    raise KeyError(f"La columna de fecha '{DATE_COL}' no existe en el CSV.")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

print("Primeras filas:")
print(df.head(3))

# ---------------------------
# 2) EDA breve
# ---------------------------
print("\n=== 2) EDA rápido ===")
print("Info:")
print(df.info())
print("\nNulos por columna (top 15):")
print(df.isna().sum().sort_values(ascending=False).head(15))

# ---------------------------
# 3) Limpieza básica
# ---------------------------
print("\n=== 3) Limpieza ===")
# Imputación simple: numéricos con mediana, categóricos con marcador
for c in NUM_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

for c in CAT_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = df[c].fillna('__MISSING__')

# ---------------------------
# 4) Ingeniería ligera: retornos/log-retornos de precio
# ---------------------------
print("\n=== 4) Ingeniería de rasgos (retornos) ===")
if all([pc in df.columns for pc in PRICE_COLS]):
    for pc in PRICE_COLS:
        # Retornos por empresa y fecha
        df[pc + '_ret'] = (
            df.sort_values([ID_COLS[0], DATE_COL])
              .groupby(ID_COLS)[pc]
              .pct_change()
        )
        df[pc + '_logret'] = np.log1p(df[pc + '_ret'])
        # Imputar primeros NA en 0.0 para continuidad
        df[pc + '_ret'] = df[pc + '_ret'].fillna(0.0)
        df[pc + '_logret'] = df[pc + '_logret'].fillna(0.0)
else:
    print("[INFO] Columnas de precio no disponibles; se omite cálculo de retornos.")

# Actualizamos lista de numéricos tras ingeniería
extra_num = [c for c in [pc + '_ret' for pc in PRICE_COLS] + [pc + '_logret' for pc in PRICE_COLS] if c in df.columns]
NUM_USED = [c for c in NUM_COLS if c in df.columns] + extra_num

# ---------------------------
# 5) Separación X / y (sin y por defecto) + codificación
# ---------------------------
print("\n=== 5) Preparación de X: codificación one-hot y escalado ===")
# Quitamos identificadores y fecha de las variables predictoras
X = df.drop(columns=[DATE_COL] + ID_COLS, errors='ignore').copy()

# One-hot en categóricas
cat_in_X = [c for c in CAT_COLS if c in X.columns]
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)

# Partición temporal por defecto utilizando la fecha de corte
cutoff = pd.to_datetime(SPLIT_DATE)
idx_train = df[DATE_COL] < cutoff
idx_test = df[DATE_COL] >= cutoff

X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()

# Escalado de numéricos (solo columnas presentes en X)
num_in_X = [c for c in NUM_USED if c in X_train.columns]
scaler = StandardScaler()
if num_in_X: # Aqui normaliza de 0 a 1
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_test[num_in_X] = scaler.transform(X_test[num_in_X])
else:
    print("[INFO] No se encontraron columnas numéricas para escalar.")

print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)

# ---------------------------
# 6) Exportación
# ---------------------------
print("\n=== 6) Exportación ===")
OUTDIR.mkdir(parents=True, exist_ok=True)
train_path = OUTDIR / 'fintech_train.parquet'
test_path = OUTDIR / 'fintech_test.parquet'

# Guardamos sólo X (sin objetivo)
X_train.to_parquet(train_path, index=False)
X_test.to_parquet(test_path, index=False)

# Guardar esquema procesado
processed_schema = {
    'source_csv': str(csv_path.resolve()),
    'source_dict': str(dict_path.resolve()),
    'date_col': DATE_COL,
    'id_cols': ID_COLS,
    'categorical_cols_used': cat_in_X,
    'numeric_cols_used': num_in_X,
    'engineered_cols': extra_num,
    'split': {
        'type': 'time_split',
        'cutoff': SPLIT_DATE,
        'train_rows': int(idx_train.sum()),
        'test_rows': int(idx_test.sum()),
    },
    'X_train_shape': list(X_train.shape),
    'X_test_shape': list(X_test.shape),
    'notes': [
        'Dataset 100% SINTÉTICO con fines académicos; no refleja métricas reales.',
        'Evitar fuga de datos: el escalador se ajusta en TRAIN y se aplica a TEST.'
    ]
}

with open(OUTDIR / 'processed_schema.json', 'w', encoding='utf-8') as f:
    json.dump(processed_schema, f, ensure_ascii=False, indent=2)

# Lista de columnas finales para referencia de modelado
with open(OUTDIR / 'features_columns.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(X_train.columns))

print("\nArchivos exportados:")
print(" -", train_path)
print(" -", test_path)
print(" -", OUTDIR / 'processed_schema.json')
print(" -", OUTDIR / 'features_columns.txt')

print("\n✔ Listo. Recuerda: este dataset es sintético para práctica académica.")
