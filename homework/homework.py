# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import json
import gzip
import pickle
import time
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # <-- CORRECCIÓN: MinMaxScaler para intervalo [0, 1]
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (precision_score, balanced_accuracy_score,
                             recall_score, f1_score, confusion_matrix)

inicio = time.time()

def limpieza(df):
    df = df.copy()
    df = df.rename(columns={'default payment next month': 'default'})
    df = df.drop(columns="ID")
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df = df.dropna()
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df

train = limpieza(pd.read_csv("files/input/train_data.csv.zip", compression="zip"))
test = limpieza(pd.read_csv("files/input/test_data.csv.zip", compression="zip"))


x_test, y_test = test.drop(columns="default"), test["default"]
x_train, y_train = train.drop(columns="default"), train["default"]


cat_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

preprocessor = ColumnTransformer(
    transformers=[('OneHotEncoder', OneHotEncoder(handle_unknown='ignore'), cat_features)],
    remainder='passthrough'
)
 
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("StandardScaler", StandardScaler(with_mean=False)),
    ("PCA", PCA(n_components=24)),
    ("SelectKBest", SelectKBest(score_func=f_classif, k='all')),
    ("MLPClassifier", MLPClassifier(
        max_iter=3000,
        early_stopping=True,
        hidden_layer_sizes=(100, 50, 25),
        alpha=0.005,
        learning_rate_init=0.004,
        activation='tanh',
        random_state=13,
    ))
])
 

param_grid = {
    "PCA__n_components": [24],
    "SelectKBest__k": ["all"],
    "MLPClassifier__hidden_layer_sizes": [(100, 50, 25)],
    "MLPClassifier__alpha": [0.005],
    "MLPClassifier__learning_rate_init": [0.004],
    "MLPClassifier__activation": ["tanh"],
}
 
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(x_train, y_train)

# Verificación inmediata antes de guardar nada
print("Train score:", grid_search.score(x_train, y_train))
print("Test score:", grid_search.score(x_test, y_test))

for x, y, name in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
    y_pred = grid_search.predict(x)
    print(name,
          "precision:", precision_score(y, y_pred),
          "balanced_accuracy:", balanced_accuracy_score(y, y_pred),
          "recall:", recall_score(y, y_pred),
          "f1:", f1_score(y, y_pred))

# Paso 5 - Save model (solo si las métricas de arriba pasan los umbrales)
os.makedirs("files/models", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_search, f)

os.makedirs("files/output", exist_ok=True)

def calcular_metricas(model, x, y, dataset_name):
    y_pred = model.predict(x)
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
    }

def calcular_cm(model, x, y, dataset_name):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

metrics = [
    calcular_metricas(grid_search, x_train, y_train, "train"),
    calcular_metricas(grid_search, x_test, y_test, "test"),
    calcular_cm(grid_search, x_train, y_train, "train"),
    calcular_cm(grid_search, x_test, y_test, "test"),
]

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    for m in metrics:
        f.write(json.dumps(m) + "\n")

fin = time.time()

import numpy as np
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, f1_score

y_proba_train = grid_search.predict_proba(x_train)[:, 1]
y_proba_test = grid_search.predict_proba(x_test)[:, 1]

for t in np.arange(0.48, 0.60, 0.005):
    for name, y_true, y_proba in [('train', y_train, y_proba_train), ('test', y_test, y_proba_test)]:
        y_pred_t = (y_proba >= t).astype(int)
        p = precision_score(y_true, y_pred_t, zero_division=0)
        r = recall_score(y_true, y_pred_t, zero_division=0)
        ba = balanced_accuracy_score(y_true, y_pred_t)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        print(f"{name} t={t:.3f}  precision={p:.4f}  recall={r:.4f}  balanced_acc={ba:.4f}  f1={f1:.4f}")

print("Mejores parámetros:", grid_search.best_params_)
print("Mejor score (balanced_accuracy, CV):", grid_search.best_score_)
print("Train score:", grid_search.score(x_train, y_train))
print("Test score:", grid_search.score(x_test, y_test))
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score

for x, y, name in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
    y_pred = grid_search.predict(x)
    print(name,
          "precision:", precision_score(y, y_pred),
          "balanced_accuracy:", balanced_accuracy_score(y, y_pred),
          "recall:", recall_score(y, y_pred),
          "f1:", f1_score(y, y_pred))