import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Cargar el dataset
print("\nCargando el dataset...")
df = pd.read_csv("incidents.byCountryYr.csv")
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# 2. Transformación de datos
print("\nAplicando transformaciones a los datos...")

# Aplicar transformación logarítmica para reducir el impacto de valores extremos
df["Freq_log"] = np.log1p(df["Freq"])
print("Se ha aplicado la transformación logarítmica a 'Freq'.")

# Convertir 'Freq' en categorías para clasificación
bins = [-1, 0, 10, 100, df["Freq"].max()]
labels = ["Cero", "Bajo", "Medio", "Alto"]
df["Freq_category"] = pd.cut(df["Freq"], bins=bins, labels=labels)
print("Se ha categorizado la variable 'Freq' en 'Cero', 'Bajo', 'Medio' y 'Alto'.")

# 3. Preparación de los datos para el modelado
print("\nPreparando los datos para el modelado...")

# Seleccionar variables predictoras y objetivo
X = df[["iyear", "country_txt"]].copy()
y = df["Freq_category"]
print(f"Se han seleccionado las variables predictoras: {list(X.columns)}.")

# Convertir la variable categórica 'country_txt' a numérica
label_encoder = LabelEncoder()
X["country_txt"] = label_encoder.fit_transform(X["country_txt"])
print("Se ha codificado 'country_txt' a valores numéricos.")

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"División de datos completada: {X_train.shape[0]} muestras para entrenamiento, {X_test.shape[0]} para prueba.")

# 4. Balanceo de clases con SMOTE (Soluciona el desbalanceo en Naive Bayes y Regresión Logística)
print("\nAplicando SMOTE para balancear las clases...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Datos balanceados con SMOTE. Ahora hay {X_train_bal.shape[0]} muestras en el conjunto de entrenamiento.")

# 5. Entrenamiento de modelos de clasificación
print("\nEntrenamiento de modelos:")

# Naive Bayes
print("\nEntrenando Naive Bayes...")
nb_model = GaussianNB()
nb_model.fit(X_train_bal, y_train_bal)
nb_preds = nb_model.predict(X_test)
print("Naive Bayes entrenado y evaluado.")

# Árbol de Decisión (ID3)
print("\nEntrenando Árbol de Decisión (ID3)...")
id3_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
id3_model.fit(X_train_bal, y_train_bal)
id3_preds = id3_model.predict(X_test)
print("Árbol de Decisión (ID3) entrenado y evaluado.")

# k-NN con k=3
print("\nEntrenando k-NN con k=3...")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_bal, y_train_bal)
knn_preds = knn_model.predict(X_test)
print("k-NN (k=3) entrenado y evaluado.")

# Regresión Logística con aumento de iteraciones y solver optimizado
print("\nEntrenando Regresión Logística con solver optimizado...")
lr_model = LogisticRegression(max_iter=2000, solver="saga")  # Soluciona el problema de convergencia
lr_model.fit(X_train_bal, y_train_bal)
lr_preds = lr_model.predict(X_test)
print("Regresión Logística entrenada y evaluada.")

# 6. Evaluación de modelos con matriz de confusión y métricas detalladas
print("\nEvaluando los modelos entrenados:")

def evaluar_modelo(nombre, y_true, y_pred):
    print(f"\nEvaluación de {nombre}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))
    print("Métricas detalladas:\n", classification_report(y_true, y_pred, zero_division=0))  # Soluciona UndefinedMetricWarning

evaluar_modelo("Naive Bayes", y_test, nb_preds)
evaluar_modelo("Árbol de Decisión (ID3)", y_test, id3_preds)
evaluar_modelo("k-NN (k=3)", y_test, knn_preds)
evaluar_modelo("Regresión Logística", y_test, lr_preds)

print("\nEvaluación completada. Todos los errores anteriores han sido corregidos.")
print("-----------------------------------------------------------------------------------------------------------------------------------")
print("¡Entrenamiento y evaluación de modelos completados!")
print("-----------------------------------------------------------------------------------------------------------------------------------")
