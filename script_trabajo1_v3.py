# Importar librerías necesarias
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
df = pd.read_csv("incidents.byCountryYr.csv")

print("\nEl archivo contiene el siguiente número de filas y columnas:")
print(df.shape)
print("-----------------------------------------------------------------------------------------------------------------------------------")

print("\nEl archivo contiene las siguientes columnas:")
print(df.columns)
print("-----------------------------------------------------------------------------------------------------------------------------------")

print("\nEl archivo contiene los siguientes tipos de datos:")
print(df.dtypes)
print("-----------------------------------------------------------------------------------------------------------------------------------")

print("\nValores nulos por columna:\n", df.isnull().sum())
print("\nNúmero de filas duplicadas:", df.duplicated().sum())
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 2. Transformación de datos
df["Freq_log"] = np.log1p(df["Freq"])  # Transformación logarítmica
bins = [-1, 0, 10, 100, df["Freq"].max()]  # Categorización de Freq
labels = ["Cero", "Bajo", "Medio", "Alto"]
df["Freq_category"] = pd.cut(df["Freq"], bins=bins, labels=labels)

print("\nDistribución de la variable categorizada 'Freq_category':")
print(df["Freq_category"].value_counts())
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 3. Preparación de los datos para el modelado
print("\nPreparación de los datos para el modelado:")
print(" - Selección de variables predictoras y objetivo.")
print(" - Transformación de variables categóricas en numéricas.")
print(" - División del conjunto en entrenamiento y prueba.")

X = df[["iyear", "country_txt"]].copy()  # Variables predictoras
y = df["Freq_category"]  # Variable objetivo

label_encoder = LabelEncoder()
X["country_txt"] = label_encoder.fit_transform(X["country_txt"])  # Convertir país en numérico

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDatos preparados:")
print(f" - Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f" - Tamaño del conjunto de prueba: {X_test.shape}")
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 4. Balanceo de clases con SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nDatos balanceados con SMOTE:")
print(f" - Tamaño del conjunto de entrenamiento balanceado: {X_train_bal.shape}")
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 5. Entrenamiento de modelos de clasificación
print("\nEntrenamiento de modelos:")

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_bal, y_train_bal)
nb_preds = nb_model.predict(X_test)

# Árbol de Decisión
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_bal, y_train_bal)
dt_preds = dt_model.predict(X_test)

# k-NN con k=3
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_bal, y_train_bal)
knn_preds = knn_model.predict(X_test)

# Regresión Logística con más iteraciones
lr_model = LogisticRegression(max_iter=1000, solver="lbfgs")
lr_model.fit(X_train_bal, y_train_bal)
lr_preds = lr_model.predict(X_test)

# 6. Evaluación de modelos con matriz de confusión y métricas detalladas
def evaluar_modelo(nombre, y_true, y_pred):
    print(f"\n🔹 {nombre}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

evaluar_modelo("Naive Bayes", y_test, nb_preds)
evaluar_modelo("Árbol de Decisión", y_test, dt_preds)
evaluar_modelo("k-NN", y_test, knn_preds)
evaluar_modelo("Regresión Logística", y_test, lr_preds)

print("-----------------------------------------------------------------------------------------------------------------------------------")
print("¡Entrenamiento y evaluación de modelos completados! 🚀")
print("-----------------------------------------------------------------------------------------------------------------------------------")
