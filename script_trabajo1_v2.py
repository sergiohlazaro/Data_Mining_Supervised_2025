import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
print("\n")
print("1. Para empezar cargamos los datos y los prepararemos para el modelado:")
print(" - Hay que cargar el dataset y explorarlo, al explorarlo revisaremos valores nulos y duplicados y corregiremos estos si los hay.")
print(" - Por último, realizaremos un análisis exploratorio de datos (EDA) para entender mejor los datos generando una serie de gráficos.")
df = pd.read_csv("incidents.byCountryYr.csv")

print("\n")
print("El archivo contiene el siguiente número de filas y columnas:")
print(df.shape)                                                                     # Muestra el número de filas y columnas
print("-----------------------------------------------------------------------------------------------------------------------------------")

print("\nEl archivo contiene las siguientes columnas:")
print(df.columns)                                                                   # Lista los nombres de las columnas
print("-----------------------------------------------------------------------------------------------------------------------------------")

print("\nEl archivo contiene los siguientes tipos de datos:")
print(df.dtypes)                                                                    # Muestra los tipos de datos
print("-----------------------------------------------------------------------------------------------------------------------------------")

print("\nValores nulos por columna:\n", df.isnull().sum())                          # Verifica valores nulos
print("\nNúmero de filas duplicadas:", df.duplicated().sum())                       # Verifica duplicados

print("\nEstadísticas descriptivas:")
print(df.describe())                                                                # Muestra estadísticas descriptivas
print("-----------------------------------------------------------------------------------------------------------------------------------")

# --- ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---

# 1️. Distribución de la variable "Freq" (Filtrar valores positivos)
df_filtered = df[df["Freq"] > 0]

plt.figure(figsize=(12, 5))
sns.histplot(df_filtered["Freq"], bins=50, kde=True)
plt.xlabel("Número de Incidentes")
plt.ylabel("Frecuencia")
plt.title("Distribución de la Frecuencia de Incidentes por País y Año")
plt.grid(axis="y")

plt.savefig("imgs1/histograma_freq.png")
print("Gráfico guardado como 'histograma_freq.png'; Ábrelo manualmente.")
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 2️. Transformación logarítmica de "Freq"
df["Freq_log"] = np.log1p(df["Freq"])                                               # log(1+Freq) para evitar log(0)

plt.figure(figsize=(12, 5))
sns.histplot(df["Freq_log"], bins=50, kde=True)
plt.xlabel("Logaritmo de Frecuencia de Incidentes")
plt.ylabel("Frecuencia")
plt.title("Distribución Logarítmica de la Frecuencia de Incidentes")
plt.grid(axis="y")

plt.savefig("imgs1/histograma_freq_log.png")
print("Gráfico guardado como 'histograma_freq_log.png'; Ábrelo manualmente.")
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 3️. Convertir "Freq" en categorías (Bajo, Medio, Alto)
                                                                                    # Método más robusto con `cut()` para evitar errores con valores repetidos
bins = [-1, 0, 10, 100, df["Freq"].max()]                                           # Definir cortes de categorías
labels = ["Cero", "Bajo", "Medio", "Alto"]
df["Freq_category"] = pd.cut(df["Freq"], bins=bins, labels=labels)

print("\nDistribución de la variable categorizada 'Freq_category':")
print(df["Freq_category"].value_counts())
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 4️. Normalizar "Freq" (Escalar entre 0 y 1)
scaler = MinMaxScaler()
df["Freq_scaled"] = scaler.fit_transform(df[["Freq"]])

print("\nEstadísticas de la variable normalizada 'Freq_scaled':")
print(df[["Freq", "Freq_scaled"]].describe())
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 5️. Evolución del número de incidentes por año
incidents_by_year = df.groupby("iyear")["Freq"].sum()

plt.figure(figsize=(12, 5))
plt.plot(incidents_by_year.index, incidents_by_year.values, marker="o", linestyle="-", color="b")
plt.xlabel("Año")
plt.ylabel("Total de Incidentes")
plt.title("Evolución del Número de Incidentes por Año")
plt.grid(True)

plt.savefig("imgs1/evolucion_incidentes.png")
print("Gráfico guardado como 'evolucion_incidentes.png'; Ábrelo manualmente.")
print("-----------------------------------------------------------------------------------------------------------------------------------")
print("\n¡Análisis de datos completado!\n")
print("2. Ahora que hemos entendido los datos, podemos pasar a la fase de clasificación supervisada:") 
print(" - Preparar los datos para el modelado.")
print(" - Entrenar los modelos de clasificación (Naive Bayes, Árboles de Decisión, k-NN, Regresión Logística).")
print(" - Evaluar el rendimiento de los modelos.\n")