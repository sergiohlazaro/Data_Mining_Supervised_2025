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

print("\n3. Preparación de los datos para el modelado:")
print(" - Selección de variables predictoras y objetivo.")
print(" - Transformación de variables categóricas en numéricas.")
print(" - División del conjunto en entrenamiento y prueba.")

# 1. Seleccionar variables predictoras y objetivo
X = df[["iyear", "country_txt"]].copy()   # Variables predictoras
y = df["Freq_category"]  # Variable objetivo

# 2. Codificar "country_txt" a valores numéricos
label_encoder = LabelEncoder()
X["country_txt"] = label_encoder.fit_transform(X["country_txt"])

# 3. Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDatos preparados:")
print(f" - Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f" - Tamaño del conjunto de prueba: {X_test.shape}")
print("-----------------------------------------------------------------------------------------------------------------------------------")

print("\n4. Entrenamiento de modelos de clasificación:")
print(" - Entrenamos Naive Bayes, Árbol de Decisión, k-NN y Regresión Logística.")
print(" - Evaluamos su rendimiento en el conjunto de prueba.")
print(" - ERRORES que surgieron y se aplico: Balanceo de clases con SMOTE.")
print(" - ERRORES que surgieron y se aplico: Aumentamos iteraciones en Regresión Logística.")
print(" - ERRORES que surgieron y se aplico: Evaluamos modelos con matriz de confusión.")

# 1️. Entrenar **Naive Bayes**
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# 2️. Entrenar **Árbol de Decisión**
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# 3️. Entrenar **k-NN (k=5)**
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

# 4️. Entrenar **Regresión Logística**
lr_model = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="multinomial")
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# 5. Evaluación de precisión
print("\nResultados de los modelos:")
print(f"🔹 Naive Bayes Accuracy: {accuracy_score(y_test, nb_preds):.4f}")
print(f"🔹 Árbol de Decisión Accuracy: {accuracy_score(y_test, dt_preds):.4f}")
print(f"🔹 k-NN Accuracy: {accuracy_score(y_test, knn_preds):.4f}")
print(f"🔹 Regresión Logística Accuracy: {accuracy_score(y_test, lr_preds):.4f}")

print("\nClasificación detallada para Naive Bayes:")
print(classification_report(y_test, nb_preds))

print("\nClasificación detallada para Árbol de Decisión:")
print(classification_report(y_test, dt_preds))

print("\nClasificación detallada para k-NN:")
print(classification_report(y_test, knn_preds))

print("\nClasificación detallada para Regresión Logística:")
print(classification_report(y_test, lr_preds))

print("-----------------------------------------------------------------------------------------------------------------------------------")
print("Surgen varios problemas con este script:")
print("1. Regresión logística no converge [lbfgs failed to converge (status=1): STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT]: solución, aumentar max_iter a 1000 o más para permitir más interacciones")
print("2. Naive Bayes y Regresión Logística solo predicen la clase mayoritaria (Cero),lo que indica que los modelos no están aprendiendo correctamente, la distribución de clases es desbalanceada, con demasiados datos en la clase Cero: solución, balancear las clases con SMOTE (aplicar balanceo de clases (undersampling, oversampling o técnicas como SMOTE))")
print("3. Advertencias de precisión indefinida (UndefinedMetricWarning) ocurre porque ciertas clases nunca son predichas, en la clasificación detallada; solución, establecer zero_division=0 en classification_report para evitar advertencias.")