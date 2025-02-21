import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("incidents.byCountryYr.csv")                               # Carga el archivo csv
print("\n")
print("El archivo contiene el siguiente número de filas y columnas:")
                                                                            # ¿Se han cargado bien los datos?
print(df.shape)                                                             # Muestra el número de filas y columnas
print("-----------------------------------------------------------------------------------------------------------------------------------")
print("\n")
print("El archivo contiene las siguientes columnas:")
print(df.columns)                                                           # Lista los nombres de las columnas
print("-----------------------------------------------------------------------------------------------------------------------------------")
print("\n")
print("El archivo contiene los siguientes tipos de datos:")
print(df.dtypes)                                                            # Muestra los tipos de datos
print("-----------------------------------------------------------------------------------------------------------------------------------")
print("\n")
print("Valores nulos por columna:\n", df.isnull().sum())                    # Verifica valores nulos
print("\nNúmero de filas duplicadas:", df.duplicated().sum())               # Verifica duplicados
print("\n")
print("Estadísticas descriptivas:")
print(df.describe())                                                        # Muestra estadísticas descriptivas
print("\n")                                                                 # iyear = año, freq = min,max,mean,std,25%,50%,75%,max
print("-----------------------------------------------------------------------------------------------------------------------------------")
                                                                            # ¿Qué información útil se puede extraer de los datos?
                                                                            # Análisis Exploratorio de Datos (EDA)
                                                                            # Configurar el tamaño del gráfico
                                                                            # Filtrar solo valores positivos de Freq (evitamos el log de 0)
df_filtered = df[df["Freq"] > 0]

plt.figure(figsize=(12, 5))

                                                                            # Histograma corregido sin escala log en el eje Y
sns.histplot(df_filtered["Freq"], bins=50, kde=True)

                                                                            # Etiquetas
plt.xlabel("Número de Incidentes")
plt.ylabel("Frecuencia")
plt.title("Distribución de la Frecuencia de Incidentes por País y Año")
plt.grid(axis="y")