# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset
print("\nCargando el dataset...")
df = pd.read_csv("incidents.byCountryYr.csv")
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# 2. Transformación de datos
print("\nAplicando transformaciones a los datos...")

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

# Convertir la variable categórica 'country_txt' a numérica
label_encoder = LabelEncoder()
X["country_txt"] = label_encoder.fit_transform(X["country_txt"])
print("Se ha codificado 'country_txt' a valores numéricos.")

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"División de datos completada: {X_train.shape[0]} muestras para entrenamiento, {X_test.shape[0]} para prueba.")

# 4. Aplicar Selección de Variables
print("\nAplicando Métodos de Selección de Variables...")

# 4.1. Filtrado Univariante (Usando Chi-cuadrado y Mutual Information)
print("\nAplicando Chi-cuadrado y Mutual Information...")

# Chi-cuadrado (solo en valores positivos)
X_chi2 = X_train.copy()
X_chi2[X_chi2 < 0] = 0  # Asegurar valores positivos para chi2
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X_chi2, y_train)
chi2_scores = chi2_selector.scores_

# Información Mutua
mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
mi_selector.fit(X_train, y_train)
mi_scores = mi_selector.scores_

# 4.2. Wrapper con Recursive Feature Elimination (RFE)
print("\nAplicando RFE con Árbol de Decisión...")
rfe_selector = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=1)
rfe_selector.fit(X_train, y_train)
rfe_ranking = rfe_selector.ranking_

# 4.3. Importancia de Características con Random Forest
print("\nEvaluando importancia de características con Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_importance = rf_model.feature_importances_

# 4.4. Crear un DataFrame con los resultados
feature_selection_results = pd.DataFrame({
    'Feature': X_train.columns,
    'Chi2 Score': chi2_scores,
    'Mutual Information Score': mi_scores,
    'RFE Ranking': rfe_ranking,
    'Random Forest Importance': rf_importance
}).sort_values(by='Random Forest Importance', ascending=False)

# Guardar los resultados en un archivo CSV
feature_selection_results.to_csv("csv_outputs/feature_selection_results_var_select.csv", index=False)
print("\nLos resultados de la selección de variables se han guardado en 'feature_selection_results_var_select.csv'.")

# Guardar el gráfico en un archivo en lugar de mostrarlo en pantalla
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_selection_results["Feature"], y=feature_selection_results["Random Forest Importance"])
plt.title("Importancia de las Variables según Random Forest")
plt.xlabel("Variable")
plt.ylabel("Importancia")

# Guardar la imagen
plt.savefig("imgs1/feature_importance_var_select.png")
print("\nEl gráfico se ha guardado como 'feature_importance_var_select.png'. Ábrelo manualmente.")

print("-----------------------------------------------------------------------------------------------------------------------------------")
print("\nProceso de selección de variables completado.")
print("-----------------------------------------------------------------------------------------------------------------------------------")