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
from sklearn.model_selection import GridSearchCV

# 1. Cargar el dataset
df = pd.read_csv("incidents.byCountryYr.csv")

# 2. Transformación de datos
df["Freq_log"] = np.log1p(df["Freq"])  # Transformación logarítmica
bins = [-1, 0, 10, 100, df["Freq"].max()]
labels = ["Cero", "Bajo", "Medio", "Alto"]
df["Freq_category"] = pd.cut(df["Freq"], bins=bins, labels=labels)

# 3. Preparación de los datos para el modelado
X = df[["iyear", "country_txt"]].copy()
y = df["Freq_category"]

label_encoder = LabelEncoder()
X["country_txt"] = label_encoder.fit_transform(X["country_txt"])  # Convertir país en numérico

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Balanceo de clases con SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 5. Entrenamiento de modelos de clasificación
print("\nEntrenamiento de modelos:")

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_bal, y_train_bal)
nb_preds = nb_model.predict(X_test)

# Árbol de Decisión (ID3)
id3_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
id3_model.fit(X_train_bal, y_train_bal)
id3_preds = id3_model.predict(X_test)

# J48 (similar a ID3 pero con poda)
j48_model = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
j48_model.fit(X_train_bal, y_train_bal)
j48_preds = j48_model.predict(X_test)

# k-NN con k=1 (IB1)
ib1_model = KNeighborsClassifier(n_neighbors=1)
ib1_model.fit(X_train_bal, y_train_bal)
ib1_preds = ib1_model.predict(X_test)

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
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

evaluar_modelo("Naive Bayes", y_test, nb_preds)
evaluar_modelo("Árbol de Decisión (ID3)", y_test, id3_preds)
evaluar_modelo("J48 (ID3 con poda)", y_test, j48_preds)
evaluar_modelo("IB1 (k-NN con k=1)", y_test, ib1_preds)
evaluar_modelo("k-NN (k=3)", y_test, knn_preds)
evaluar_modelo("Regresión Logística", y_test, lr_preds)

print("-----------------------------------------------------------------------------------------------------------------------------------")
print("¡Entrenamiento y evaluación de modelos completados! 🚀")
print("-----------------------------------------------------------------------------------------------------------------------------------")

# 🔹 1. Optimización del Árbol de Decisión (ID3)
print("\nAjustando hiperparámetros para ID3...")

param_grid_id3 = {
    "max_depth": [10, 15, 20, None],  # Ajustar la profundidad
    "min_samples_split": [2, 5, 10]  # Número mínimo de muestras para dividir un nodo
}

grid_id3 = GridSearchCV(DecisionTreeClassifier(criterion="entropy", random_state=42),
                        param_grid_id3, cv=5, scoring="accuracy", n_jobs=-1)
grid_id3.fit(X_train_bal, y_train_bal)

best_id3 = grid_id3.best_estimator_
id3_preds_opt = best_id3.predict(X_test)

print(f"Mejor configuración para ID3: {grid_id3.best_params_}")
evaluar_modelo("Árbol de Decisión (ID3 Optimizado)", y_test, id3_preds_opt)

# 🔹 2. Optimización de k-NN
print("\nAjustando hiperparámetros para k-NN...")

param_grid_knn = {
    "n_neighbors": [3, 5, 7],  # Probar diferentes valores de k
    "weights": ["uniform", "distance"],  # Pesos iguales o inversamente proporcionales a la distancia
}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring="accuracy", n_jobs=-1)
grid_knn.fit(X_train_bal, y_train_bal)

best_knn = grid_knn.best_estimator_
knn_preds_opt = best_knn.predict(X_test)

print(f"Mejor configuración para k-NN: {grid_knn.best_params_}")
evaluar_modelo("k-NN Optimizado", y_test, knn_preds_opt)

print("\nOptimización completada. Evaluación de modelos optimizados realizada.")
print("-----------------------------------------------------------------------------------------------------------------------------------")
print("¡Entrenamiento y evaluación de modelos completados! 🚀")
print("-----------------------------------------------------------------------------------------------------------------------------------")