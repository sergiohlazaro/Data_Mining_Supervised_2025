import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Cargar el dataset
print("Cargando el dataset y explorando los datos...")
df = pd.read_csv("incidents.byCountryYr.csv")

# 2. Transformación de datos
df["Freq_log"] = np.log1p(df["Freq"])  
bins = [-1, 0, 10, 100, df["Freq"].max()]
labels = ["Cero", "Bajo", "Medio", "Alto"]
df["Freq_category"] = pd.cut(df["Freq"], bins=bins, labels=labels)

# 3. Selección de variables (EXCLUYENDO 'iyear')
X = df[["country_txt"]].copy()
y = df["Freq_category"]

label_encoder = LabelEncoder()
X["country_txt"] = label_encoder.fit_transform(X["country_txt"])  

# División del conjunto en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Balanceo de clases con SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 5. Optimización de Naive Bayes
print("\nOptimizando Naive Bayes...")
param_grid_nb = {"var_smoothing": np.logspace(-9, 0, 10)}
grid_nb = GridSearchCV(GaussianNB(), param_grid_nb, cv=5, scoring="accuracy")
grid_nb.fit(X_train_bal, y_train_bal)
best_nb = grid_nb.best_estimator_
nb_preds = best_nb.predict(X_test)
print(f"Mejor configuración para Naive Bayes: {grid_nb.best_params_}")

# 6. Optimización de Árbol de Decisión
print("\nOptimizando Árbol de Decisión (ID3)...")
param_grid_dt = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(criterion="entropy", random_state=42), param_grid_dt, cv=5, scoring="accuracy")
grid_dt.fit(X_train_bal, y_train_bal)
best_dt = grid_dt.best_estimator_
dt_preds = best_dt.predict(X_test)
print(f"Mejor configuración para Árbol de Decisión: {grid_dt.best_params_}")

# 7. Optimización de k-NN
print("\nOptimizando k-NN...")
param_grid_knn = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring="accuracy")
grid_knn.fit(X_train_bal, y_train_bal)
best_knn = grid_knn.best_estimator_
knn_preds = best_knn.predict(X_test)
print(f"Mejor configuración para k-NN: {grid_knn.best_params_}")

# 8. Optimización de Regresión Logística
print("\nOptimizando Regresión Logística...")
param_grid_lr = {"C": np.logspace(-4, 2, 10), "solver": ["lbfgs"]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring="accuracy")
grid_lr.fit(X_train_bal, y_train_bal)
best_lr = grid_lr.best_estimator_
lr_preds = best_lr.predict(X_test)
print(f"Mejor configuración para Regresión Logística: {grid_lr.best_params_}")

# 9. Evaluación de modelos optimizados
def evaluar_modelo(nombre, y_true, y_pred):
    print(f"\n{nombre}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

evaluar_modelo("Naive Bayes Optimizado", y_test, nb_preds)
evaluar_modelo("Árbol de Decisión (ID3) Optimizado", y_test, dt_preds)
evaluar_modelo("k-NN Optimizado", y_test, knn_preds)
evaluar_modelo("Regresión Logística Optimizada", y_test, lr_preds)

print("\n-------------------------------------------------------------------")
print("¡Optimización y evaluación de modelos completadas! 🚀")
print("-------------------------------------------------------------------")
