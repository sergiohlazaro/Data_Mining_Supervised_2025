# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector

# 1. Cargar el dataset
print("Cargando el dataset y explorando los datos...")
df = pd.read_csv("incidents.byCountryYr.csv")

# 2. Transformación de datos
print("Transformando datos y categorizando la variable objetivo...")
df["Freq_log"] = np.log1p(df["Freq"])  
bins = [-1, 0, 10, 100, df["Freq"].max()]
labels = ["Cero", "Bajo", "Medio", "Alto"]
df["Freq_category"] = pd.cut(df["Freq"], bins=bins, labels=labels)

# 3. Selección de variables (INCLUYENDO 'iyear' para RFE)
X = df[["country_txt", "iyear"]].copy()  # Agregamos 'iyear'
y = df["Freq_category"]

# Codificación de variables categóricas
label_encoder = LabelEncoder()
X["country_txt"] = label_encoder.fit_transform(X["country_txt"])  

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicando SMOTE para balancear las clases
print("Aplicando SMOTE para balancear las clases...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Tamaño del conjunto de entrenamiento balanceado: {X_train_bal.shape}")

# 4. Métodos Wrapper: RFE (Recursive Feature Elimination)
print("\nAplicando RFE (Recursive Feature Elimination)...")
estimator = DecisionTreeClassifier(random_state=42)
rfe = RFE(estimator, n_features_to_select=1)  # Eliminaremos hasta quedarnos con 1 variable
rfe.fit(X_train_bal, y_train_bal)
X_train_rfe = X_train_bal.iloc[:, rfe.support_]
X_test_rfe = X_test.iloc[:, rfe.support_]

print("Variables seleccionadas con RFE:", X_train_rfe.columns.tolist())

# 5. Métodos Wrapper: Sequential Feature Selection (SFS)
print("\nAplicando Sequential Feature Selection (SFS)...")
sfs = SequentialFeatureSelector(estimator, n_features_to_select=1, direction="forward")
sfs.fit(X_train_bal, y_train_bal)
X_train_sfs = X_train_bal.iloc[:, sfs.support_]
X_test_sfs = X_test.iloc[:, sfs.support_]

print("Variables seleccionadas con SFS:", X_train_sfs.columns.tolist())

# 6. Métodos Embedded: Random Forest Feature Importance
print("\nEvaluando importancia de variables con Random Forest...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_bal, y_train_bal)
feature_importances = pd.Series(rf.feature_importances_, index=X_train_bal.columns)
print("Importancia de variables según Random Forest:\n", feature_importances)

# Guardamos el gráfico en un archivo en lugar de mostrarlo en pantalla
plt.figure(figsize=(8, 4))
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title("Importancia de las Variables según Random Forest")
plt.xlabel("Variable")
plt.ylabel("Importancia")
plt.xticks(rotation=45)
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig("imgs1/feature_importance_rf.png")

# Cerrar la figura para evitar posibles problemas de memoria
plt.close()

# 7. Métodos Embedded: Lasso Regression (L1 Regularization)
print("\nAplicando Lasso Regression (L1 Regularization)...")
lasso = Lasso(alpha=0.01)  # Regularización L1
lasso.fit(X_train_bal, label_encoder.fit_transform(y_train_bal))

# Variables seleccionadas por Lasso
lasso_importances = pd.Series(abs(lasso.coef_), index=X_train_bal.columns)
selected_lasso = lasso_importances[lasso_importances > 0].index.tolist()
print("Variables seleccionadas con Lasso:", selected_lasso)

# 8. Reentrenamiento de modelos con selección de variables aplicada
print("\nEntrenamiento de modelos con selección de variables aplicada:")

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_bal, y_train_bal)
nb_preds = nb_model.predict(X_test)

# Árbol de Decisión
id3_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
id3_model.fit(X_train_bal, y_train_bal)
id3_preds = id3_model.predict(X_test)

# k-NN con k=3
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_bal, y_train_bal)
knn_preds = knn_model.predict(X_test)

# Regresión Logística con más iteraciones
lr_model = LogisticRegression(max_iter=1000, solver="lbfgs")
lr_model.fit(X_train_bal, y_train_bal)
lr_preds = lr_model.predict(X_test)

# 9. Evaluación de modelos
def evaluar_modelo(nombre, y_true, y_pred):
    print(f"\n{nombre}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

evaluar_modelo("Naive Bayes", y_test, nb_preds)
evaluar_modelo("Árbol de Decisión (ID3)", y_test, id3_preds)
evaluar_modelo("k-NN (k=3)", y_test, knn_preds)
evaluar_modelo("Regresión Logística", y_test, lr_preds)

print("-----------------------------------------------------------------------------------------------------------------------------------")
print("¡Reentrenamiento y evaluación de modelos completados tras la selección de variables Wrapper y Embedded! 🚀")
print("-----------------------------------------------------------------------------------------------------------------------------------")
