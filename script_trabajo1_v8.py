import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
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

# 5. Reajuste de modelos con mejor configuración
print("\nReajustando modelos con configuraciones corregidas...")
nb_model = GaussianNB(var_smoothing=1e-9)
dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3, weights="uniform")
lr_model = LogisticRegression(max_iter=1000, solver="liblinear", C=0.01, penalty='l1')

# Entrenamiento de modelos corregidos
nb_model.fit(X_train_bal, y_train_bal)
dt_model.fit(X_train_bal, y_train_bal)
knn_model.fit(X_train_bal, y_train_bal)
lr_model.fit(X_train_bal, y_train_bal)

# Predicciones
nb_preds = nb_model.predict(X_test)
dt_preds = dt_model.predict(X_test)
knn_preds = knn_model.predict(X_test)
lr_preds = lr_model.predict(X_test)

# 6. Validación Cruzada (k-Fold Cross Validation)
print("\nRealizando Validación Cruzada...")
kfold_scores = cross_val_score(knn_model, X_train_bal, y_train_bal, cv=5)
print("Validación Cruzada k-NN (Accuracy Promedio):", np.mean(kfold_scores))

# 7. Combinación de Modelos (Voting Classifier)
print("\nProbando combinación de modelos con Voting Classifier...")
voting_clf = VotingClassifier(estimators=[('knn', knn_model), ('dt', dt_model)], voting='hard')
voting_clf.fit(X_train_bal, y_train_bal)
voting_preds = voting_clf.predict(X_test)

# 8. Evaluación de modelos
print("\nEvaluación de modelos finales:")
def evaluar_modelo(nombre, y_true, y_pred):
    print(f"\n🔹 {nombre}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

evaluar_modelo("Naive Bayes", y_test, nb_preds)
evaluar_modelo("Árbol de Decisión (ID3)", y_test, dt_preds)
evaluar_modelo("k-NN", y_test, knn_preds)
evaluar_modelo("Regresión Logística", y_test, lr_preds)
evaluar_modelo("Voting Classifier", y_test, voting_preds)

print("\n-------------------------------------------------------------------")
print("¡Reajuste, validación cruzada y combinación de modelos completados! 🚀")
print("-------------------------------------------------------------------")
