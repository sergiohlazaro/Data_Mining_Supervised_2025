import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.svm import SVC
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

# 5. Entrenamiento de los mejores modelos
print("\nEntrenando modelos principales...")
dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3, weights="uniform")

dt_model.fit(X_train_bal, y_train_bal)
knn_model.fit(X_train_bal, y_train_bal)

# 6. Optimización de la combinación de modelos con Voting Classifier
print("\nOptimizando combinación de modelos con Voting Classifier...")
voting_clf = VotingClassifier(estimators=[('knn', knn_model), ('dt', dt_model)], voting='soft')
voting_clf.fit(X_train_bal, y_train_bal)
voting_preds = voting_clf.predict(X_test)

# 7. Implementación de Stacking Classifier
print("\nImplementando Stacking Classifier...")
stacking_clf = StackingClassifier(
    estimators=[('knn', knn_model), ('dt', dt_model)],
    final_estimator=SVC(probability=True)
)
stacking_clf.fit(X_train_bal, y_train_bal)
stacking_preds = stacking_clf.predict(X_test)

# 8. Evaluación de modelos
print("\nEvaluación de modelos combinados:")
def evaluar_modelo(nombre, y_true, y_pred):
    print(f"\n{nombre}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

evaluar_modelo("Voting Classifier", y_test, voting_preds)
evaluar_modelo("Stacking Classifier", y_test, stacking_preds)

print("\n-------------------------------------------------------------------")
print("¡Optimización, Voting y Stacking completados! 🚀")
print("-------------------------------------------------------------------")
