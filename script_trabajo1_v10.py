# -------------------------------
# PUNTO 1: ANÁLISIS EXPLORATORIO DE DATOS
# -------------------------------
# EDA - Análisis Exploratorio de Datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta de imágenes si no existe
output_dir = "imgs1"
os.makedirs(output_dir, exist_ok=True)

# Cargar el CSV
df = pd.read_csv("mlb_teams.csv")  # Asegúrate de que el archivo esté en la misma carpeta o usa ruta absoluta

# 1. Porcentaje de valores nulos por columna
null_percent = df.isnull().mean().sort_values(ascending=False) * 100
null_percent = null_percent[null_percent > 0]
print("\n")
print("Porcentaje de valores nulos:")
print(null_percent)

# 2. Estadísticas descriptivas para columnas numéricas
print("\nEstadísticas descriptivas (numéricas):")
print(df.describe(include='number'))

# 3. Estadísticas descriptivas para columnas categóricas
print("\nEstadísticas descriptivas (categóricas):")
print(df.describe(include='object'))

# 4. Mapa de calor de valores nulos
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Mapa de Calor de Valores Nulos")
plt.xlabel("Variables")
plt.ylabel("Registros")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mapa_valores_nulos_mlb_teams.png"))
plt.close()

# 5. Matriz de correlación
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title("Matriz de Correlación de Variables Numéricas")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "matriz_correlacion_mlb_teams.png"))
plt.close()
print("\n")

# -------------------------------
# PUNTO 2: PREPROCESAMIENTO
# -------------------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 1. Eliminamos columnas irrelevantes para el análisis
df_clean = df.drop(columns=['rownames', 'team_name', 'ball_park'])

# 2. Imputación de columnas numéricas (media)
num_cols = df_clean.select_dtypes(include='number').columns
imputer_num = SimpleImputer(strategy='mean')
df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])

# 3. Codificación y/o imputación de columnas categóricas
cat_cols = df_clean.select_dtypes(include='object').columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])

# 4. Codificamos las columnas categóricas con LabelEncoder
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le  # guardamos para referencia futura si quieres invertirlo

# 5. Verificación final
print("\nDataset limpio y preprocesado:")
print(df_clean.info())
print("\nEjemplo de primeras filas:\n", df_clean.head())

# (Opcional) Guardar preprocesado como CSV para futuras ejecuciones
df_clean.to_csv("mlb_teams_preprocesado.csv", index=False)
print("\n")

# -------------------------------
# PUNTO 3: DEFINICIÓN DE LA VARIABLE OBJETIVO
# -------------------------------
print("En lugar de elegir una sola variable target, vamos a entrenar y comparar modelos para estas 3 variables binarias: division_winner; league_winner; world_series_winner")
print("De esta manera:")
print("Evaluaremos qué modelos funcionan mejor para predecir cada tipo de éxito competitivo. Analizaremos si hay patrones comunes (por ejemplo, si un equipo gana la liga, ¿cuánto influye eso en ganar la Serie Mundial?) y compararemos precisión, recall, F1-score para cada una...")
print("\n")

# -------------------------------
# PUNTO 4: SELECCIÓN DE VARIABLES
# -------------------------------
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

# Variables objetivo que analizaremos
target_vars = ['division_winner', 'league_winner', 'world_series_winner']

# Función modular para aplicar filtrado univariante
def seleccionar_variables_univariantes(df, target_var, k_top=10):
    print(f"\n🔍 Analizando selección univariante para target: '{target_var}'")
    
    # Separar variables predictoras de la variable objetivo
    X = df.drop(columns=target_vars)
    y = df[target_var]

    # Escalar X entre 0 y 1 para chi2
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Mutual Information
    mi_scores = mutual_info_classif(X_scaled, y, discrete_features='auto')
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # Chi-cuadrado (requiere valores positivos)
    chi2_scores, _ = chi2(X_scaled, y)
    chi2_series = pd.Series(chi2_scores, index=X.columns).sort_values(ascending=False)

    # Mostrar top k variables seleccionadas
    print(f"\n🔹 Top {k_top} por Mutual Information:")
    print(mi_series.head(k_top))
    print(f"\n🔹 Top {k_top} por Chi-cuadrado:")
    print(chi2_series.head(k_top))

    # (Opcional) guardar como archivo CSV para estudio posterior
    resumen = pd.DataFrame({
        'Mutual Information': mi_series,
        'Chi2': chi2_series
    }).sort_values(by='Mutual Information', ascending=False)
    
    resumen.to_csv(f"seleccion_variables_{target_var}.csv")

    return resumen

# Ejecutamos para cada target
selecciones = {}
for target in target_vars:
    selecciones[target] = seleccionar_variables_univariantes(df_clean, target)
print("\n")

# -------------------------------
# PUNTO 5: MODELADO SUPERVISADO
# -------------------------------
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Modelos a evaluar
modelos = {
    "KNN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Decision Tree (entropy)": DecisionTreeClassifier(criterion='entropy', random_state=42),
    "Decision Tree (J48-like)": DecisionTreeClassifier(
        criterion='entropy',
        ccp_alpha=0.01,
        class_weight='balanced',
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
}

# Métricas a calcular
metricas = {
    'Accuracy': make_scorer(accuracy_score),
    'F1': make_scorer(f1_score),
    'Precision': make_scorer(precision_score),
    'Recall': make_scorer(recall_score)
}

# Evaluación cruzada
def evaluar_modelos(df, target, modelos, metricas):
    print(f"\n📊 Resultados para target: '{target}'")
    
    X = df.drop(columns=target_vars)  # usamos solo predictores
    y = df[target]

    resultados = {}

    for nombre, modelo in modelos.items():
        print(f"\n🔹 Modelo: {nombre}")
        resultados[nombre] = {}

        for metrica_nombre, scorer in metricas.items():
            score = cross_val_score(modelo, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring=scorer)
            mean_score = score.mean()
            resultados[nombre][metrica_nombre] = round(mean_score, 4)
            print(f"   {metrica_nombre}: {mean_score:.4f}")

    # Convertimos a DataFrame para guardar o visualizar
    resultados_df = pd.DataFrame(resultados).T
    resultados_df.to_csv(f"resultados_modelos_{target}.csv")
    return resultados_df

# Ejecutamos para cada variable objetivo
resultados_modelado = {}
for target in target_vars:
    resultados_modelado[target] = evaluar_modelos(df_clean, target, modelos, metricas)

# -------------------------------
# EVALUACIÓN ADICIONAL - J48 con SMOTE
# -------------------------------
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

resultados_j48_smote = {}

for target in target_vars:
    print(f"\n🌱 Evaluando J48 con SMOTE para target: {target}")
    
    modelos_j48_smote = {
        "Decision Tree (J48-SMOTE)": Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('tree', DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.01, random_state=42))
        ])
    }

    resultados_j48_smote[target] = evaluar_modelos(df_clean, target, modelos_j48_smote, metricas)
    resultados_j48_smote[target].to_csv(f"resultados_j48_smote_{target}.csv")

# -------------------------------
# OPCIÓN A: VISUALIZACIÓN DE RESULTADOS
# -------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta de imágenes si no existe
os.makedirs("imgs1", exist_ok=True)

# Visualización por target y métrica
for target, df_resultados in resultados_modelado.items():
    for metrica in df_resultados.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df_resultados.index, y=df_resultados[metrica])
        plt.title(f"{metrica} - Modelos para {target}")
        plt.ylabel(metrica)
        plt.xlabel("Modelo")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"imgs1/{target}_{metrica.lower().replace(' ', '_')}.png")
        plt.close()

# -------------------------------
# OPCIÓN C: Random Forest para los 3 targets
# -------------------------------
from sklearn.ensemble import RandomForestClassifier

# Añadir Random Forest a los modelos existentes
rf_modelos = {
    "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Reutilizamos las métricas definidas antes
resultados_rf = {}

for target in target_vars:
    print(f"\n🌲 Random Forest para target: {target}")
    resultados_rf[target] = evaluar_modelos(df_clean, target, rf_modelos, metricas)
print("\n")

# -------------------------------
# OPCIÓN C: VISUALIZACIÓN DE RESULTADOS - RANDOM FOREST
# -------------------------------
# Extraer las métricas de Random Forest para cada target
rf_summary = pd.DataFrame({
    target: resultados_rf[target].loc["Random Forest (100 trees)"]
    for target in target_vars
}).T  # Transponemos para que cada fila sea un target

# Generar un gráfico por métrica
for metrica in rf_summary.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=rf_summary.index, y=rf_summary[metrica], hue=rf_summary.index, palette="Blues_d", legend=False)
    plt.title(f"{metrica} - Random Forest (100 árboles)")
    plt.ylabel(metrica)
    plt.xlabel("Target")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"imgs1/randomforest_{metrica.lower().replace(' ', '_')}_targets.png")
    plt.close()

# -------------------------------
# PUNTO 6: COMBINACIÓN DE CLASIFICADORES
# -------------------------------
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# MÉTRICAS PERSONALIZADAS
# -------------------------------
# Para evitar errores de métricas indefinidas
metricas = {
    'Accuracy': make_scorer(accuracy_score),
    'F1': make_scorer(f1_score, zero_division=0),
    'Precision': make_scorer(precision_score, zero_division=0),
    'Recall': make_scorer(recall_score, zero_division=0)
}

# -------------------------------
# EVALUACIÓN MODULAR
# -------------------------------
def evaluar_modelos(df, target, modelos, metricas):
    print(f"\n📊 Resultados para target: '{target}'")
    
    X = df.drop(columns=target_vars)
    y = df[target]

    resultados = {}

    for nombre, modelo in modelos.items():
        print(f"\n🔹 Modelo: {nombre}")
        resultados[nombre] = {}

        for metrica_nombre, scorer in metricas.items():
            score = cross_val_score(modelo, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring=scorer)
            mean_score = score.mean()
            resultados[nombre][metrica_nombre] = round(mean_score, 4)
            print(f"   {metrica_nombre}: {mean_score:.4f}")

    return pd.DataFrame(resultados).T

# -------------------------------
# MODELOS COMBINADOS
# -------------------------------
# Modelos base
modelos_base = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('nb', GaussianNB()),
    ('dt', DecisionTreeClassifier(criterion='entropy', random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))
]

# Metamodelo para stacking
meta_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

# Ensambladores
combinadores = {
    "VotingClassifier": VotingClassifier(estimators=modelos_base, voting='hard'),
    "StackingClassifier": StackingClassifier(estimators=modelos_base, final_estimator=meta_model, cv=5)
}

# -------------------------------
# EVALUACIÓN Y GUARDADO
# -------------------------------
# Crear carpeta para gráficos si no existe
os.makedirs("imgs1", exist_ok=True)

resultados_ensamble = {}

for target in target_vars:
    print(f"\n🤝 Evaluación de clasificadores combinados para target: {target}")
    resultados_ensamble[target] = evaluar_modelos(df_clean, target, combinadores, metricas)
    resultados_ensamble[target].to_csv(f"resultados_ensamble_{target}.csv")

# -------------------------------
# VISUALIZACIÓN DE RESULTADOS
# -------------------------------
for target, df_resultados in resultados_ensamble.items():
    for metrica in df_resultados.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=df_resultados.index, y=df_resultados[metrica], hue=df_resultados.index, palette="Purples_d", legend=False)
        plt.title(f"{metrica} - Modelos Combinados para {target}")
        plt.ylabel(metrica)
        plt.xlabel("Modelo")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"imgs1/{target}_ensambles_{metrica.lower().replace(' ', '_')}.png")
        plt.close()
print("\n")