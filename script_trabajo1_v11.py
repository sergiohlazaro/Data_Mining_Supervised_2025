# -------------------------------
# CONFIGURACIÓN DE CARPETAS
# -------------------------------
import os
from pathlib import Path

folders_imgs = [
    "imgs1/eda",
    "imgs1/seleccion_variables",
    "imgs1/modelos_supervisados",
    "imgs1/j48_smote",
    "imgs1/random_forest",
    "imgs1/combinados"
]

folders_csv = [
    "resultados/seleccion_variables",
    "resultados/modelos_supervisados",
    "resultados/j48_smote",
    "resultados/random_forest",
    "resultados/combinados"
]

for folder in folders_imgs + folders_csv:
    Path(folder).mkdir(parents=True, exist_ok=True)

# -------------------------------
# PUNTO 1: ANÁLISIS EXPLORATORIO DE DATOS
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el CSV
df = pd.read_csv("mlb_teams.csv")

# 1. Porcentaje de valores nulos
null_percent = df.isnull().mean().sort_values(ascending=False) * 100
null_percent = null_percent[null_percent > 0]
print("\nPorcentaje de valores nulos:")
print(null_percent)

# 2. Estadísticas descriptivas
print("\nEstadísticas descriptivas (numéricas):")
print(df.describe(include='number'))
print("\nEstadísticas descriptivas (categóricas):")
print(df.describe(include='object'))

# 3. Gráfico de valores nulos
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Mapa de Calor de Valores Nulos")
plt.xlabel("Variables")
plt.ylabel("Registros")
plt.tight_layout()
plt.savefig("imgs1/eda/mapa_valores_nulos.png")
plt.close()

# 4. Matriz de correlación
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title("Matriz de Correlación de Variables Numéricas")
plt.tight_layout()
plt.savefig("imgs1/eda/matriz_correlacion.png")
plt.close()

# -------------------------------
# PUNTO 2: PREPROCESAMIENTO
# -------------------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Eliminar columnas irrelevantes
df_clean = df.drop(columns=['rownames', 'team_name', 'ball_park'])

# Imputación numérica
num_cols = df_clean.select_dtypes(include='number').columns
imputer_num = SimpleImputer(strategy='mean')
df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])

# Imputación categórica y codificación
cat_cols = df_clean.select_dtypes(include='object').columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

print("\nDataset limpio y preprocesado:")
print(df_clean.info())
print("\nEjemplo de primeras filas:\n", df_clean.head())
df_clean.to_csv("resultados/modelos_supervisados/mlb_teams_preprocesado.csv", index=False)

# -------------------------------
# PUNTO 3: DEFINICIÓN DE LA VARIABLE OBJETIVO
# -------------------------------
print("\nTrabajaremos con tres targets: division_winner, league_winner, world_series_winner\n")
target_vars = ['division_winner', 'league_winner', 'world_series_winner']

# -------------------------------
# PUNTO 4: SELECCIÓN DE VARIABLES
# -------------------------------
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

def seleccionar_variables_univariantes(df, target_var, k_top=10):
    print(f"\n🔍 Selección univariante para target: '{target_var}'")
    X = df.drop(columns=target_vars)
    y = df[target_var]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    mi_scores = mutual_info_classif(X_scaled, y, discrete_features='auto')
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    chi2_scores, _ = chi2(X_scaled, y)
    chi2_series = pd.Series(chi2_scores, index=X.columns).sort_values(ascending=False)
    resumen = pd.DataFrame({
        'Mutual Information': mi_series,
        'Chi2': chi2_series
    }).sort_values(by='Mutual Information', ascending=False)
    resumen.to_csv(f"resultados/seleccion_variables/{target_var}_seleccion_variables.csv")
    return resumen

selecciones = {}
for target in target_vars:
    selecciones[target] = seleccionar_variables_univariantes(df_clean, target)

# -------------------------------
# VISUALIZACIÓN DE SELECCIÓN DE VARIABLES
# -------------------------------
for target, df_vars in selecciones.items():
    top_mi = df_vars['Mutual Information'].nlargest(10)
    top_chi2 = df_vars['Chi2'].nlargest(10)

    # Mutual Information
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_mi.values, y=top_mi.index, hue=top_mi.index, palette="coolwarm", legend=False)
    plt.title(f"Top 10 Variables - Mutual Information ({target})")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(f"imgs1/seleccion_variables/{target}_top10_mutual_info.png")
    plt.close()

    # Chi2
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_chi2.values, y=top_chi2.index, hue=top_chi2.index, palette="magma", legend=False)
    plt.title(f"Top 10 Variables - Chi2 ({target})")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(f"imgs1/seleccion_variables/{target}_top10_chi2.png")
    plt.close()

# -------------------------------
# PUNTO 5: MODELADO SUPERVISADO
# -------------------------------
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

modelos = {
    "KNN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Decision Tree (entropy)": DecisionTreeClassifier(criterion='entropy', random_state=42),
    "Decision Tree (J48-like)": DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.01, class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
}

metricas = {
    'Accuracy': make_scorer(accuracy_score),
    'F1': make_scorer(f1_score, zero_division=0),
    'Precision': make_scorer(precision_score, zero_division=0),
    'Recall': make_scorer(recall_score, zero_division=0)
}

def evaluar_modelos(df, target, modelos, metricas, carpeta_resultado="resultados/modelos_supervisados", sufijo="_resultados_modelos"):
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
    resultados_df = pd.DataFrame(resultados).T
    Path(carpeta_resultado).mkdir(parents=True, exist_ok=True)
    resultados_df.to_csv(f"{carpeta_resultado}/{target}{sufijo}.csv")
    return resultados_df

resultados_modelado = {}
for target in target_vars:
    resultados_modelado[target] = evaluar_modelos(df_clean, target, modelos, metricas, "resultados/modelos_supervisados", "_supervisado")

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
        plt.savefig(f"imgs1/modelos_supervisados/{target}_{metrica.lower().replace(' ', '_')}.png")
        plt.close()

# -------------------------------
# J48 CON SMOTE
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
    resultados_j48_smote[target] = evaluar_modelos(df_clean, target, modelos_j48_smote, metricas, "resultados/j48_smote", "_smote")

# -------------------------------
# VISUALIZACIÓN DE RESULTADOS - J48 CON SMOTE
# -------------------------------
for target, df_resultados in resultados_j48_smote.items():
    for metrica in df_resultados.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=df_resultados.index, y=df_resultados[metrica], hue=df_resultados.index, palette="Greens_d", legend=False)
        plt.title(f"{metrica} - J48 con SMOTE para {target}")
        plt.ylabel(metrica)
        plt.xlabel("Modelo")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"imgs1/j48_smote/{target}_j48_smote_{metrica.lower().replace(' ', '_')}.png")
        plt.close()

# -------------------------------
# RANDOM FOREST
# -------------------------------
from sklearn.ensemble import RandomForestClassifier
rf_modelos = {
    "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42)
}

resultados_rf = {}
for target in target_vars:
    print(f"\n🌲 Random Forest para target: {target}")
    resultados_rf[target] = evaluar_modelos(df_clean, target, rf_modelos, metricas, "resultados/random_forest", "_rf")

rf_summary = pd.DataFrame({
    target: resultados_rf[target].loc["Random Forest (100 trees)"]
    for target in target_vars
}).T

for metrica in rf_summary.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=rf_summary.index, y=rf_summary[metrica], hue=rf_summary.index, palette="Blues_d", legend=False)
    plt.title(f"{metrica} - Random Forest (100 árboles)")
    plt.ylabel(metrica)
    plt.xlabel("Target")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"imgs1/random_forest/randomforest_{metrica.lower().replace(' ', '_')}_targets.png")
    plt.close()

# -------------------------------
# COMBINACIÓN DE CLASIFICADORES
# -------------------------------
from sklearn.ensemble import VotingClassifier, StackingClassifier

modelos_base = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('nb', GaussianNB()),
    ('dt', DecisionTreeClassifier(criterion='entropy', random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))
]

meta_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

combinadores = {
    "VotingClassifier": VotingClassifier(estimators=modelos_base, voting='hard'),
    "StackingClassifier": StackingClassifier(estimators=modelos_base, final_estimator=meta_model, cv=5)
}

resultados_ensamble = {}
for target in target_vars:
    print(f"\n🤝 Clasificadores combinados para target: {target}")
    resultados_ensamble[target] = evaluar_modelos(df_clean, target, combinadores, metricas, "resultados/combinados", "_combinado")

for target, df_resultados in resultados_ensamble.items():
    for metrica in df_resultados.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=df_resultados.index, y=df_resultados[metrica], hue=df_resultados.index, palette="Purples_d", legend=False)
        plt.title(f"{metrica} - Modelos Combinados para {target}")
        plt.ylabel(metrica)
        plt.xlabel("Modelo")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"imgs1/combinados/{target}_ensambles_{metrica.lower().replace(' ', '_')}.png")
        plt.close()
print("\n")

# -------------------------------
# TABLA RESUMEN CONSOLIDADA DE RESULTADOS
# -------------------------------
import glob

# Función para cargar todos los CSV de una carpeta y combinarlos
def cargar_resultados_carpeta(ruta_carpeta, sufijo):
    resumen = []
    for path_csv in glob.glob(f"{ruta_carpeta}/*{sufijo}.csv"):
        nombre_target = Path(path_csv).stem.replace(sufijo, "").replace("_", "").replace("__", "_")
        df = pd.read_csv(path_csv, index_col=0)
        df['Target'] = nombre_target
        df['Modelo'] = df.index
        resumen.append(df)
    return pd.concat(resumen, axis=0)

# Cargar todos los resultados
df_supervisados = cargar_resultados_carpeta("resultados/modelos_supervisados", "_supervisado")
df_j48_smote = cargar_resultados_carpeta("resultados/j48_smote", "_smote")
df_rf = cargar_resultados_carpeta("resultados/random_forest", "_rf")
df_combinados = cargar_resultados_carpeta("resultados/combinados", "_combinado")

# Consolidar en una sola tabla
df_consolidado = pd.concat([
    df_supervisados.assign(Tipo="Modelos Supervisados"),
    df_j48_smote.assign(Tipo="J48 + SMOTE"),
    df_rf.assign(Tipo="Random Forest"),
    df_combinados.assign(Tipo="Combinados"),
])

# Reordenar columnas
columnas_finales = ['Target', 'Tipo', 'Modelo', 'Accuracy', 'F1', 'Precision', 'Recall']
df_consolidado = df_consolidado[columnas_finales]

# Guardar la tabla consolidada
Path("resultados/resumen").mkdir(parents=True, exist_ok=True)
df_consolidado.to_csv("resultados/resumen/resumen_modelos.csv", index=False)

# Mostrar por pantalla
print("📊 Tabla resumen de todos los modelos:")
print(df_consolidado.to_string(index=False))
print("\n")