# 📊 Minería de Datos: Clasificación Supervisada

## 📁 Estructura del proyecto
# ├── imgs1/
# │ ├── eda/
# │ ├── seleccion_variables/
# │ ├── modelos_supervisados/
# │ ├── j48_smote/
# │ ├── random_forest/
# │ ├── combinados/
# │ └── fss/
# ├── resultados/
# │ ├── seleccion_variables/
# │ ├── modelos_supervisados/
# │ ├── j48_smote/
# │ ├── random_forest/
# │ ├── combinados/
# │ ├── fss/
# │ └── resumen/
# ├── mlb_teams.csv
# └── script_trabajo1.py


---

## ✅ Objetivos

- Cargar y analizar un dataset con características suficientes.
- Preprocesar los datos de manera adecuada.
- Seleccionar variables mediante análisis univariante y multivariante.
- Aplicar múltiples modelos supervisados.
- Comparar resultados e interpretar modelos transparentes.
- Aplicar técnicas de balanceo y combinaciones de clasificadores.
- Generar visualizaciones y consolidar los resultados.

---

## 📄 Dataset

Se ha utilizado un dataset con información histórica de los equipos de béisbol de la MLB, incluyendo estadísticas de temporada y resultados.

- Observaciones: 2784
- Variables: 39 (después del preprocesado)
- Targets: 
  - `division_winner`
  - `league_winner`
  - `world_series_winner`

---

## ⚙️ Proceso desarrollado

### 1. Análisis Exploratorio (EDA)
- Porcentaje de valores nulos.
- Estadísticas descriptivas.
- Matriz de correlación.
- Visualizaciones guardadas en `imgs1/eda`.

### 2. Preprocesamiento
- Imputación de valores nulos.
- Codificación de variables categóricas.
- Limpieza de columnas irrelevantes.

### 3. Selección de Variables
- **Univariante**: Mutual Information y Chi².
- **Multivariante (FSS)**:
  - Filtrado por importancia (Random Forest).
  - Wrapper con RFE + Regresión Logística.

### 4. Modelado Supervisado
Modelos aplicados:
- KNN (k=1, k=3)
- Naive Bayes
- Árbol de decisión (entropy)
- Árbol tipo J48 (con poda)
- Regresión Logística

### 5. Técnicas Avanzadas
- Árbol con **SMOTE** para datos desbalanceados.
- **Random Forest** con 100 árboles.
- **Modelos combinados**: Voting y Stacking.

### 6. Interpretación de Modelos Transparentes
- **JRip**: Reglas inferidas.
- **Árboles**: Condiciones y caminos de decisión.
- **Regresión logística**: Coeficientes y efecto de variables.
- **TAN**: Supuesto razonado de estructura y evaluación.

---

## 📈 Resultados

Los resultados se consolidan en `resultados/resumen/resumen_modelos.csv` e incluyen:

- Accuracy
- F1 Score
- Precisión
- Recall

Gráficos comparativos generados automáticamente para cada métrica y target.

---

## 🔎 Conclusiones

- El dataset ha sido procesado correctamente y se han cumplido los requisitos de la práctica.
- Los modelos como **Random Forest**, **Árboles de decisión** y **StackingClassifier** han mostrado el mejor rendimiento.
- Se ha trabajado con métricas relevantes para clases desbalanceadas.
- Todos los resultados han sido visualizados y exportados correctamente.

---

## 🧠 Requisitos de la práctica cubiertos

- ✅ Carga, análisis y limpieza de datos.
- ✅ Preprocesamiento completo.
- ✅ Selección de variables (univariante y multivariante).
- ✅ Modelado con múltiples algoritmos.
- ✅ Evaluación y comparación de resultados.
- ✅ Interpretación de modelos transparentes.
- ✅ Aplicación de técnicas avanzadas: SMOTE, combinados.
- ✅ Visualización, exportación de resultados y estructura de carpetas.

---

## 🚀 Ejecución

Para ejecutar el script:

```bash
python script_trabajo1.py

---

