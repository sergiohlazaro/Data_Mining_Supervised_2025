# Mineria de Datos

## 📌 Descripción del Proyecto
Este proyecto tiene como objetivo la experimentación con distintos **modelos de clasificación supervisada** en el contexto de **Minería de Datos**. Se ha aplicado un enfoque de **preprocesamiento de datos, selección de variables, optimización de hiperparámetros y combinación de modelos** para determinar cuál ofrece el mejor rendimiento.

---
## 📂 Estructura del Proyecto

```
📁 Mineria_de_Datos_Proyecto
│-- 📄 script_trabajo1_v1.py  # Versión inicial con modelos básicos
│-- 📄 script_trabajo1_v2.py  # Ajustes en el preprocesamiento de datos
│-- 📄 script_trabajo1_v3.py  # Balanceo de clases con SMOTE
│-- 📄 script_trabajo1_v4.py  # Selección de variables
│-- 📄 script_trabajo1_v5.py  # Implementación de validación cruzada
│-- 📄 script_trabajo1_v6.py  # Optimización de hiperparámetros
│-- 📄 script_trabajo1_v7.py  # Combinación de modelos
│-- 📄 script_trabajo1_v8.py  # Modelos finales y comparación
│-- 📄 README.md  # Documentación del proyecto
│-- 📁 imgs1/   # Carpeta con imágenes generadas
│   ├── incidents.byCountryYr.csv  # Dataset de incidentes
│-- 📁 documentation/  # Resultados, documentos...
│   ├── ...
│   ├── ...
```

---
## 🚀 Instalación y Configuración

### **📦 Requerimientos**
Asegúrate de tener **Python 3.9 o superior** instalado junto con las siguientes librerías:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

### **📂 Uso**
1. **Descargar el repositorio**
```bash
git clone https://github.com/tu_usuario/mineria_de_datos.git
cd mineria_de_datos
```
2. **Ejecutar un script específico**
```bash
python script_trabajo1_v1.py
```
3. **Explorar resultados** en la terminal.

---
## 📊 Modelos Utilizados
A continuación, se listan algunos de los modelos implementados, junto con sus resultados **antes y después de la optimización**:

### **📌 Algoritmos de Clasificación**
| Modelo | Accuracy (Inicial) | Accuracy (Optimizado) |
|--------|-------------------|----------------------|
| IB1 (k=1) | 56.66% | 56.22% |
| IBk (k=3) | 57.40% | 64.26% |
| Naive Bayes | 36.17% | 23.77% |
| Árbol de Decisión (ID3) | 62.30% | 57.69% |
| J48 (ID3 con poda) | 45.49% | N/A |
| Regresión Logística | 37.99% | 19.65% |
| JRip (Reglas RIPPER - WEKA) | 60.11% | N/A |
| Voting Classifier | 60.78% | 63.28% |
| Stacking Classifier | 58.33% | 57.01% |

✅ **Mejor modelo final:** **Weighted Voting Classifier** (Accuracy **63.77%**)

---
## 📌 Selección de Variables
Se implementaron distintos métodos para reducir la dimensionalidad del dataset:
- **Filtrado Univariante**: Chi-cuadrado y Mutual Information.
- **Wrapper Methods**: Recursive Feature Elimination (RFE), Sequential Feature Selection (SFS).
- **Embedded Methods**: Random Forest Feature Importance, Lasso Regression.

🔎 **Resultados:**
- **Eliminar `iyear` afectó negativamente a Naive Bayes y Regresión Logística**.
- **ID3 y k-NN mantuvieron su rendimiento tras la selección de variables**.

---
## 🔧 Optimización de Modelos
Se utilizaron técnicas como `GridSearchCV` para ajustar hiperparámetros:

| Modelo | Mejores Hiperparámetros |
|--------|------------------------|
| k-NN | `n_neighbors=3`, `weights='uniform'` |
| Árbol de Decisión (ID3) | `max_depth=None`, `min_samples_split=2` |
| Naive Bayes | `var_smoothing=1.0` |
| Regresión Logística | `C=0.0001`, `solver='lbfgs'` |

**Impacto:** La optimización mejoró k-NN e ID3, pero **Naive Bayes y Regresión Logística no se beneficiaron**.

---
## 📊 Validación del Modelo
Se aplicó **Validación Cruzada k-Fold (k=5)** para evaluar la estabilidad de los modelos:

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn_model, X_train_bal, y_train_bal, cv=5)
```

✅ **Mejor rendimiento en validación cruzada:**
- **k-NN: 49.75% accuracy promedio**.
- **Voting Classifier superó el 60% tras la validación**.

---
## 🔄 Combinación de Modelos
Se probaron combinaciones de modelos para mejorar la precisión final:

| Modelo Combinado | Accuracy Final |
|-----------------|---------------|
| **Voting Classifier** | **63.28%** |
| **Stacking Classifier** (Regresión Logística) | **59.31%** |
| **Weighted Voting Classifier** | **63.77%** ✅ Mejor opción |

📌 **Conclusión:** La combinación de modelos mejoró el rendimiento, destacando **Weighted Voting Classifier como el mejor modelo**.

---
## 📌 Conclusiones
✅ **k-NN (k=3) y Weighted Voting fueron los modelos más efectivos.**
❌ **Naive Bayes y Regresión Logística no fueron adecuados para este dataset.**
📊 **La combinación de modelos fue clave para obtener mejores resultados.**

---


