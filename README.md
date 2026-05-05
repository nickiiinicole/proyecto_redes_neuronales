# Proyecto  — Redes Neuronales (Clasificación de Crédito)

Este repositorio contiene el notebook **FinalProjectUD4.ipynb**, un proyecto de la UD4 (Redes Neuronales) para **clasificación supervisada** con datos tabulares.

## ¿De qué va el proyecto?
Se trabaja con el dataset **`credits.csv`** (1000 instancias) para predecir la variable objetivo **`class`**:
- `good` → crédito otorgado
- `bad` → crédito denegado

El notebook recorre un flujo completo: **carga de datos → EDA → preprocesado → entrenamiento/evaluación de redes neuronales → comparación con otros modelos**.

> En la plantilla del proyecto se ofrece también el dataset alternativo `glass.csv` (multiclase), pero en este notebook se usa el de **crédito**.

## Pasos principales del notebook
### 1) Importación
- Lectura del dataset: `creditos = pd.read_csv('credits.csv')`.

### 2) EDA + Preprocesamiento
- Análisis exploratorio con visualizaciones (matplotlib/seaborn/plotly) y revisión de valores nulos/outliers.
- Preprocesado aplicado:
  - Se elimina `ID`.
  - Se imputan nulos de `own_telephone` con la **moda**.
  - Se convierte el target a binario: `good → 1`, `bad → 0`.
  - `train_test_split` estratificado (80/20) para mantener proporción de clases.
  - **One-Hot Encoding** con `pd.get_dummies(..., drop_first=True)` y alineado de columnas entre train y test.
  - Escalado con **StandardScaler**.

### 3) Redes Neuronales (Keras/TensorFlow)
Se entrenan dos redes densas (clasificación binaria, salida sigmoide), usando:
- `EarlyStopping` (monitor `val_loss`, `patience=25`, `restore_best_weights=True`)
- `class_weight='balanced'` (vía `compute_class_weight`) para tratar el **desbalanceo de clases**

Modelos:
- **Modelo 1 (Red_Simple):** Dense(32, relu) → Dense(1, sigmoid)
- **Modelo 2 (Red_Compleja):** Dense(64, relu) + Dropout(0.1) → Dense(32, relu) + Dropout(0.1) → Dense(1, sigmoid)
  - Adam con *learning rate* más bajo (0.0005)

Evaluación en test (20%) con métricas orientadas a la clase de riesgo (`bad` = 0), incluyendo:
- Accuracy / **Balanced Accuracy**
- F1 y Recall para clase 0
- ROC-AUC
- Matrices de confusión y curvas de aprendizaje (loss)

### 4) Comparación con otros modelos supervisados
Se comparan modelos clásicos con **validación cruzada estratificada (10-fold)** y métrica `balanced_accuracy`:
- Logistic Regression, KNN, Decision Tree, **Random Forest**, SVM

El notebook selecciona **Random Forest** como candidato final por estabilidad y rendimiento (especialmente en la clase minoritaria), y lo evalúa en test con:
- Matriz de confusión
- Curvas ROC y Precision-Recall (enfocadas a detectar `bad`)
- Importancia de variables (Top 15)

## Cómo ejecutar
1. Abre **FinalProjectUD4.ipynb** en Jupyter Notebook / JupyterLab / VS Code.
2. Asegúrate de tener Python y las dependencias principales:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn plotly
```

3. Ejecuta las celdas en orden.

## Reproducibilidad
- Se fija semilla `seed = 42` para NumPy y TensorFlow.

---

Archivos principales:
- `FinalProjectUD4.ipynb`: notebook del proyecto
- `credits.csv`: dataset utilizado
- `glass.csv`: dataset alternativo (no usado en este notebook)
