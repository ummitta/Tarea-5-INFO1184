# ============================================================
# INFO1184 - Inteligencia de Negocios
# Tarea 5 - Análisis de datos y visualización
# Dataset: Hepatitis - UCI
# ============================================================

from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.decomposition import PCA


# ============================================================
# 1. CARGA DEL DATASET
# ============================================================

hepatitis = fetch_ucirepo(id=46)

X = hepatitis.data.features
y = hepatitis.data.targets

df = pd.concat([X, y], axis=1)

target_col = y.columns[0]

print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())

print("\nInformación general:")
print(df.info())

print("\nValores faltantes:")
print(df.isnull().sum())


# ============================================================
# 2. LIMPIEZA DE DATOS
# ============================================================

# Eliminar columnas con demasiados datos faltantes
porcentaje_faltantes = df.isnull().mean() * 100

print("\nPorcentaje de faltantes por columna:")
print(porcentaje_faltantes.sort_values(ascending=False))

# Umbral: eliminar columnas con más de 40% de datos faltantes
columnas_eliminar = porcentaje_faltantes[porcentaje_faltantes > 40].index.tolist()

# No eliminar la variable objetivo
if target_col in columnas_eliminar:
    columnas_eliminar.remove(target_col)

print("\nColumnas eliminadas por muchos faltantes:")
print(columnas_eliminar)

df_limpio = df.drop(columns=columnas_eliminar)

X = df_limpio.drop(columns=[target_col])
y = df_limpio[target_col]


# ============================================================
# 3. GRÁFICO 1: VALORES FALTANTES
# ============================================================

plt.figure(figsize=(10, 5))
msno.matrix(df_limpio)
plt.title("Valores faltantes después de eliminar columnas poco útiles")
plt.savefig("01_valores_faltantes.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 4. GRÁFICO 2: DISTRIBUCIÓN DE LA CLASE OBJETIVO
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x=target_col)
plt.title("Distribución de la clase objetivo")
plt.xlabel("Clase")
plt.ylabel("Cantidad de pacientes")
plt.savefig("02_distribucion_clase.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nDistribución de clases:")
print(y.value_counts())


# ============================================================
# 5. SELECCIÓN DE VARIABLES ÚTILES
# ============================================================

# En este dataset muchas variables están codificadas como números.
# Separamos las numéricas.
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Detectar variables binarias: columnas con solo 2 valores diferentes
variables_binarias = []
variables_continuas = []

for col in numeric_cols:
    valores_unicos = X[col].dropna().nunique()

    if valores_unicos <= 2:
        variables_binarias.append(col)
    else:
        variables_continuas.append(col)

print("\nVariables binarias/categóricas codificadas:")
print(variables_binarias)

print("\nVariables continuas:")
print(variables_continuas)


# ============================================================
# 6. GRÁFICO 3: MATRIZ DE CORRELACIÓN
# ============================================================

# Para la matriz se usan variables continuas y la clase objetivo.
# Esto evita que variables binarias generen interpretaciones raras.
df_corr = df_limpio[variables_continuas + [target_col]]

plt.figure(figsize=(10, 7))
corr = df_corr.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación de variables clínicas")
plt.savefig("03_matriz_correlacion.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 7. PREPROCESAMIENTO: IMPUTACIÓN Y NORMALIZACIÓN
# ============================================================

# Todas las columnas numéricas serán imputadas y normalizadas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ]
)


# ============================================================
# 8. DIVISIÓN DE DATOS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTamaño de entrenamiento:", X_train.shape)
print("Tamaño de prueba:", X_test.shape)


# ============================================================
# 9. MODELADO
# ============================================================

modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

resultados = []

for nombre, modelo in modelos.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", modelo)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print("\n======================================")
    print("Modelo:", nombre)
    print("======================================")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.savefig(f"matriz_confusion_{nombre}.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 10. GRÁFICO 4: COMPARACIÓN DE MODELOS
# ============================================================

df_resultados = pd.DataFrame(resultados)

print("\nResultados de modelos:")
print(df_resultados)

df_resultados_melt = df_resultados.melt(
    id_vars="Modelo",
    value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
    var_name="Métrica",
    value_name="Valor"
)

plt.figure(figsize=(9, 5))
sns.barplot(data=df_resultados_melt, x="Modelo", y="Valor", hue="Métrica")
plt.title("Comparación de modelos de clasificación")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.savefig("04_comparacion_modelos.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 11. GRÁFICO 5: IMPORTANCIA DE VARIABLES
# ============================================================

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=100))
])

rf_pipeline.fit(X_train, y_train)

rf_model = rf_pipeline.named_steps["classifier"]

importancias = rf_model.feature_importances_

df_importancias = pd.DataFrame({
    "Variable": num_cols,
    "Importancia": importancias
}).sort_values(by="Importancia", ascending=False)

print("\nImportancia de variables:")
print(df_importancias)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_importancias, x="Importancia", y="Variable")
plt.title("Importancia de variables según Random Forest")
plt.savefig("05_importancia_variables.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 12. GRÁFICO 6: PCA
# ============================================================

X_preprocesado = preprocessor.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocesado)

df_pca = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Clase": y
})

print("\nVarianza explicada por PCA:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Clase", s=70)
plt.title("PCA del dataset Hepatitis")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.savefig("06_pca_hepatitis.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 13. EXPORTAR RESULTADOS
# ============================================================

df_resultados.to_csv("resultados_modelos.csv", index=False)
df_importancias.to_csv("importancia_variables.csv", index=False)

print("\nProceso finalizado.")
print("Se generaron imágenes y archivos CSV para el informe.")