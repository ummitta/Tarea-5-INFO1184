from ucimlrepo import fetch_ucirepo

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Gráficos
def distribucion_edades(df):
    plt.figure(figsize=(8, 5))

    plt.hist(df["Age"], bins=10)

    plt.title("Distribución de edades")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")

    plt.show()
    return


def edad_supervivencia(df):
    plt.figure(figsize=(6, 5))

    muere = df[df["Class"] == "Muere"]["Age"]
    sobrevive = df[df["Class"] == "Sobrevive"]["Age"]

    plt.boxplot(
        [muere, sobrevive],
        tick_labels=["Muere", "Sobrevive"]
    )

    plt.title("Edad según resultado del paciente")
    plt.xlabel("Resultado")
    plt.ylabel("Edad")
    plt.grid(True)

    plt.show()
    return


def ascitis_supervivencia(df):
    tabla = pd.crosstab(df["Ascites"], df["Class"])

    tabla.plot(kind="bar", figsize=(7, 5))

    plt.title("Ascitis vs resultado")
    plt.xlabel("Ascitis")
    plt.ylabel("Cantidad de pacientes")
    plt.xticks(rotation=0)
    plt.grid(True)

    plt.show()
    return


def albumina_supervivencia(df):
    plt.figure(figsize=(6, 5))

    muere = df[df["Class"] == "Muere"]["Albumin"]
    sobrevive = df[df["Class"] == "Sobrevive"]["Albumin"]

    plt.boxplot(
        [muere, sobrevive],
        tick_labels=["Muere", "Sobrevive"]
    )

    plt.title("Albúmina según resultado")
    plt.xlabel("Resultado")
    plt.ylabel("Nivel de albúmina")
    plt.grid(True)

    plt.show()
    return


def bilirrubina_supervivencia(df):
    plt.figure(figsize=(6, 5))

    muere = df[df["Class"] == "Muere"]["Bilirubin"]
    sobrevive = df[df["Class"] == "Sobrevive"]["Bilirubin"]

    plt.boxplot(
        [muere, sobrevive],
        tick_labels=["Muere", "Sobrevive"]
    )

    plt.title("Bilirrubina según resultado")
    plt.xlabel("Resultado")
    plt.ylabel("Bilirrubina")
    plt.grid(True)

    plt.savefig(
        "bilirrubina_supervivencia.png",
        bbox_inches="tight"
    )

    plt.show()
    return


def matriz_correlacion(df, vars_categoricas):
    # Se crea una copia para no modificar los valores del DataFrame original
    df_correlacion = df.copy()

    # Convierte la variable objetivo en números
    df_correlacion["Class"] = df_correlacion["Class"].map({
        "Muere": 0,
        "Sobrevive": 1
    })

    # Convierte el sexo en números
    df_correlacion["Sex"] = df_correlacion["Sex"].map({
        "Hombre": 0,
        "Mujer": 1
    })

    # Convierte las variables de tipo Sí/No en números
    for col in vars_categoricas:
        df_correlacion[col] = df_correlacion[col].map({
            "No": 0,
            "Si": 1
        })

    # Calcula las correlaciones
    corr = df_correlacion.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))

    plt.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    # Agrega el valor numérico de cada correlación
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    plt.title("Matriz de correlación")
    plt.tight_layout()

    plt.show()
    return


# Etapas de la metodología CRISP-DM
def comprension_datos(df) -> None:
    print(df.info())
    print(f"{'':=^32}")

    print("\nDimensiones:")
    print(df.shape)
    print(f"{'':=^32}")

    print("\nValores faltantes:")
    print(df.isnull().sum())
    print(f"{'':=^32}")

    print("\nResumen estadístico:")
    print(df.describe())
    print(f"{'':=^32}")

    return


def preparacion_datos(df) -> None:
    # Reemplaza los valores numéricos faltantes por la mediana
    vars_numericas = [
        "Age",
        "Bilirubin",
        "Alk Phosphate",
        "Sgot",
        "Albumin",
        "Protime"
    ]

    for col in vars_numericas:
        df[col] = df[col].fillna(df[col].median())

    # Reemplaza los valores categóricos faltantes por la moda
    vars_categoricas = [
        "Sex",
        "Steroid",
        "Antivirals",
        "Fatigue",
        "Malaise",
        "Anorexia",
        "Liver Big",
        "Liver Firm",
        "Spleen Palpable",
        "Spiders",
        "Ascites",
        "Varices",
        "Histology"
    ]

    for col in vars_categoricas:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nValores faltantes después de la preparación:")
    print(df.isnull().sum())
    print(f"{'':=^32}")

    # Cambia los valores por etiquetas más fáciles de interpretar
    df["Class"] = df["Class"].map({
        1: "Muere",
        2: "Sobrevive"
    })

    df["Sex"] = df["Sex"].map({
        1: "Hombre",
        2: "Mujer"
    })

    for col in vars_categoricas[1:]:
        df[col] = df[col].map({
            1: "No",
            2: "Si"
        })

    # Gráficos
    distribucion_edades(df)
    edad_supervivencia(df)
    ascitis_supervivencia(df)
    albumina_supervivencia(df)
    bilirrubina_supervivencia(df)

    # No se incluye Sex porque se transforma por separado
    variables_si_no = vars_categoricas[1:]

    matriz_correlacion(df, variables_si_no)

    return


def main() -> None:
    # Carga el dataset de hepatitis
    hepatitis = fetch_ucirepo(id=46)

    X = hepatitis.data.features
    y = hepatitis.data.targets

    # Une las variables en un único DataFrame
    df = pd.concat([X, y], axis=1)

    # Etapas de la metodología CRISP-DM
    comprension_datos(df)
    preparacion_datos(df)

    return


if __name__ == "__main__":
    main()