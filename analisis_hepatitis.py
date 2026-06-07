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