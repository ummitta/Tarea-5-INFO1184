from ucimlrepo import fetch_ucirepo

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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