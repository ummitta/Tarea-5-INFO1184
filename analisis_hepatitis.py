from ucimlrepo import fetch_ucirepo

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main() -> None:
    hepatitis = fetch_ucirepo(id=46)

    X = hepatitis.data.features
    y = hepatitis.data.targets

    df = pd.concat([X, y], axis=1)

    comprension_datos(df)

    preparacion_datos(df)

    return

def preparacion_datos(df) -> None:
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

    return


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


if __name__ == "__main__":
    main()