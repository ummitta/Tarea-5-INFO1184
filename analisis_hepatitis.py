from ucimlrepo import fetch_ucirepo

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main() -> None:
    hepatitis = fetch_ucirepo(id=46)

    X = hepatitis.data.features
    y = hepatitis.data.targets

    df = pd.concat([X, y], axis=1)

    return


if __name__ == "__main__":
    main()