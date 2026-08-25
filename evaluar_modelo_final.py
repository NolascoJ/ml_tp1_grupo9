from pathlib import Path

import pandas as pd
from sklearn.metrics import root_mean_squared_error

from modelo_polinomico import crear_modelo_polinomico


ARCHIVO_TRAIN = Path("insurance_train.csv")
ARCHIVO_TEST = Path("test/insurance_test.csv")
GRADO = 2


def main() -> None:
    df_train = pd.read_csv(ARCHIVO_TRAIN)
    df_test = pd.read_csv(ARCHIVO_TEST)

    X_train = df_train.drop(columns="charges")
    y_train = df_train["charges"]
    X_test = df_test.drop(columns="charges")
    y_test = df_test["charges"]

    modelo = crear_modelo_polinomico(GRADO)
    modelo.fit(X_train, y_train)

    predicciones_test = modelo.predict(X_test)
    rmse_test = root_mean_squared_error(y_test, predicciones_test)

    print("Evaluación final en el conjunto de test")
    print(
        "Modelo: regresión polinómica de "
        f"grado {GRADO} sin regularización"
    )
    print(f"Elementos de entrenamiento: {len(df_train)}")
    print(f"Elementos de test: {len(df_test)}")
    print(f"RMSE test: {rmse_test:,.2f}")


if __name__ == "__main__":
    main()
