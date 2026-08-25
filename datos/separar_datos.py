from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_ORIGINAL = RAIZ_PROYECTO / "data" / "raw" / "insurance.csv"
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "insurance_train.csv"
ARCHIVO_TEST = RAIZ_PROYECTO / "data" / "test" / "insurance_test.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    """Separa el dataset en train y test."""
    df = pd.read_csv(ARCHIVO_ORIGINAL)

    columnas_requeridas = {
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
        "charges",
    }

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Se restablece el índice sin modificar ninguna de las variables.
    ARCHIVO_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO_TEST.parent.mkdir(parents=True, exist_ok=True)
    train.reset_index(drop=True).to_csv(ARCHIVO_TRAIN, index=False)
    test.reset_index(drop=True).to_csv(ARCHIVO_TEST, index=False)

    print(f"Train guardado en {ARCHIVO_TRAIN}: {len(train)} filas")
    print(f"Test reservado en {ARCHIVO_TEST}: {len(test)} filas")
    print("No abras ni uses el archivo de test hasta la evaluación final.")


if __name__ == "__main__":
    main()
