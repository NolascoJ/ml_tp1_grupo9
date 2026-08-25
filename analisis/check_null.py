from pathlib import Path

import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "insurance_train.csv"

df = pd.read_csv(ARCHIVO_TRAIN)

print(df.dtypes)

print('\n')

print(df.isnull().sum())
