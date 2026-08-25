from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_SALIDA = RAIZ_PROYECTO / "data" / "raw" / "insurance.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mirichoi0218/insurance",
    "insurance.csv",
)

ARCHIVO_SALIDA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(ARCHIVO_SALIDA, index=False)
print(df.head())
print(
    f"\nDataset exportado en {ARCHIVO_SALIDA}: "
    f"{df.shape[0]} filas, {df.shape[1]} columnas"
)
