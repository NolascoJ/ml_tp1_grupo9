import kagglehub
from kagglehub import KaggleDatasetAdapter


df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mirichoi0218/insurance",
    "insurance.csv",
)

df.to_csv("insurance.csv", index=False)
print(df.head())
print(f"\nDataset exportado: {df.shape[0]} filas, {df.shape[1]} columnas")
