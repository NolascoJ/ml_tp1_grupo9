from pathlib import Path

import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "insurance_train.csv"
FACTOR_IQR = 1.5


def resumen_outliers(serie: pd.Series) -> tuple[dict[str, float], pd.Series]:
    """Calcula los límites IQR y la máscara de outliers de una variable."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - FACTOR_IQR * iqr
    limite_superior = q3 + FACTOR_IQR * iqr
    es_outlier = (serie < limite_inferior) | (serie > limite_superior)

    resumen = {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Límite inferior": limite_inferior,
        "Límite superior": limite_superior,
        "Outliers": es_outlier.sum(),
        "Porcentaje": es_outlier.mean() * 100,
    }
    return resumen, es_outlier


def main() -> None:
    """Analiza outliers del conjunto de entrenamiento usando el criterio IQR."""
    df = pd.read_csv(ARCHIVO_TRAIN)
    columnas_numericas = df.select_dtypes(include="number").columns
    mascaras_outliers = []

    print("Análisis de outliers con el criterio IQR (1.5 × IQR)")
    print("Se analiza únicamente insurance_train.csv; no se modifica ningún dato.\n")

    for columna in columnas_numericas:
        resumen, mascara = resumen_outliers(df[columna])
        mascaras_outliers.append(mascara)

        print(f"{columna}:")
        print(f"  Q1 = {resumen['Q1']:.2f} | Q3 = {resumen['Q3']:.2f} | IQR = {resumen['IQR']:.2f}")
        print(
            "  Límites: "
            f"[{resumen['Límite inferior']:.2f}, {resumen['Límite superior']:.2f}]"
        )
        print(
            f"  Outliers: {resumen['Outliers']} "
            f"({resumen['Porcentaje']:.2f}% del train)\n"
        )

    filas_con_outlier = pd.concat(mascaras_outliers, axis=1).any(axis=1)
    print(
        "Filas con al menos un outlier en variables numéricas: "
        f"{filas_con_outlier.sum()} ({filas_con_outlier.mean() * 100:.2f}% del train)"
    )


if __name__ == "__main__":
    main()
