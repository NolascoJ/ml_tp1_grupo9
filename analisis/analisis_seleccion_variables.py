"""Análisis de selección de variables para predecir ``charges``.

El script usa solamente el conjunto de entrenamiento y aplica:

* correlación de Pearson a los predictores numéricos;
* ANOVA F-test a los predictores categóricos;
* información mutua a todos los predictores;
* matriz de correlación entre variables numéricas;

No se aplica Chi-cuadrado contra ``charges`` porque la variable objetivo es
continua. Todos los resultados se guardan en ``resultados_seleccion_variables``.
"""

import os
from pathlib import Path
from tempfile import gettempdir

# Evita depender de que el directorio de configuración del usuario sea
# escribible (por ejemplo, al ejecutar dentro de un contenedor).
_cache_matplotlib = Path(gettempdir()) / "ml_tp1_matplotlib"
_cache_matplotlib.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_matplotlib))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "insurance_train.csv"
CARPETA_SALIDA = RAIZ_PROYECTO / "resultados" / "seleccion_variables"
CARPETA_GRAFICOS = CARPETA_SALIDA / "graficos"
VARIABLE_OBJETIVO = "charges"

COLUMNAS_NUMERICAS = ["age", "bmi", "children"]
COLUMNAS_CATEGORICAS = ["sex", "smoker", "region"]
COLUMNAS_PREDICTORAS = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS

RANDOM_STATE = 42
N_VECINOS_MI = 3


def cargar_datos() -> pd.DataFrame:
    """Carga y valida las columnas necesarias del conjunto de entrenamiento."""
    df = pd.read_csv(ARCHIVO_TRAIN)
    requeridas = set(COLUMNAS_PREDICTORAS + [VARIABLE_OBJETIVO])
    faltantes = requeridas.difference(df.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    if df[list(requeridas)].isna().any().any():
        raise ValueError("Hay valores faltantes en las columnas analizadas.")

    return df


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    """Ajusta, guarda y cierra una figura."""
    fig.tight_layout()
    fig.savefig(CARPETA_GRAFICOS / nombre, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analizar_pearson(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Pearson, su p-valor y R cuadrado para variables numéricas."""
    filas = []
    for columna in COLUMNAS_NUMERICAS:
        resultado = pearsonr(df[columna], df[VARIABLE_OBJETIVO])
        intervalo = resultado.confidence_interval(confidence_level=0.95)
        filas.append(
            {
                "feature": columna,
                "correlacion_pearson": resultado.statistic,
                "r_cuadrado": resultado.statistic**2,
                "p_value": resultado.pvalue,
                "ic_95_inferior": intervalo.low,
                "ic_95_superior": intervalo.high,
            }
        )

    tabla = pd.DataFrame(filas).sort_values(
        "correlacion_pearson", key=lambda serie: serie.abs(), ascending=False
    )
    tabla.to_csv(CARPETA_SALIDA / "resultados_pearson.csv", index=False)

    grafico = tabla.sort_values("correlacion_pearson")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colores = ["#d95f59" if valor < 0 else "#4c78a8" for valor in grafico["correlacion_pearson"]]
    barras = ax.barh(grafico["feature"], grafico["correlacion_pearson"], color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Correlación de Pearson con charges")
    ax.set_title("Relación lineal de las variables numéricas")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.bar_label(barras, labels=[f"{v:.3f}" for v in grafico["correlacion_pearson"]], padding=4)
    guardar_figura(fig, "01_correlacion_pearson.png")
    return tabla


def analizar_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica ANOVA F de una vía entre cada categoría y charges."""
    filas = []
    for columna in COLUMNAS_CATEGORICAS:
        grupos = [
            grupo[VARIABLE_OBJETIVO].to_numpy()
            for _, grupo in df.groupby(columna, observed=True)
        ]
        resultado = f_oneway(*grupos)

        media_global = df[VARIABLE_OBJETIVO].mean()
        suma_cuadrados_total = ((df[VARIABLE_OBJETIVO] - media_global) ** 2).sum()
        suma_cuadrados_entre = sum(
            len(grupo)
            * (grupo[VARIABLE_OBJETIVO].mean() - media_global) ** 2
            for _, grupo in df.groupby(columna, observed=True)
        )
        filas.append(
            {
                "feature": columna,
                "F": resultado.statistic,
                "p_value": resultado.pvalue,
                "eta_cuadrado": suma_cuadrados_entre / suma_cuadrados_total,
            }
        )

    tabla = pd.DataFrame(filas).sort_values("F", ascending=False)
    tabla.to_csv(CARPETA_SALIDA / "resultados_anova.csv", index=False)

    grafico = tabla.sort_values("F")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(grafico["feature"], grafico["F"], color="#efa900")
    ax.set_xscale("log")
    ax.set_xlabel("F-value (escala logarítmica)")
    ax.set_ylabel("Feature")
    ax.set_title("ANOVA F-test")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    guardar_figura(fig, "02_anova_f_test.png")

    resumen_grupos = (
        pd.concat(
            [
                df.groupby(columna, observed=True)[VARIABLE_OBJETIVO]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
                .rename(columns={columna: "categoria"})
                .assign(feature=columna)
                for columna in COLUMNAS_CATEGORICAS
            ],
            ignore_index=True,
        )
        [["feature", "categoria", "count", "mean", "median", "std"]]
    )
    resumen_grupos.to_csv(CARPETA_SALIDA / "resumen_categorias.csv", index=False)
    return tabla


def analizar_informacion_mutua(df: pd.DataFrame) -> pd.DataFrame:
    """Estima información mutua con variables categóricas marcadas discretas."""
    X = df[COLUMNAS_PREDICTORAS].copy()
    X[COLUMNAS_CATEGORICAS] = OrdinalEncoder().fit_transform(
        X[COLUMNAS_CATEGORICAS]
    )
    discretas = [
        columna in COLUMNAS_CATEGORICAS + ["children"] for columna in X.columns
    ]
    valores = mutual_info_regression(
        X,
        df[VARIABLE_OBJETIVO],
        discrete_features=discretas,
        n_neighbors=N_VECINOS_MI,
        random_state=RANDOM_STATE,
    )
    tabla = pd.DataFrame(
        {"feature": X.columns, "informacion_mutua": valores}
    ).sort_values("informacion_mutua", ascending=False)
    tabla.to_csv(
        CARPETA_SALIDA / "resultados_informacion_mutua.csv", index=False
    )

    grafico = tabla.sort_values("informacion_mutua")
    fig, ax = plt.subplots(figsize=(8, 5))
    colores = plt.get_cmap("viridis_r")(
        np.linspace(0.08, 0.92, len(grafico))
    )
    ax.barh(
        grafico["feature"],
        grafico["informacion_mutua"],
        color=colores,
    )
    ax.set_xlabel("MI Score")
    ax.set_ylabel("Feature")
    ax.set_title("Mutual Information")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    guardar_figura(fig, "03_informacion_mutua.png")
    return tabla


def graficar_matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    """Genera la matriz de correlación para las columnas numéricas."""
    columnas = COLUMNAS_NUMERICAS + [VARIABLE_OBJETIVO]
    matriz = df[columnas].corr(method="pearson")
    matriz.to_csv(CARPETA_SALIDA / "matriz_correlacion.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    imagen = ax.imshow(matriz, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columnas)), labels=columnas, rotation=35, ha="right")
    ax.set_yticks(range(len(columnas)), labels=columnas)
    ax.set_title("Matriz de correlación de Pearson")
    for fila in range(len(columnas)):
        for columna in range(len(columnas)):
            valor = matriz.iloc[fila, columna]
            ax.text(
                columna,
                fila,
                f"{valor:.2f}",
                ha="center",
                va="center",
                color="white" if abs(valor) > 0.55 else "black",
            )
    fig.colorbar(imagen, ax=ax, label="Correlación")
    guardar_figura(fig, "04_matriz_correlacion.png")
    return matriz


def crear_resumen(
    pearson: pd.DataFrame,
    anova: pd.DataFrame,
    informacion_mutua: pd.DataFrame,
) -> pd.DataFrame:
    """Combina los resultados de filtros sin tomar decisiones automáticas."""
    resumen = pd.DataFrame({"feature": COLUMNAS_PREDICTORAS})
    resumen = resumen.merge(
        pearson[["feature", "correlacion_pearson", "p_value"]].rename(
            columns={"p_value": "p_value_pearson"}
        ),
        on="feature",
        how="left",
    )
    resumen = resumen.merge(
        anova[["feature", "F", "p_value", "eta_cuadrado"]].rename(
            columns={"p_value": "p_value_anova"}
        ),
        on="feature",
        how="left",
    )
    resumen = resumen.merge(informacion_mutua, on="feature", how="left")
    resumen["decision_inicial"] = "conservar"
    resumen.to_csv(CARPETA_SALIDA / "resumen_filtros.csv", index=False)
    return resumen


def main() -> None:
    """Ejecuta todos los análisis usando exclusivamente el conjunto train."""
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()

    print(f"Dataset de entrenamiento: {len(df)} observaciones")
    print("Chi² no se aplica: charges es una variable continua.\n")

    pearson = analizar_pearson(df)
    anova = analizar_anova(df)
    informacion_mutua = analizar_informacion_mutua(df)
    graficar_matriz_correlacion(df)
    resumen = crear_resumen(pearson, anova, informacion_mutua)

    print("Resultados de filtros")
    print(resumen.round(4).to_string(index=False))
    print(f"\nTablas y gráficos guardados en: {CARPETA_SALIDA.resolve()}")


if __name__ == "__main__":
    main()
