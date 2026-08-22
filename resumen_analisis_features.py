from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder


ARCHIVO_TRAIN = "insurance_train.csv"
ARCHIVO_SALIDA = "resumen_analisis_features.png"
VARIABLE_OBJETIVO = "charges"
NIVEL_SIGNIFICACION = 0.05
SEMILLA = 42


def calcular_informacion_mutua(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """Calcula MI respetando la naturaleza discreta de cada predictor."""
    X_codificado = X.copy()
    categoricas = X.select_dtypes(exclude="number").columns.tolist()
    X_codificado[categoricas] = OrdinalEncoder().fit_transform(X[categoricas])

    discretas = set(categoricas + ["children"])
    mascara_discretas = [columna in discretas for columna in X.columns]

    valores = mutual_info_regression(
        X_codificado,
        y,
        discrete_features=mascara_discretas,
        random_state=SEMILLA,
    )
    return pd.Series(valores, index=X.columns).sort_values(ascending=False)


def calcular_anova(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Aplica F de regresión a numéricas y ANOVA de una vía a categóricas."""
    resultados = []
    numericas = X.select_dtypes(include="number").columns
    categoricas = X.select_dtypes(exclude="number").columns

    for columna in numericas:
        valor_f, p_value = f_regression(X[[columna]], y)
        resultados.append((columna, valor_f[0], p_value[0]))

    for columna in categoricas:
        grupos = [
            grupo[VARIABLE_OBJETIVO].to_numpy()
            for _, grupo in df.groupby(columna, observed=True)
        ]
        valor_f, p_value = f_oneway(*grupos)
        resultados.append((columna, valor_f, p_value))

    tabla = pd.DataFrame(resultados, columns=["feature", "F", "p_value"])
    tabla["p_ajustado"] = (tabla["p_value"] * len(tabla)).clip(upper=1)
    tabla["significativa"] = tabla["p_ajustado"] < NIVEL_SIGNIFICACION
    return tabla.sort_values("F", ascending=False).reset_index(drop=True)


def calcular_chi_cuadrado(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula χ² y V de Cramér entre los predictores categóricos."""
    categoricas = X.select_dtypes(exclude="number").columns.tolist()
    matriz = pd.DataFrame(1.0, index=categoricas, columns=categoricas)
    resultados = []

    for feature_1, feature_2 in combinations(categoricas, 2):
        contingencia = pd.crosstab(X[feature_1], X[feature_2])
        chi2, p_value, _, _ = chi2_contingency(contingencia)
        filas, columnas = contingencia.shape
        n = contingencia.to_numpy().sum()
        denominador = n * min(filas - 1, columnas - 1)
        valor_v = (chi2 / denominador) ** 0.5 if denominador else 0.0
        matriz.loc[feature_1, feature_2] = valor_v
        matriz.loc[feature_2, feature_1] = valor_v
        resultados.append(
            {
                "par": f"{feature_1} vs. {feature_2}",
                "chi2": chi2,
                "p_value": p_value,
            }
        )

    tabla = pd.DataFrame(resultados)
    tabla["p_ajustado"] = (tabla["p_value"] * len(tabla)).clip(upper=1)
    tabla["significativa"] = tabla["p_ajustado"] < NIVEL_SIGNIFICACION
    return matriz, tabla


def graficar_matriz(
    ax: plt.Axes,
    matriz: pd.DataFrame,
    titulo: str,
    etiqueta_color: str,
    mapa_color: str,
    minimo: float,
    maximo: float,
) -> None:
    """Grafica una matriz de asociaciones con sus valores anotados."""
    imagen = ax.imshow(matriz, cmap=mapa_color, vmin=minimo, vmax=maximo)
    nombres = matriz.columns
    ax.set_xticks(range(len(nombres)), labels=nombres, rotation=35, ha="right")
    ax.set_yticks(range(len(nombres)), labels=nombres)
    ax.set_title(titulo)

    for fila in range(len(nombres)):
        for columna in range(len(nombres)):
            valor = matriz.iloc[fila, columna]
            ax.text(
                columna,
                fila,
                f"{valor:.2f}",
                ha="center",
                va="center",
                color="white" if abs(valor) >= 0.5 else "black",
            )

    plt.colorbar(imagen, ax=ax, label=etiqueta_color, fraction=0.046, pad=0.04)


def main() -> None:
    """Genera un resumen gráfico de los análisis univariados de features."""
    df_train = pd.read_csv(ARCHIVO_TRAIN)
    X = df_train.drop(columns=VARIABLE_OBJETIVO)
    y = df_train[VARIABLE_OBJETIVO]
    X_numerico = X.select_dtypes(include="number")

    pearson_objetivo = X_numerico.corrwith(y).sort_values()
    matriz_pearson = X_numerico.corr(method="pearson")
    informacion_mutua = calcular_informacion_mutua(X, y)
    tabla_anova = calcular_anova(df_train, X, y)
    matriz_cramer, tabla_chi2 = calcular_chi_cuadrado(X)

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))

    colores_pearson = [
        "#c95b52" if valor < 0 else "#5875c4"
        for valor in pearson_objetivo
    ]
    axes[0, 0].barh(
        pearson_objetivo.index,
        pearson_objetivo.values,
        color=colores_pearson,
    )
    axes[0, 0].axvline(0, color="black", linewidth=0.8)
    axes[0, 0].set_xlim(-1, 1)
    axes[0, 0].set_xlabel("Correlación de Pearson")
    axes[0, 0].set_title("Pearson con charges")
    axes[0, 0].grid(axis="x", linestyle="--", alpha=0.3)

    graficar_matriz(
        axes[0, 1],
        matriz_pearson,
        "Pearson entre predictores numéricos",
        "Correlación",
        "coolwarm",
        -1,
        1,
    )

    mi_grafico = informacion_mutua.iloc[::-1]
    axes[1, 0].barh(mi_grafico.index, mi_grafico.values, color="#5875c4")
    axes[1, 0].set_xlabel("Información mutua estimada")
    axes[1, 0].set_title("Información mutua con charges")
    axes[1, 0].grid(axis="x", linestyle="--", alpha=0.3)

    anova_grafico = tabla_anova.sort_values("F", ascending=True)
    colores_anova = [
        "#3a9d5d" if significativa else "#efa900"
        for significativa in anova_grafico["significativa"]
    ]
    axes[1, 1].barh(
        anova_grafico["feature"],
        anova_grafico["F"],
        color=colores_anova,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("F-value (escala logarítmica)")
    axes[1, 1].set_title("ANOVA F-test con charges")
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.3)

    graficar_matriz(
        axes[2, 0],
        matriz_cramer,
        "χ² entre categóricas — V de Cramér",
        "V de Cramér",
        "Blues",
        0,
        1,
    )

    chi2_grafico = tabla_chi2.sort_values("chi2", ascending=True)
    colores_chi2 = [
        "#3a9d5d" if significativa else "#efa900"
        for significativa in chi2_grafico["significativa"]
    ]
    axes[2, 1].barh(
        chi2_grafico["par"],
        chi2_grafico["chi2"],
        color=colores_chi2,
    )
    axes[2, 1].set_xlabel("Estadístico χ²")
    axes[2, 1].set_title("Test χ² entre predictores categóricos")
    axes[2, 1].grid(axis="x", linestyle="--", alpha=0.3)

    for indice, fila in enumerate(chi2_grafico.itertuples(index=False)):
        axes[2, 1].text(
            fila.chi2,
            indice,
            f"  χ²={fila.chi2:.2f} | p-aj.={fila.p_ajustado:.3f}",
            va="center",
            ha="left",
        )

    fig.suptitle("Resumen de relaciones y selección de features", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ARCHIVO_SALIDA, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado en {ARCHIVO_SALIDA}")
    plt.show()


if __name__ == "__main__":
    main()
