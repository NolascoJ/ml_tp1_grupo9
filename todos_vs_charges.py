import matplotlib.pyplot as plt
import pandas as pd


ARCHIVO_TRAIN = "insurance_train.csv"
ARCHIVO_SALIDA = "todos_vs_charges.png"
VARIABLE_OBJETIVO = "charges"


def graficar_numerica(
    ax: plt.Axes,
    df: pd.DataFrame,
    columna: str,
) -> None:
    """Grafica una variable numérica contra charges."""
    ax.scatter(
        df[columna],
        df[VARIABLE_OBJETIVO],
        alpha=0.55,
        color="#5875c4",
        edgecolors="none",
    )
    ax.set_xlabel(columna)
    ax.set_ylabel(VARIABLE_OBJETIVO)
    ax.set_title(f"{columna} vs. {VARIABLE_OBJETIVO}")
    ax.grid(alpha=0.25)


def graficar_categorica(
    ax: plt.Axes,
    df: pd.DataFrame,
    columna: str,
) -> None:
    """Grafica la distribución de charges para cada categoría."""
    categorias = df[columna].dropna().unique()
    grupos = [
        df.loc[df[columna] == categoria, VARIABLE_OBJETIVO]
        for categoria in categorias
    ]

    boxplot = ax.boxplot(grupos, tick_labels=categorias, patch_artist=True)
    for caja in boxplot["boxes"]:
        caja.set_facecolor("#8ea6df")

    ax.set_xlabel(columna)
    ax.set_ylabel(VARIABLE_OBJETIVO)
    ax.set_title(f"{columna} vs. {VARIABLE_OBJETIVO}")
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    """Grafica todos los predictores contra charges."""
    df_train = pd.read_csv(ARCHIVO_TRAIN)
    predictores = df_train.drop(columns=VARIABLE_OBJETIVO)
    columnas_numericas = predictores.select_dtypes(include="number").columns
    columnas_categoricas = predictores.select_dtypes(exclude="number").columns

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    indice = 0
    for columna in columnas_numericas:
        graficar_numerica(axes[indice], df_train, columna)
        indice += 1

    for columna in columnas_categoricas:
        graficar_categorica(axes[indice], df_train, columna)
        indice += 1

    for ax in axes[indice:]:
        ax.axis("off")

    fig.suptitle("Variables predictoras vs. charges — entrenamiento", fontsize=15)
    fig.tight_layout()
    fig.savefig(ARCHIVO_SALIDA, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado en {ARCHIVO_SALIDA}")
    plt.show()


if __name__ == "__main__":
    main()
