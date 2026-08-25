import matplotlib.pyplot as plt
import pandas as pd


# El análisis exploratorio se realiza solamente sobre el conjunto de entrenamiento.
df = pd.read_csv("insurance_train.csv")
ARCHIVO_SALIDA = "distribuciones_variables.png"
numericas = df.select_dtypes(include="number").columns
categoricas = df.select_dtypes(exclude="number").columns

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for columna, ax in zip(numericas, axes):
    df[columna].plot.hist(bins=20, edgecolor="black", ax=ax)
    ax.set_title(columna)
    ax.set_ylabel("Frecuencia")

for columna, ax in zip(categoricas, axes[len(numericas):]):
    df[columna].value_counts().plot.bar(ax=ax)
    ax.set_title(columna)
    ax.set_xlabel("")
    ax.set_ylabel("Cantidad")
    rotacion = 20 if columna == "region" else 0
    ax.tick_params(axis="x", rotation=rotacion, labelsize=9)

for ax in axes[len(numericas) + len(categoricas):]:
    ax.axis("off")

plt.tight_layout()
fig.savefig(ARCHIVO_SALIDA, dpi=180, bbox_inches="tight")
print(f"Gráfico guardado en {ARCHIVO_SALIDA}")
plt.show()
