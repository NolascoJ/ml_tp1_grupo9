import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("insurance.csv")
numericas = df.select_dtypes(include="number").columns
categoricas = df.select_dtypes(exclude="number").columns

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for columna, ax in zip(numericas, axes):
    df[columna].plot.hist(bins=20, edgecolor="black", ax=ax)
    ax.set_title(columna)

for columna, ax in zip(categoricas, axes[len(numericas):]):
    df[columna].value_counts().plot.bar(ax=ax)
    ax.set_title(columna)
    ax.set_xlabel("")
    ax.set_ylabel("Cantidad")
    ax.tick_params(axis="x", rotation=0)

for ax in axes[len(numericas) + len(categoricas):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
