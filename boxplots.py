import matplotlib.pyplot as plt
import pandas as pd


# El análisis exploratorio se realiza solamente sobre el conjunto de entrenamiento.
df = pd.read_csv("insurance_train.csv")
ARCHIVO_SALIDA = "boxplots_variables_numericas.png"

axes = df.plot.box(subplots=True, layout=(2, 2), figsize=(10, 7))
fig = axes.iloc[0].get_figure()
plt.tight_layout()
fig.savefig(ARCHIVO_SALIDA, dpi=180, bbox_inches="tight")
print(f"Gráfico guardado en {ARCHIVO_SALIDA}")
plt.show()
