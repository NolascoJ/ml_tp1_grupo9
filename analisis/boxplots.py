from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
# El análisis exploratorio se realiza solamente sobre el conjunto de entrenamiento.
df = pd.read_csv(RAIZ_PROYECTO / "data" / "processed" / "insurance_train.csv")
ARCHIVO_SALIDA = RAIZ_PROYECTO / "resultados" / "graficos" / "boxplots_variables_numericas.png"

axes = df.plot.box(subplots=True, layout=(2, 2), figsize=(10, 7))
fig = axes.iloc[0].get_figure()
plt.tight_layout()
ARCHIVO_SALIDA.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(ARCHIVO_SALIDA, dpi=180, bbox_inches="tight")
print(f"Gráfico guardado en {ARCHIVO_SALIDA}")
plt.show()
