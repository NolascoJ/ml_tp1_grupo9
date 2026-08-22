import matplotlib.pyplot as plt
import pandas as pd


# El análisis exploratorio se realiza solamente sobre el conjunto de entrenamiento.
df = pd.read_csv("insurance_train.csv")
df.plot.box(subplots=True, layout=(2, 2), figsize=(10, 7))
plt.tight_layout()
plt.show()
