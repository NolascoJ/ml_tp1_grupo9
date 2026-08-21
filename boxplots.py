import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("insurance.csv")
df.plot.box(subplots=True, layout=(2, 2), figsize=(10, 7))
plt.tight_layout()
plt.show()
