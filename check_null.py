import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("insurance_train.csv")

print(df.dtypes)

print('\n')

print(df.isnull().sum())
