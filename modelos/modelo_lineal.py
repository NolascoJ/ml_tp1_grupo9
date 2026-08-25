from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_validate

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "insurance_train.csv"

df_train = pd.read_csv(ARCHIVO_TRAIN)

X_train = df_train.drop(columns="charges")
y_train = df_train["charges"]

columnas_categoricas = ["sex", "smoker", "region"]
columnas_numericas = ["age", "bmi", "children"]

preprocesador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
            columnas_categoricas,
        ),
        ("numericas", StandardScaler(), columnas_numericas),
    ]
)

modelo_lineal = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("modelo", LinearRegression()),
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

resultados = cross_validate(
    modelo_lineal,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    return_train_score=True,
)

rmse_train = -resultados["train_score"]
rmse_validacion = -resultados["test_score"]

print("RMSE por fold")
for i, (train, validacion) in enumerate(
    zip(rmse_train, rmse_validacion), start=1
):
    print(
        f"Fold {i}: "
        f"train = {train:,.2f} | "
        f"validación = {validacion:,.2f}"
    )

print("\nResumen")
print(f"RMSE train promedio: {rmse_train.mean():,.2f}")
print(f"RMSE validación promedio: {rmse_validacion.mean():,.2f}")
print(f"Desvío RMSE validación: {rmse_validacion.std():,.2f}")
