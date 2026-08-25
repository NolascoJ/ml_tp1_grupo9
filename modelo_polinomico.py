from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


ARCHIVO_TRAIN = "insurance_train.csv"
GRADOS = (1, 2, 3)
GRADO_LASSO = 2
LAMBDAS_LASSO = (0.001, 0.01, 0.1, 1)
N_SPLITS = 5
RANDOM_STATE = 42

COLUMNAS_CATEGORICAS = ["sex", "smoker", "region"]
COLUMNAS_NUMERICAS = ["age", "bmi", "children"]


def crear_modelo_polinomico(
    grado: int,
    lambda_lasso: Optional[float] = None,
) -> Pipeline:
    """Crea un pipeline polinómico con regresión lineal o Lasso."""
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                COLUMNAS_CATEGORICAS,
            ),
            ("numericas", StandardScaler(), COLUMNAS_NUMERICAS),
        ]
    )

    if lambda_lasso is None:
        regresor = LinearRegression()
    else:
        regresor = Lasso(alpha=lambda_lasso, max_iter=100_000)

    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            (
                "polinomio",
                PolynomialFeatures(degree=grado, include_bias=False),
            ),
            # L1 penaliza coeficientes: todas las features polinómicas deben
            # quedar en una escala comparable dentro de cada fold.
            ("escalado_polinomio", StandardScaler()),
            ("modelo", regresor),
        ]
    )


def evaluar_modelo(
    modelo: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
) -> dict:
    """Evalúa un pipeline y devuelve sus RMSE y modelos ajustados."""
    return cross_validate(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        return_estimator=True,
    )


def main() -> None:
    """Evalúa grados polinómicos mediante 5-fold cross-validation."""
    df_train = pd.read_csv(ARCHIVO_TRAIN)
    X_train = df_train.drop(columns="charges")
    y_train = df_train["charges"]

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("Regresión polinómica con 5-fold cross-validation")
    print("El conjunto test permanece reservado.\n")

    resumen_modelos = []

    for grado in GRADOS:
        modelo = crear_modelo_polinomico(grado)
        resultados = evaluar_modelo(modelo, X_train, y_train, cv)

        rmse_train = -resultados["train_score"]
        rmse_validacion = -resultados["test_score"]
        cantidad_features = resultados["estimator"][0].named_steps[
            "polinomio"
        ].n_output_features_

        print(f"Grado {grado} ({cantidad_features} features transformadas)")
        for fold, (train, validacion) in enumerate(
            zip(rmse_train, rmse_validacion), start=1
        ):
            print(
                f"  Fold {fold}: train = {train:,.2f} | "
                f"validación = {validacion:,.2f}"
            )

        print(f"  RMSE train promedio: {rmse_train.mean():,.2f}")
        print(f"  RMSE validación promedio: {rmse_validacion.mean():,.2f}")
        print(f"  Desvío RMSE validación: {rmse_validacion.std():,.2f}\n")

        resumen_modelos.append(
            {
                "modelo": "Sin regularización",
                "grado": grado,
                "lambda": None,
                "features": cantidad_features,
                "features_activas": None,
                "rmse_train": rmse_train.mean(),
                "rmse_validacion": rmse_validacion.mean(),
                "desvio_validacion": rmse_validacion.std(),
            }
        )

    for lambda_lasso in LAMBDAS_LASSO:
        modelo = crear_modelo_polinomico(GRADO_LASSO, lambda_lasso)
        resultados = evaluar_modelo(modelo, X_train, y_train, cv)

        rmse_train = -resultados["train_score"]
        rmse_validacion = -resultados["test_score"]
        cantidad_features = resultados["estimator"][0].named_steps[
            "polinomio"
        ].n_output_features_
        features_activas = [
            (abs(estimador.named_steps["modelo"].coef_) > 1e-8).sum()
            for estimador in resultados["estimator"]
        ]
        promedio_features_activas = sum(features_activas) / len(features_activas)

        print(
            f"Grado {GRADO_LASSO} con Lasso "
            f"(lambda = {lambda_lasso:g})"
        )
        for fold, (train, validacion) in enumerate(
            zip(rmse_train, rmse_validacion), start=1
        ):
            print(
                f"  Fold {fold}: train = {train:,.2f} | "
                f"validación = {validacion:,.2f}"
            )

        print(f"  RMSE train promedio: {rmse_train.mean():,.2f}")
        print(f"  RMSE validación promedio: {rmse_validacion.mean():,.2f}")
        print(f"  Desvío RMSE validación: {rmse_validacion.std():,.2f}")
        print(
            "  Features activas promedio: "
            f"{promedio_features_activas:.1f} de {cantidad_features}\n"
        )

        resumen_modelos.append(
            {
                "modelo": "Lasso",
                "grado": GRADO_LASSO,
                "lambda": lambda_lasso,
                "features": cantidad_features,
                "features_activas": promedio_features_activas,
                "rmse_train": rmse_train.mean(),
                "rmse_validacion": rmse_validacion.mean(),
                "desvio_validacion": rmse_validacion.std(),
            }
        )

    print("Comparación final")
    print(
        "Modelo             | Grado | Lambda | Activas | "
        "RMSE train | RMSE validación | Desvío"
    )
    for resultado in resumen_modelos:
        lambda_texto = (
            "-" if resultado["lambda"] is None else f"{resultado['lambda']:g}"
        )
        activas_texto = (
            "-"
            if resultado["features_activas"] is None
            else f"{resultado['features_activas']:.1f}/{resultado['features']}"
        )
        print(
            f"{resultado['modelo']:<18} | "
            f"{resultado['grado']:>5} | "
            f"{lambda_texto:>6} | "
            f"{activas_texto:>7} | "
            f"{resultado['rmse_train']:>10,.2f} | "
            f"{resultado['rmse_validacion']:>15,.2f} | "
            f"{resultado['desvio_validacion']:>7,.2f}"
        )


if __name__ == "__main__":
    main()
