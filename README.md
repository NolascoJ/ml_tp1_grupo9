# TP1 - Regresión y evaluación de modelos

Implementación de modelos de regresión para predecir `charges` en el dataset
Insurance Charges.

## Estructura

```text
.
├── analisis/             # Limpieza, EDA y análisis de variables
├── data/
│   ├── raw/              # Dataset descargado sin modificar
│   ├── processed/        # Conjunto de entrenamiento
│   └── test/             # Conjunto reservado para evaluación final
├── datos/                # Descarga y separación train/test
├── modelos/              # Modelos y evaluación final
├── resultados/
│   ├── graficos/         # Gráficos generados por los análisis
│   ├── seleccion_variables/
├── requirements.txt
└── TP1. Regresión e Introducción a la evaluación de modelos.pdf
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Flujo de trabajo

Desde la raíz del proyecto, ejecutar en este orden:

```bash
python3 datos/exportar_dataset.py
python3 datos/separar_datos.py
python3 analisis/check_null.py
python3 analisis/analizar_outliers.py
python3 analisis/histogramas.py
python3 analisis/boxplots.py
python3 analisis/todos_vs_charges.py
python3 analisis/analisis_seleccion_variables.py
python3 modelos/modelo_lineal.py
python3 modelos/modelo_polinomico.py
```

El conjunto `data/test/insurance_test.csv` debe mantenerse reservado durante
el análisis y la validación cruzada. Una vez elegido el modelo, se evalúa una
sola vez mediante:

```bash
python3 modelos/evaluar_modelo_final.py
```
