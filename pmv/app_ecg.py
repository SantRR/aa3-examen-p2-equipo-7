"""PMV de CardioRitmo: tamizaje de latidos con el modelo de ecg/modelo/.

Uso (desde la raíz del repo):  python pmv/app_ecg.py  →  http://127.0.0.1:7860
Entrada: CSV sin encabezado, un latido por fila, 140 valores por latido.
"""

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import gradio as gr
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
MODELO = keras.models.load_model(AQUI.parent / "ecg" / "modelo" / "modelo_ecg.keras")
EJEMPLO = AQUI / "ecg_ejemplo.csv"
UMBRAL = 0.5
PUNTOS = 140


def leer_latidos(ruta):
    """Lee y valida el CSV; devuelve una matriz (latidos, 140) normalizada por latido."""
    datos = pd.read_csv(ruta, header=None)
    if datos.shape[1] != PUNTOS:
        raise gr.Error(
            f"Cada fila debe tener {PUNTOS} valores; el archivo tiene {datos.shape[1]}."
        )
    X = datos.apply(pd.to_numeric, errors="coerce").to_numpy("float32")
    if np.isnan(X).any():
        raise gr.Error("El archivo contiene valores vacíos o no numéricos.")
    desv = X.std(axis=1, keepdims=True)
    if (desv == 0).any():
        raise gr.Error("Hay latidos planos (sin variación); revise el registro.")
    return (X - X.mean(axis=1, keepdims=True)) / desv


def analizar(archivo):
    X = leer_latidos(archivo if archivo else EJEMPLO)
    prob = MODELO.predict(X[..., None], verbose=0).ravel()
    marcados = prob >= UMBRAL

    resumen = (
        f"### {marcados.sum()} de {len(X)} latidos marcados para revisión "
        f"({marcados.mean():.0%})\n"
        "El cardiólogo solo necesita revisar los latidos marcados. "
        "Esta herramienta es un apoyo al tamizaje y no sustituye el diagnóstico médico."
    )

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for x in X[~marcados]:
        ax.plot(x, color="#4C72B0", alpha=0.15, lw=0.8)
    for x in X[marcados]:
        ax.plot(x, color="#DD5555", alpha=0.6, lw=1)
    ax.plot([], color="#4C72B0", label="normal")
    ax.plot([], color="#DD5555", label="marcado para revisión")
    ax.set_xlabel("punto del latido")
    ax.set_ylabel("amplitud normalizada")
    ax.legend(loc="lower left")
    fig.tight_layout()

    tabla = pd.DataFrame(
        {
            "latido": np.arange(1, len(X) + 1),
            "probabilidad de anormal": prob.round(3),
            "resultado": np.where(marcados, "REVISAR", "normal"),
        }
    ).sort_values("probabilidad de anormal", ascending=False)
    return resumen, fig, tabla


with gr.Blocks(title="CardioRitmo · Tamizaje de latidos") as demo:
    gr.Markdown(
        "# CardioRitmo · Tamizaje de latidos\n"
        "Suba el registro del Holter (CSV, un latido por fila, 140 puntos). "
        "El modelo marca los latidos anormales para que el cardiólogo revise solo esos."
    )
    with gr.Row():
        archivo = gr.File(label="Registro (CSV)", file_types=[".csv"], scale=1)
        boton = gr.Button("Analizar", variant="primary", scale=0)
    resumen = gr.Markdown()
    grafica = gr.Plot(label="Latidos del registro")
    tabla = gr.Dataframe(label="Detalle por latido (ordenado por probabilidad)")

    salidas = [resumen, grafica, tabla]
    boton.click(analizar, inputs=archivo, outputs=salidas)
    demo.load(analizar, inputs=archivo, outputs=salidas)  # muestra el ejemplo al abrir

if __name__ == "__main__":
    # autoverificación: el registro de ejemplo se lee y produce una fila por latido
    assert len(analizar(None)[2]) == len(pd.read_csv(EJEMPLO, header=None))
    demo.launch()
