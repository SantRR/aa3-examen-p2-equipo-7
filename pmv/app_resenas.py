"""PMV de Compra Fácil Bajío: bandeja de atención priorizada por reseñas negativas.

Uso (desde la raíz del repo):  python pmv/app_resenas.py  →  http://127.0.0.1:7861
Entrada: una reseña escrita a mano, o un CSV con una columna "texto" (una reseña por fila).
"""

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import gradio as gr
import joblib
import keras
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
CARPETA_MODELO = AQUI.parent / "resenas" / "modelo"
MODELO = keras.models.load_model(CARPETA_MODELO / "modelo_resenas.keras")
VECTORIZADOR = joblib.load(CARPETA_MODELO / "vectorizador_tfidf.joblib")
EJEMPLO = AQUI / "resenas_ejemplo.csv"
UMBRAL = 0.5


def probabilidad_negativa(textos):
    """Probabilidad de que cada reseña sea negativa (mismo preprocesamiento que el DEV)."""
    X = VECTORIZADOR.transform(textos).astype("float32").toarray()
    return MODELO.predict(X, verbose=0).ravel()


def clasificar_una(texto):
    texto = (texto or "").strip()
    if not texto:
        raise gr.Error("Escriba una reseña para clasificarla.")
    p = float(probabilidad_negativa([texto])[0])
    if p >= UMBRAL:
        veredicto = f"## 🔴 NEGATIVA · atender primero\nProbabilidad de que sea negativa: **{p:.0%}**"
    else:
        veredicto = f"## 🔵 Positiva\nProbabilidad de que sea negativa: **{p:.0%}**"
    return veredicto, {"negativa": p, "positiva": 1 - p}


def leer_resenas(ruta):
    """Lee y valida el CSV; devuelve la lista de reseñas."""
    try:
        datos = pd.read_csv(ruta)
    except Exception as e:
        raise gr.Error(f"No se pudo leer el archivo como CSV: {e}")
    if "texto" not in datos.columns:
        raise gr.Error('El CSV debe tener una columna llamada "texto".')
    textos = datos["texto"].dropna().astype(str).str.strip()
    textos = textos[textos != ""]
    if textos.empty:
        raise gr.Error("El archivo no contiene reseñas.")
    return textos.tolist()


def bandeja(archivo):
    textos = leer_resenas(archivo if archivo else EJEMPLO)
    prob = probabilidad_negativa(textos)
    negativas = prob >= UMBRAL
    resumen = (
        f"### {negativas.sum()} de {len(textos)} reseñas para atender primero "
        f"({negativas.mean():.0%})\n"
        "La bandeja está ordenada de la más a la menos probable de ser negativa. "
        "El modelo solo ordena el trabajo: todas las reseñas deben revisarse, porque "
        "alrededor de 8 de cada 100 negativas no las detecta."
    )
    orden = np.argsort(-prob)
    tabla = pd.DataFrame(
        {
            "prioridad": np.where(negativas, "🔴 ATENDER PRIMERO", "🔵 positiva")[orden],
            "prob. negativa": [f"{p:.1%}" for p in prob[orden]],
            "reseña": np.array(textos, dtype=object)[orden],
        }
    )
    return resumen, tabla


with gr.Blocks(title="Compra Fácil Bajío · Atención a clientes") as demo:
    gr.Markdown(
        "# Compra Fácil Bajío · Atención a clientes\n"
        "El modelo detecta las reseñas negativas para que atención a clientes responda "
        "primero a quien está molesto. Umbral de decisión: 0.5."
    )
    with gr.Tab("Bandeja priorizada"):
        with gr.Row():
            archivo = gr.File(label='Reseñas del día (CSV con columna "texto")', file_types=[".csv"], scale=1)
            boton = gr.Button("Priorizar", variant="primary", scale=0)
        resumen = gr.Markdown()
        tabla = gr.Dataframe(label="Bandeja (ordenada por probabilidad de negativa)", wrap=True,
                             column_widths=["20%", "12%", "68%"])
        boton.click(bandeja, inputs=archivo, outputs=[resumen, tabla])
        demo.load(bandeja, inputs=archivo, outputs=[resumen, tabla])  # muestra el ejemplo al abrir
    with gr.Tab("Probar una reseña"):
        entrada = gr.Textbox(label="Reseña", lines=3, placeholder="Escriba o pegue una reseña…")
        boton_una = gr.Button("Clasificar", variant="primary")
        veredicto = gr.Markdown()
        barras = gr.Label(label="Probabilidades")
        boton_una.click(clasificar_una, inputs=entrada, outputs=[veredicto, barras])
        entrada.submit(clasificar_una, inputs=entrada, outputs=[veredicto, barras])
        gr.Examples(
            ["Llegó roto y nadie me responde, no lo recomiendo.",
             "Excelente calidad, llegó antes de lo previsto.",
             "Está bien pero la batería dura muy poco."],
            inputs=entrada,
        )

if __name__ == "__main__":
    # autoverificación: el ejemplo se lee y produce una fila por reseña
    assert len(bandeja(None)[1]) == len(pd.read_csv(EJEMPLO))
    demo.launch(server_port=7861)
