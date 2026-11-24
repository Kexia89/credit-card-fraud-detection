

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

def preparar_europeo(df):

    df = df.copy()
    if "monto" not in df and "Amount" in df:
        df["monto"] = df["Amount"]
    if "hora" not in df and "Time" in df:
        df["hora"] = (df["Time"] // 3600) % 24
    if "log_monto" not in df and "monto" in df:
        df["log_monto"] = np.log1p(df["monto"])
    if "hora_sin" not in df and "hora" in df:
        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    return df

def predecir(bundle, df):

    if bundle.get("version_formato") != 1:
        raise ValueError("Versión de artefacto no compatible")
    if bundle["dataset"] == "europeo":
        df = preparar_europeo(df)

    faltantes = [c for c in bundle["variables"] if c not in df.columns]
    if faltantes:
        detalle = ", ".join(faltantes[:10])
        mensaje = f"Faltan {len(faltantes)} variables requeridas: {detalle}"
        if bundle["dataset"] == "sparkov":
            mensaje += (". Sparkov necesita variables históricas calculadas en orden temporal; "
                        "consulta la sección de inferencia del README.")
        raise ValueError(mensaje)

    x = df[bundle["variables"]]
    probabilidades = []
    for nombre, modelo in bundle["estimadores"].items():
        entrada = x
        if nombre == "Regresión logística" and bundle.get("escalador") is not None:
            entrada = bundle["escalador"].transform(x)
        probabilidades.append(modelo.predict_proba(entrada)[:, 1])
    prob = np.mean(probabilidades, axis=0)
    pred = (prob >= bundle["umbral"]).astype(int)
    return prob, pred

def main():
    parser = argparse.ArgumentParser(description="Predice fraude con un modelo ya entrenado")
    parser.add_argument("--model", required=True, help="archivo .joblib generado durante entrenamiento")
    parser.add_argument("--input", required=True, help="CSV de transacciones o variables preparadas")
    parser.add_argument("--output", default="predicciones.csv", help="CSV de salida")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    datos = pd.read_csv(args.input)
    prob, pred = predecir(bundle, datos)
    salida = pd.DataFrame({"probabilidad_fraude": prob, "prediccion_fraude": pred})
    ruta = Path(args.output)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta, index=False)
    print(f"Predicciones guardadas: {ruta.resolve()} ({pred.sum():,} alertas de {len(pred):,})")

if __name__ == "__main__":
    main()
