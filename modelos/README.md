# Modelos entrenados

Esta carpeta recibe los artefactos creados por `fraud_detection.py`:

- `<base>_mejor_modelo.joblib`: paquete completo para inferencia, con estimador,
  columnas, umbral y escalador cuando aplica.
- `<base>_metadata.json`: descripción legible del paquete y versiones utilizadas.
- `<base>_xgboost.json` o `<base>_lightgbm.txt`: formato nativo del booster ganador.

Los binarios entrenados no se versionan en GitHub porque pueden ser grandes y están
ligados a versiones concretas de las bibliotecas. Se regeneran con:

```bash
python fraud_detection.py --dataset ambas --data-dir data --model-dir modelos
```

Importante: `joblib` utiliza serialización tipo pickle. Nunca cargues un modelo de una
fuente no confiable.
