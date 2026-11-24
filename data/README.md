# Datos

Los archivos CSV no se suben al repositorio: son grandes y su distribución debe respetar
las condiciones publicadas por cada fuente. Descárgalos de Kaggle y colócalos en esta carpeta:

| Archivo | Base | Fuente |
|---|---|---|
| `creditcard.csv` | Europea (PCA), sept. 2013 | [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `fraudTrain.csv` | Sparkov (simulada), 2019-2020 | [Kaggle – Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
| `fraudTest.csv`  | Sparkov (simulada), 2019-2020 | [Kaggle – Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |

```
data/
├── creditcard.csv
├── fraudTrain.csv
└── fraudTest.csv
```

El script también busca en subcarpetas, así que `data/Data2/creditcard.csv` funciona igual.

Antes de descargar, inicia sesión en Kaggle y acepta, cuando corresponda, las condiciones
de uso mostradas en la página del dataset. Los patrones de `.gitignore` impiden agregar los
CSV accidentalmente al repositorio.
