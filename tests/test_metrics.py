import numpy as np

from fraud_detection import bootstrap_ic, mcnemar, metricas, umbral_optimo

def test_metricas_desbalanceadas_y_conteos():
    y = np.array([0, 0, 0, 0, 1, 1])
    prob = np.array([0.05, 0.10, 0.20, 0.80, 0.70, 0.90])
    resultado = metricas(y, prob, 0.50)
    assert (resultado["TP"], resultado["FP"], resultado["FN"], resultado["TN"]) == (2, 1, 0, 3)
    assert resultado["Recall"] == 1.0
    assert resultado["FPR por 1000"] == 250.0
    assert resultado["Lift PR-AUC"] > 1.0

def test_umbral_proviene_de_precision_recall():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.05, 0.40, 0.60, 0.90])
    assert 0.0 < umbral_optimo(y, prob) <= 1.0

def test_bootstrap_estratificado_y_mcnemar_exacto():
    y = np.array([0] * 20 + [1] * 5)
    prob = np.linspace(0.01, 0.99, len(y))
    intervalos, distribucion = bootstrap_ic(y, prob, 0.80, n=20)
    assert len(distribucion) == 20
    assert set(intervalos) == {"PR-AUC", "ROC-AUC", "F1", "Precisión", "Recall", "MCC"}

    pred_a = (prob >= 0.80).astype(int)
    pred_b = (prob >= 0.70).astype(int)
    prueba = mcnemar(y, pred_a, pred_b)
    assert prueba["metodo"] == "exacto_binomial"
    assert prueba["discordantes"] == prueba["solo_a_correcto"] + prueba["solo_b_correcto"]
    assert 0.0 <= prueba["p_valor"] <= 1.0
