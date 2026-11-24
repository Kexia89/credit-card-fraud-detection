

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, matthews_corrcoef,
    balanced_accuracy_score,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

SEMILLA = 42
np.random.seed(SEMILLA)

COLOR_LEGIT = "#2E86AB"
COLOR_FRAUDE = "#E63946"
PALETA_MODELOS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.titleweight": "bold"})

def buscar_archivo(nombre, carpeta_base):

    carpeta_base = Path(carpeta_base)
    directo = carpeta_base / nombre
    if directo.exists():
        return directo
    encontrados = list(carpeta_base.rglob(nombre))
    if not encontrados:
        raise FileNotFoundError(
            f"No encontré '{nombre}' dentro de {carpeta_base.resolve()}. "
            "Revisa data/README.md para descargar las bases."
        )
    return encontrados[0]

def guardar(fig, carpeta, nombre):
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / nombre)
    plt.close(fig)

def titulo(texto):
    print("\n" + "=" * 80)
    print(f"  {texto}")
    print("=" * 80)

def preparar_europeo(carpeta_datos, rapido=False):
    ruta = buscar_archivo("creditcard.csv", carpeta_datos)
    df = pd.read_csv(ruta)
    print(f"Leído {ruta.name}: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicados eliminados: {antes - len(df):,}")

    if rapido:

        fraudes = df[df.Class == 1]
        legit = df[df.Class == 0].sample(40_000, random_state=SEMILLA)
        df = pd.concat([fraudes, legit]).sample(frac=1, random_state=SEMILLA)

    df = df.rename(columns={"Class": "objetivo", "Amount": "monto"})

    df["hora"] = (df["Time"] // 3600) % 24
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["log_monto"] = np.log1p(df["monto"])

    columnas_v = [f"V{i}" for i in range(1, 29)]
    variables = columnas_v + ["log_monto", "hora_sin", "hora_cos"]

    X, y = df[variables], df["objetivo"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEMILLA)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEMILLA)

    return dict(nombre="europeo", df=df, variables=variables,
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test)

def distancia_haversine(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def preparar_sparkov(carpeta_datos, rapido=False):
    columnas = ["trans_date_trans_time", "cc_num", "category", "amt", "gender", "lat", "long",
                "city_pop", "dob", "unix_time", "merch_lat", "merch_long", "is_fraud"]
    train = pd.read_csv(buscar_archivo("fraudTrain.csv", carpeta_datos), usecols=columnas)
    test = pd.read_csv(buscar_archivo("fraudTest.csv", carpeta_datos), usecols=columnas)
    train["particion"], test["particion"] = "train", "test"
    print(f"Leídos fraudTrain ({len(train):,}) y fraudTest ({len(test):,})")

    df = pd.concat([train, test], ignore_index=True)
    df["fecha"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "fecha"]).reset_index(drop=True)

    df["hora"] = df["fecha"].dt.hour
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["edad"] = (df["fecha"] - pd.to_datetime(df["dob"])).dt.days / 365.25
    df["log_monto"] = np.log1p(df["amt"])
    df["log_poblacion"] = np.log1p(df["city_pop"])
    df["genero_m"] = (df["gender"] == "M").astype(int)
    df["distancia_km"] = distancia_haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["madrugada"] = df["hora"].between(22, 23) | df["hora"].between(0, 3)
    df["madrugada"] = df["madrugada"].astype(int)

    g = df.groupby("cc_num")

    df["seg_desde_anterior"] = g["fecha"].diff().dt.total_seconds().fillna(1e7)
    df["log_seg_desde_anterior"] = np.log1p(df["seg_desde_anterior"])

    media_hist = g["amt"].transform(lambda s: s.shift().expanding().mean())
    df["ratio_monto_hist"] = (df["amt"] / media_hist).fillna(1.0)
    df["n_trans_previas"] = g.cumcount()

    df = df.set_index("fecha")
    rolling = df.groupby("cc_num")["amt"].rolling("24h")
    df["trans_24h"] = rolling.count().values - 1
    df["monto_24h"] = rolling.sum().values - df["amt"].values
    df = df.reset_index()

    dummies = pd.get_dummies(df["category"], prefix="cat", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df = df.rename(columns={"is_fraud": "objetivo", "amt": "monto"})

    variables = ["log_monto", "hora", "dia_semana", "edad", "log_poblacion", "genero_m",
                 "distancia_km", "madrugada", "log_seg_desde_anterior", "ratio_monto_hist",
                 "n_trans_previas", "trans_24h", "monto_24h"] + list(dummies.columns)

    tr = df[df.particion == "train"].sort_values("unix_time")
    te = df[df.particion == "test"]
    if rapido:
        tr = pd.concat([tr[tr.objetivo == 1], tr[tr.objetivo == 0].sample(150_000, random_state=SEMILLA)]).sort_values("unix_time")
        te = pd.concat([te[te.objetivo == 1], te[te.objetivo == 0].sample(80_000, random_state=SEMILLA)])

    corte = int(len(tr) * 0.8)
    tr_fit, val = tr.iloc[:corte], tr.iloc[corte:]

    return dict(nombre="sparkov", df=df, variables=variables,
                X_train=tr_fit[variables], y_train=tr_fit["objetivo"],
                X_val=val[variables], y_val=val["objetivo"],
                X_test=te[variables], y_test=te["objetivo"])

def eda_y_pruebas(datos, carpeta):
    df, nombre = datos["df"], datos["nombre"]
    titulo(f"[{nombre}] Exploración y pruebas de hipótesis")
    resultados = {}

    conteo = df["objetivo"].value_counts().sort_index()
    tasa = conteo[1] / conteo.sum()
    print(f"Legítimas: {conteo[0]:,} | Fraudes: {conteo[1]:,} | Tasa de fraude: {tasa:.4%}")
    resultados["tasa_fraude"] = float(tasa)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(["Legítima", "Fraude"], conteo.values, color=[COLOR_LEGIT, COLOR_FRAUDE])
    ax[0].set_yscale("log"); ax[0].set_title("Número de transacciones (escala log)")
    for i, v in enumerate(conteo.values):
        ax[0].text(i, v, f"{v:,}", ha="center", va="bottom")
    ax[1].pie(conteo.values, labels=["Legítima", "Fraude"], autopct="%1.2f%%",
              colors=[COLOR_LEGIT, COLOR_FRAUDE], startangle=90, wedgeprops=dict(width=0.4))
    ax[1].set_title("Proporción")
    fig.suptitle(f"Desbalance de clases — base {nombre}")
    guardar(fig, carpeta, "01_balance_clases.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for clase, color, etiqueta in [(0, COLOR_LEGIT, "Legítima"), (1, COLOR_FRAUDE, "Fraude")]:
        sns.kdeplot(np.log1p(df.loc[df.objetivo == clase, "monto"]), ax=ax[0], fill=True,
                    color=color, label=etiqueta, common_norm=False, alpha=0.4)
    ax[0].set_xlabel("log(1 + monto)"); ax[0].legend(); ax[0].set_title("Densidad del monto")
    sns.boxplot(data=df, x="objetivo", y="monto", ax=ax[1], palette=[COLOR_LEGIT, COLOR_FRAUDE], showfliers=False)
    ax[1].set_xticklabels(["Legítima", "Fraude"]); ax[1].set_xlabel(""); ax[1].set_title("Monto (sin atípicos)")
    guardar(fig, carpeta, "02_monto_por_clase.png")

    m_f = df.loc[df.objetivo == 1, "monto"]
    m_l = df.loc[df.objetivo == 0, "monto"]
    u, p_u = stats.mannwhitneyu(m_f, m_l, alternative="two-sided")

    efecto = 2 * u / (len(m_f) * len(m_l)) - 1
    print(f"Mann-Whitney U (monto fraude vs legítima): U={u:,.0f}, p={p_u:.2e}, r_rb={efecto:.3f}")
    print(f"   Mediana fraude = {m_f.median():.2f} | Mediana legítima = {m_l.median():.2f}")
    resultados["mann_whitney_monto"] = {"U": float(u), "p_valor": float(p_u), "r_biserial": float(efecto)}

    tabla_hora = pd.crosstab(df["hora"], df["objetivo"])
    chi2, p_chi, gl, _ = stats.chi2_contingency(tabla_hora)
    v_cramer = np.sqrt(chi2 / (tabla_hora.values.sum() * (min(tabla_hora.shape) - 1)))
    print(f"Chi-cuadrada (hora vs fraude): chi2={chi2:,.1f}, gl={gl}, p={p_chi:.2e}, V de Cramér={v_cramer:.3f}")
    resultados["chi2_hora"] = {"chi2": float(chi2), "gl": int(gl), "p_valor": float(p_chi), "v_cramer": float(v_cramer)}

    fig, ax = plt.subplots(figsize=(11, 4))
    tasa_hora = df.groupby("hora")["objetivo"].mean() * 100
    ax.bar(tasa_hora.index, tasa_hora.values, color=COLOR_FRAUDE, alpha=0.85)
    ax.axhline(tasa * 100, ls="--", color="black", lw=1, label="tasa global")
    ax.set_xlabel("Hora del día"); ax.set_ylabel("% fraude"); ax.legend()
    ax.set_title(f"Tasa de fraude por hora — base {nombre}")
    guardar(fig, carpeta, "03_fraude_por_hora.png")

    if nombre == "europeo":
        corr = df[[f"V{i}" for i in range(1, 29)] + ["log_monto", "objetivo"]].corr()["objetivo"].drop("objetivo")
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(corr.index, corr.values, color=np.where(corr.values > 0, COLOR_FRAUDE, COLOR_LEGIT))
        ax.set_title("Correlación de Pearson de cada componente con el fraude")
        plt.xticks(rotation=60)
        guardar(fig, carpeta, "04_correlacion_objetivo.png")

        top = corr.abs().head(6).index
        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        for a, col in zip(axes.ravel(), top):
            for clase, color in [(0, COLOR_LEGIT), (1, COLOR_FRAUDE)]:
                sns.kdeplot(df.loc[df.objetivo == clase, col], ax=a, fill=True, color=color, alpha=0.4, common_norm=False)

            ks = stats.ks_2samp(df.loc[df.objetivo == 1, col], df.loc[df.objetivo == 0, col]).statistic
            a.set_title(f"{col}  (KS = {ks:.2f})")
        fig.suptitle("Componentes más discriminantes (rojo = fraude)")
        fig.tight_layout()
        guardar(fig, carpeta, "05_distribuciones_top.png")
    else:
        cat = df.groupby("category")["objetivo"].agg(["mean", "sum"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(cat.index, cat["mean"] * 100, color=COLOR_FRAUDE)
        ax.set_xlabel("% fraude"); ax.set_title("Tasa de fraude por categoría de comercio")
        guardar(fig, carpeta, "04_fraude_por_categoria.png")

        tabla_cat = pd.crosstab(df["category"], df["objetivo"])
        chi2c, pc, glc, _ = stats.chi2_contingency(tabla_cat)
        vc = np.sqrt(chi2c / (tabla_cat.values.sum() * (min(tabla_cat.shape) - 1)))
        print(f"Chi-cuadrada (categoría vs fraude): chi2={chi2c:,.1f}, gl={glc}, p={pc:.2e}, V de Cramér={vc:.3f}")
        resultados["chi2_categoria"] = {"chi2": float(chi2c), "gl": int(glc), "p_valor": float(pc), "v_cramer": float(vc)}

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for a, col, xlab in zip(axes, ["edad", "log_seg_desde_anterior", "ratio_monto_hist"],
                                ["Edad (años)", "log(seg desde la compra anterior)", "monto / promedio histórico"]):
            datos_col = df[[col, "objetivo"]].copy()
            if col == "ratio_monto_hist":
                datos_col[col] = datos_col[col].clip(upper=20)
            for clase, color in [(0, COLOR_LEGIT), (1, COLOR_FRAUDE)]:
                sns.kdeplot(datos_col.loc[datos_col.objetivo == clase, col], ax=a, fill=True,
                            color=color, alpha=0.4, common_norm=False)
            a.set_xlabel(xlab)
        fig.suptitle("Variables construidas: fraude (rojo) vs legítima (azul)")
        guardar(fig, carpeta, "05_variables_construidas.png")

    return resultados

def construir_modelos(y_train):

    peso = (y_train == 0).sum() / (y_train == 1).sum()
    modelos = {
        "Regresión logística": LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEMILLA),
    }
    if XGBClassifier is not None:
        modelos["XGBoost"] = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                           colsample_bytree=0.8, scale_pos_weight=np.sqrt(peso),
                                           eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEMILLA)
    if LGBMClassifier is not None:

        modelos["LightGBM"] = LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                             min_child_samples=50, reg_lambda=5.0,
                                             subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=np.sqrt(peso),
                                             n_jobs=8, random_state=SEMILLA, verbose=-1)
    return modelos

def submuestreo(X, y, proporcion=20):

    idx_f = y[y == 1].index
    idx_l = y[y == 0].sample(min(len(idx_f) * proporcion, (y == 0).sum()), random_state=SEMILLA).index
    idx = idx_f.union(idx_l)
    return X.loc[idx], y.loc[idx]

def umbral_optimo(y, prob):

    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = np.nanargmax(f1[:-1])
    return float(t[i])

def metricas(y, prob, umbral):
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    n = len(y)
    prevalencia = (tp + fn) / n
    especificidad = tn / (tn + fp) if tn + fp else 0.0
    vpn = tn / (tn + fn) if tn + fn else 0.0
    tasa_alertas = (tp + fp) / n
    return {
        "ROC-AUC": roc_auc_score(y, prob),
        "PR-AUC": average_precision_score(y, prob),
        "Lift PR-AUC": average_precision_score(y, prob) / prevalencia if prevalencia else np.nan,
        "Precisión": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "F2": fbeta_score(y, pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "Exactitud balanceada": balanced_accuracy_score(y, pred),
        "Especificidad": especificidad,
        "VPN": vpn,
        "FPR por 1000": 1000 * fp / (tn + fp) if tn + fp else 0.0,
        "Alertas por 1000": 1000 * tasa_alertas,
        "Umbral": umbral, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }

def bootstrap_ic(y, prob, umbral, n=500):

    rng = np.random.default_rng(SEMILLA)
    y, prob = np.asarray(y), np.asarray(prob)
    idx_pos, idx_neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(idx_pos) or not len(idx_neg):
        raise ValueError("El bootstrap requiere observaciones de ambas clases")
    distribuciones = {k: [] for k in ["PR-AUC", "ROC-AUC", "F1", "Precisión", "Recall", "MCC"]}
    for _ in range(n):
        i = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        yi, probi = y[i], prob[i]
        pred = (probi >= umbral).astype(int)
        distribuciones["PR-AUC"].append(average_precision_score(yi, probi))
        distribuciones["ROC-AUC"].append(roc_auc_score(yi, probi))
        distribuciones["F1"].append(f1_score(yi, pred))
        distribuciones["Precisión"].append(precision_score(yi, pred, zero_division=0))
        distribuciones["Recall"].append(recall_score(yi, pred))
        distribuciones["MCC"].append(matthews_corrcoef(yi, pred))
    ic = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {k: ic(v) for k, v in distribuciones.items()}, np.asarray(distribuciones["PR-AUC"])

def mcnemar(y, pred_a, pred_b):

    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    a_bien, b_bien = pred_a == y, pred_b == y
    solo_a = int(np.sum(a_bien & ~b_bien))
    solo_b = int(np.sum(~a_bien & b_bien))
    discordantes = solo_a + solo_b
    p = 1.0 if discordantes == 0 else stats.binomtest(
        min(solo_a, solo_b), discordantes, p=0.5, alternative="two-sided"
    ).pvalue
    return {"metodo": "exacto_binomial", "solo_a_correcto": solo_a,
            "solo_b_correcto": solo_b, "discordantes": discordantes, "p_valor": float(p)}

def guardar_modelo(datos, mejor, resumen, ajustados, escalador, arboles, carpeta_modelos):

    carpeta_modelos = Path(carpeta_modelos)
    carpeta_modelos.mkdir(parents=True, exist_ok=True)
    nombre_base = datos["nombre"]
    es_ensamble = mejor == "Ensamble (promedio)"
    nombres = arboles if es_ensamble else [mejor]
    bundle = {
        "version_formato": 1,
        "dataset": nombre_base,
        "nombre_modelo": mejor,
        "tipo": "ensamble_promedio" if es_ensamble else "modelo_individual",
        "estimadores": {n: ajustados[n] for n in nombres},
        "variables": list(datos["variables"]),
        "umbral": float(resumen.loc[mejor, "Umbral"]),
        "escalador": escalador if mejor == "Regresión logística" else None,
        "metrica_seleccion": "PR-AUC",
        "metricas_test": {k: float(v) for k, v in resumen.loc[mejor].items()},
        "versiones": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "nota_inferencia": (
            "Sparkov requiere las mismas variables históricas calculadas en orden temporal."
            if nombre_base == "sparkov" else
            "La entrada puede prepararse desde Time, Amount y V1..V28."
        ),
    }
    ruta_joblib = carpeta_modelos / f"{nombre_base}_mejor_modelo.joblib"
    joblib.dump(bundle, ruta_joblib, compress=3)
    artefactos = [str(ruta_joblib)]

    if not es_ensamble:
        modelo = ajustados[mejor]
        if mejor == "XGBoost" and hasattr(modelo, "save_model"):
            ruta_nativa = carpeta_modelos / f"{nombre_base}_xgboost.json"
            modelo.save_model(ruta_nativa)
            artefactos.append(str(ruta_nativa))
        elif mejor == "LightGBM" and hasattr(modelo, "booster_"):
            ruta_nativa = carpeta_modelos / f"{nombre_base}_lightgbm.txt"
            modelo.booster_.save_model(str(ruta_nativa))
            artefactos.append(str(ruta_nativa))

    metadata = {k: v for k, v in bundle.items() if k not in {"estimadores", "escalador"}}
    ruta_metadata = carpeta_modelos / f"{nombre_base}_metadata.json"
    ruta_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    artefactos.append(str(ruta_metadata))
    print(f"Modelo guardado: {ruta_joblib.resolve()}")
    return artefactos

def entrenar_y_evaluar(datos, carpeta, carpeta_modelos=None):
    nombre = datos["nombre"]
    titulo(f"[{nombre}] Entrenamiento de modelos")
    X_tr, y_tr = datos["X_train"], datos["y_train"]
    X_val, y_val = datos["X_val"], datos["y_val"]
    X_te, y_te = datos["X_test"], datos["y_test"]
    print(f"Train: {len(X_tr):,} ({y_tr.sum():,} fraudes) | Val: {len(X_val):,} | Test: {len(X_te):,} ({y_te.sum():,} fraudes)")

    escalador = StandardScaler().fit(X_tr)
    modelos = construir_modelos(y_tr)
    tabla, probs, preds, ajustados = {}, {}, {}, {}

    for nom, modelo in modelos.items():
        t0 = time.time()
        if nom == "Regresión logística":
            Xa, ya = submuestreo(X_tr, y_tr, 50) if len(X_tr) > 400_000 else (X_tr, y_tr)
            modelo.fit(escalador.transform(Xa), ya)
            p_val = modelo.predict_proba(escalador.transform(X_val))[:, 1]
            p_te = modelo.predict_proba(escalador.transform(X_te))[:, 1]
        else:
            Xa, ya = submuestreo(X_tr, y_tr, 20) if (nom == "Random Forest" and len(X_tr) > 400_000) else (X_tr, y_tr)
            modelo.fit(Xa, ya)
            p_val = modelo.predict_proba(X_val)[:, 1]
            p_te = modelo.predict_proba(X_te)[:, 1]

        u = umbral_optimo(y_val, p_val)
        m = metricas(y_te, p_te, u)
        tabla[nom], probs[nom], ajustados[nom] = m, p_te, modelo
        preds[nom] = (p_te >= u).astype(int)
        print(f"  {nom:<20} PR-AUC={m['PR-AUC']:.4f}  ROC-AUC={m['ROC-AUC']:.4f}  "
              f"F1={m['F1']:.4f}  Prec={m['Precisión']:.4f}  Rec={m['Recall']:.4f}  ({time.time() - t0:.0f}s)")

    arboles = [n for n in probs if n != "Regresión logística"]
    if len(arboles) >= 2:
        p_val_ens = np.mean([
            ajustados[n].predict_proba(X_val)[:, 1] for n in arboles], axis=0)
        p_te_ens = np.mean([probs[n] for n in arboles], axis=0)
        u = umbral_optimo(y_val, p_val_ens)
        tabla["Ensamble (promedio)"] = metricas(y_te, p_te_ens, u)
        probs["Ensamble (promedio)"] = p_te_ens
        preds["Ensamble (promedio)"] = (p_te_ens >= u).astype(int)
        m = tabla["Ensamble (promedio)"]
        print(f"  {'Ensamble (promedio)':<20} PR-AUC={m['PR-AUC']:.4f}  ROC-AUC={m['ROC-AUC']:.4f}  F1={m['F1']:.4f}")

    resumen = pd.DataFrame(tabla).T.sort_values("PR-AUC", ascending=False)
    mejor = resumen.index[0]
    mejor_f1 = resumen["F1"].idxmax()
    print(f"\nMejor modelo por PR-AUC: {mejor}")
    if mejor_f1 != mejor:
        print(f"Mejor modelo por F1 al umbral elegido: {mejor_f1} (objetivo distinto)")

    ic, dist_pr = bootstrap_ic(y_te, probs[mejor], resumen.loc[mejor, "Umbral"])
    print(f"IC 95 % bootstrap ({mejor}): PR-AUC [{ic['PR-AUC'][0]:.4f}, {ic['PR-AUC'][1]:.4f}] | "
          f"ROC-AUC [{ic['ROC-AUC'][0]:.4f}, {ic['ROC-AUC'][1]:.4f}] | F1 [{ic['F1'][0]:.4f}, {ic['F1'][1]:.4f}]")
    segundo = resumen.index[1]
    prueba_mc = mcnemar(y_te, preds[mejor], preds[segundo])
    print(f"McNemar exacto {mejor} vs {segundo}: "
          f"solo primero={prueba_mc['solo_a_correcto']}, solo segundo={prueba_mc['solo_b_correcto']}, "
          f"p={prueba_mc['p_valor']:.4f}")

    artefactos_modelo = []
    if carpeta_modelos is not None:
        artefactos_modelo = guardar_modelo(
            datos, mejor, resumen, ajustados, escalador, arboles, carpeta_modelos
        )

    graficas_modelos(datos, resumen, probs, preds, ajustados, mejor, dist_pr, carpeta)

    exportar = resumen.round(4).rename_axis("Modelo").reset_index()
    for col in ["TP", "FP", "FN", "TN"]:
        exportar[col] = exportar[col].astype(int)
    exportar.to_csv(carpeta / "metricas_modelos.csv", index=False)
    return {
        "resumen": resumen, "probs": probs, "y_test": y_te, "mejor": mejor, "mejor_f1": mejor_f1,
        "ic95": ic, "mcnemar": {"modelos": [mejor, segundo], **prueba_mc},
        "artefactos_modelo": artefactos_modelo,
    }

def graficas_modelos(datos, resumen, probs, preds, ajustados, mejor, dist_pr, carpeta):
    nombre, y_te = datos["nombre"], datos["y_test"]
    orden = list(resumen.index)
    colores = dict(zip(orden, PALETA_MODELOS))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for n in orden:
        fpr, tpr, _ = roc_curve(y_te, probs[n])
        ax[0].plot(fpr, tpr, lw=2, color=colores[n], label=f"{n} ({resumen.loc[n, 'ROC-AUC']:.3f})")
        p, r, _ = precision_recall_curve(y_te, probs[n])
        ax[1].plot(r, p, lw=2, color=colores[n], label=f"{n} ({resumen.loc[n, 'PR-AUC']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC")
    ax[1].axhline(y_te.mean(), color="grey", ls="--", lw=1, label="azar")
    ax[1].set(xlabel="Recall", ylabel="Precisión", title="Curva Precisión-Recall")
    ax[0].legend(loc="lower right", fontsize=9); ax[1].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Comparación de modelos — base {nombre}")
    guardar(fig, carpeta, "06_curvas_roc_pr.png")

    cm = confusion_matrix(y_te, preds[mejor])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    etiquetas = np.array([[f"{v:,}\n({v / cm[i].sum():.1%})" for v in fila] for i, fila in enumerate(cm)])
    sns.heatmap(cm, annot=etiquetas, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legítima", "Fraude"], yticklabels=["Legítima", "Fraude"])
    ax.set(xlabel="Predicción", ylabel="Real", title=f"Matriz de confusión — {mejor}")
    guardar(fig, carpeta, "07_matriz_confusion.png")

    cols = ["PR-AUC", "ROC-AUC", "F1", "Precisión", "Recall", "MCC"]
    fig, ax = plt.subplots(figsize=(13, 5))
    resumen[cols].T.plot(kind="bar", ax=ax, color=[colores[n] for n in orden], width=0.8)
    ax.set_ylim(0, 1.05); ax.set_title(f"Métricas en el conjunto de prueba — base {nombre}")
    plt.xticks(rotation=0); ax.legend(fontsize=9, ncol=3, loc="lower right")
    guardar(fig, carpeta, "08_metricas_modelos.png")

    candidato = mejor if mejor in ajustados and hasattr(ajustados[mejor], "feature_importances_") else \
        next((n for n in orden if n in ajustados and hasattr(ajustados[n], "feature_importances_")), None)
    if candidato:
        imp = pd.Series(ajustados[candidato].feature_importances_, index=datos["variables"])
        imp = (imp / imp.sum()).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp.index, imp.values, color="#2A9D8F")
        ax.set_title(f"Importancia relativa de variables — {candidato}")
        guardar(fig, carpeta, "09_importancia_variables.png")

    p, r, t = precision_recall_curve(y_te, probs[mejor])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, p[:-1], label="Precisión", color=COLOR_LEGIT, lw=2)
    ax.plot(t, r[:-1], label="Recall", color=COLOR_FRAUDE, lw=2)
    ax.axvline(resumen.loc[mejor, "Umbral"], color="black", ls="--", label="umbral elegido")
    ax.set(xlabel="Umbral de probabilidad", title=f"Compromiso precisión / recall — {mejor}")
    ax.legend()
    guardar(fig, carpeta, "10_umbral.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(dist_pr, kde=True, color="#264653", ax=ax)
    for q in np.percentile(dist_pr, [2.5, 97.5]):
        ax.axvline(q, color=COLOR_FRAUDE, ls="--")
    ax.set(xlabel="PR-AUC", title=f"Distribución bootstrap de la PR-AUC (IC 95 %) — {mejor}")
    guardar(fig, carpeta, "11_bootstrap_pr_auc.png")

def comparar_bases(res, carpeta):
    titulo("Comparación entre bases")
    filas = []
    for base, r in res.items():
        for modelo, m in r["resumen"].iterrows():
            filas.append({"Base": base, "Modelo": modelo,
                          **m[["PR-AUC", "Lift PR-AUC", "ROC-AUC", "F1", "F2", "Precisión",
                               "Recall", "MCC", "FPR por 1000", "Alertas por 1000"]].to_dict()})
    comp = pd.DataFrame(filas)
    comp.round(4).to_csv(carpeta / "comparacion_bases.csv", index=False)
    print(comp.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, (base, r) in enumerate(res.items()):
        for n, c in zip(r["resumen"].index, PALETA_MODELOS):
            p, rc, _ = precision_recall_curve(r["y_test"], r["probs"][n])
            ax[i].plot(rc, p, color=c, lw=2, label=f"{n} ({r['resumen'].loc[n, 'PR-AUC']:.3f})")
        ax[i].set(title=f"Precisión-Recall — {base}", xlabel="Recall", ylabel="Precisión")
        ax[i].legend(fontsize=8, loc="lower left")
    guardar(fig, carpeta, "comparacion_curvas_pr.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=comp.melt(id_vars=["Base", "Modelo"], value_vars=["PR-AUC", "F1", "MCC"]),
                x="variable", y="value", hue="Base", palette=["#264653", "#E76F51"], ax=ax, errorbar="sd")
    ax.set(xlabel="", ylabel="valor", ylim=(0, 1.05), title="Desempeño promedio de los modelos por base (barras = ±1 de)")
    guardar(fig, carpeta, "comparacion_metricas.png")

def main():
    parser = argparse.ArgumentParser(description="Detección de fraude en tarjetas de crédito")
    parser.add_argument("--dataset", choices=["europeo", "sparkov", "ambas"], default="ambas")
    parser.add_argument("--data-dir", default="data", help="carpeta donde están los csv (se busca en subcarpetas)")
    parser.add_argument("--out-dir", default="resultados")
    parser.add_argument("--model-dir", default="modelos", help="carpeta para modelos entrenados")
    parser.add_argument("--no-guardar-modelo", action="store_true", help="no exporta artefactos de inferencia")
    parser.add_argument("--rapido", action="store_true", help="usa una muestra para pruebas rápidas")
    args = parser.parse_args()

    salida = Path(args.out_dir)
    bases = ["europeo", "sparkov"] if args.dataset == "ambas" else [args.dataset]
    resultados, reporte = {}, {}

    for base in bases:
        carpeta = salida / base
        carpeta.mkdir(parents=True, exist_ok=True)
        datos = preparar_europeo(args.data_dir, args.rapido) if base == "europeo" else preparar_sparkov(args.data_dir, args.rapido)
        pruebas = eda_y_pruebas(datos, carpeta)
        carpeta_modelos = None if args.no_guardar_modelo else Path(args.model_dir)
        res = entrenar_y_evaluar(datos, carpeta, carpeta_modelos)
        resultados[base] = res
        reporte[base] = {
            "pruebas_estadisticas": pruebas,
            "mejor_modelo": res["mejor"],
            "criterio_mejor_modelo": "PR-AUC",
            "mejor_modelo_f1": res["mejor_f1"],
            "metricas_test": {k: round(float(v), 4) for k, v in res["resumen"].loc[res["mejor"]].items()},
            "ic95_bootstrap": res["ic95"],
            "mcnemar": res["mcnemar"],
            "artefactos_modelo": res["artefactos_modelo"],
        }

    if len(resultados) == 2:
        comparar_bases(resultados, salida)

    with open(salida / "reporte.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"\nListo. Gráficas y métricas en: {salida.resolve()}")

if __name__ == "__main__":
    main()
