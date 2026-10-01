"""Carga de las 7 ramas (modelos base) y sus configuraciones candidatas.

Cada rama es un DataFrame indexado por participante (código 1..80). Una fila completamente
vacía significa "rama no disponible para ese participante" (p. ej. el 66 no tiene audio): la
rama se entrena y predice solo con quienes la tienen, y la fusión promedia las ramas
disponibles de cada participante.

| rama               | modalidad | qué es                                                        | explicable |
|--------------------|-----------|---------------------------------------------------------------|-----------|
| au                 | rostro    | 66 AU / pose / emoción (py-feat), sin n_frames_detected       | SHAP+LIME |
| rostro_temporal    | rostro    | 256 features temporales (MediaPipe, v2)                        | SHAP+LIME |
| egemaps            | voz       | 88 eGeMAPS, media por participante (audio editado)             | SHAP+LIME |
| voz_microventanas  | voz       | wav2vec2-large-robust, 0.19 s cada 2.5 s, última capa (réplica) | proxy     |
| voz_densa          | voz       | ídem, micro-ventanas de 0.19 s contiguas (todo el audio)        | proxy     |
| texto_emb          | texto     | embeddings de frases (Whisper -> mpnet multilingüe)            | proxy     |
| texto_lexico       | texto     | 7 features léxicas interpretables                               | SHAP+LIME |

"proxy": en los embeddings, SHAP por dimensión no es interpretable. Se explica con (1) la
contribución de la modalidad en la fusión y (2) su correlación con features interpretables
de la misma modalidad.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C

INFO = {
    "au": {"modalidad": "rostro", "interpretable": True, "tipo": "tabular"},
    "rostro_temporal": {"modalidad": "rostro", "interpretable": True, "tipo": "tabular"},
    "egemaps": {"modalidad": "voz", "interpretable": True, "tipo": "tabular"},
    "voz_microventanas": {"modalidad": "voz", "interpretable": False, "tipo": "embedding"},
    "voz_densa": {"modalidad": "voz", "interpretable": False, "tipo": "embedding"},
    "texto_emb": {"modalidad": "texto", "interpretable": False, "tipo": "embedding"},
    "texto_lexico": {"modalidad": "texto", "interpretable": True, "tipo": "lexico"},
}


def _base():
    return [("var", VarianceThreshold(0.0)), ("sc", StandardScaler())]


def configs(tipo):
    """(pipeline, grid) candidatos por tipo de rama. La imputación se hace fuera, por fold."""
    if tipo == "tabular":
        return {
            "l1logreg": (Pipeline(_base() + [("clf", LogisticRegression(
                penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5000))]),
                {"clf__C": [0.05, 0.1, 0.3, 1.0]}),
            "anova_logreg": (Pipeline(_base() + [("sel", SelectKBest(f_classif)), ("clf", LogisticRegression(
                class_weight="balanced", max_iter=5000))]),
                {"sel__k": [4, 8], "clf__C": [0.3, 1.0]}),
            "anova_rf": (Pipeline(_base() + [("sel", SelectKBest(f_classif)), ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=3, class_weight="balanced",
                random_state=C.SEED, n_jobs=1))]),
                {"sel__k": [4, 8]}),
        }
    if tipo == "lexico":
        return {
            "logreg": (Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(
                class_weight="balanced", max_iter=5000))]), {"clf__C": [0.01, 0.1, 1.0]}),
            "rf": (Pipeline([("clf", RandomForestClassifier(
                n_estimators=300, max_depth=3, min_samples_leaf=3, class_weight="balanced",
                random_state=C.SEED, n_jobs=1))]), {}),
        }
    if tipo == "embedding":
        return {
            "emb_logreg": (Pipeline([("sc", StandardScaler()), ("pca", "passthrough"), ("clf", LogisticRegression(
                class_weight="balanced", max_iter=5000))]),
                {"pca": ["passthrough", PCA(16, random_state=C.SEED), PCA(32, random_state=C.SEED)],
                 "clf__C": [1e-3, 1e-2, 1e-1]}),
            "emb_et": (Pipeline([("clf", ExtraTreesClassifier(
                n_estimators=300, max_features="sqrt", min_samples_leaf=3, class_weight="balanced",
                random_state=C.SEED, n_jobs=1))]), {}),
        }
    raise ValueError(tipo)


def _emb_df(filas, pids, prefijo):
    df = pd.DataFrame.from_dict(filas, orient="index")
    df.columns = [f"{prefijo}_{i:04d}" for i in range(df.shape[1])]
    return df.reindex(pids)


def cargar(dx):
    """-> (pids, y, ramas{nombre: DataFrame indexado por pid})"""
    au = pd.read_csv(C.AU_CSV).rename(columns={"video_id": "pid"}).set_index("pid")
    y = au[f"target_{dx}"].astype(int)            # etiqueta del proyecto ('DX IA', cruzada por código)
    pids = np.array(sorted(y.index))
    ramas = {}
    au_cols = [c for c in au.columns if c not in ("error", "n_frames_detected", "target_ansiedad",
                                                  "target_depresion")]
    ramas["au"] = au.loc[pids, au_cols].astype(float)

    vt = pd.read_csv(C.VT_CSV).set_index("codigo")
    ramas["rostro_temporal"] = vt.reindex(pids)[[c for c in vt.columns if c != "participant"]].astype(float)

    eg = pd.read_csv(C.EGEMAPS[dx])
    eg["pid"] = eg["audio_id"].astype(str).str[:3].astype(int)
    eg_cols = [c for c in eg.columns if c not in ("audio_id", "seg_idx", "label", "pid")]
    ramas["egemaps"] = eg.groupby("pid")[eg_cols].mean().reindex(pids)

    z = np.load(C.R_NPZ)
    ramas["voz_microventanas"] = _emb_df({p: z[f"{p:03d}"].mean(0) for p in pids if f"{p:03d}" in z.files},
                                         pids, "emb")
    rd = {p: np.load(C.RD_DIR / f"{p:03d}.npz")["d019"] for p in pids if (C.RD_DIR / f"{p:03d}.npz").exists()}
    ramas["voz_densa"] = _emb_df(rd, pids, "emb")

    t = np.load(C.TXT_NPZ)
    tp = [int(p) for p in t["pids"]]
    ramas["texto_emb"] = _emb_df(dict(zip(tp, t["emb"])), pids, "emb")
    ramas["texto_lexico"] = pd.DataFrame(t["lex"], index=tp, columns=[str(n) for n in t["lex_nombres"]]).reindex(pids)
    return pids, y.loc[pids].values, ramas
