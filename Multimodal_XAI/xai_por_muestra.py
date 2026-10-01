"""Predicción, SHAP y LIME POR MUESTRA + búsqueda de la mejor combinación multimodal,
para DEPRESIÓN y ANSIEDAD (modelos independientes por eje).

Esquema (lo pidió Mateo): RepeatedStratifiedKFold(16, R): en cada fold se entrena con 75
participantes y se prueba con 5. Las 5 predicciones y sus explicaciones se agregan a las
matrices, y así sucesivamente hasta cubrir a los 80. Con R repeticiones, cada participante
queda explicado R veces y las matrices finales promedian.

Por fold y por rama (7 ramas, ver ramas.py):
  1. imputación por mediana con estadísticas SOLO del train del fold;
  2. elección de configuración por CV interna (5-fold) SOLO dentro de los 75 de train;
  3. predicción de los 5 de test;
  4. ramas interpretables: SHAP (Permutation sobre TODO el pipeline, fondo = muestra del
     train) y LIME (sin discretizar, fondo = train) de cada muestra de test.
Fusión: soft vote (promedio de probabilidades de las ramas disponibles). Se evalúan las 127
combinaciones de 1 a 7 ramas. AUTO elige la combinación por AUC OOF DENTRO del train y la
aplica al test: es la estimación honesta de "quedarse con la mejor" (el máximo de la tabla
de combinaciones es optimista por construcción).

Diferencias con build_oof_matrix_v4_balanceado.py (ver auditoría en el repo Multimodal):
  - ambos ejes, cada uno con su etiqueta; sin destilación (tenía fuga);
  - sin n_frames_detected; embeddings deterministas (sin attention pooling aleatorio);
  - selección de configuración anidada y combinación elegida sin mirar el test.

Salidas en stacking_output/<dx>/ (gitignored): ver ensamblar().
Uso:  python Multimodal_XAI/xai_por_muestra.py --dx depresion [--repeticiones 1] [--minutos 8]
"""
import argparse
import itertools
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)

import config as C
import ramas as R

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- modelos
def cv_interna():
    return StratifiedKFold(C.N_INNER, shuffle=True, random_state=C.SEED)


def elegir(X, y, cfgs):
    """Elige la configuración por AUC de CV interna; devuelve (nombre, estimador ajustado
    con todo X, predicciones OOF dentro de X con esa configuración)."""
    mejor = (None, -np.inf, None)
    for nombre, (pipe, grid) in cfgs.items():
        if grid:
            gs = GridSearchCV(clone(pipe), grid, cv=cv_interna(), scoring="roc_auc", n_jobs=-1).fit(X, y)
            score, est = gs.best_score_, gs.best_estimator_
        else:
            score = cross_val_score(clone(pipe), X, y, cv=cv_interna(), scoring="roc_auc", n_jobs=-1).mean()
            est = clone(pipe).fit(X, y)
        if score > mejor[1]:
            mejor = (nombre, score, est)
    nombre, _, est = mejor
    oof = cross_val_predict(clone(est), X, y, cv=cv_interna(), method="predict_proba", n_jobs=-1)[:, 1]
    return nombre, est, oof


# --------------------------------------------------------------------------- explicaciones
def explicar_shap(est, Xtr, Xte):
    """SHAP (Permutation, 2 permutaciones antitéticas) sobre el pipeline completo, en
    probabilidad de la clase positiva. Fondo: muestra del train del fold."""
    bg = shap.utils.sample(Xtr, min(C.BG_SHAP, len(Xtr)), random_state=C.SEED)
    ex = shap.explainers.Permutation(lambda X: est.predict_proba(X)[:, 1],
                                     shap.maskers.Independent(bg, max_samples=len(bg)))
    return ex(Xte, max_evals=2 * (2 * Xte.shape[1] + 1), silent=True).values


def explicar_lime(est, Xtr, Xte, semilla):
    """LIME tabular sin discretizar (lime 0.2.0.1 + scipy>=1.11 rompe al discretizar):
    peso lineal local por feature para la clase positiva."""
    ex = LimeTabularExplainer(Xtr, mode="classification", discretize_continuous=False,
                              random_state=semilla)
    out = np.zeros(Xte.shape)
    for i in range(len(Xte)):
        e = ex.explain_instance(Xte[i], est.predict_proba, num_features=Xte.shape[1],
                                num_samples=C.LIME_SAMPLES, labels=(1,))
        for j, w in e.as_map()[1]:
            out[i, j] = w
    return out


# --------------------------------------------------------------------------- combinaciones
def combinaciones(nombres):
    return [c for k in range(1, len(nombres) + 1) for c in itertools.combinations(nombres, k)]


def prob_combo(probs, combo, index):
    """Soft vote sobre las ramas del combo DISPONIBLES para cada participante."""
    return pd.concat([probs[b].reindex(index) for b in combo], axis=1).mean(axis=1)


def auc_seguro(y, s):
    m = ~np.isnan(s)
    return roc_auc_score(y[m], s[m]) if m.sum() > 2 and len(np.unique(y[m])) == 2 else np.nan


# --------------------------------------------------------------------------- un fold
def procesar_fold(pids, y, ramas, tr, te, semilla):
    out = {"pid": pids[te], "y": y[te], "prob": {}, "oof": {}, "config": {},
           "shap": {}, "lime": {}, "filas_expl": {}}
    for b, X in ramas.items():
        info = R.INFO[b]
        disp = X.notna().any(axis=1).values
        trb, teb = tr[disp[tr]], te[disp[te]]
        A = X.values.astype(float)
        Xtr, Xte = A[trb], A[teb]
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)
        nombre, est, oof = elegir(Xtr, y[trb], R.configs(info["tipo"]))
        out["config"][b] = nombre
        out["oof"][b] = pd.Series(oof, index=pids[trb])
        out["prob"][b] = pd.Series(est.predict_proba(Xte)[:, 1] if len(teb) else [], index=pids[teb],
                                   dtype=float)
        if info["interpretable"] and len(teb):
            out["shap"][b] = explicar_shap(est, Xtr, Xte)
            out["lime"][b] = explicar_lime(est, Xtr, Xte, semilla)
            out["filas_expl"][b] = pids[teb]
    # AUTO: la combinación con mayor AUC OOF dentro del train, aplicada al test
    idx_tr = pd.Index(pids[tr])
    y_tr = pd.Series(y[tr], index=idx_tr)
    aucs = {"+".join(c): auc_seguro(y_tr.values, prob_combo(out["oof"], c, idx_tr).values)
            for c in combinaciones(list(ramas))}
    out["auc_train"] = aucs
    out["auto"] = max((k for k in aucs if not np.isnan(aucs[k])), key=aucs.get)
    out["prob_auto"] = prob_combo(out["prob"], out["auto"].split("+"), pd.Index(pids[te]))
    return out


# --------------------------------------------------------------------------- ensamblar
def ensamblar(dx, pids, y, ramas, cache, n_rep):
    dest = C.OUT / dx
    res = []
    for f in sorted(cache.glob("r*_f*.pkl")):
        if int(f.name[1:f.name.index("_")]) < n_rep:
            res.append(pickle.load(open(f, "rb")))
    nombres = list(ramas)

    # 1) predicciones por repetición y promedio
    filas = []
    for r in res:
        d = pd.DataFrame({"subject_id": r["pid"], "repeat": r["repeat"], "fold": r["fold"]}).set_index("subject_id")
        for b in nombres:
            d[f"prob_{b}"] = r["prob"][b].reindex(d.index)
        d["prob_AUTO"] = r["prob_auto"].reindex(d.index).values
        d["AUTO_combinacion"] = r["auto"]
        d["y_true"] = r["y"]
        filas.append(d.reset_index())
    D = pd.concat(filas, ignore_index=True).sort_values(["repeat", "subject_id"])
    D.to_csv(dest / f"oof_matrix_detail_by_repeat_{dx}.csv", index=False)
    pcols = [f"prob_{b}" for b in nombres] + ["prob_AUTO"]
    F = D.groupby("subject_id")[pcols].mean()
    F["y_true"] = D.groupby("subject_id")["y_true"].first()
    F.reset_index().to_csv(dest / f"oof_matrix_final_{dx}.csv", index=False)

    # 2) todas las combinaciones: AUC estándar (todos los participantes juntos) por repetición
    tab = []
    for c in combinaciones(nombres) + [("AUTO",)]:
        rep = []
        for _, g in D.groupby("repeat"):
            s = g["prob_AUTO"].values if c == ("AUTO",) else g[[f"prob_{b}" for b in c]].mean(axis=1).values
            rep.append(auc_seguro(g["y_true"].values, s))
        n = int(D.groupby("repeat").apply(lambda g: g[[f"prob_{b}" for b in c]].notna().any(axis=1).sum()
                                           if c != ("AUTO",) else len(g)).iloc[0])
        tab.append({"combinacion": "+".join(c), "n_modelos": 0 if c == ("AUTO",) else len(c),
                    "modalidades": "+".join(sorted({R.INFO[b]["modalidad"] for b in c})) if c != ("AUTO",) else "",
                    "n_participantes": n, "auc_media": np.nanmean(rep),
                    "auc_de": np.nanstd(rep) if len(rep) > 1 else np.nan})
    T = pd.DataFrame(tab).sort_values("auc_media", ascending=False)
    T.round(4).to_csv(dest / f"combinaciones_{dx}.csv", index=False)
    mejor = T[T.combinacion != "AUTO"].iloc[0]["combinacion"].split("+")

    # 3) SHAP y LIME por muestra (ramas interpretables), promedio sobre repeticiones
    for tipo in ("shap", "lime"):
        bloques = []
        for b in nombres:
            if not R.INFO[b]["interpretable"]:
                continue
            partes = [pd.DataFrame(r[tipo][b], index=r["filas_expl"][b], columns=ramas[b].columns)
                      for r in res if b in r[tipo]]
            M = pd.concat(partes).groupby(level=0).mean()
            M.columns = [f"{b}__{c}" for c in M.columns]
            bloques.append(M)
        X = pd.concat(bloques, axis=1).reindex(pids)
        X.index.name = "subject_id"
        X.reset_index().to_csv(dest / f"{tipo}_matrix_{dx}.csv", index=False)

    # 4) contribución de cada modelo/modalidad a la fusión ganadora (descomposición exacta del
    #    soft vote: (p_rama - media OOF de la rama en el train del fold) / n_ramas_disponibles)
    contrib = []
    for r in res:
        d = pd.DataFrame(index=pd.Index(r["pid"], name="subject_id"))
        disponibles = pd.concat([r["prob"][b].reindex(d.index) for b in mejor], axis=1).notna().sum(axis=1)
        for b in mejor:
            d[f"contrib_{b}"] = (r["prob"][b].reindex(d.index) - r["oof"][b].mean()) / disponibles
        contrib.append(d)
    Cm = pd.concat(contrib).groupby(level=0).mean().reindex(pids)
    Cm["fusion_" + "+".join(mejor)] = F.reindex(pids)[[f"prob_{b}" for b in mejor]].mean(axis=1)
    Cm.reset_index().to_csv(dest / f"contribucion_modelos_{dx}.csv", index=False)

    # 5) explicabilidad de los embeddings por proxy: correlación entre la probabilidad OOF de la
    #    rama de embeddings y las features interpretables de la misma modalidad
    proxy = []
    pares = {"voz_microventanas": "egemaps", "voz_densa": "egemaps", "texto_emb": "texto_lexico"}
    for emb, ref in pares.items():
        p = F[f"prob_{emb}"]
        for col in ramas[ref].columns:
            v = ramas[ref][col].reindex(F.index)
            m = p.notna() & v.notna()
            if m.sum() > 10 and v[m].std() > 0:
                rho, pv = spearmanr(p[m], v[m])
                proxy.append({"rama_embedding": emb, "feature_interpretable": f"{ref}__{col}",
                              "rho_spearman": rho, "p_valor": pv})
    Pq = pd.DataFrame(proxy)
    Pq["abs_rho"] = Pq.rho_spearman.abs()
    Pq.sort_values(["rama_embedding", "abs_rho"], ascending=[True, False]).drop(columns="abs_rho") \
      .round(4).to_csv(dest / f"explicacion_embeddings_{dx}.csv", index=False)

    # 6) resumen legible
    auto = T[T.combinacion == "AUTO"].iloc[0]
    elecciones = pd.Series([r["auto"] for r in res]).value_counts()
    configs_ = {b: pd.Series([r["config"][b] for r in res]).value_counts().to_dict() for b in nombres}
    lineas = [f"=== {dx.upper()} — {len(pids)} participantes ({int(y.sum())} positivos 'DX IA'), "
              f"{len(res)} folds (16 x {n_rep}), AUC estándar ===",
              "", "Modelos solos:"]
    for b in nombres:
        fila = T[T.combinacion == b].iloc[0]
        lineas.append(f"  {b:20s} AUC={fila.auc_media:.3f}  (n={fila.n_participantes})")
    lineas += ["", "Mejor combinación por número de modelos:"]
    for k in range(1, len(nombres) + 1):
        fila = T[T.n_modelos == k].iloc[0]
        lineas.append(f"  {k} modelo(s): {fila.combinacion:60s} AUC={fila.auc_media:.3f}")
    lineas += ["", f"MEJOR combinación de la tabla (optimista): {'+'.join(mejor)}  AUC={T.iloc[0 if T.iloc[0].combinacion != 'AUTO' else 1].auc_media:.3f}",
               f"AUTO (estimación honesta de 'quedarse con la mejor'): AUC={auto.auc_media:.3f}",
               f"AUTO eligió: {elecciones.head(5).to_dict()}",
               "", "Configuración elegida por rama (CV interna):"] + [f"  {b}: {v}" for b, v in configs_.items()]
    (dest / f"resumen_{dx}.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas), flush=True)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", choices=list(C.DXS) + ["all"], default="all")
    ap.add_argument("--repeticiones", type=int, default=1)
    ap.add_argument("--minutos", type=float, default=8.0)
    ap.add_argument("--solo-folds", type=int, default=None, help="prueba rápida: procesa N folds y sale")
    args = ap.parse_args()
    t0 = time.time()
    for dx in (C.DXS if args.dx == "all" else [args.dx]):
        pids, y, ramas = R.cargar(dx)
        cache = C.OUT / dx / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        splits = list(RepeatedStratifiedKFold(n_splits=C.N_SPLITS, n_repeats=args.repeticiones,
                                              random_state=C.SEED).split(pids, y))
        hechos = 0
        for i, (tr, te) in enumerate(splits):
            rep, fold = divmod(i, C.N_SPLITS)
            f = cache / f"r{rep}_f{fold:02d}.pkl"
            if f.exists():
                continue
            if (time.time() - t0) / 60 > args.minutos or (args.solo_folds and hechos >= args.solo_folds):
                n = len(list(cache.glob("r*_f*.pkl")))
                print(f"[{dx}] pausa: {n}/{len(splits)} folds en caché ({time.time() - t0:.0f}s)", flush=True)
                return
            t1 = time.time()
            res = procesar_fold(pids, y, ramas, tr, te, C.SEED + i)
            res["repeat"], res["fold"] = rep, fold
            pickle.dump(res, open(f, "wb"))
            hechos += 1
            print(f"[{dx}] fold {i + 1}/{len(splits)} ({time.time() - t1:.0f}s)  AUTO={res['auto']}", flush=True)
        ensamblar(dx, pids, y, ramas, cache, args.repeticiones)


if __name__ == "__main__":
    main()
