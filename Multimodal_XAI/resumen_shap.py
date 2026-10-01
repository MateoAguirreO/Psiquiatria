"""Resumen de SHAP/LIME por muestra para el artículo: qué características contribuyen más,
en qué dirección, y con qué robustez. Una fila por feature y eje.

Columnas:
  rama, auc_rama          AUC estándar del modelo de esa rama (de combinaciones_<eje>.csv);
                          solo tiene sentido interpretar features de ramas que predicen (> 0.55)
  importancia             |SHAP| medio sobre los 80 participantes (probabilidad)
  direccion               rho de Spearman entre el VALOR de la feature y su SHAP:
                          > 0 = más valor sube el riesgo; < 0 = más valor lo baja
  lime_concuerda          el signo medio de LIME coincide con la dirección SHAP
  estabilidad_top10       en cuántas de las 5 repeticiones la feature queda en el top-10 de su rama
  auc_univariado, q_fdr   la feature sola contra la etiqueta (Mann-Whitney, FDR por rama):
                          control independiente del modelo

Uso:  python Multimodal_XAI/resumen_shap.py
"""
import pickle

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

import config as C
import ramas as R


def resumen(dx):
    pids, y, ram = R.cargar(dx)
    dest = C.OUT / dx
    S = pd.read_csv(dest / f"shap_matrix_{dx}.csv").set_index("subject_id")
    L = pd.read_csv(dest / f"lime_matrix_{dx}.csv").set_index("subject_id")
    comb = pd.read_csv(dest / f"combinaciones_{dx}.csv").set_index("combinacion")
    res = [pickle.load(open(f, "rb")) for f in sorted((dest / "cache").glob("r*_f*.pkl"))]
    yser = pd.Series(y, index=pids)
    filas = []
    for b in [r for r in R.INFO if R.INFO[r]["interpretable"]]:
        cols = [c for c in S.columns if c.startswith(f"{b}__")]
        # estabilidad: top-10 de |SHAP| por repetición
        top_por_rep = []
        for rep in sorted({r["repeat"] for r in res}):
            partes = [pd.DataFrame(r["shap"][b], index=r["filas_expl"][b], columns=ram[b].columns)
                      for r in res if r["repeat"] == rep and b in r["shap"]]
            imp = pd.concat(partes).abs().mean()
            top_por_rep.append(set(imp.sort_values(ascending=False).head(10).index))
        pvals = []
        for c in cols:
            f = c.split("__", 1)[1]
            v = ram[b][f].reindex(S.index)
            s = S[c]
            m = v.notna() & s.notna()
            if m.sum() < 10 or v[m].std() == 0:
                continue
            yy = yser.reindex(S.index)[m]
            rho = spearmanr(v[m], s[m])[0]
            filas.append({"rama": b, "auc_rama": comb.loc[b, "auc_media"], "feature": f,
                          "importancia": s.abs().mean(), "direccion": rho,
                          "lime_concuerda": bool(np.sign(L[c].mean()) == np.sign(rho)),
                          "estabilidad_top10": sum(f in t for t in top_por_rep),
                          "auc_univariado": roc_auc_score(yy, v[m]),
                          "p_mw": mannwhitneyu(v[m][yy == 1], v[m][yy == 0]).pvalue})
            pvals.append(len(filas) - 1)
        if pvals:
            q = false_discovery_control([filas[i]["p_mw"] for i in pvals])
            for i, qi in zip(pvals, q):
                filas[i]["q_fdr"] = qi
    T = pd.DataFrame(filas).sort_values(["auc_rama", "importancia"], ascending=[False, False])
    T.round(4).to_csv(dest / f"shap_resumen_{dx}.csv", index=False)
    print(f"\n=== {dx.upper()} ===")
    for b, g in T.groupby("rama", sort=False):
        print(f"-- {b} (AUC rama {g.auc_rama.iloc[0]:.3f})")
        print(g.head(8)[["feature", "importancia", "direccion", "lime_concuerda", "estabilidad_top10",
                         "auc_univariado", "q_fdr"]].round(3).to_string(index=False))
    return T


if __name__ == "__main__":
    for dx in C.DXS:
        resumen(dx)
