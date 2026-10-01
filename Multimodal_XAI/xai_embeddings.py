"""Explicabilidad de los modelos de EMBEDDINGS (ansiedad y depresión), según el estado del
arte revisado en Multimodal/experimento_embeddings/ESTADO_DEL_ARTE_XAI_EMBEDDINGS.md.

MÉTODO 3, probing dirigido a la tarea (Dixit et al. 2024, arXiv:2409.09511):
  1. Importancia de cada dimensión del embedding PARA EL CLASIFICADOR (|coef| de una
     logística L2 sobre dimensiones estandarizadas, promediada en 16 folds; la C se elige
     en CV interna).
  2. Probes Ridge que predicen cada feature interpretable desde (a) todas las dimensiones,
     (b) las k=50 más importantes y (c) 50 al azar (línea base, 20 sorteos).
  3. Si una feature se predice casi igual de bien con las 50 importantes que con todas, y
     claramente mejor que con 50 al azar, esa propiedad está codificada en las dimensiones
     que usa el clasificador.
  Ramas: voz_microventanas y voz_densa -> 88 eGeMAPS (por categoría: frecuencia, energía,
  espectral, temporal); texto_emb -> 7 features léxicas.

MÉTODO 5, atribución temporal por oclusión + transcripción:
  Para cada participante, con el modelo del fold en que NO estuvo (16 folds, la configuración
  se elige en CV interna), se quita cada fragmento de la entrevista (fragmento = segmento
  con marca de tiempo de Whisper) y se mide cuánto cambia su probabilidad de riesgo:
    contribución = p(completo) - p(sin el fragmento).
  Ramas: voz_microventanas (micro-ventana k en t = 2.5*k s), egemaps (segmento k en
  t = recorte_inicial + 2.5*k s) y texto_emb (embedding de cada fragmento).

Salidas en stacking_output/<eje>/ (gitignored; las transcripciones son dato clínico):
  probing_embeddings_<eje>.csv, probing_resumen_<eje>.csv,
  atribucion_temporal_<eje>.csv, atribucion_temporal_<eje>.md
Uso:  python Multimodal_XAI/xai_embeddings.py --metodo probing|temporal|ambos --dx depresion|ansiedad|all
"""
import argparse
import json
import warnings

import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import ramas as R
from xai_por_muestra import elegir

warnings.filterwarnings("ignore")
TRANS = C.DATA / "features" / "transcripciones"
AUDIO_ED = C.DATA / "features" / "audio_editado_denoised"
FRAG_NPZ = C.DATA / "features" / "texto_fragmentos.npz"
K_TOP, N_RAND = 50, 20


# =========================================================================== método 3
def categoria_egemaps(f):
    """Categorías de eGeMAPS (Eyben et al. 2016), las mismas que usa Dixit et al. 2024."""
    g = f.lower()
    if any(k in g for k in ("voicedsegmentspersec", "voicedsegmentlength", "unvoicedsegmentlength",
                            "loudnesspeakspersec")):
        return "temporal"
    if any(k in g for k in ("f0semitone", "jitter", "frequency_sma3nz", "bandwidth")):
        return "frecuencia"
    if any(k in g for k in ("loudness", "shimmer", "hnr", "equivalentsoundlevel")):
        return "energia"
    return "espectral"


def importancia_dimensiones(X, y):
    imp, oof = np.zeros(X.shape[1]), np.zeros(len(y))
    for tr, te in StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED).split(X, y):
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
                          {"logisticregression__C": [1e-3, 1e-2, 1e-1]},
                          cv=StratifiedKFold(C.N_INNER, shuffle=True, random_state=C.SEED), scoring="roc_auc").fit(X[tr], y[tr])
        imp += np.abs(gs.best_estimator_[-1].coef_[0])
        oof[te] = gs.predict_proba(X[te])[:, 1]
    return imp / C.N_SPLITS, roc_auc_score(y, oof)


def r2_probe(X, t):
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))
    return r2_score(t, cross_val_predict(pipe, X, t, cv=KFold(5, shuffle=True, random_state=C.SEED)))


def probing(dx):
    pids, y, ram = R.cargar(dx)
    rng = np.random.default_rng(C.SEED)
    filas, resumen = [], []
    for emb, ref in (("voz_microventanas", "egemaps"), ("voz_densa", "egemaps"), ("texto_emb", "texto_lexico")):
        ok = ram[emb].notna().any(axis=1).values & ram[ref].notna().any(axis=1).values
        X, T, yy = ram[emb].values[ok], ram[ref][ok], y[ok]
        imp, auc = importancia_dimensiones(X, yy)
        top = np.argsort(imp)[::-1][:K_TOP]
        rands = [rng.choice(X.shape[1], K_TOP, replace=False) for _ in range(N_RAND)]
        for f in T.columns:
            t = T[f].values.astype(float)
            if np.nanstd(t) == 0:
                continue
            r_all, r_top = r2_probe(X, t), r2_probe(X[:, top], t)
            r_rand = float(np.mean([r2_probe(X[:, r], t) for r in rands]))
            filas.append({"rama_embedding": emb, "feature": f,
                          "categoria": categoria_egemaps(f) if ref == "egemaps" else "lexico",
                          "r2_todas": r_all, "r2_top50": r_top, "r2_azar50": r_rand,
                          "ratio_top_vs_todas": r_top / r_all if r_all > 0 else np.nan,
                          "ganancia_top_vs_azar": r_top - r_rand})
        resumen.append({"rama_embedding": emb, "auc_clasificador_logistico_oof": round(auc, 3),
                        "n": int(ok.sum())})
        print(f"[{dx}] {emb}: AUC clasificador (logística, OOF) = {auc:.3f}", flush=True)
    P = pd.DataFrame(filas)
    dest = C.OUT / dx
    P.round(4).to_csv(dest / f"probing_embeddings_{dx}.csv", index=False)
    # resumen por categoría, solo con features bien codificadas por el embedding (R2 todas >= 0.2)
    bien = P[P.r2_todas >= 0.2]
    cat = bien.groupby(["rama_embedding", "categoria"]).agg(
        n_features=("feature", "size"), r2_todas=("r2_todas", "mean"), r2_top50=("r2_top50", "mean"),
        r2_azar50=("r2_azar50", "mean"), ratio=("ratio_top_vs_todas", "mean"),
        ganancia_vs_azar=("ganancia_top_vs_azar", "mean")).reset_index()
    cat = cat.merge(pd.DataFrame(resumen), on="rama_embedding")
    cat.round(3).to_csv(dest / f"probing_resumen_{dx}.csv", index=False)
    print(cat.round(3).to_string(index=False))
    top_feats = bien.sort_values("ganancia_top_vs_azar", ascending=False).groupby("rama_embedding").head(6)
    print("Features más ligadas a las dimensiones que usa el clasificador:")
    print(top_feats[["rama_embedding", "feature", "categoria", "r2_todas", "r2_top50", "r2_azar50"]]
          .round(3).to_string(index=False), flush=True)


# =========================================================================== método 5
def fragmentos(pid):
    d = json.loads((TRANS / f"{pid:03d}.json").read_text(encoding="utf-8"))
    out = []
    for i, f in enumerate(d["fragmentos"]):
        ini = f["ini"] if f["ini"] is not None else 0.0
        fin = f["fin"] if f["fin"] is not None else d["duracion_s"]
        if f["texto"]:
            out.append((i, float(ini), float(fin), f["texto"]))
    return out, d["audio"]


def embeddings_fragmentos(pids):
    """Embedding (mpnet multilingüe, normalizado) de cada fragmento; la media por participante
    reproduce la rama texto_emb (verificado abajo)."""
    if FRAG_NPZ.exists():
        z = np.load(FRAG_NPZ, allow_pickle=True)
        return {int(k): z[k] for k in z.files}
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cuda")
    out = {}
    for p in pids:
        frag, _ = fragmentos(p)
        out[p] = st.encode([t for _, _, _, t in frag], batch_size=32, normalize_embeddings=True)
    np.savez_compressed(FRAG_NPZ, **{str(k): v for k, v in out.items()})
    return out


def unidades_temporales(pids, dx):
    """Por rama: {pid: (tiempos (n,), vectores (n, d))} de las unidades que se promedian."""
    zr = np.load(C.R_NPZ)
    voz = {p: (np.arange(len(zr[f"{p:03d}"])) * 2.5, zr[f"{p:03d}"]) for p in pids if f"{p:03d}" in zr.files}
    eg = pd.read_csv(C.EGEMAPS[dx])
    eg["pid"] = eg["audio_id"].astype(str).str[:3].astype(int)
    cols = [c for c in eg.columns if c not in ("audio_id", "seg_idx", "label", "pid")]
    ege = {}
    for p, g in eg.groupby("pid"):
        w = AUDIO_ED / f"{p:03d}_recortado_denoised.wav"
        if not w.exists():
            continue
        yv, _ = librosa.load(str(w), sr=16000, mono=True)
        yv = np.append(yv[0], yv[1:] - 0.97 * yv[:-1])          # mismo preprocesamiento que el entrenamiento
        _, idx = librosa.effects.trim(yv, top_db=20)
        g = g.sort_values("seg_idx")
        ege[p] = (idx[0] / 16000 + g["seg_idx"].values * 2.5, g[cols].values.astype(float))
    return {"voz_microventanas": voz, "egemaps": ege}


def temporal(dx):
    pids, y, ram = R.cargar(dx)
    frag_emb = embeddings_fragmentos(pids)
    tm = np.stack([frag_emb[p].mean(0) for p in pids])
    print(f"[{dx}] verificación: media de fragmentos == rama texto_emb -> "
          f"{np.allclose(tm, ram['texto_emb'].values, atol=1e-4)}", flush=True)
    unid = unidades_temporales(pids, dx)
    unid["texto_emb"] = {p: (np.array([ini for _, ini, _, _ in fragmentos(p)[0]]), frag_emb[p]) for p in pids}
    filas = []
    for tr, te in StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED).split(pids, y):
        for b in ("voz_microventanas", "egemaps", "texto_emb"):
            X = ram[b]
            disp = X.notna().any(axis=1).values
            trb = tr[disp[tr]]
            _, est, _ = elegir(X.values[trb].astype(float), y[trb], R.configs(R.INFO[b]["tipo"]))
            for i in te:
                p = pids[i]
                if p not in unid[b]:
                    continue
                t_u, V = unid[b][p]
                frag, tipo_audio = fragmentos(p)
                p_full = est.predict_proba(V.mean(0, keepdims=True))[0, 1]
                for j, (fi, ini, fin, texto) in enumerate(frag):
                    if b == "texto_emb":
                        dentro = np.arange(len(frag)) == j
                    else:
                        dentro = (t_u >= ini) & (t_u < fin)
                    if dentro.sum() == 0 or dentro.all():
                        contrib = 0.0
                    else:
                        contrib = p_full - est.predict_proba(V[~dentro].mean(0, keepdims=True))[0, 1]
                    filas.append({"subject_id": p, "y_true": int(y[i]), "rama": b, "p_participante": p_full,
                                  "fragmento": fi, "ini_s": round(ini, 1), "fin_s": round(fin, 1),
                                  "unidades_en_fragmento": int(dentro.sum()), "contribucion": contrib,
                                  "audio_transcrito": tipo_audio, "texto": texto})
    A = pd.DataFrame(filas)
    dest = C.OUT / dx
    A.round(4).to_csv(dest / f"atribucion_temporal_{dx}.csv", index=False)
    escribir_resumen_temporal(dx, A, dest)


def escribir_resumen_temporal(dx, A, dest):
    """Ejemplos legibles: los 5 positivos con mayor riesgo OOF promedio de las 3 ramas, con los
    3 fragmentos que más SUBIERON el riesgo (suma de contribuciones de las ramas)."""
    piv = A.pivot_table(index=["subject_id", "fragmento"], columns="rama", values="contribucion").fillna(0)
    piv["total"] = piv.sum(axis=1)
    meta = A.drop_duplicates(["subject_id", "fragmento"]).set_index(["subject_id", "fragmento"])
    riesgo = A.drop_duplicates(["subject_id", "rama"]).groupby("subject_id").agg(
        p=("p_participante", "mean"), y=("y_true", "first"))
    lineas = [f"# Atribución temporal ({dx}): qué momentos de la entrevista movieron el riesgo", "",
              "Modelos que no vieron al participante (16 folds). Contribución = p(completo) - p(sin el fragmento), "
              "sumada en las ramas voz_microventanas, egemaps y texto_emb. **Contiene fragmentos de entrevistas: "
              "dato clínico, no compartir.**", ""]
    for pid in riesgo[riesgo.y == 1].sort_values("p", ascending=False).head(5).index:
        lineas.append(f"## Participante {pid} — riesgo medio OOF {riesgo.loc[pid, 'p']:.2f} (positivo en 'DX IA')")
        sub = piv.loc[pid].sort_values("total", ascending=False).head(3)
        for frag, fila in sub.iterrows():
            m = meta.loc[(pid, frag)]
            txt = m["texto"] if len(m["texto"]) <= 220 else m["texto"][:220] + "…"
            partes = ", ".join(f"{r}={fila[r]:+.3f}" for r in ("voz_microventanas", "egemaps", "texto_emb") if r in fila)
            lineas.append(f"- **{m['ini_s']:.0f}–{m['fin_s']:.0f} s** (total {fila['total']:+.3f}; {partes}): «{txt}»")
        lineas.append("")
    (dest / f"atribucion_temporal_{dx}.md").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas[:20]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metodo", choices=["probing", "temporal", "ambos"], default="ambos")
    ap.add_argument("--dx", choices=list(C.DXS) + ["all"], default="all")
    a = ap.parse_args()
    for dx in (C.DXS if a.dx == "all" else [a.dx]):
        (C.OUT / dx).mkdir(parents=True, exist_ok=True)
        if a.metodo in ("probing", "ambos"):
            probing(dx)
        if a.metodo in ("temporal", "ambos"):
            temporal(dx)


if __name__ == "__main__":
    main()
