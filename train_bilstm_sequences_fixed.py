"""
train_bilstm_sequences_fixed.py
Consume las secuencias frame-a-frame y entrena/compara 5 arquitecturas.
División de 3 vías (FIT/CALIB/VAL) para eliminar data leakage.
Métricas: AUC y F1-macro con threshold=0.5 (fijo).
"""
from __future__ import annotations
import argparse
import glob
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

NON_FEATURE_COLS = {"video_id", "frame_order", "timestamp_ms", "frame", "approx_time", "input"}

def _normalize_video_id(vid: str) -> str:
    vid = str(vid).strip()
    try:
        return str(int(float(vid)))
    except (ValueError, TypeError):
        return vid

def load_sequences(sequences_dir: str) -> dict[str, pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(sequences_dir, "*.parquet")) + glob.glob(os.path.join(sequences_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No se encontraron secuencias en '{sequences_dir}'.")
    sequences = {}
    for f in files:
        video_id = _normalize_video_id(os.path.splitext(os.path.basename(f))[0])
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        sequences[video_id] = df
    return sequences

def list_candidate_columns(sequences: dict[str, pd.DataFrame]) -> list[str]:
    all_df = pd.concat(sequences.values(), ignore_index=True)
    candidate_cols = [c for c in all_df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in candidate_cols if pd.to_numeric(all_df[c], errors="coerce").notna().mean() > 0]
    if not numeric_cols:
        raise ValueError("Ninguna columna numérica encontrada; revisa las secuencias generadas.")
    return numeric_cols

def filter_high_nan_columns(train_concat: pd.DataFrame, candidate_cols: list[str], max_nan_frac: float = 0.5) -> list[str]:
    keep = [c for c in candidate_cols if train_concat[c].isna().mean() <= max_nan_frac]
    if not keep:
        raise ValueError("Ninguna columna pasó el filtro de NaNs en este fold.")
    return keep

class SequenceDataset(Dataset):
    def __init__(self, video_ids, sequences, feature_cols, labels, max_seq_len):
        self.video_ids = video_ids
        self.sequences = sequences
        self.feature_cols = feature_cols
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        df = self.sequences[vid][self.feature_cols]
        arr = df.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        if len(arr) > self.max_seq_len:
            idxs = np.linspace(0, len(arr) - 1, self.max_seq_len).astype(int)
            arr = arr[idxs]
        return torch.from_numpy(arr), self.labels[vid]

def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 24, dropout: float = 0.5):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(packed)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(self.dropout(h_cat)).squeeze(-1)

class BiGRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 24, dropout: float = 0.5):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(self.dropout(h_cat)).squeeze(-1)

class CNN1DClassifier(nn.Module):
    def __init__(self, n_features: int, n_filters: int = 24, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n_filters, 1)

    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None].to(x.device)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(dim=1) / lengths.to(x.device).unsqueeze(-1).clamp(min=1)
        return self.head(self.dropout(pooled)).squeeze(-1)

class CNNBiGRUClassifier(nn.Module):
    def __init__(self, n_features: int, n_filters: int = 24, hidden_size: int = 24, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Conv1d(n_features, n_filters, kernel_size=5, padding=2)
        self.rnn = nn.GRU(n_filters, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = x.transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(self.dropout(h_cat)).squeeze(-1)

class MLPMeanPoolClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 24, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None].to(x.device)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(x.device).unsqueeze(-1).clamp(min=1)
        h = torch.relu(self.fc1(pooled))
        return self.head(self.dropout(h)).squeeze(-1)

ARCHITECTURES = {
    "bilstm": BiLSTMClassifier,
    "bigru": BiGRUClassifier,
    "cnn1d": CNN1DClassifier,
    "cnn_bigru": CNNBiGRUClassifier,
    "mlp_meanpool": MLPMeanPoolClassifier,
}

def build_model(architecture: str, n_features: int, hidden_size: int, dropout: float = 0.5) -> nn.Module:
    cls = ARCHITECTURES[architecture]
    if architecture == "cnn1d":
        return cls(n_features, n_filters=hidden_size, dropout=dropout)
    if architecture == "cnn_bigru":
        return cls(n_features, n_filters=hidden_size, hidden_size=hidden_size, dropout=dropout)
    return cls(n_features, hidden_size=hidden_size, dropout=dropout)

def train_one_fold(model, fit_loader, calib_loader, val_loader, device, max_epochs=60, patience=8, lr=1e-3, weight_decay=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_calib_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        for x, lengths, y in fit_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        calib_losses = []
        with torch.no_grad():
            for x, lengths, y in calib_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                calib_losses.append(criterion(logits, y).item())
        
        calib_loss = float(np.mean(calib_losses))
        
        if calib_loss < best_calib_loss:
            best_calib_loss = calib_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, lengths, y in val_loader:
            x = x.to(device)
            logits = model(x, lengths)
            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(y)
    
    y_pred = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    
    return y_true, y_pred, model, best_calib_loss, epoch + 1

def run_cv(sequences_dir, labels_csv, target, max_seq_len, n_splits, n_repeats,
           hidden_size, batch_size, seed, architecture, patience, max_epochs,
           calib_frac=0.25, precomputed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if precomputed is None:
        sequences = load_sequences(sequences_dir)
        candidate_cols = list_candidate_columns(sequences)
        print(f"{len(candidate_cols)} columnas candidatas: {candidate_cols}")
        
        labels_df = pd.read_csv(labels_csv)
        labels_df["video_id"] = labels_df["video_id"].apply(_normalize_video_id)
        labels_df = labels_df.dropna(subset=[target])
        label_map = dict(zip(labels_df["video_id"], labels_df[target].astype(int)))
        
        video_ids = [vid for vid in sequences.keys() if vid in label_map]
        missing = set(sequences.keys()) - set(label_map.keys())
        if missing:
            print(f"Aviso: {len(missing)} video_id sin etiqueta, se excluyen: {sorted(missing)}")
        
        y_all = np.array([label_map[v] for v in video_ids])
        print(f"N={len(video_ids)} videos, distribución: {dict(zip(*np.unique(y_all, return_counts=True)))}")
        
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        folds = list(rskf.split(video_ids, y_all))
        
        precomputed = {
            "sequences": sequences, "candidate_cols": candidate_cols, "label_map": label_map,
            "video_ids": video_ids, "y_all": y_all, "folds": folds,
        }
    
    sequences = precomputed["sequences"]
    candidate_cols = precomputed["candidate_cols"]
    label_map = precomputed["label_map"]
    video_ids = precomputed["video_ids"]
    folds = precomputed["folds"]
    
    fold_aucs = []
    fold_f1_macro = []
    detailed_records = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        set_global_seed(seed + fold_idx)
        
        train_ids = [video_ids[i] for i in train_idx]
        val_ids = [video_ids[i] for i in val_idx]
        y_train = np.array([label_map[vid] for vid in train_ids])
        
        try:
            fit_ids, calib_ids = train_test_split(
                train_ids, test_size=calib_frac, stratify=y_train, random_state=seed + fold_idx
            )
            y_calib = np.array([label_map[vid] for vid in calib_ids])
            if len(np.unique(y_calib)) < 2 or len(calib_ids) < 4:
                raise ValueError("calib set insuficiente")
            fallback = False
        except Exception:
            fit_ids = train_ids
            calib_ids = []
            fallback = True
            print(f"[{architecture}] Fold {fold_idx}: WARNING - fallback (sin calib estratificado)")
        
        fit_concat = pd.concat([sequences[v][candidate_cols].apply(pd.to_numeric, errors="coerce") for v in fit_ids], ignore_index=True)
        feature_cols = filter_high_nan_columns(fit_concat, candidate_cols)
        
        if fold_idx == 0:
            dropped = set(candidate_cols) - set(feature_cols)
            if dropped:
                print(f"[{architecture}] Fold 0: descartadas por NaN: {sorted(dropped)}")
        
        fit_concat = fit_concat[feature_cols]
        scaler = StandardScaler().fit(fit_concat.fillna(fit_concat.mean()))
        
        scaled_sequences = {}
        for v in fit_ids + calib_ids + val_ids:
            df = sequences[v][feature_cols].apply(pd.to_numeric, errors="coerce")
            df = df.fillna(fit_concat.mean())
            scaled = pd.DataFrame(scaler.transform(df), columns=feature_cols)
            scaled_sequences[v] = scaled
        
        fit_ds = SequenceDataset(fit_ids, scaled_sequences, feature_cols, label_map, max_seq_len)
        calib_ds = SequenceDataset(calib_ids, scaled_sequences, feature_cols, label_map, max_seq_len) if not fallback else None
        val_ds = SequenceDataset(val_ids, scaled_sequences, feature_cols, label_map, max_seq_len)
        
        fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if calib_ds else fit_loader
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        model = build_model(architecture, n_features=len(feature_cols), hidden_size=hidden_size).to(device)
        
        y_true, y_pred, _, best_calib_loss, epochs_used = train_one_fold(
            model, fit_loader, calib_loader, val_loader, device, max_epochs=max_epochs, patience=patience
        )
        
        if len(np.unique(y_true)) < 2:
            print(f"[{architecture}] Fold {fold_idx}: val con una sola clase, se omite.")
            continue
        
        # ── MÉTRICAS ──────────────────────────────────────────────────
        auc = roc_auc_score(y_true, y_pred)
        pred_05 = (y_pred >= 0.5).astype(int)
        f1_macro = f1_score(y_true, pred_05, average="macro", zero_division=0)
        
        fold_aucs.append(auc)
        fold_f1_macro.append(f1_macro)
        
        print(f"[{architecture}] Fold {fold_idx}: AUC={auc:.3f} | F1-macro@0.5={f1_macro:.3f} | "
              f"n_fit={len(fit_ids)}, n_calib={len(calib_ids)}, n_val={len(val_ids)}")
        
        detailed_records.append({
            "architecture": architecture, "fold": fold_idx,
            "n_fit": len(fit_ids), "n_calib": len(calib_ids), "n_val": len(val_ids),
            "n_features": len(feature_cols),
            "auc": auc, "f1_macro_thr05": f1_macro,
            "best_calib_loss": best_calib_loss, "epochs_used": epochs_used,
        })
    
    n_eff = len(fold_aucs)
    print(f"[{architecture}] AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f} ({n_eff} folds)")
    print(f"[{architecture}] F1-macro@0.5: {np.mean(fold_f1_macro):.3f} ± {np.std(fold_f1_macro):.3f}\n")
    
    return fold_aucs, precomputed, detailed_records

def run_all_architectures(sequences_dir, labels_csv, target, max_seq_len, n_splits,
                          n_repeats, hidden_size, batch_size, seed, patience, max_epochs,
                          calib_frac=0.25):
    results = {}
    precomputed = None
    all_detailed = []
    
    for arch in ARCHITECTURES:
        fold_aucs, precomputed, detailed_records = run_cv(
            sequences_dir, labels_csv, target, max_seq_len, n_splits, n_repeats,
            hidden_size, batch_size, seed, arch, patience, max_epochs, calib_frac, precomputed=precomputed,
        )
        results[arch] = fold_aucs
        all_detailed.extend(detailed_records)
    
    print("=" * 80)
    print(f"RESUMEN COMPARATIVO — target={target}")
    print("=" * 80)
    
    summary_rows = []
    for arch, aucs in results.items():
        arch_details = [d for d in all_detailed if d["architecture"] == arch]
        f1_vals = [d["f1_macro_thr05"] for d in arch_details]
        
        summary_rows.append({
            "architecture": arch,
            "auc_mean": np.mean(aucs) if aucs else float("nan"),
            "auc_std": np.std(aucs) if aucs else float("nan"),
            "f1_macro_thr05_mean": np.mean(f1_vals) if f1_vals else float("nan"),
            "f1_macro_thr05_std": np.std(f1_vals) if f1_vals else float("nan"),
            "n_folds": len(aucs),
        })
        
        print(f"{arch:15s}  AUC={summary_rows[-1]['auc_mean']:.3f}±{summary_rows[-1]['auc_std']:.3f} | "
              f"F1-macro@0.5={summary_rows[-1]['f1_macro_thr05_mean']:.3f}±{summary_rows[-1]['f1_macro_thr05_std']:.3f}  "
              f"({len(aucs)} folds)")
    
    summary_df = pd.DataFrame(summary_rows).sort_values("auc_mean", ascending=False)
    detailed_df = pd.DataFrame(all_detailed)
    
    return summary_df, detailed_df

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena modelos con división 3 vías (FIT/CALIB/VAL). Reporta AUC y F1-macro@0.5.")
    parser.add_argument("--sequences_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--target", type=str, default="target_depresion", choices=["target_ansiedad", "target_depresion"])
    parser.add_argument("--architecture", type=str, default="all", choices=list(ARCHITECTURES.keys()) + ["all"])
    parser.add_argument("--max_seq_len", type=int, default=600)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--calib_frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--detail_csv", type=str, default=None)
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    
    if args.architecture == "all":
        summary_df, detailed_df = run_all_architectures(
            sequences_dir=args.sequences_dir, labels_csv=args.labels_csv, target=args.target,
            max_seq_len=args.max_seq_len, n_splits=args.n_splits, n_repeats=args.n_repeats,
            hidden_size=args.hidden_size, batch_size=args.batch_size, seed=args.seed,
            patience=args.patience, max_epochs=args.max_epochs, calib_frac=args.calib_frac,
        )
        print("\n" + summary_df.to_string(index=False))
        if args.output_csv:
            summary_df.to_csv(args.output_csv, index=False)
            print(f"\nTabla comparativa guardada en '{args.output_csv}'.")
        if args.detail_csv:
            detailed_df.to_csv(args.detail_csv, index=False)
            print(f"Detalle por fold guardado en '{args.detail_csv}'.")
    else:
        _, _, detailed_records = run_cv(
            sequences_dir=args.sequences_dir, labels_csv=args.labels_csv, target=args.target,
            max_seq_len=args.max_seq_len, n_splits=args.n_splits, n_repeats=args.n_repeats,
            hidden_size=args.hidden_size, batch_size=args.batch_size, seed=args.seed,
            architecture=args.architecture, patience=args.patience, max_epochs=args.max_epochs,
            calib_frac=args.calib_frac,
        )
        if args.detail_csv:
            pd.DataFrame(detailed_records).to_csv(args.detail_csv, index=False)
            print(f"Detalle por fold guardado en '{args.detail_csv}'.")