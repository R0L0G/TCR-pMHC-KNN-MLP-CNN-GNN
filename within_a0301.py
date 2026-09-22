"""
Ewaluacja w obrębie samego HLA-A0301 (allel prezentujący IE-1 CMV).
Flaga MHC stała -> liczy się wyłącznie tożsamość TCR. Bazowy odsetek pozytywów
~53%, więc losowy klasyfikator ma PR-AUC ~0,53 i ROC 0,5.

Predykcje MLP/CNN/GNN: z Wyniki_Diff/aligned_predictions.npz (wyrównane po Name).
Predykcja k-NN: doliczana tu (konfiguracja jak w pracy: train=pełny, oba łańcuchy
+ flaga, k=10 cosine), wyrównana po Name do tej samej kolejności.

Uruchom: conda run -n mgr_thesis python within_a0301.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ALLELE = "HLA-A0301"
OUT = "Wyniki_Audyt/within_a0301.json"


def bootstrap_ci(y, p, fn, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    sc = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        sc.append(fn(y[idx], p[idx]))
    return float(fn(y, p)), float(np.percentile(sc, 2.5)), float(np.percentile(sc, 97.5))


def main():
    te = pd.read_pickle("training_ready_data/Test.data.pkl")
    tr = pd.read_pickle("training_ready_data/Train_data.pkl")
    d = np.load("Wyniki_Diff/aligned_predictions.npz", allow_pickle=True)
    names = d["names"]
    y = d["y"].astype(int)
    preds = {"MLP": d["p_mlp"], "CNN": d["p_cnn"], "GNN": d["p_gnn"]}

    # Name -> allele
    name2allele = dict(zip(te["Name"], te["HLA_MHC"]))
    alleles = np.array([name2allele[n] for n in names])

    # ---- k-NN doliczony, wyrównany po Name ----
    emb = pd.read_pickle("graph_ready_data/tcr_embeddings.pkl")
    amap = {n: np.asarray(a, np.float32) for n, a in zip(emb["Name"], emb["Embeddings_alpha"])}
    bmap = {n: np.asarray(b, np.float32) for n, b in zip(emb["Name"], emb["Embeddings_beta"])}
    most_freq = tr["HLA_MHC"].value_counts().index[0]

    def feats(df):
        X, yy = [], []
        for t, m, lab in zip(df["TCR_name"], df["HLA_MHC"], df["Binding"]):
            f = 1.0 if m == most_freq else 0.0
            X.append(np.concatenate([amap[t], bmap[t], [f]]))
            yy.append(int(lab))
        return np.asarray(X), np.asarray(yy)

    Xtr, ytr = feats(tr)
    name2row = {n: (t, m) for n, t, m in zip(te["Name"], te["TCR_name"], te["HLA_MHC"])}
    te_sub = pd.DataFrame({
        "Name": names,
        "TCR_name": [name2row[n][0] for n in names],
        "HLA_MHC": [name2row[n][1] for n in names],
        "Binding": y,
    })
    Xte, _ = feats(te_sub)
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=-1)
    knn.fit(Xtr, ytr)
    preds["k-NN"] = knn.predict_proba(Xte)[:, 1]

    # ---- maska A0301 ----
    mask = alleles == ALLELE
    yA = y[mask]
    base = float(yA.mean())
    print(f"W obrębie {ALLELE}: n={mask.sum()}, pozytywów={yA.sum()} ({base*100:.1f}%)")
    print(f"Baseline losowy: PR-AUC={base:.3f}  ROC=0,500\n")
    print(f"{'Model':8} {'PR-AUC [95% CI]':28} {'ROC-AUC [95% CI]':28} {'ΔPR vs los.'}")

    out = {"allele": ALLELE, "n": int(mask.sum()), "n_pos": int(yA.sum()),
           "pos_rate": base, "models": {}}
    for name in ["k-NN", "MLP", "CNN", "GNN"]:
        pA = preds[name][mask]
        pr = bootstrap_ci(yA, pA, average_precision_score)
        rc = bootstrap_ci(yA, pA, roc_auc_score)
        out["models"][name] = {"pr_auc": pr, "roc_auc": rc, "delta_pr_vs_random": pr[0] - base}
        print(f"{name:8} {pr[0]:.3f} [{pr[1]:.3f}; {pr[2]:.3f}]      "
              f"{rc[0]:.3f} [{rc[1]:.3f}; {rc[2]:.3f}]      {pr[0]-base:+.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nZapisano: {OUT}")


if __name__ == "__main__":
    main()
