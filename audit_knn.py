"""
Audyt pewnościowy wyników k-NN (BEZ treningu sieci, sam sklearn).

Cel: potwierdzić, że raportowane PR-AUC k-NN (0,548 na zbiorze GNN, 33% poz.)
jest (1) reprodukowalne tym samym pipeline'em co praca, (2) realnym sygnałem,
a nie strukturalnym artefaktem/przeciekiem.

Testy:
  R)  Reprodukcja: dokładnie jak evaluate_on_gnn_testset.py (alpha+beta+mhc_flag).
  P)  Permutacja etykiet train (×N ziaren) -> oczekiwany spad do ~pos_rate.
  M)  Ablacja flagi MHC -> ile wnosi MHC.
  C)  Tylko-CDR3a / tylko-CDR3b -> który łańcuch niesie sygnał.

Uruchom z /home/tarnickil/MGR :  conda run -n mgr_thesis python audit_knn.py
"""
import os, glob, json
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

TEST_GRAPHS_DIR = "Graphs/Test_graphs"
N_PERM = 10
SEED = 42
OUT = "Wyniki_Audyt"


def load_gnn_test_names(graphs_dir):
    names = []
    for f in sorted(glob.glob(os.path.join(graphs_dir, "*.pt"))):
        rel = torch.load(f, weights_only=False)["target_relation"]
        names.append(rel["Name"])
    return set(names)


def build_emap():
    emb = pd.read_pickle("graph_ready_data/tcr_embeddings.pkl")
    a = {n: np.asarray(x, dtype=np.float32)
         for n, x in zip(emb["Name"], emb["Embeddings_alpha"])}
    b = {n: np.asarray(x, dtype=np.float32)
         for n, x in zip(emb["Name"], emb["Embeddings_beta"])}
    return a, b


def features(df, a_map, b_map, most_freq_allele, use_mhc=True, chain="both"):
    rows, y = [], []
    for tcr, mhc, lab in zip(df["TCR_name"], df["HLA_MHC"], df["Binding"]):
        a, b = a_map[tcr], b_map[tcr]
        if chain == "both":
            v = np.concatenate([a, b])
        elif chain == "alpha":
            v = a
        else:
            v = b
        if use_mhc:
            v = np.concatenate([v, [1.0 if mhc == most_freq_allele else 0.0]])
        rows.append(v)
        y.append(int(lab))
    return np.asarray(rows), np.asarray(y)


def fit_knn(Xtr, ytr, Xte):
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=-1)
    knn.fit(Xtr, ytr)
    return knn.predict_proba(Xte)[:, 1]


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_pickle("training_ready_data/Train_data.pkl")
    te = pd.read_pickle("training_ready_data/Test.data.pkl")
    a_map, b_map = build_emap()
    most_freq = tr["HLA_MHC"].value_counts().index[0]   # == list(mhc_id.keys())[0]
    print(f"Najczęstszy allel (mhc_flag=1): {most_freq}")

    print("Wczytywanie nazw podgrafów testowych GNN ...")
    gnn_names = load_gnn_test_names(TEST_GRAPHS_DIR)
    te_sub = te[te["Name"].isin(gnn_names)].reset_index(drop=True)
    pos_rate = float(te_sub["Binding"].mean())
    print(f"Zbiór testowy GNN: n={len(te_sub)}, pos={pos_rate*100:.1f}%")

    results = {"pos_rate": pos_rate, "n_test": len(te_sub)}

    # ---- R) Reprodukcja (both + mhc) ----
    Xtr, ytr = features(tr, a_map, b_map, most_freq, use_mhc=True, chain="both")
    Xte, yte = features(te_sub, a_map, b_map, most_freq, use_mhc=True, chain="both")
    p = fit_knn(Xtr, ytr, Xte)
    results["reproduction"] = {
        "pr_auc": float(average_precision_score(yte, p)),
        "roc_auc": float(roc_auc_score(yte, p)),
        "expected_pr_auc_paper": 0.5477,
    }
    print(f"\n[R] Reprodukcja: PR-AUC={results['reproduction']['pr_auc']:.4f} "
          f"ROC={results['reproduction']['roc_auc']:.4f}  (praca: 0,5477)")

    # ---- P) Permutacja etykiet train ----
    rng = np.random.default_rng(SEED)
    perm_pr, perm_roc = [], []
    for i in range(N_PERM):
        ysh = ytr.copy()
        rng.shuffle(ysh)
        pp = fit_knn(Xtr, ysh, Xte)
        perm_pr.append(average_precision_score(yte, pp))
        perm_roc.append(roc_auc_score(yte, pp))
    results["label_permutation"] = {
        "n": N_PERM,
        "pr_auc_mean": float(np.mean(perm_pr)), "pr_auc_sd": float(np.std(perm_pr)),
        "roc_auc_mean": float(np.mean(perm_roc)), "roc_auc_sd": float(np.std(perm_roc)),
        "pr_baseline": pos_rate, "roc_baseline": 0.5,
    }
    lp = results["label_permutation"]
    print(f"[P] Permutacja (×{N_PERM}): PR-AUC={lp['pr_auc_mean']:.4f}±{lp['pr_auc_sd']:.4f} "
          f"(baseline {pos_rate:.3f}) | ROC={lp['roc_auc_mean']:.4f}±{lp['roc_auc_sd']:.4f} (baseline 0,5)")

    # ---- M) Ablacja flagi MHC ----
    Xtr_n, _ = features(tr, a_map, b_map, most_freq, use_mhc=False, chain="both")
    Xte_n, _ = features(te_sub, a_map, b_map, most_freq, use_mhc=False, chain="both")
    p = fit_knn(Xtr_n, ytr, Xte_n)
    results["no_mhc_flag"] = {"pr_auc": float(average_precision_score(yte, p)),
                              "roc_auc": float(roc_auc_score(yte, p))}
    print(f"[M] Bez flagi MHC: PR-AUC={results['no_mhc_flag']['pr_auc']:.4f} "
          f"ROC={results['no_mhc_flag']['roc_auc']:.4f}")

    # ---- C) Łańcuchy ----
    for ch in ("alpha", "beta"):
        Xtr_c, _ = features(tr, a_map, b_map, most_freq, use_mhc=True, chain=ch)
        Xte_c, _ = features(te_sub, a_map, b_map, most_freq, use_mhc=True, chain=ch)
        p = fit_knn(Xtr_c, ytr, Xte_c)
        results[f"chain_{ch}"] = {"pr_auc": float(average_precision_score(yte, p)),
                                  "roc_auc": float(roc_auc_score(yte, p))}
        print(f"[C] Tylko {ch}: PR-AUC={results[f'chain_{ch}']['pr_auc']:.4f} "
              f"ROC={results[f'chain_{ch}']['roc_auc']:.4f}")

    with open(os.path.join(OUT, "knn_controls.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nZapisano: {OUT}/knn_controls.json")


if __name__ == "__main__":
    main()
