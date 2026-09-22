"""
Inferencja (BEZ treningu) trzech modeli na jednolitym zbiorze testowym GNN
(7707 podgrafów, ~33% pozytywów) i sparowany bootstrap różnic metryk:
    CNN - MLP   oraz   CNN - GNN(L=1)

Wyrównanie predykcji po kolumnie 'Name' (inner join), więc niezależne od
kolejności iteracji loaderów. Surowe predykcje zapisywane do Wyniki_Diff/.

Checkpointy (gotowe, tylko wczytywane):
    best_mlp.pt
    best_cnn.pt
    Wyniki_Ablation/gcn_L1/checkpoint.pt
"""
import os, glob, json
import numpy as np
import pandas as pd
import torch

from model_base import (
    model_MLP, TCRCMVConvNet, TCRCMVGraphNet,
    get_mlp_loader, get_cnn_loader, get_gnn_loader,
    Train_data, Val_data, Test_data,
    tcr_embbedings, cnn_tcr_embeddings, mhc_id,
)
from ablation_gnn import _collect_preds
from evaluate_on_gnn_testset import (
    load_names_from_graphs, filter_data, collect_preds,
)
from sklearn.metrics import average_precision_score, roc_auc_score

DEVICE = "cuda:0"
TEST_GRAPHS_DIR = "Graphs/Test_graphs"
OUT_DIR = "Wyniki_Diff"
N_BOOT = 1000
SEED = 42


def mlp_cnn_predictions():
    """Zwraca DataFrame[Name, y_true, p_mlp, p_cnn] dla zbioru testowego GNN."""
    test_names = load_names_from_graphs(TEST_GRAPHS_DIR)
    test_sub = filter_data(Test_data, test_names)   # zachowuje kolejność Test_data
    names = test_sub["Name"].values

    # MLP
    mlp = model_MLP(in_dim=1282, hidden_dim=256).to(DEVICE)
    mlp.load_state_dict(torch.load("best_mlp.pt", weights_only=True))
    mlp_loader = get_mlp_loader(test_sub, tcr_embbedings, mhc_id,
                                shuffle=False, class_balanced=False, num_workers=0)
    y_mlp, p_mlp = collect_preds(mlp, mlp_loader)

    # CNN
    cnn = TCRCMVConvNet().to(DEVICE)
    cnn.load_state_dict(torch.load("best_cnn.pt", weights_only=True))
    cnn_loader = get_cnn_loader(test_sub, cnn_tcr_embeddings, mhc_id,
                                shuffle=False, class_balanced=False, num_workers=0)
    y_cnn, p_cnn = collect_preds(cnn, cnn_loader)

    assert np.array_equal(y_mlp, y_cnn), "MLP/CNN: niespójne etykiety (różna kolejność)"
    return pd.DataFrame({
        "Name": names, "y_true": y_mlp.astype(int),
        "p_mlp": p_mlp, "p_cnn": p_cnn,
    })


def gnn_predictions():
    """Zwraca DataFrame[Name, y_true_gnn, p_gnn] w kolejności posortowanych plików."""
    files = sorted(glob.glob(os.path.join(TEST_GRAPHS_DIR, "*.pt")))
    names = [torch.load(f, weights_only=False)["target_relation"]["Name"] for f in files]

    gnn = TCRCMVGraphNet(n_gcn_layers=1).to(DEVICE)
    gnn.load_state_dict(torch.load("Wyniki_Ablation/gcn_L1/checkpoint.pt", weights_only=True))
    gnn_loader = get_gnn_loader(TEST_GRAPHS_DIR, shuffle=False,
                                class_balanced=False, num_workers=0)
    y_gnn, p_gnn = _collect_preds(gnn, gnn_loader, DEVICE)

    assert len(names) == len(y_gnn), f"GNN: {len(names)} nazw vs {len(y_gnn)} predykcji"
    return pd.DataFrame({
        "Name": names, "y_true_gnn": y_gnn.astype(int), "p_gnn": p_gnn,
    })


def paired_bootstrap(y, p_a, p_b, metric, n_boot=N_BOOT, seed=SEED):
    """Sparowany bootstrap różnicy metryki: metric(a) - metric(b).
    Ten sam zestaw wylosowanych indeksów dla obu modeli w każdej iteracji."""
    rng = np.random.default_rng(seed)
    n = len(y)
    point = metric(y, p_a) - metric(y, p_b)
    diffs = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.sum() == 0 or ys.sum() == n:   # zdegenerowana próba — pomiń
            continue
        diffs[k] = metric(ys, p_a[idx]) - metric(ys, p_b[idx])
        k += 1
    diffs = diffs[:k]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac_gt0 = float((diffs > 0).mean())
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "p_boot_gt0": frac_gt0, "n_valid": int(k)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("→ Inferencja MLP/CNN ...")
    df_sc = mlp_cnn_predictions()
    print(f"  MLP/CNN: {len(df_sc)} próbek, pos={df_sc['y_true'].mean()*100:.1f}%")

    print("→ Inferencja GNN(L=1) ...")
    df_g = gnn_predictions()
    print(f"  GNN: {len(df_g)} próbek, pos={df_g['y_true_gnn'].mean()*100:.1f}%")

    # Wyrównanie po Name (inner join)
    df = df_sc.merge(df_g, on="Name", how="inner")
    assert (df["y_true"] == df["y_true_gnn"]).all(), "Niespójne etykiety po złączeniu!"
    df = df.drop(columns=["y_true_gnn"])
    print(f"→ Wyrównano po Name: {len(df)} wspólnych próbek")

    df.to_csv(os.path.join(OUT_DIR, "aligned_predictions.csv"), index=False)
    np.savez(os.path.join(OUT_DIR, "aligned_predictions.npz"),
             y=df["y_true"].values, p_mlp=df["p_mlp"].values,
             p_cnn=df["p_cnn"].values, p_gnn=df["p_gnn"].values,
             names=df["Name"].values)

    y = df["y_true"].values.astype(int)
    p_mlp, p_cnn, p_gnn = df["p_mlp"].values, df["p_cnn"].values, df["p_gnn"].values

    out = {"n_test": int(len(y)), "pos_fraction": float(y.mean()),
           "point_metrics": {
               "pr_auc":  {"mlp": float(average_precision_score(y, p_mlp)),
                           "cnn": float(average_precision_score(y, p_cnn)),
                           "gnn": float(average_precision_score(y, p_gnn))},
               "roc_auc": {"mlp": float(roc_auc_score(y, p_mlp)),
                           "cnn": float(roc_auc_score(y, p_cnn)),
                           "gnn": float(roc_auc_score(y, p_gnn))}},
           "differences": {}}

    for mname, metric in [("pr_auc", average_precision_score),
                          ("roc_auc", roc_auc_score)]:
        out["differences"][mname] = {
            "CNN_minus_MLP": paired_bootstrap(y, p_cnn, p_mlp, metric),
            "CNN_minus_GNN": paired_bootstrap(y, p_cnn, p_gnn, metric),
        }

    with open(os.path.join(OUT_DIR, "diff_ci.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n==== RÓŻNICE (punkt [95% CI], p_boot>0) ====")
    for mname in ("pr_auc", "roc_auc"):
        print(f"\n{mname.upper()}:")
        for comp, d in out["differences"][mname].items():
            print(f"  {comp}: {d['point']:+.4f} "
                  f"[{d['lo']:+.4f}; {d['hi']:+.4f}]  p_boot>0={d['p_boot_gt0']:.3f}")
    print(f"\nZapisano: {OUT_DIR}/diff_ci.json, aligned_predictions.csv/.npz")


if __name__ == "__main__":
    main()
