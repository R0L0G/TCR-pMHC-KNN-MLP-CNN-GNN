"""
Re-ewaluacja MLP, CNN i k-NN na tych samych 7708 próbkach co GNN
(podgrafy testowe z Graphs/Test_graphs/ — subsample 1:2 pos:neg, ~33% pos).
Wyniki zapisywane do Wyniki_MLP/, Wyniki_CNN/, Wyniki_KNN/ z sufiksem _gnnset.
"""
import os, json, torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from model_base import (
    model_MLP, TCRCMVConvNet,
    get_mlp_loader, get_cnn_loader,
    Train_data, Val_data, Test_data,
    tcr_embbedings, cnn_tcr_embeddings, mhc_id,
)
from Evaluator import evaluate, print_results

DEVICE = "cuda:0"
TEST_GRAPHS_DIR = "Graphs/Test_graphs"
VAL_GRAPHS_DIR  = "Graphs/Validation_graphs"


def load_names_from_graphs(graphs_dir):
    """Zwraca listę (Name, Binding) dla każdego podgrafu w katalogu."""
    entries = []
    for fname in sorted(os.listdir(graphs_dir)):
        if not fname.endswith(".pt"):
            continue
        g = torch.load(os.path.join(graphs_dir, fname), weights_only=False)
        rel = g["target_relation"]
        entries.append({"Name": rel["Name"], "Binding": int(rel["Binding"])})
    return pd.DataFrame(entries)


def filter_data(full_df, names_df):
    """Filtruje full_df do wierszy obecnych w names_df (po kolumnie Name)."""
    return full_df[full_df["Name"].isin(names_df["Name"])].reset_index(drop=True)


def collect_preds(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                features, labels = [t.to(DEVICE) for t in batch]
                logits = model(features).squeeze(-1)
            else:
                alpha, beta, mhc_id_t, labels = [t.to(DEVICE) for t in batch]
                logits = model(alpha, beta, mhc_id_t)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def save_results(results, out_dir, suffix=""):
    hits_key = next((k for k in results if k.startswith("hits_at_")), None)
    data = {
        "model_name": results["model_name"],
        "pr_auc":     list(results["pr_auc"]),
        "roc_auc":    list(results["roc_auc"]),
        "hits_at_100": float(results[hits_key]) if hits_key else float("nan"),
        "threshold":  float(results["threshold"]),
        "f1":         float(results["f1"]),
        "mcc":        float(results["mcc"]),
        "n_test":     int(results["n_test"]),
        "pos_fraction": float(results["pos_fraction"]),
    }
    path = os.path.join(out_dir, f"results{suffix}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {path}")


if __name__ == "__main__":
    print("Wczytywanie nazw z podgrafów testowych i walidacyjnych...")
    test_names = load_names_from_graphs(TEST_GRAPHS_DIR)
    val_names  = load_names_from_graphs(VAL_GRAPHS_DIR)
    print(f"  Test graphs: {len(test_names)} ({test_names['Binding'].mean()*100:.1f}% pos)")
    print(f"  Val  graphs: {len(val_names)}  ({val_names['Binding'].mean()*100:.1f}% pos)")

    test_sub = filter_data(Test_data, test_names)
    val_sub  = filter_data(Val_data,  val_names)
    print(f"  Test_data subset: {len(test_sub)}, {test_sub['Binding'].mean()*100:.1f}% pos")
    print(f"  Val_data  subset: {len(val_sub)},  {val_sub['Binding'].mean()*100:.1f}% pos")

    # ── MLP ──────────────────────────────────────────────────────────────────
    print("\n[MLP]")
    mlp = model_MLP(in_dim=1282, hidden_dim=256).to(DEVICE)
    mlp.load_state_dict(torch.load("best_mlp.pt", weights_only=True))

    mlp_val_loader  = get_mlp_loader(val_sub,  tcr_embbedings, mhc_id, shuffle=False,
                                     class_balanced=False, num_workers=0)
    mlp_test_loader = get_mlp_loader(test_sub, tcr_embbedings, mhc_id, shuffle=False,
                                     class_balanced=False, num_workers=0)
    y_val,  y_val_p  = collect_preds(mlp, mlp_val_loader)
    y_test, y_test_p = collect_preds(mlp, mlp_test_loader)

    r = evaluate(y_test, y_test_p, y_val=y_val, y_val_proba=y_val_p,
                 model_name="MLP_gnnset", save_plots_dir="Wyniki_MLP",
                 show_plots=False)
    print_results(r)
    save_results(r, "Wyniki_MLP", "_gnnset")

    # ── CNN ──────────────────────────────────────────────────────────────────
    print("\n[CNN]")
    cnn = TCRCMVConvNet().to(DEVICE)
    cnn.load_state_dict(torch.load("best_cnn.pt", weights_only=True))

    cnn_val_loader  = get_cnn_loader(val_sub,  cnn_tcr_embeddings, mhc_id, shuffle=False,
                                     class_balanced=False, num_workers=0)
    cnn_test_loader = get_cnn_loader(test_sub, cnn_tcr_embeddings, mhc_id, shuffle=False,
                                     class_balanced=False, num_workers=0)
    y_val,  y_val_p  = collect_preds(cnn, cnn_val_loader)
    y_test, y_test_p = collect_preds(cnn, cnn_test_loader)

    r = evaluate(y_test, y_test_p, y_val=y_val, y_val_proba=y_val_p,
                 model_name="CNN_gnnset", save_plots_dir="Wyniki_CNN",
                 show_plots=False)
    print_results(r)
    save_results(r, "Wyniki_CNN", "_gnnset")

    # ── k-NN ─────────────────────────────────────────────────────────────────
    print("\n[k-NN]")
    def build_features(df):
        rows = []
        mhc_keys = list(mhc_id.keys())
        for _, row in df.iterrows():
            t = tcr_embbedings[tcr_embbedings["TCR_name"] == row["TCR_name"]].iloc[0]
            alpha = np.array(t["Embeddings_alpha"])
            beta  = np.array(t["Embeddings_beta"])
            flag  = np.array([1.0 if row["HLA_MHC"] == mhc_keys[0] else 0.0])
            rows.append(np.concatenate([alpha, beta, flag]))
        return np.array(rows)

    print("  Budowanie cech treningu k-NN...")
    X_train = build_features(Train_data)
    y_train = Train_data["Binding"].values.astype(int)
    X_val   = build_features(val_sub)
    y_val   = val_sub["Binding"].values.astype(int)
    X_test  = build_features(test_sub)
    y_test  = test_sub["Binding"].values.astype(int)

    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)
    y_val_p  = knn.predict_proba(X_val)[:, 1]
    y_test_p = knn.predict_proba(X_test)[:, 1]

    r = evaluate(y_test, y_test_p, y_val=y_val, y_val_proba=y_val_p,
                 model_name="k-NN_gnnset", save_plots_dir="Wyniki_KNN",
                 show_plots=False)
    print_results(r)
    save_results(r, "Wyniki_KNN", "_gnnset")

    print("\nGotowe. Porównanie na tym samym zbiorze co GNN (~33% pos).")
