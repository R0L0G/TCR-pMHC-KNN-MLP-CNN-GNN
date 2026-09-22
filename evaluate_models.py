"""
Ewaluacja MLP, CNN i k-NN na zbiorze testowym z class_balanced=False.
Zapisuje wyniki JSON do Wyniki_MLP/, Wyniki_CNN/, Wyniki_KNN/.
"""
import os, json
import torch
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from model_base import (
    model_MLP, TCRCMVConvNet,
    get_mlp_loader, get_cnn_loader,
    Train_data, Val_data, Test_data,
    tcr_embbedings, cnn_tcr_embeddings, mhc_id,
)
from Evaluator import evaluate, print_results

DEVICE = "cuda:0"


def collect_preds(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                features, labels = [t.to(device) for t in batch]
                logits = model(features).squeeze(-1)
            else:
                alpha, beta, mhc_id_t, labels = [t.to(device) for t in batch]
                logits = model(alpha, beta, mhc_id_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    return np.array(all_labels), np.array(all_probs)


def eval_and_save(model, val_loader, test_loader, model_name, out_dir, checkpoint):
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model = model.to(DEVICE)

    y_val, y_val_proba = collect_preds(model, val_loader, DEVICE)
    y_test, y_test_proba = collect_preds(model, test_loader, DEVICE)

    os.makedirs(out_dir, exist_ok=True)
    results = evaluate(
        y_test, y_test_proba,
        y_val=y_val, y_val_proba=y_val_proba,
        pos_rate=float(Test_data["Binding"].mean()),
        model_name=model_name,
        save_plots_dir=out_dir,
        show_plots=False,
    )
    print_results(results)

    hits_key = next((k for k in results if k.startswith("hits_at_")), None)
    serializable = {
        "model_name": model_name,
        "pr_auc":     list(results["pr_auc"]),
        "roc_auc":    list(results["roc_auc"]),
        "hits_at_100": float(results[hits_key]) if hits_key else float("nan"),
        "threshold":  float(results["threshold"]),
        "f1":         float(results["f1"]),
        "mcc":        float(results["mcc"]),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[{model_name}] zapisano → {out_dir}/results.json")
    return results


def eval_knn():
    import pandas as pd
    from Evaluator import evaluate, print_results

    # Buduj macierze cech: ESM2(α) + ESM2(β) + binary MHC flag
    def build_features(df):
        rows = []
        for _, row in df.iterrows():
            tcr = tcr_embbedings[tcr_embbedings["TCR_name"] == row["TCR_name"]].iloc[0]
            alpha = np.array(tcr["Embeddings_alpha"])
            beta  = np.array(tcr["Embeddings_beta"])
            mhc_flag = np.array([1.0 if row["HLA_MHC"] == list(mhc_id.keys())[0] else 0.0])
            rows.append(np.concatenate([alpha, beta, mhc_flag]))
        return np.array(rows)

    print("Budowanie cech k-NN (może chwilę potrwać)...")
    X_train = build_features(Train_data)
    y_train = Train_data["Binding"].values.astype(int)
    X_val   = build_features(Val_data)
    y_val   = Val_data["Binding"].values.astype(int)
    X_test  = build_features(Test_data)
    y_test  = Test_data["Binding"].values.astype(int)

    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)

    y_val_proba  = knn.predict_proba(X_val)[:, 1]
    y_test_proba = knn.predict_proba(X_test)[:, 1]

    out_dir = "Wyniki_KNN"
    os.makedirs(out_dir, exist_ok=True)
    results = evaluate(
        y_test, y_test_proba,
        y_val=y_val, y_val_proba=y_val_proba,
        pos_rate=float(Test_data["Binding"].mean()),
        model_name="k-NN",
        save_plots_dir=out_dir,
        show_plots=False,
    )
    print_results(results)

    hits_key = next((k for k in results if k.startswith("hits_at_")), None)
    serializable = {
        "model_name": "k-NN",
        "pr_auc":     list(results["pr_auc"]),
        "roc_auc":    list(results["roc_auc"]),
        "hits_at_100": float(results[hits_key]) if hits_key else float("nan"),
        "threshold":  float(results["threshold"]),
        "f1":         float(results["f1"]),
        "mcc":        float(results["mcc"]),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[k-NN] zapisano → {out_dir}/results.json")
    return results


if __name__ == "__main__":
    # CNN
    cnn_val_loader  = get_cnn_loader(Val_data,  cnn_tcr_embeddings, mhc_id, shuffle=False, class_balanced=False, num_workers=0)
    cnn_test_loader = get_cnn_loader(Test_data, cnn_tcr_embeddings, mhc_id, shuffle=False, class_balanced=False, num_workers=0)
    eval_and_save(
        TCRCMVConvNet(),
        cnn_val_loader, cnn_test_loader,
        model_name="CNN", out_dir="Wyniki_CNN", checkpoint="best_cnn.pt",
    )

    # k-NN
    eval_knn()

    print("\nGotowe. Wyniki w Wyniki_MLP/, Wyniki_CNN/, Wyniki_KNN/")
