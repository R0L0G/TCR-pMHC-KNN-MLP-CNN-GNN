"""
Ablacja schematu ważenia klas dla MLP (pkt 8 recenzji nr 2, wariant C).
Trzy reżimy: sampler-only / pos_weight-only / oba (obecny).
Pokazuje wpływ na operating point (recall, FP) vs. metryki rankingowe (PR-AUC, ROC-AUC).
NIE rusza best_mlp.pt ani tabeli głównej. Uruchom z /home/tarnickil/MGR:
    conda run -n mgr_thesis python ablation_weighting.py
"""
import os, json
import torch, torch.multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)
import numpy as np

from model_base import (
    model_MLP, train_model, get_mlp_loader,
    Train_data, Val_data, Test_data, tcr_embbedings, mhc_id,
)
from Evaluator import evaluate

DEVICE = "cuda:0"
SEED = 42
OUT = "Wyniki_Ablation_Weighting"

n_pos = int((Train_data["Binding"] == 1).sum())
n_neg = int((Train_data["Binding"] == 0).sum())

# (nazwa, class_balanced, num_pos, num_neg); pos_weight = num_neg/num_pos
REGIMES = [
    ("sampler_only",   True,  1,     1),      # pos_weight = 1
    ("posweight_only", False, n_pos, n_neg),  # pos_weight ~ 8.7, bez samplera
    ("both",           True,  n_pos, n_neg),  # obecny: sampler + pos_weight
]


def collect_preds(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for feats, labels in loader:
            logits = model(feats.to(DEVICE)).squeeze(-1)
            ps.extend(torch.sigmoid(logits).cpu().numpy())
            ys.extend(labels.numpy())
    return np.array(ys), np.array(ps)


def main():
    os.makedirs(OUT, exist_ok=True)
    val_loader  = get_mlp_loader(Val_data,  tcr_embbedings, mhc_id, shuffle=False, class_balanced=False, num_workers=0)
    test_loader = get_mlp_loader(Test_data, tcr_embbedings, mhc_id, shuffle=False, class_balanced=False, num_workers=0)
    rows = {}
    for name, balanced, npos, nneg in REGIMES:
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = model_MLP(in_dim=1282, hidden_dim=256)
        train_loader = get_mlp_loader(Train_data, tcr_embbedings, mhc_id,
                                      shuffle=True, class_balanced=balanced, num_workers=0)
        train_model(model, train_loader, val_loader, num_pos=npos, num_neg=nneg,
                    device=DEVICE, save_path=os.path.join(OUT, f"mlp_{name}.pt"))
        y_val, p_val = collect_preds(model, val_loader)
        y_test, p_test = collect_preds(model, test_loader)
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        res = evaluate(y_test, p_test, y_val=y_val, y_val_proba=p_val,
                       pos_rate=float(Test_data["Binding"].mean()),
                       model_name=f"MLP_{name}", save_plots_dir=os.path.join(OUT, name),
                       show_plots=False)
        tn, fp, fn, tp = (int(v) for v in np.array(res["confusion_matrix"]).ravel())
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        rows[name] = {
            "pos_weight": nneg / npos, "class_balanced": balanced,
            "pr_auc": [round(x, 3) for x in res["pr_auc"]],
            "roc_auc": [round(x, 3) for x in res["roc_auc"]],
            "f1": round(float(res["f1"]), 3), "mcc": round(float(res["mcc"]), 3),
            "recall": round(recall, 3), "fp": fp, "tp": tp, "fn": fn, "tn": tn,
            "threshold": round(float(res["threshold"]), 3),
        }
        print(f"[{name}] PR-AUC={res['pr_auc'][0]:.3f} ROC={res['roc_auc'][0]:.3f} "
              f"recall={recall:.3f} FP={fp} F1={res['f1']:.3f}")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("Zapisano ->", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
