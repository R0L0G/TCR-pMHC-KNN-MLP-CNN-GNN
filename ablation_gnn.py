"""
Ablation study GNN: GCNConv L=1,2,3,4 i GATConv (drop_rate=0.0).
Wszystkie warianty trenowane z identycznym seedem i hiperparametrami.
Wyniki: Wyniki_Ablation/<wariant>/checkpoint.pt + results.json + plots/
Uruchomienie: python ablation_gnn.py
"""
import csv
import os
import json
import traceback

import torch
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dhg.nn import GATConv

# Import z model_base — ładuje wszystkie dane i tworzy loadery (wolne, oczekiwane)
from model_base import (
    TCRCMVGraphNet, train_model, step,
    get_gnn_loader,
    Train_data, Val_data, Test_data,
)
from Evaluator import evaluate, print_results

DEVICE = "cuda:0"
ABLATION_DIR = "Wyniki_Ablation"
SEED = 42


class TCRCMVGraphNetGAT(nn.Module):
    """Identyczna architektura co TCRCMVGraphNet, ale z GATConv zamiast GCNConv.
    drop_rate=0.0 zapobiega NaN-crash w DHG 0.9.6 (atten_drop zeruje wagi uwagi)."""

    def __init__(
        self,
        d_esm2=1280,
        d_mhc_onehot=2,
        d_target_flag=4,
        d_hidden=256,
        n_gat_layers=3,
        dropout_node=0.3,
        dropout_gat=0.3,
        dropout_classifier=0.3,
    ):
        super().__init__()
        self.name = f"TCRCMVGraphNetGAT_L{n_gat_layers}"
        self.d_hidden = d_hidden
        self.n_gat_layers = n_gat_layers

        self.tcr_projection = nn.Sequential(
            nn.Linear(d_esm2, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(d_hidden + d_mhc_onehot + d_target_flag, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout_node),
        )
        self.gat_layers = nn.ModuleList([
            GATConv(d_hidden, d_hidden, drop_rate=0.0, use_bn=False)
            for _ in range(n_gat_layers)
        ])
        self.gat_dropouts = nn.ModuleList([
            nn.Dropout(dropout_gat) for _ in range(n_gat_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(d_hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.classifier[-1].weight, gain=1.0)

    def forward(self, X_features, graph, target_idx, d_esm2=1280, d_mhc=2):
        X_esm2   = X_features[:, :d_esm2]
        X_mhc    = X_features[:, d_esm2:d_esm2 + d_mhc]
        X_target = X_features[:, d_esm2 + d_mhc:]

        h_esm2    = self.tcr_projection(X_esm2)
        h_concat  = torch.cat([h_esm2, X_mhc, X_target], dim=1)
        h_0       = self.node_fusion(h_concat)

        all_layers = [h_0]
        h = h_0
        for gat_layer, dropout in zip(self.gat_layers, self.gat_dropouts):
            h_new = gat_layer(h, graph)
            h_new = F.gelu(h_new)
            h_new = dropout(h_new)
            h     = h_new + h
            all_layers.append(h)

        h_stacked = torch.stack(all_layers, dim=0)
        h_jk      = h_stacked.max(dim=0).values
        h_target  = h_jk[target_idx]
        return self.classifier(h_target).squeeze(-1)


def _collect_preds(model, loader, device):
    """Zbiera (y_true, y_proba) z DataLoadera bez gradientów."""
    # pos_weight=1.0: wartość loss jest ignorowana; criterion przekazywane do step() tylko
    # dlatego że step() tego wymaga — interesują nas wyłącznie logits i labels.
    pos_weight = torch.tensor([1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            _, logits, labels = step(model, batch, criterion, device)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def run_variant(name, model, out_dir):
    """Trenuje jeden wariant GNN i zapisuje checkpoint + wyniki + wykresy."""
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # num_pos/num_neg dla pos_weight w BCEWithLogitsLoss (stosunek 1:2 jak w model_base.main())
    num_pos = int((Train_data["Binding"] == 1).sum())
    num_neg = 2 * num_pos

    train_loader = get_gnn_loader(
        "Graphs/Training_graphs", class_balanced=True, num_workers=0)
    val_loader   = get_gnn_loader(
        "Graphs/Validation_graphs", shuffle=False, class_balanced=False, num_workers=0)
    test_loader  = get_gnn_loader(
        "Graphs/Test_graphs",       shuffle=False, class_balanced=False, num_workers=0)

    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")

    print(f"\n{'='*60}\nTrening: {name}\n{'='*60}")
    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_pos=num_pos,
        num_neg=num_neg,
        device=DEVICE,
        save_path=checkpoint_path,
    )

    # Wczytaj najlepszy checkpoint przed ewaluacją
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    y_val,  y_val_proba  = _collect_preds(model, val_loader,  DEVICE)
    y_test, y_test_proba = _collect_preds(model, test_loader, DEVICE)

    results = evaluate(
        y_test, y_test_proba,
        y_val=y_val, y_val_proba=y_val_proba,
        pos_rate=float(Test_data["Binding"].mean()),
        model_name=name,
        save_plots_dir=plots_dir,
        show_plots=False,
    )
    print_results(results)

    # Serializacja do JSON (bez obiektów matplotlib)
    hits_key = [k for k in results if k.startswith("hits_at_")][0]
    results_json = {
        "model_name":      results["model_name"],
        "n_test":          results["n_test"],
        "n_positives":     results["n_positives"],
        "pos_fraction":    results["pos_fraction"],
        "baseline":        results["baseline"],
        "pr_auc":          list(results["pr_auc"]),
        "roc_auc":         list(results["roc_auc"]),
        hits_key:          float(results[hits_key]),
        "threshold":       results["threshold"],
        "f1":              results["f1"],
        "mcc":             results["mcc"],
        "best_val_pr_auc": train_result["best_val_pr_auc"],
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"[{name}] Zapisano wyniki: {out_dir}")
    return results_json


def main():
    os.makedirs(ABLATION_DIR, exist_ok=True)

    variants = [
        ("gcn_L1", lambda: TCRCMVGraphNet(n_gcn_layers=1)),
        ("gcn_L2", lambda: TCRCMVGraphNet(n_gcn_layers=2)),
        ("gcn_L3", lambda: TCRCMVGraphNet(n_gcn_layers=3)),
        ("gcn_L4", lambda: TCRCMVGraphNet(n_gcn_layers=4)),
        ("gat",    lambda: TCRCMVGraphNetGAT()),
    ]

    summary = []
    for name, model_fn in variants:
        out_dir = os.path.join(ABLATION_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        try:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            r = run_variant(name, model_fn(), out_dir)
            summary.append(r)
        except Exception:
            print(f"\n[{name}] BŁĄD — pomijam wariant:")
            print(traceback.format_exc())
            err_path = os.path.join(out_dir, "error.log")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            print(f"  Szczegóły błędu: {err_path}")

    if not summary:
        print("Żaden wariant nie zakończył się sukcesem.")
        return

    # Tabela podsumowująca na stdout
    print("\n" + "="*72)
    print("ABLATION STUDY — PODSUMOWANIE")
    print("="*72)
    hdr = f"{'Wariant':<12} {'PR-AUC':>10} {'95% CI PR':>20} {'ROC-AUC':>10} {'Hits@100':>10} {'F1':>8} {'MCC':>8}"
    print(hdr)
    print("-"*72)
    for r in summary:
        pr   = r["pr_auc"]
        roc  = r["roc_auc"]
        hits_key = [k for k in r if k.startswith("hits_at_")][0]
        hits = r[hits_key]
        print(f"{r['model_name']:<12} {pr[0]:>10.3f} "
              f"[{pr[1]:.3f}; {pr[2]:.3f}]{'':<6} "
              f"{roc[0]:>10.3f} {hits:>10.3f} {r['f1']:>8.3f} {r['mcc']:>8.3f}")

    # Zapis do CSV
    csv_path = os.path.join(ABLATION_DIR, "summary_table.csv")
    # hits_at_100: zakłada k_for_hits=100 (domyślne w Evaluator.evaluate)
    fieldnames = ["model_name", "pr_auc_point", "pr_auc_lower", "pr_auc_upper",
                  "roc_auc_point", "roc_auc_lower", "roc_auc_upper",
                  "hits_at_100", "f1", "mcc", "threshold", "best_val_pr_auc"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in summary:
            hits_key = [k for k in r if k.startswith("hits_at_")][0]
            writer.writerow({
                "model_name":      r["model_name"],
                "pr_auc_point":    r["pr_auc"][0],
                "pr_auc_lower":    r["pr_auc"][1],
                "pr_auc_upper":    r["pr_auc"][2],
                "roc_auc_point":   r["roc_auc"][0],
                "roc_auc_lower":   r["roc_auc"][1],
                "roc_auc_upper":   r["roc_auc"][2],
                "hits_at_100":     r.get(hits_key, ""),
                "f1":              r["f1"],
                "mcc":             r["mcc"],
                "threshold":       r["threshold"],
                "best_val_pr_auc": r.get("best_val_pr_auc", ""),
            })
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
