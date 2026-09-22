"""
Regeneracja krzywej PR dla GNN (gcn_L1) z POPRAWNYM baseline 0,333 (pkt 4 recenzji nr 2).
Używa zapisanych, wyrównanych predykcji (bez retreningu).
Uruchom: conda run -n mgr_thesis python regen_gnn_pr.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

OUT = "Praca Magisterska/figures/gcn_L1_pr.png"

d = np.load("Wyniki_Diff/aligned_predictions.npz", allow_pickle=True)
y = d["y"].astype(int)
p = d["p_gnn"].astype(float)

baseline = float(y.mean())            # ~0,333 dla podpróbki n=7707
ap = average_precision_score(y, p)
prec, rec, _ = precision_recall_curve(y, p)

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, lw=2, label=f"PR-AUC = {ap:.3f}")
plt.axhline(baseline, ls="--", color="gray", label=f"Random ({baseline:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("gcn_L1 - PR Curve")
plt.xlim(0, 1); plt.ylim(0, 1.02); plt.legend(loc="upper right"); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Zapisano {OUT}: PR-AUC={ap:.3f}, baseline={baseline:.3f}")
