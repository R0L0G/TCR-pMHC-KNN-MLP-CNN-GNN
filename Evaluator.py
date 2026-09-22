# """
# Moduł ewaluacyjny dla projektu predykcji TCR-CMV.

# Zwraca komplet metryk z confidence intervals oraz wykresy.
# Wspiera modele MLP/CNN (PyTorch) oraz k-NN (sklearn-like z .predict_proba).

# Główna funkcja: evaluate_model(model, val_df, test_df, ...) -> dict
# """

# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     average_precision_score,
#     roc_auc_score,
#     f1_score,
#     matthews_corrcoef,
#     precision_recall_curve,
#     roc_curve,
#     confusion_matrix,
# )
# from sklearn.utils import resample
# from typing import Callable, Optional, Tuple, Dict, Any


# # ---------------------------------------------------------------------------
# # 1. Pojedyncze metryki z bootstrap CI
# # ---------------------------------------------------------------------------

# def bootstrap_metric(
#     y_true: np.ndarray,
#     y_pred_proba: np.ndarray,
#     metric_fn: Callable,
#     n_bootstrap: int = 1000,
#     ci_level: float = 0.95,
#     random_state: int = 42,
# ) -> Tuple[float, float, float]:
#     """
#     Liczy metrykę z bootstrap confidence interval.

#     Parameters
#     ----------
#     y_true : np.ndarray
#         Prawdziwe etykiety {0, 1}
#     y_pred_proba : np.ndarray
#         Predykowane prawdopodobienstwa
#     metric_fn : Callable
#         Funkcja metryki przyjmujaca (y_true, y_pred_proba) lub (y_true, y_pred_binary)
#     n_bootstrap : int
#         Liczba iteracji bootstrap
#     ci_level : float
#         Poziom CI (np. 0.95 dla 95%)
#     random_state : int
#         Seed dla powtarzalnosci

#     Returns
#     -------
#     point_estimate, lower_bound, upper_bound
#     """
#     rng = np.random.RandomState(random_state)
#     scores = []
#     n = len(y_true)

#     for _ in range(n_bootstrap):
#         indices = rng.randint(0, n, n)
#         # Sprawdzamy czy probka ma obie klasy (potrzebne dla AUC)
#         if len(np.unique(y_true[indices])) < 2:
#             continue
#         score = metric_fn(y_true[indices], y_pred_proba[indices])
#         scores.append(score)

#     point = metric_fn(y_true, y_pred_proba)
#     alpha = (1 - ci_level) / 2
#     lower = np.percentile(scores, alpha * 100)
#     upper = np.percentile(scores, (1 - alpha) * 100)

#     return point, lower, upper


# def hits_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 100) -> float:
#     """
#     Hits@K w definicji recall: ilu pozytywow z prawdziwych jest w top-K predykcji.

#     Returns
#     -------
#     Frakcja pozytywow w top-K
#     """
#     if y_true.sum() == 0:
#         return 0.0
#     sorted_indices = np.argsort(-y_pred_proba)
#     top_k_indices = sorted_indices[:k]
#     return y_true[top_k_indices].sum() / y_true.sum()


# def find_best_threshold(
#     y_val: np.ndarray, y_val_proba: np.ndarray, n_thresholds: int = 99
# ) -> Tuple[float, float]:
#     """
#     Znajduje threshold maksymalizujacy F1 na zbiorze walidacyjnym.

#     Returns
#     -------
#     best_threshold, best_f1_on_val
#     """
#     thresholds = np.linspace(0.01, 0.99, n_thresholds)
#     f1_scores = []
#     for t in thresholds:
#         y_pred_binary = (y_val_proba >= t).astype(int)
#         f1_scores.append(f1_score(y_val, y_pred_binary, zero_division=0))
#     best_idx = int(np.argmax(f1_scores))
#     return float(thresholds[best_idx]), float(f1_scores[best_idx])


# # ---------------------------------------------------------------------------
# # 2. Predict wrapper - obsluguje rozne typy modeli
# # ---------------------------------------------------------------------------

# def predict_proba(
#     model: Any,
#     X: np.ndarray,
#     device: str = "cpu",
#     batch_size: int = 256,
# ) -> np.ndarray:
#     """
#     Uniwersalny wrapper na predykcje prawdopodobienstw.

#     Obsluguje:
#     - sklearn-like (z .predict_proba)
#     - PyTorch nn.Module (forward zwracajacy logits lub prawdopodobienstwa)

#     Parameters
#     ----------
#     model : sklearn-like albo torch.nn.Module
#     X : np.ndarray
#         Cechy wejsciowe
#     device : str
#         'cpu' lub 'cuda' (tylko dla PyTorch)
#     batch_size : int
#         Tylko dla PyTorch

#     Returns
#     -------
#     Wektor prawdopodobienstw klasy 1
#     """
#     # sklearn-like
#     if hasattr(model, "predict_proba"):
#         proba = model.predict_proba(X)
#         if proba.ndim == 2:
#             return proba[:, 1]
#         return proba.ravel()

#     # PyTorch
#     if isinstance(model, torch.nn.Module):
#         model.eval()
#         model.to(device)
#         all_proba = []
#         with torch.no_grad():
#             for i in range(0, len(X), batch_size):
#                 batch = X[i : i + batch_size]
#                 if isinstance(batch, np.ndarray):
#                     batch = torch.from_numpy(batch).float()
#                 batch = batch.to(device)
#                 logits = model(batch)
#                 # Zakladamy ze model zwraca logity lub prawdopodobienstwa w [0,1]
#                 if logits.dim() > 1 and logits.shape[1] > 1:
#                     # Multi-output, bierzemy klase 1
#                     proba = torch.softmax(logits, dim=1)[:, 1]
#                 else:
#                     # Single output, traktujemy jako logit
#                     proba = torch.sigmoid(logits.squeeze(-1))
#                 all_proba.append(proba.cpu().numpy())
#         return np.concatenate(all_proba)

#     raise TypeError(f"Nieobslugiwany typ modelu: {type(model)}")


# # ---------------------------------------------------------------------------
# # 3. Wykresy
# # ---------------------------------------------------------------------------

# def plot_pr_curve(
#     y_true: np.ndarray,
#     y_pred_proba: np.ndarray,
#     pr_auc: float,
#     title: str = "Precision-Recall Curve",
#     save_path: Optional[str] = None,
# ) -> plt.Figure:
#     """Krzywa Precision-Recall."""
#     precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
#     baseline = y_true.sum() / len(y_true)

#     fig, ax = plt.subplots(figsize=(7, 6))
#     ax.plot(recall, precision, lw=2, label=f"Model (PR-AUC = {pr_auc:.3f})")
#     ax.axhline(
#         baseline,
#         color="gray",
#         ls="--",
#         lw=1,
#         label=f"Random (baseline = {baseline:.3f})",
#     )
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")
#     ax.set_title(title)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1.05])
#     ax.legend(loc="best")
#     ax.grid(alpha=0.3)

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     return fig


# def plot_roc_curve(
#     y_true: np.ndarray,
#     y_pred_proba: np.ndarray,
#     roc_auc: float,
#     title: str = "ROC Curve",
#     save_path: Optional[str] = None,
# ) -> plt.Figure:
#     """Krzywa ROC."""
#     fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

#     fig, ax = plt.subplots(figsize=(7, 6))
#     ax.plot(fpr, tpr, lw=2, label=f"Model (ROC-AUC = {roc_auc:.3f})")
#     ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Random")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title(title)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1.05])
#     ax.legend(loc="best")
#     ax.grid(alpha=0.3)

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     return fig


# def plot_confusion_matrix(
#     y_true: np.ndarray,
#     y_pred_binary: np.ndarray,
#     title: str = "Confusion Matrix (best F1 threshold)",
#     save_path: Optional[str] = None,
# ) -> plt.Figure:
#     """Macierz pomylek z wartosciami liczbowymi."""
#     cm = confusion_matrix(y_true, y_pred_binary)
#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(cm, cmap="Blues")
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_xticklabels(["Predicted 0", "Predicted 1"])
#     ax.set_yticklabels(["True 0", "True 1"])
#     ax.set_title(title)

#     # Liczby w komorkach
#     for i in range(2):
#         for j in range(2):
#             color = "white" if cm[i, j] > cm.max() / 2 else "black"
#             ax.text(
#                 j, i, str(cm[i, j]), ha="center", va="center",
#                 color=color, fontsize=14, fontweight="bold",
#             )

#     plt.colorbar(im, ax=ax)
#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     return fig


# def plot_score_distribution(
#     y_true: np.ndarray,
#     y_pred_proba: np.ndarray,
#     threshold: Optional[float] = None,
#     title: str = "Score Distribution",
#     save_path: Optional[str] = None,
# ) -> plt.Figure:
#     """Histogram predykcji per klasa - pokazuje rozdzielnosc modelu."""
#     fig, ax = plt.subplots(figsize=(7, 5))
#     ax.hist(
#         y_pred_proba[y_true == 0], bins=50, alpha=0.6, label="Negative (y=0)",
#         color="steelblue", density=True,
#     )
#     ax.hist(
#         y_pred_proba[y_true == 1], bins=50, alpha=0.6, label="Positive (y=1)",
#         color="coral", density=True,
#     )
#     if threshold is not None:
#         ax.axvline(
#             threshold, color="black", ls="--", lw=1.5,
#             label=f"Threshold = {threshold:.3f}",
#         )
#     ax.set_xlabel("Predicted probability")
#     ax.set_ylabel("Density")
#     ax.set_title(title)
#     ax.legend(loc="best")
#     ax.grid(alpha=0.3)

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     return fig


# # ---------------------------------------------------------------------------
# # 4. Glowna funkcja ewaluacyjna
# # ---------------------------------------------------------------------------

# def evaluate_model(
#     model: Any,
#     val_df: pd.DataFrame,
#     test_df: pd.DataFrame,
#     feature_cols: list,
#     label_col: str = "Binding",
#     mhc_col: Optional[str] = "HLA_MHC",
#     device: str = "cpu",
#     batch_size: int = 256,
#     n_bootstrap: int = 1000,
#     k_for_hits: int = 100,
#     model_name: str = "Model",
#     save_plots_dir: Optional[str] = None,
#     show_plots: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Pelna ewaluacja modelu na zbiorach val i test.

#     Parameters
#     ----------
#     model : Any
#         Model z .predict_proba (sklearn) lub torch.nn.Module
#     val_df : pd.DataFrame
#         Zbior walidacyjny - uzywany do wyboru threshold dla F1
#     test_df : pd.DataFrame
#         Zbior testowy - na nim raportowane finalne metryki
#     feature_cols : list
#         Lista nazw kolumn z cechami w DataFrame
#     label_col : str
#         Nazwa kolumny z etykieta (domyslnie 'Binding')
#     mhc_col : str, optional
#         Nazwa kolumny MHC dla per-MHC breakdown. None = pomijaj
#     device : str
#         'cpu' lub 'cuda' (tylko dla PyTorch)
#     batch_size : int
#         Tylko dla PyTorch
#     n_bootstrap : int
#         Liczba iteracji bootstrap dla CI
#     k_for_hits : int
#         K dla Hits@K
#     model_name : str
#         Nazwa modelu w tytulach wykresow
#     save_plots_dir : str, optional
#         Katalog do zapisania wykresow. None = nie zapisuj
#     show_plots : bool
#         Czy wyswietlac wykresy (False dla batch processing)

#     Returns
#     -------
#     dict z kluczami:
#         'pr_auc': (point, lower, upper)
#         'roc_auc': (point, lower, upper)
#         'hits_at_k': float
#         'f1': float
#         'best_threshold': float
#         'mcc': float
#         'precision_at_threshold': float
#         'recall_at_threshold': float
#         'confusion_matrix': np.ndarray (2x2)
#         'per_mhc': dict (jesli mhc_col != None)
#         'figures': dict z matplotlib Figures
#     """
#     # Wyciagnij cechy i etykiety
#     X_val = val_df.loc[:, feature_cols].values
#     y_val = val_df[label_col].values.astype(int)
#     X_test = test_df[feature_cols].values
#     y_test = test_df[label_col].values.astype(int)



#     # Predykcje
#     y_val_proba = predict_proba(model, X_val, device=device, batch_size=batch_size)
#     y_test_proba = predict_proba(model, X_test, device=device, batch_size=batch_size)

#     # 1. Threshold-free metryki na test (z bootstrap CI)
#     pr_auc, pr_lower, pr_upper = bootstrap_metric(
#         y_test, y_test_proba, average_precision_score, n_bootstrap=n_bootstrap,
#     )
#     roc_auc, roc_lower, roc_upper = bootstrap_metric(
#         y_test, y_test_proba, roc_auc_score, n_bootstrap=n_bootstrap,
#     )

#     # 2. Hits@K
#     hits_k = hits_at_k(y_test, y_test_proba, k=k_for_hits)

#     # 3. Best threshold z val, F1 + MCC na test
#     best_threshold, val_f1 = find_best_threshold(y_val, y_val_proba)
#     y_test_pred_binary = (y_test_proba >= best_threshold).astype(int)
#     test_f1 = f1_score(y_test, y_test_pred_binary, zero_division=0)
#     test_mcc = matthews_corrcoef(y_test, y_test_pred_binary)

#     # 4. Precision/Recall przy threshold
#     cm = confusion_matrix(y_test, y_test_pred_binary)
#     tn, fp, fn, tp = cm.ravel()
#     precision_at_thr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall_at_thr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

#     # 5. Wykresy
#     figures = {}
#     save_path = lambda name: (
#         f"{save_plots_dir}/{model_name}_{name}.png" if save_plots_dir else None
#     )

#     figures["pr_curve"] = plot_pr_curve(
#         y_test, y_test_proba, pr_auc,
#         title=f"{model_name} - Precision-Recall Curve (test)",
#         save_path=save_path("pr_curve"),
#     )
#     figures["roc_curve"] = plot_roc_curve(
#         y_test, y_test_proba, roc_auc,
#         title=f"{model_name} - ROC Curve (test)",
#         save_path=save_path("roc_curve"),
#     )
#     figures["confusion_matrix"] = plot_confusion_matrix(
#         y_test, y_test_pred_binary,
#         title=f"{model_name} - Confusion Matrix (threshold = {best_threshold:.3f})",
#         save_path=save_path("confusion_matrix"),
#     )
#     figures["score_distribution"] = plot_score_distribution(
#         y_test, y_test_proba, threshold=best_threshold,
#         title=f"{model_name} - Score Distribution (test)",
#         save_path=save_path("score_distribution"),
#     )

#     if not show_plots:
#         for fig in figures.values():
#             plt.close(fig)

#     # 6. Per-MHC breakdown (opcjonalny)
#     per_mhc_results = None
#     if mhc_col is not None and mhc_col in test_df.columns:
#         per_mhc_results = {}
#         for mhc_value in test_df[mhc_col].unique():
#             mask = (test_df[mhc_col] == mhc_value).values
#             if mask.sum() < 10:
#                 continue
#             y_sub = y_test[mask]
#             proba_sub = y_test_proba[mask]
#             if len(np.unique(y_sub)) < 2:
#                 # Jedna klasa, AUC niezdefiniowane
#                 per_mhc_results[mhc_value] = {
#                     "n_samples": int(mask.sum()),
#                     "n_positives": int(y_sub.sum()),
#                     "pr_auc": None,
#                     "roc_auc": None,
#                     "note": "Tylko jedna klasa - AUC niezdefiniowane",
#                 }
#                 continue
#             per_mhc_results[mhc_value] = {
#                 "n_samples": int(mask.sum()),
#                 "n_positives": int(y_sub.sum()),
#                 "pos_fraction": float(y_sub.mean()),
#                 "pr_auc": float(average_precision_score(y_sub, proba_sub)),
#                 "roc_auc": float(roc_auc_score(y_sub, proba_sub)),
#                 "hits_at_k": float(hits_at_k(y_sub, proba_sub, k=k_for_hits)),
#             }

#     # 7. Wyniki
#     results = {
#         "model_name": model_name,
#         "n_test_samples": len(y_test),
#         "n_test_positives": int(y_test.sum()),
#         "test_pos_fraction": float(y_test.mean()),
#         "pr_auc": (float(pr_auc), float(pr_lower), float(pr_upper)),
#         "roc_auc": (float(roc_auc), float(roc_lower), float(roc_upper)),
#         f"hits_at_{k_for_hits}": float(hits_k),
#         "best_threshold": float(best_threshold),
#         "f1": float(test_f1),
#         "mcc": float(test_mcc),
#         "precision_at_threshold": float(precision_at_thr),
#         "recall_at_threshold": float(recall_at_thr),
#         "confusion_matrix": cm,
#         "per_mhc": per_mhc_results,
#         "figures": figures,
#     }

#     return results


# # ---------------------------------------------------------------------------
# # 5. Helper do drukowania wynikow w czytelnej formie
# # ---------------------------------------------------------------------------

# def print_results(results: Dict[str, Any]) -> None:
#     """Wypisuje wyniki ewaluacji w czytelnym formacie."""
#     print("=" * 70)
#     print(f"Wyniki ewaluacji: {results['model_name']}")
#     print("=" * 70)
#     print(f"N test samples: {results['n_test_samples']}")
#     print(f"N test positives: {results['n_test_positives']} "
#           f"({results['test_pos_fraction']:.1%})")
#     print()

#     pr = results["pr_auc"]
#     roc = results["roc_auc"]
#     print(f"PR-AUC:  {pr[0]:.3f}  (95% CI: {pr[1]:.3f} - {pr[2]:.3f})")
#     print(f"ROC-AUC: {roc[0]:.3f}  (95% CI: {roc[1]:.3f} - {roc[2]:.3f})")

#     hits_keys = [k for k in results if k.startswith("hits_at_")]
#     if hits_keys:
#         k = hits_keys[0]
#         print(f"{k.replace('_', '@').upper()}: {results[k]:.3f}")

#     print(f"\nBest threshold (z val set): {results['best_threshold']:.3f}")
#     print(f"F1 (test):  {results['f1']:.3f}")
#     print(f"MCC (test): {results['mcc']:.3f}")
#     print(f"Precision @ threshold: {results['precision_at_threshold']:.3f}")
#     print(f"Recall @ threshold:    {results['recall_at_threshold']:.3f}")

#     print(f"\nConfusion matrix:")
#     cm = results["confusion_matrix"]
#     print(f"               Pred 0    Pred 1")
#     print(f"  True 0   {cm[0,0]:>8}  {cm[0,1]:>8}")
#     print(f"  True 1   {cm[1,0]:>8}  {cm[1,1]:>8}")

#     if results.get("per_mhc"):
#         print(f"\nPer-MHC breakdown:")
#         for mhc, sub in results["per_mhc"].items():
#             print(f"  {mhc} (n={sub['n_samples']}, pos={sub['n_positives']}):")
#             if sub.get("pr_auc") is None:
#                 print(f"    {sub.get('note', 'brak danych')}")
#             else:
#                 print(f"    PR-AUC = {sub['pr_auc']:.3f}, "
#                       f"ROC-AUC = {sub['roc_auc']:.3f}, "
#                       f"Hits@K = {sub['hits_at_k']:.3f}")
#     print("=" * 70)


# # ---------------------------------------------------------------------------
# # 6. Helper do porownania wielu modeli w jednej tabeli
# # ---------------------------------------------------------------------------

# def compare_models(results_list: list) -> pd.DataFrame:
#     """
#     Tworzy DataFrame z porownaniem wielu modeli.

#     Parameters
#     ----------
#     results_list : list of dict
#         Lista wynikow z evaluate_model()

#     Returns
#     -------
#     pd.DataFrame z kolumnami: model_name, PR-AUC, ROC-AUC, Hits@K, F1, MCC
#     """
#     rows = []
#     for r in results_list:
#         pr = r["pr_auc"]
#         roc = r["roc_auc"]
#         hits_keys = [k for k in r if k.startswith("hits_at_")]
#         hits_value = r[hits_keys[0]] if hits_keys else None
#         rows.append(
#             {
#                 "Model": r["model_name"],
#                 "PR-AUC": f"{pr[0]:.3f} ({pr[1]:.3f}-{pr[2]:.3f})",
#                 "ROC-AUC": f"{roc[0]:.3f} ({roc[1]:.3f}-{roc[2]:.3f})",
#                 f"{hits_keys[0]}": f"{hits_value:.3f}" if hits_value is not None else "N/A",
#                 "F1": f"{r['f1']:.3f}",
#                 "MCC": f"{r['mcc']:.3f}",
#                 "Threshold": f"{r['best_threshold']:.3f}",
#             }
#         )
#     return pd.DataFrame(rows)
"""
Ewaluator dla modeli klasyfikacji binarnej.

Działa na gotowych predykcjach (y_true, y_pred_proba) — niezależnie od typu modelu.
Zwraca metryki z 95% CI i wykresy.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_curve, confusion_matrix,
)


def bootstrap_ci(y_true, y_pred, metric_fn, n=1000, ci=0.95, seed=42):
    """Bootstrap CI dla metryki. Zwraca (point, lower, upper)."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    point = metric_fn(y_true, y_pred)
    alpha = (1 - ci) / 2
    return point, np.percentile(scores, alpha * 100), np.percentile(scores, (1 - alpha) * 100)


def hits_at_k(y_true, y_pred_proba, k=100):
    """Frakcja pozytywów wśród top-K predykcji."""
    if y_true.sum() == 0:
        return 0.0
    top_k = np.argsort(-y_pred_proba)[:k]
    return y_true[top_k].sum() / y_true.sum()


def find_best_threshold(y_val, y_val_proba):
    """Threshold maksymalizujący F1 na val set."""
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_val, y_val_proba >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def evaluate(
    y_test, y_test_proba,
    y_val=None, y_val_proba=None,
    mhc_test=None,
    pos_rate=None,
    model_name="Model",
    n_bootstrap=1000,
    k_for_hits=100,
    save_plots_dir=None,
    show_plots=True,
):
    """
    Pełna ewaluacja modelu z gotowych predykcji.

    Parameters
    ----------
    y_test, y_test_proba : np.ndarray
        Etykiety i predykcje na test secie.
    y_val, y_val_proba : np.ndarray, optional
        Do dobrania threshold dla F1. Jeśli None -> threshold=0.5.
    mhc_test : np.ndarray, optional
        MHC per próbka test (do per-MHC breakdown).
    pos_rate : float, optional
        Prawdziwy udział pozytywów w zbiorze testowym — używany jako baseline na wykresie
        PR-AUC. Należy podać gdy y_test pochodzi z DataLoadera z WeightedRandomSampler
        (class_balanced=True), bo wtedy y_test.mean() ≈ 0.5 zamiast prawdziwego rozkładu.
        Przykład: pos_rate=Test_data["Binding"].mean()
    model_name : str
        Nazwa modelu w tytułach wykresów i plikach.
    save_plots_dir : str, optional
        Katalog do zapisu PNG.

    Returns
    -------
    dict z metrykami i wykresami.
    """
    import warnings

    # Konwersja na numpy (gdyby ktoś podał torch.Tensor lub listę)
    y_test = np.asarray(y_test).astype(int)
    y_test_proba = np.asarray(y_test_proba).astype(float)

    observed_rate = float(y_test.mean())

    # Jeśli pos_rate nie podano — użyj observed_rate, ale ostrzeż gdy wygląda na balanced
    if pos_rate is None:
        if 0.40 <= observed_rate <= 0.60:
            warnings.warn(
                f"[{model_name}] y_test.mean() = {observed_rate:.3f} — wygląda na zbalansowany "
                f"loader (WeightedRandomSampler?). Jeśli zbiór testowy jest niezbalansowany, "
                f"podaj pos_rate=Test_data['Binding'].mean() żeby baseline PR był poprawny, "
                f"a metryki odpowiadały prawdziwemu rozkładowi klas.",
                UserWarning,
                stacklevel=2,
            )
        baseline = observed_rate
    else:
        baseline = float(pos_rate)

    # Threshold-free metryki z CI
    pr_auc = bootstrap_ci(y_test, y_test_proba, average_precision_score, n_bootstrap)
    roc_auc = bootstrap_ci(y_test, y_test_proba, roc_auc_score, n_bootstrap)

    # Hits@K
    hits = hits_at_k(y_test, y_test_proba, k_for_hits)

    # Threshold z val (jeśli brak val -> 0.5)
    if y_val is not None and y_val_proba is not None:
        threshold = find_best_threshold(
            np.asarray(y_val).astype(int), np.asarray(y_val_proba).astype(float)
        )
    else:
        threshold = 0.5

    y_pred = (y_test_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Wykresy
    figures = _make_plots(y_test, y_test_proba, y_pred, pr_auc[0], roc_auc[0],
                          threshold, model_name, save_plots_dir, baseline=baseline)
    if not show_plots:
        for f in figures.values():
            plt.close(f)

    # Per-MHC breakdown
    per_mhc = None
    if mhc_test is not None:
        mhc_test = np.asarray(mhc_test)
        per_mhc = {}
        for m in np.unique(mhc_test):
            mask = mhc_test == m
            if mask.sum() < 10 or len(np.unique(y_test[mask])) < 2:
                per_mhc[str(m)] = {"n_samples": int(mask.sum()),
                                    "n_positives": int(y_test[mask].sum()),
                                    "note": "Za mało danych lub jedna klasa"}
                continue
            per_mhc[str(m)] = {
                "n_samples": int(mask.sum()),
                "n_positives": int(y_test[mask].sum()),
                "pr_auc": float(average_precision_score(y_test[mask], y_test_proba[mask])),
                "roc_auc": float(roc_auc_score(y_test[mask], y_test_proba[mask])),
            }

    return {
        "model_name": model_name,
        "n_test": len(y_test),
        "n_positives": int(y_test.sum()),
        "pos_fraction": observed_rate,
        "baseline": baseline,
        "pr_auc": pr_auc,                # (point, lower, upper)
        "roc_auc": roc_auc,
        f"hits_at_{k_for_hits}": float(hits),
        "threshold": float(threshold),
        "f1": float(f1),
        "mcc": float(mcc),
        "confusion_matrix": cm,
        "per_mhc": per_mhc,
        "figures": figures,
    }


def _make_plots(y_test, y_proba, y_pred, pr_auc, roc_auc, threshold, name, save_dir,
                baseline=None):
    """Cztery wykresy: PR curve, ROC curve, confusion matrix, score distribution."""
    figs = {}
    save = lambda n: f"{save_dir}/{name}_{n}.png" if save_dir else None

    if baseline is None:
        baseline = float(y_test.mean())

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(baseline, ls="--", color="gray", label=f"Random ({baseline:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{name} - PR Curve",
           xlim=[0, 1], ylim=[0, 1.05])
    ax.legend(); ax.grid(alpha=0.3)
    if save("pr"): fig.savefig(save("pr"), dpi=150, bbox_inches="tight")
    figs["pr_curve"] = fig

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Random")
    ax.set(xlabel="FPR", ylabel="TPR", title=f"{name} - ROC Curve",
           xlim=[0, 1], ylim=[0, 1.05])
    ax.legend(); ax.grid(alpha=0.3)
    if save("roc"): fig.savefig(save("roc"), dpi=150, bbox_inches="tight")
    figs["roc_curve"] = fig

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(f"{name} - Confusion Matrix (thr={threshold:.3f})")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color,
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    if save("cm"): fig.savefig(save("cm"), dpi=150, bbox_inches="tight")
    figs["confusion_matrix"] = fig

    # Score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_proba[y_test == 0], bins=50, alpha=0.6, label="Negative", density=True, color="steelblue")
    ax.hist(y_proba[y_test == 1], bins=50, alpha=0.6, label="Positive", density=True, color="coral")
    ax.axvline(threshold, ls="--", color="black", label=f"Threshold = {threshold:.3f}")
    ax.set(xlabel="Predicted probability", ylabel="Density",
           title=f"{name} - Score Distribution")
    ax.legend(); ax.grid(alpha=0.3)
    if save("scores"): fig.savefig(save("scores"), dpi=150, bbox_inches="tight")
    figs["score_distribution"] = fig

    return figs


def print_results(r):
    """Wypisanie wyników w czytelnej formie."""
    print("=" * 60)
    print(f"Model: {r['model_name']}")
    print(f"Test: {r['n_test']} próbek, {r['n_positives']} pozytywów ({r['pos_fraction']:.1%})"
          + (f"  [baseline PR: {r['baseline']:.3f}]" if r.get('baseline') != r.get('pos_fraction') else ""))
    print(f"PR-AUC:  {r['pr_auc'][0]:.3f}  (95% CI: {r['pr_auc'][1]:.3f} - {r['pr_auc'][2]:.3f})")
    print(f"ROC-AUC: {r['roc_auc'][0]:.3f}  (95% CI: {r['roc_auc'][1]:.3f} - {r['roc_auc'][2]:.3f})")
    hits_key = [k for k in r if k.startswith("hits_at_")][0]
    hits_label = hits_key.replace('hits_at_', 'Hits@')
    print(f"{hits_label}: {r[hits_key]:.3f}")
    print(f"Threshold: {r['threshold']:.3f} | F1: {r['f1']:.3f} | MCC: {r['mcc']:.3f}")
    cm = r["confusion_matrix"]
    print(f"Confusion: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    if r.get("per_mhc"):
        print("\nPer-MHC:")
        for m, sub in r["per_mhc"].items():
            if "note" in sub:
                print(f"  {m}: {sub['note']}")
            else:
                print(f"  {m} (n={sub['n_samples']}, pos={sub['n_positives']}): "
                      f"PR-AUC={sub['pr_auc']:.3f}, ROC-AUC={sub['roc_auc']:.3f}")
    print("=" * 60)


def compare_models(results_list):
    """DataFrame porównujący wiele modeli."""
    import pandas as pd
    rows = []
    for r in results_list:
        hits_key = [k for k in r if k.startswith("hits_at_")][0]
        rows.append({
            "Model": r["model_name"],
            "PR-AUC": f"{r['pr_auc'][0]:.3f} ({r['pr_auc'][1]:.3f}-{r['pr_auc'][2]:.3f})",
            "ROC-AUC": f"{r['roc_auc'][0]:.3f} ({r['roc_auc'][1]:.3f}-{r['roc_auc'][2]:.3f})",
            hits_key: f"{r[hits_key]:.3f}",
            "F1": f"{r['f1']:.3f}",
            "MCC": f"{r['mcc']:.3f}",
        })
    return pd.DataFrame(rows)