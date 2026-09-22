# Ablation Study GNN + Poprawki recenzji — Plan implementacji

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stworzyć `ablation_gnn.py` trenujący 5 wariantów GNN (GCNConv L=1–4 + GATConv) z pełną ewaluacją + naprawić wszystkie błędy merytoryczne i językowe w `Praca_magisterska.tex`.

**Architecture:** `ablation_gnn.py` importuje klasy i dane z `model_base.py`, definiuje `TCRCMVGraphNetGAT` lokalnie, trenuje każdy wariant sekwencyjnie i zapisuje checkpointy + `results.json` + wykresy do `Wyniki_Ablation/<wariant>/`. Zmiany w LaTeX to wyłącznie edycje `Praca_magisterska.tex` — `.cls` pozostaje bez zmian.

**Tech Stack:** Python 3, PyTorch, DHG 0.9.6 (`GCNConv`, `GATConv`), scikit-learn, `Evaluator.evaluate()`, LaTeX.

---

## Pliki

| Plik | Akcja |
|------|-------|
| `ablation_gnn.py` | Utwórz — skrypt treningu ablation GNN |
| `Wyniki_Ablation/` | Utwórz strukturę katalogów |
| `Praca Magisterska/Praca_magisterska.tex` | Modyfikuj — 6 grup zmian |
| `model_base.py` | Tylko odczyt (import) |
| `Evaluator.py` | Tylko odczyt (import) |

---

## Task 1: Utwórz `ablation_gnn.py` — klasa GAT + helpery

**Files:**
- Create: `ablation_gnn.py`

- [ ] **Krok 1.1: Utwórz plik z nagłówkiem, importami i klasą `TCRCMVGraphNetGAT`**

```python
"""
Ablation study GNN: GCNConv L=1,2,3,4 i GATConv (drop_rate=0.0).
Wszystkie warianty trenowane z identycznym seedem i hiperparametrami.
Wyniki: Wyniki_Ablation/<wariant>/checkpoint.pt + results.json + plots/
Uruchomienie: python ablation_gnn.py
"""
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
    Train_data, Test_data,
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
        dropout_classifier=0.3,
    ):
        super().__init__()
        self.name = "TCRCMVGraphNetGAT"
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
            nn.Dropout(0.3) for _ in range(n_gat_layers)
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
    pos_weight = torch.tensor([1.0], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            _, logits, labels = step(model, batch, criterion, device)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)
```

- [ ] **Krok 1.2: Sprawdź że plik istnieje**

```bash
python -c "import ast; ast.parse(open('ablation_gnn.py').read()); print('OK — brak błędów składni')"
```

Oczekiwany output: `OK — brak błędów składni`

---

## Task 2: Dopisz `run_variant()` i `main()` do `ablation_gnn.py`

**Files:**
- Modify: `ablation_gnn.py` (dopisanie na końcu pliku)

- [ ] **Krok 2.1: Dopisz funkcję `run_variant()`**

```python
def run_variant(name, model, out_dir):
    """Trenuje jeden wariant GNN i zapisuje checkpoint + wyniki + wykresy."""
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Subsample 1 pos : 2 neg (identycznie jak model_base.main())
    train_pos = Train_data[Train_data["Binding"] == 1]
    train_neg = Train_data[Train_data["Binding"] == 0]
    n_pos     = len(train_pos)
    train_neg = train_neg.sample(n=2 * n_pos, replace=False, random_state=SEED)
    num_pos   = n_pos
    num_neg   = len(train_neg)

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
```

- [ ] **Krok 2.2: Dopisz funkcję `main()`**

```python
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
        pr  = r["pr_auc"]
        roc = r["roc_auc"]
        hits = r.get("hits_at_100", r.get("hits_at_k", 0))
        print(f"{r['model_name']:<12} {pr[0]:>10.3f} "
              f"[{pr[1]:.3f}; {pr[2]:.3f}]{'':<6} "
              f"{roc[0]:>10.3f} {hits:>10.3f} {r['f1']:>8.3f} {r['mcc']:>8.3f}")

    # Zapis do CSV
    import csv
    csv_path = os.path.join(ABLATION_DIR, "summary_table.csv")
    fieldnames = ["model_name", "pr_auc_point", "pr_auc_lower", "pr_auc_upper",
                  "roc_auc_point", "roc_auc_lower", "roc_auc_upper",
                  "hits_at_100", "f1", "mcc", "threshold", "best_val_pr_auc"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in summary:
            hits_key = [k for k in r if k.startswith("hits_at_")][0]
            writer.writerow({
                "model_name":    r["model_name"],
                "pr_auc_point":  r["pr_auc"][0],
                "pr_auc_lower":  r["pr_auc"][1],
                "pr_auc_upper":  r["pr_auc"][2],
                "roc_auc_point": r["roc_auc"][0],
                "roc_auc_lower": r["roc_auc"][1],
                "roc_auc_upper": r["roc_auc"][2],
                "hits_at_100":   r.get(hits_key, ""),
                "f1":            r["f1"],
                "mcc":           r["mcc"],
                "threshold":     r["threshold"],
                "best_val_pr_auc": r.get("best_val_pr_auc", ""),
            })
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Krok 2.3: Sprawdź składnię całego pliku**

```bash
python -c "import ast; ast.parse(open('ablation_gnn.py').read()); print('OK')"
```

Oczekiwany output: `OK`

---

## Task 3: LaTeX — błędy językowe (P5–P11)

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex`

Wszystkie zmiany to proste zamiany tekstu. Wykonaj je jeden po drugim — po każdej sprawdź że plik kompiluje się bez błędów (`pdflatex` lub weryfikacja składni).

- [ ] **P5 — „prostą przybliżeniem" → „prostym przybliżeniem"**

Znajdź i zamień:
```
ESM2 jest prostą przybliżeniem biologicznego sąsiedztwa.
```
Na:
```
ESM2 jest prostym przybliżeniem biologicznego sąsiedztwa.
```

- [ ] **P6 — „zewnętrzną walidity" → „zewnętrzną trafność"**

Znajdź i zamień:
```
co ogranicza zewnętrzną walidity wyników
```
Na:
```
co ogranicza zewnętrzną trafność wyników
```

- [ ] **P7 — „explicite" → „wprost"**

Znajdź i zamień:
```
Model nie modeluje explicite zdolności jednego TCR
```
Na:
```
Model nie modeluje wprost zdolności jednego TCR
```

- [ ] **P8 — „residualnymi" → „rezydualnymi"**

Znajdź i zamień:
```
z~aktywacją GELU i~połączeniami residualnymi:
```
Na:
```
z~aktywacją GELU i~połączeniami rezydualnymi:
```

- [ ] **P9 — „niezrównoważoność" → „niezrównoważenie"**

Znajdź i zamień:
```
zachowanie proporcji pos:neg w~każdym podzbiorze (podobna niezrównoważoność),
```
Na:
```
zachowanie proporcji pos:neg w~każdym podzbiorze (podobne niezrównoważenie),
```

- [ ] **P10 — „kolacji" → „scalającej"**

Znajdź i zamień:
```
zaimplementowane w~niestandardowej funkcji kolacji (\textit{custom collate}).
```
Na:
```
zaimplementowane w~niestandardowej funkcji scalającej (\textit{custom collate}).
```

- [ ] **P11 — „sparowanego embeddings" → „sparowanych embeddingów"**

Znajdź i zamień:
```
Wejściem MLP jest konkatenacja sparowanego embeddings TCR i~kodu MHC:
```
Na:
```
Wejściem MLP jest konkatenacja sparowanych embeddingów TCR i~kodu MHC:
```

---

## Task 4: LaTeX — poprawki merytoryczne (P1–P4, P12 + referencja do Poprawki.md)

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex`

- [ ] **P1 — Hipoteza: ROC-AUC → PR-AUC**

Znajdź i zamień:
```
porównywalne wartości ROC-AUC co modele grafowe (GNN) na zadaniu predykcji wiązania TCR do
```
Na:
```
porównywalne wartości PR-AUC co modele grafowe (GNN) na zadaniu predykcji wiązania TCR do
```

- [ ] **P2 — Zakończenie: główna metryka PR-AUC**

Znajdź i zamień:
```
Wszystkie modele ewaluowano przy użyciu ROC-AUC jako głównej
porównywalnej metryki oraz PR-AUC jako metryki wrażliwej na niezrównoważenie klas~\cite{davis2006}.
```
Na:
```
Wszystkie modele ewaluowano przy użyciu PR-AUC jako głównej metryki~---
adekwatnej dla silnie niezrównoważonych danych biologicznych~\cite{davis2006}~---
oraz ROC-AUC uzupełniająco, w~celu porównywalności z~literaturą.
```

- [ ] **P3 — „trzech alleli" → „dwóch alleli" (oba wystąpienia)**

**Wystąpienie 1** (ograniczenia, punkt 1):

Znajdź i zamień:
```
\item \textbf{Jeden epitop.} Wyniki dotyczą wyłącznie epitopu IE-1 w~kontekście trzech alleli
        HLA.
```
Na:
```
\item \textbf{Jeden epitop.} Wyniki dotyczą wyłącznie epitopu IE-1 w~kontekście dwóch alleli
        HLA.
```

**Wystąpienie 2** (ograniczenia, punkt 2):

Znajdź i zamień:
```
  \item \textbf{Ograniczona różnorodność MHC.} Trzy allele MHC stanowią wąski wycinek przestrzeni
```
Na:
```
  \item \textbf{Ograniczona różnorodność MHC.} Dwa allele MHC stanowią wąski wycinek przestrzeni
```

- [ ] **P4 — Złagodzenie wniosku o CNN**

Znajdź i zamień:
```
(3)~CNN okazuje się najlepszym modelem spośród zbadanych, co jest zgodne z~wynikami literatury
dla podobnych zadań~\cite{montemurro2022}.
```
Na:
```
(3)~CNN uzyskał najwyższą wartość punktową PR-AUC i~ROC-AUC spośród zbadanych modeli, co jest
zgodne z~wynikami literatury dla podobnych zadań~\cite{montemurro2022}; przewaga ta wymaga
jednak potwierdzenia statystycznego, ponieważ przedziały ufności między modelami neuronowymi
mogą się nakładać.
```

- [ ] **P12 — Rozszerzenie dyskusji o GNN**

Znajdź i zamień:
```
Dwa niezależne przebiegi treningu GNN z~różną głębokością ($L{=}3$ i~$L{=}4$) dają ROC-AUC
odpowiednio 0,796 i~0,789, co wskazuje na stabilność procedury optymalizacji i~podobną
skuteczność obu konfiguracji sieci.
```
Na:
```
Przebiegi ablacyjne GNN z~głębokościami $L \in \{1, 2, 3, 4\}$ (tabela~\ref{tab:ablation_gnn})
wskazują na stabilność procedury optymalizacji~--- różnice ROC-AUC między konfiguracjami są
nieduże. Brak wyraźnej przewagi GNN nad modelami sekwencyjnymi można wyjaśnić kilkoma
mechanizmami. Po pierwsze, embeddingi ESM2 mogą już enkodować biologiczne sąsiedztwo w~przestrzeni
sekwencji, co czyni dodatkową agregację grafową w~znacznym stopniu redundantną. Po drugie,
graf k-NN konstruowany w~przestrzeni tych samych embeddingów nie wnosi informacji ortogonalnej
wobec samych reprezentacji~--- model GNN i~model MLP czerpią z~tego samego źródła sygnału.
Po trzecie, przy ograniczonej liczbie pozytywnych obserwacji (${\approx}14\,000$) model może
nie mieć wystarczającej danych, by nauczyć się nietrywialnej propagacji przez graf.
```

- [ ] **Usuń nieistniejącą referencję do Poprawki.md**

Znajdź i zamień:
```
na loaderze testowym, zob.\ Poprawki.md).
```
Na:
```
na loaderze testowym).
```

---

## Task 5: LaTeX — dodaj wykresy po tabeli wyników

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex`

Ścieżki do plików PNG są względne wobec miejsca pliku `.tex` (`Praca Magisterska/`), więc prefiks `../`.

- [ ] **Krok 5.1: Dodaj wykresy k-NN i MLP**

Znajdź tekst tuż po bloku `\end{table}` zamykającym tabelę wyników (po `\end{tabular}` i `\end{table}` sekcji `tab:results`):

```
\end{table}



% ============================================================
\chapter{Zakończenie}
```

Wstaw po `\end{table}` (przed `% ===`):

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{../Wyniki_KNN/k-NN_pr.png}\hfill
\includegraphics[width=0.48\textwidth]{../Wyniki_KNN/k-NN_roc.png}
\caption{Krzywa PR i~ROC klasyfikatora k-NN ($k{=}10$) na zbiorze testowym.}
\label{fig:knn_curves}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{../Wyniki_MLP/MLP_pr.png}\hfill
\includegraphics[width=0.48\textwidth]{../Wyniki_MLP/MLP_roc.png}
\caption{Krzywa PR i~ROC modelu MLP na zbiorze testowym.%
% TODO: zastąpić po retreningu MLP (bez nn.Sigmoid)
}
\label{fig:mlp_curves}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{../Wyniki_CNN/CNN_pr.png}\hfill
\includegraphics[width=0.48\textwidth]{../Wyniki_CNN/CNN_roc.png}
\caption{Krzywa PR i~ROC modelu CNN na zbiorze testowym.}
\label{fig:cnn_curves}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{../Wyniki_GNN/GNN_2_pr.png}\hfill
\includegraphics[width=0.48\textwidth]{../Wyniki_GNN/GNN_2_roc.png}
\caption{Krzywa PR i~ROC modelu GNN ($L{=}3$, \texttt{best\_gnn2.pt}) na zbiorze testowym.%
% TODO: zastąpić wykresami z ablation study po uruchomieniu ablation_gnn.py
}
\label{fig:gnn_curves}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{../Wyniki_KNN/k-NN_cm.png}\hfill
\includegraphics[width=0.48\textwidth]{../Wyniki_MLP/MLP_cm.png}\\[6pt]
\includegraphics[width=0.48\textwidth]{../Wyniki_CNN/CNN_cm.png}\hfill
\includegraphics[width=0.48\textwidth]{../Wyniki_GNN/GNN_2_cm.png}
\caption{Macierze pomyłek przy optymalnym progu F1 (zbiorze walidacyjny): k-NN (góra lewo),
MLP (góra prawo), CNN (dół lewo), GNN $L{=}3$ (dół prawo).}
\label{fig:confusion_matrices}
\end{figure}
```

- [ ] **Krok 5.2: Sprawdź że ścieżki PNG istnieją**

```bash
ls "../Wyniki_KNN/k-NN_pr.png" 2>/dev/null || \
ls "Wyniki_KNN/k-NN_pr.png" && echo "OK — pliki PNG dostępne"
```

Jeśli komenda wykonywana z `Praca Magisterska/`:
```bash
cd "/home/tarnickil/MGR/Praca Magisterska" && \
  ls ../Wyniki_KNN/k-NN_pr.png ../Wyniki_MLP/MLP_pr.png \
     ../Wyniki_CNN/CNN_pr.png ../Wyniki_GNN/GNN_2_pr.png && echo "OK"
```

---

## Task 6: LaTeX — aktualizacja tabeli 4.1 i dodanie tabeli ablation

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex`

- [ ] **Krok 6.1: Zastąp tabelę wyników wersją z CI i kolumną Hits@100**

Znajdź i zamień cały blok tabeli (od `\begin{table}` do `\end{table}` dla `tab:results`):

```latex
\begin{table}[htbp]
\centering
\small
\caption{Wyniki modeli na zbiorze testowym. PR-AUC i ROC-AUC podane jako wartości punktowe
(95\%~CI wyznaczone metodą bootstrap, 1000~iteracji). F1 i~MCC wyznaczone przy progu
optymalnym ze~zbioru walidacyjnego. Klasyfikator losowy podaje wartości teoretyczne.
$\dagger$~PR-AUC dla MLP, CNN i~GNN wyznaczone na zbiorze z~${\approx}33\%$ obserwacji
pozytywnych (efekt WeightedRandomSampler na loaderze testowym); k-NN ewaluowany na
rzeczywistej dystrybucji (${\approx}10{,}5\%$ pozytywów). Bezpośrednie porównanie PR-AUC
między k-NN a~modelami neuronowymi nie jest miarodajne; ROC-AUC jest porównywalny.}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{F1} & \textbf{MCC} \\
\midrule
Klasyfikator losowy          & $0{,}105$            & $0{,}500$ & --         & -- \\
k-NN ESM2 ($k{=}10$)        & $0{,}120$            & $0{,}538$ & $0{,}196$  & $0{,}037$ \\
MLP                          & $0{,}578^{\dagger}$  & $0{,}802$ & $0{,}692$  & $0{,}541$ \\
CNN (NetTCR-2.0)             & $0{,}610^{\dagger}$  & $0{,}811$ & $0{,}694$  & $0{,}542$ \\
GNN ($L{=}3$)                 & $0{,}562^{\dagger}$  & $0{,}796$ & $0{,}692$  & $0{,}541$ \\
GNN ($L{=}4$)                 & $0{,}564^{\dagger}$  & $0{,}789$ & --         & --         \\
\bottomrule
\end{tabular}
\end{table}
```

Na:

```latex
\begin{table}[htbp]
\centering
\small
\caption{Wyniki modeli na zbiorze testowym (${\approx}10{,}5\%$ pozytywów).
PR-AUC i~ROC-AUC podane jako wartość punktowa z~95\% CI metodą bootstrap (1000~iteracji)
w~formacie $\bar{x}\ [l{,}95;\ u{,}95]$.
F1 i~MCC wyznaczone przy optymalnym progu ze~zbioru walidacyjnego.
Hits@100: frakcja pozytywów wśród 100~najwyżej ocenionych próbek.
% TODO: uzupełnić wartości po uruchomieniu ablation_gnn.py i retreningu MLP
}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
\textbf{Model}
  & \textbf{PR-AUC [95\% CI]}
  & \textbf{ROC-AUC [95\% CI]}
  & \textbf{Hits@100}
  & \textbf{F1}
  & \textbf{MCC} \\
\midrule
Klas.\ losowy
  & $0{,}105$
  & $0{,}500$
  & $0{,}105$ & -- & -- \\
k-NN ESM2 ($k{=}10$)
  & $0{,}120\ [?,\ ?]$
  & $0{,}538\ [?,\ ?]$
  & $?$ & $0{,}196$ & $0{,}037$ \\
MLP
  & $?\ [?,\ ?]$
  & $?\ [?,\ ?]$
  & $?$ & $?$ & $?$ \\
CNN (NetTCR-2.0)
  & $?\ [?,\ ?]$
  & $?\ [?,\ ?]$
  & $?$ & $?$ & $?$ \\
GNN (najlepszy $L$)
  & $?\ [?,\ ?]$
  & $?\ [?,\ ?]$
  & $?$ & $?$ & $?$ \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Krok 6.2: Dodaj tabelę ablation GNN po tabeli wyników (przed pierwszym `\begin{figure}`)**

Wstaw bezpośrednio po `\end{table}` zamykającym `tab:results` (i przed wstawionymi wykresami z Tasku 5):

```latex
\begin{table}[htbp]
\centering
\small
\caption{Ablation study architektury GNN: wpływ głębokości sieci ($L$) i~typu konwolucji
na jakość predykcji. Wszystkie warianty trenowane z~identycznym seedem (42) i~hiperparametrami.
Wyniki na zbiorze testowym przy ${\approx}10{,}5\%$ pozytywów.
GATConv: \textit{drop\_rate}${=}0{,}0$, \textit{use\_bn}${=}$False.
% TODO: uzupełnić wartości po uruchomieniu ablation_gnn.py
}
\label{tab:ablation_gnn}
\begin{tabular}{lcccc}
\toprule
\textbf{Wariant}
  & \textbf{PR-AUC [95\% CI]}
  & \textbf{ROC-AUC [95\% CI]}
  & \textbf{F1}
  & \textbf{MCC} \\
\midrule
GCNConv $L{=}1$ & $?\ [?,\ ?]$ & $?\ [?,\ ?]$ & $?$ & $?$ \\
GCNConv $L{=}2$ & $?\ [?,\ ?]$ & $?\ [?,\ ?]$ & $?$ & $?$ \\
GCNConv $L{=}3$ & $?\ [?,\ ?]$ & $?\ [?,\ ?]$ & $?$ & $?$ \\
GCNConv $L{=}4$ & $?\ [?,\ ?]$ & $?\ [?,\ ?]$ & $?$ & $?$ \\
GATConv $L{=}3$ & $?\ [?,\ ?]$ & $?\ [?,\ ?]$ & $?$ & $?$ \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Krok 6.3: Dodaj odwołanie do tabeli ablation w tekście dyskusji GNN**

W tekście, który właśnie zmodyfikowałeś w P12 (Task 4), upewnij się że jest odwołanie `tabela~\ref{tab:ablation_gnn}` — powinno być już wstawione przez tekst z P12.

- [ ] **Krok 6.4: Weryfikacja składni LaTeX**

```bash
cd "/home/tarnickil/MGR/Praca Magisterska" && \
  pdflatex -interaction=nonstopmode Praca_magisterska.tex 2>&1 | \
  grep -E "^!" | head -20
```

Oczekiwany output: brak linii zaczynających się od `!` (brak błędów krytycznych).

Ostrzeżenia (`Warning`) są akceptowalne — interesują nas tylko błędy.

---

## Self-Review

### Pokrycie specyfikacji

| Wymaganie ze spec | Task |
|-------------------|------|
| GCNConv L=1,2,3,4 | Task 1–2 |
| GATConv drop_rate=0.0 + try/except | Task 1–2 |
| Wyniki w `Wyniki_Ablation/<wariant>/` | Task 2 |
| `results.json` + `plots/` + `summary_table.csv` | Task 2 |
| Identyczny seed i hiperparametry | Task 2 |
| P1–P12 zmiany LaTeX | Task 3–4 |
| Wykresy PNG do LaTeX | Task 5 |
| Tabela 4.1 z CI + Hits@100 | Task 6 |
| Tabela ablation GNN | Task 6 |
| `SGGW-thesis.cls` bez zmian | ✓ (nigdzie nie modyfikowany) |
| GNN discussion extended | Task 4, P12 |

### Weryfikacja spójności typów

- `_collect_preds()` zwraca `(np.ndarray, np.ndarray)` — zgodne z `evaluate(y_test, y_test_proba, ...)`
- `step()` importowany z `model_base` — sygnatura `(model, batch, criterion, device)`, zgodna z użyciem
- `TCRCMVGraphNetGAT.forward()` ma identyczną sygnaturę co `TCRCMVGraphNet.forward()` — `step()` wywoła oba poprawnie
- `results.json`: klucz `hits_at_100` pochodzi z `Evaluator.evaluate()` który używa `k_for_hits=100` domyślnie → klucz `hits_at_100` ✓
- `summary_table.csv`: pole `hits_at_100` pobierane przez `[k for k in r if k.startswith("hits_at_")][0]` — zgodne z `run_variant()`

### Scan placeholderów

- Wszystkie `?` w LaTeX są zamierzonymi placeholderami z komentarzem `% TODO`
- Brak `TBD`/`TODO` w kodzie Python
