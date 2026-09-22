# Design: Ablation Study GNN + Poprawki recenzji

**Data:** 2026-06-08  
**Zakres:** Skrypt ablation study dla GNN + poprawki merytoryczne i redakcyjne w pracy magisterskiej

---

## 1. Kontekst

Praca magisterska wymaga dwóch rodzajów poprawek po recenzji:

- **Grupę A** — zmiany w LaTeX możliwe teraz (błędy językowe, metryki, wnioski, wykresy)
- **Grupę B** — wymagają nowych liczb z modelu (CI bootstrapu, poprawione PR-AUC); obsługiwane przez placeholdery w LaTeX

Sigmoid bug w MLP (`nn.Sigmoid()` przed `BCEWithLogitsLoss`) został już usunięty przez użytkownika z `model_base.py`. Checkpoint `best_mlp.pt` jest nieważny i wymaga retreningu — MLP będzie trenowany osobno przez użytkownika.

---

## 2. Składowe projektu

### 2.1 Skrypt `ablation_gnn.py`

Nowy plik w katalogu głównym `/home/tarnickil/MGR/`.

**Trenowane warianty (w kolejności):**

| Wariant | Klasa | Parametry |
|---------|-------|-----------|
| GCNConv L=1 | `TCRCMVGraphNet` | `n_gcn_layers=1` |
| GCNConv L=2 | `TCRCMVGraphNet` | `n_gcn_layers=2` |
| GCNConv L=3 | `TCRCMVGraphNet` | `n_gcn_layers=3` |
| GCNConv L=4 | `TCRCMVGraphNet` | `n_gcn_layers=4` |
| GATConv | `TCRCMVGraphNetGAT` | `drop_rate=0.0`, `use_bn=False` |

Wszystkie warianty: identyczny seed (42), identyczne hiperparametry treningu, `class_balanced=False` na val/test loaderach.

**`TCRCMVGraphNetGAT`** — nowa klasa zdefiniowana w `ablation_gnn.py`. Identyczna architektura co `TCRCMVGraphNet`, ale z `GATConv` zamiast `GCNConv`. GATConv owinięty w `try/except` — przy błędzie loguje komunikat i kontynuuje pozostałe warianty.

**Struktura wyników:**

```
Wyniki_Ablation/
  gcn_L1/
    checkpoint.pt
    results.json        # pr_auc, roc_auc (z CI), hits_at_100, threshold, f1, mcc
    plots/
      *_pr.png, *_roc.png, *_cm.png, *_scores.png
  gcn_L2/   (j.w.)
  gcn_L3/   (j.w.)
  gcn_L4/   (j.w.)
  gat/      (j.w., lub katalog pusty z error.log jeśli crash)
  summary_table.csv     # tabela porównawcza wszystkich zakończonych wariantów
```

**Uruchomienie:**
```bash
python ablation_gnn.py
```

Skrypt importuje `TCRCMVGraphNet`, `train_model`, `get_gnn_loader`, `GNNDataset` oraz dane z `model_base.py`. Logi per-wariant trafiają na stdout + do pliku `Wyniki_Ablation/<wariant>/train.log`.

### 2.2 Poprawki `Praca_magisterska.tex`

Plik `SGGW-thesis.cls` pozostaje bez zmian.

**Poprawki merytoryczne (Group A):**

| # | Linia | Zmiana |
|---|-------|--------|
| P1 | 144 | Hipoteza: „porównywalne wartości ROC-AUC" → „porównywalne wartości PR-AUC" |
| P2 | 989–990 | Zakończenie: ROC-AUC jako główna metryka → PR-AUC jako główna, ROC-AUC uzupełniająca |
| P3 | 1032 | „trzech alleli" → „dwóch alleli" |
| P4 | 1106 | Złagodzenie: „CNN okazuje się najlepszym modelem" → sformułowanie warunkowe z odwołaniem do braku CI między modelami |
| P5–P11 | różne | 7 błędów językowych (walidity, explicite, prostą przybliżeniem, residualnymi, niezrównoważoność, kolacji, sparowanego embeddings) |
| P12 | ~1006 | Rozszerzenie dyskusji o GNN: 3–4 zdania o powodach braku przewagi nad modelami sekwencyjnymi |
| P13 | po tabeli 4.1 | Dodanie wykresów PNG z `Wyniki_CNN/`, `Wyniki_GNN/`, `Wyniki_MLP/`, `Wyniki_KNN/` przez `\includegraphics` |

**Placeholdery (Group B — uzupełnić po ablation):**

- Tabela 4.1: obecne wartości punktowe zastąpione przez `\textbf{XX,XXX} [YY,YYY; ZZ,ZZZ]` + komentarz `% TODO: wstaw po ablation`
- Nowa kolumna **Hits@100** w tabeli 4.1 z placeholderami
- Przypis `†` pod tabelą: zaktualizowany by informować, że dane zostaną uzupełnione

---

## 3. Czego projekt NIE obejmuje

- Retraining MLP — użytkownik uruchomi go osobno po przygotowaniu skryptu ablation
- Wiele seedów (3–5×) — poza zakresem (punkt 7 recenzji, priorytet niski)
- Modyfikacja hiperparametrów GNN — ablation dotyczy tylko architektury (L, typ conv)
- Zmiany w `SGGW-thesis.cls`

---

## 4. Kolejność implementacji

1. Stwórz `Wyniki_Ablation/` i strukturę podfolderów
2. Napisz `ablation_gnn.py` z klasą `TCRCMVGraphNetGAT` i pętlą po wariantach
3. Poprawki P1–P12 w `Praca_magisterska.tex`
4. Dodaj wykresy (P13)
5. Wstaw placeholdery do tabeli 4.1 z kolumną Hits@100
