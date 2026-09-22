# Poprawki recenzji nr 2 (Poprawki 2.txt) — design

**Data:** 2026-06-14
**Źródło zarzutów:** `Poprawki 2.txt` (druga runda recenzji)
**Plik docelowy:** `Praca Magisterska/Praca_magisterska.tex` (+ regeneracja jednego rysunku)
**Zakres:** wszystkie punkty recenzji (1–11)

## Cel

Naprawić błędy merytoryczne i metryczne wskazane w drugiej rundzie recenzji. Wszystkie
punkty zweryfikowano empirycznie wobec danych (`graph_ready_data/`, `training_ready_data/`),
kodu (`model_base.py`, `Evaluator.py`, `evaluate_on_gnn_testset.py`, `ablation_gnn.py`) i
tekstu pracy. Werdykt: recenzent ma rację we wszystkich punktach merytorycznych; dwa
(k-NN+MHC, sigmoid) to głównie doprecyzowanie tekstu — kod jest poprawny.

## Ustalenia z autorem (decyzje)

1. **Zakres:** wszystkie punkty 1–11.
2. **Hits@100 (pkt 3):** przemianować na **Recall@100 / HitRate@100** — kod już liczy recall;
   wartości modeli bez zmian, poprawić podpis i baseline. (NIE przeliczamy na Precision@100.)
3. **Podwójne ważenie (pkt 8):** wariant **C** — dyskusja/ograniczenie **+ mała, samodzielna
   ablacja tylko dla MLP** (sampler-only / pos_weight-only / oba), osobna tabela. NIE ruszamy
   tabeli głównej 5.1; NIE retrenujemy CNN/GNN.
4. **Bez retreningu** modeli głównych. Jedyna regeneracja artefaktu: rys. PR GNN z zapisanych
   predykcji.

## Dowody z danych (kluczowe)

Zbiór modelowany (`Train/Val/Test.data.pkl`) = epitop **IE-1 CMV**, dwa kompleksy pMHC:

| pMHC w danych | peptyd | allel | białko | pozytywy (wszystkie splity) |
|---|---|---|---|---|
| `A0301_KLGGALQAK_IE-1_CMV` | KLGGALQAK | HLA-A*03:01 | IE-1 | ~8784 (train) |
| `A2402_AYAQKIFKI_IE-1_CMV` | AYAQKIFKI | HLA-A*24:02 | IE-1 | **30** (22+3+5) |

- `QYDPVAALF` to **pp65 / A*24:02** (plik `A2402_QYDPVAALF_pp65_CMV`), **nieobecny** w zbiorze modelowanym.
- A*24:02 **prezentuje** IE-1 (AYAQKIFKI) i ma 30 realnych pozytywów (~0,04%) — twierdzenie
  „nie prezentuje IE-1 / negatywne z definicji" jest błędne.
- `pos_weight`: MLP/CNN ≈ 8,7 (76782/8806), GNN = 2,0; loadery treningowe `class_balanced=True`
  (sampler 50/50) → nadkorekta ku pozytywom (objaw: GNN recall 98%, FP=2190).
- k-NN: `evaluate_on_gnn_testset.py:128-134` dołącza flagę MHC → porównywalny.
- Hits@K: `Evaluator.py:601-606` liczy `pozytywy_w_topK / suma_pozytywów` = Recall@K.

## Plan zmian (per punkt)

### Część A — Biologia (pkt 1–2)
- **A1 (linia 214):** przepisać opis zadania na dwa kompleksy pMHC IE-1 CMV
  (KLGGALQAK/A*03:01, AYAQKIFKI/A*24:02). Usunąć `QYDPVAALF` i atrybucję IE-1/A*03:01.
- **A2 (linie 83, 679, 1346–1348, 1444, 1480, 1502, 1594):** zamienić „A*24:02 nie prezentuje
  IE-1 / negatywne z definicji" na: A*24:02 prezentuje IE-1 (AYAQKIFKI), lecz ma skrajnie mało
  pozytywów (30; ~0,04%) — ta dysproporcja, nie nieprezentowanie, napędza artefakt allelu.
  Wnioski o artefakcie (predyktor allelowy ROC 0,777/0,82; spadek do 0,55–0,59) pozostają.
- **A3 (tabela ~linia 671):** uspójnić „≈30, rzadki sygnał" z nowym opisem (30 pozytywów jako
  realne, lecz rzadkie obserwacje A*24:02/AYAQKIFKI).

### Część B — Metryki i rysunki (pkt 3–5)
- **B1 (pkt 3):** definicję (814–823), podpis tab. 5.1 (linia 1161) i nazwę kolumny zmienić na
  **Recall@100 (HitRate@100)**; baseline losowy `0,333 → 0,013` (=100/7707). Wartości modeli
  bez zmian. Sprawdzić też tab. ablacji (jeśli używa Hits@100).
- **B2 (pkt 4):** przegenerować rys. 5.4 (`figures/gcn_L1_pr.png`, ew. ROC) z baseline PR
  **0,333**, z `Wyniki_Diff/aligned_predictions.npz` (p_gnn/y) lub checkpointu `gcn_L1`. Bez retreningu.
- **B3 (pkt 5):** podpis rys. 5.5 (linia 1262) „zbiór walidacyjny" → „zbiór testowy (n=7707),
  próg dobrany na walidacji".

### Część C — Metodologia i dodatki (pkt 6–11)
- **C1 (pkt 6):** dodać tabelę „rozkład naturalny" (k-NN/MLP/CNN na pełnym teście ~10,6%,
  baseline PR 0,106) z `results.json` w `Wyniki_*`. GNN pominąć z notką (podgrafy tylko dla
  podpróbki). Wnioski oprzeć na rozkładzie naturalnym + A0301-only.
- **C2 (pkt 7):** uwypuklić `sec:within_allele` jako główną ocenę merytoryczną — wzmianka w
  streszczeniu/wnioskach + zdanie ramujące agregat jako częściowo artefaktowy; odnośniki w przód.
  (Sekcja zostaje na miejscu.)
- **C3 (pkt 8 — wariant C):** akapit dyskusji/ograniczenia (redundancja sampler+pos_weight,
  nadkorekta ku pozytywom, objaw recall/FP; brak wpływu na ranking PR-AUC/ROC-AUC) **+** mała
  ablacja MLP: 3 reżimy (sampler-only / pos_weight-only / oba), osobna tabela. Wymaga krótkiego
  skryptu treningowego MLP ×3 i nowej tabeli; nie rusza tab. 5.1.
- **C4 (pkt 9):** doprecyzować, że k-NN = `[ESM2α; ESM2β; flaga MHC]` → porównywalny z modelami
  (sekcja k-NN, ~linia 897).
- **C5 (pkt 10):** wzory wyjścia (eq. 920 MLP i analogiczne CNN/GNN) na `s=f(x)`, `p̂=σ(s)`,
  z adnotacją „do straty trafia logit s; sigmoid tylko do ewaluacji". Usunąć sigmoid z wyjścia
  (relikt starej architektury; obecny kod zwraca logit).
- **C6 (pkt 11):** dodać jednozdaniowe ograniczenie — bootstrap per-obserwacja; bardziej
  rygorystyczny byłby cluster bootstrap po unikatowych TCR.

## Poza zakresem
- Retrening CNN/GNN ani zmiana tabeli głównej 5.1.
- Przeliczanie Hits na Precision@100.
- Fizyczne przenoszenie sekcji within-allele (tylko reframing + odnośniki).

## Ryzyka
- Edycje biologiczne (A1–A3) dotykają ~10 miejsc — ryzyko przeoczenia wystąpienia; po edycji
  pełny grep `QYDPVAALF`, „nie prezentuje", „z definicji".
- Ablacja MLP (C3) musi być spójna metodologicznie z resztą (ten sam seed, splity, ewaluacja).
- Regeneracja rys. GNN (B2) musi użyć tej samej wersji predykcji co tab. 5.1 (n=7707, 33%).
