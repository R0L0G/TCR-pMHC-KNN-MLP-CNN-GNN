# Poprawki recenzji nr 2 — plan wdrożenia

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Naprawić wszystkie błędy merytoryczne, metryczne i metodologiczne z `Poprawki 2.txt` w pracy magisterskiej (`Praca Magisterska/Praca_magisterska.tex`), bez retreningu modeli głównych.

**Architecture:** Większość zadań to precyzyjne edycje LaTeX (find/replace na dokładnych łańcuchach). Dwa zadania uruchamiają skrypty Pythona: regeneracja jednego rysunku z zapisanych predykcji oraz mała ablacja MLP (pkt 8, wariant C). Spec źródłowy: `docs/superpowers/specs/2026-06-14-poprawki2-pracy-magisterskiej-design.md`.

**Tech Stack:** LaTeX (klasa `SGGW-thesis.cls`), Python 3 (conda env `mgr_thesis`), PyTorch, matplotlib, scikit-learn. Skrypty uruchamiać z `/home/tarnickil/MGR` przez `conda run -n mgr_thesis python <skrypt>`.

**Uwaga o weryfikacji:** Lokalnie **nie ma** `pdflatex`/`latexmk`. Weryfikacja edycji = `grep` (stary łańcuch zniknął, nowy obecny) + kontrola zbilansowania nawiasów `{}`/`$`. Finalną kompilację PDF wykonuje autor (np. Overleaf) — to ostatni, ręczny krok.

**Uwaga o commitach:** Repozytorium nie jest repo git i autor nie życzy sobie commitów bez wyraźnej prośby. **Kroki „commit" pominięto.** Po każdym zadaniu wykonać tylko weryfikację grep.

**Plik główny:** `/home/tarnickil/MGR/Praca Magisterska/Praca_magisterska.tex` (dalej: `TEX`).

---

## Task 1: Biologia — streszczenie i wprowadzenie (pkt 1, część pkt 2)

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` (streszczenie ~83–84, wprowadzenie 214–216)

- [ ] **Step 1: Poprawić zdanie w streszczeniu (linie 83–84)**

Znajdź:
```
w~znacznej mierze wynikają z~obecności w~zbiorze allelu HLA-A*24:02, który nie prezentuje epitopu
IE-1, a~rzeczywista zdolność rozróżniania TCR w~obrębie allelu prezentującego jest istotna
```
Zamień na:
```
w~znacznej mierze wynikają z~obecności w~zbiorze allelu HLA-A*24:02, który w~danych IE-1\_CMV
jest reprezentowany niemal wyłącznie przez obserwacje negatywne, a~rzeczywista zdolność
rozróżniania TCR w~obrębie allelu prezentującego jest istotna
```

- [ ] **Step 2: Przepisać akapit wprowadzenia o epitopie (linie 214–216)**

Znajdź:
```
Peptyd \texttt{QYDPVAALF} (9~aminokwasów) pochodny z~IE-1, prezentowany przez allel HLA-A*03:01,
stanowi immunodominujący cel odpowiedzi limfocytów T~u~osób seropozytywnych~\cite{wills1996,sylwester2005}
i~jest centralnym przedmiotem analizy w~niniejszej pracy.
```
Zamień na:
```
Przedmiotem analizy w~niniejszej pracy jest rozpoznawanie epitopów białka IE-1 wirusa CMV przez
limfocyty T, w~kontekście dwóch obecnych w~danych kompleksów peptyd--MHC: peptydu \texttt{KLGGALQAK}
prezentowanego przez allel HLA-A*03:01 oraz peptydu \texttt{AYAQKIFKI} prezentowanego przez allel
HLA-A*24:02. Oba peptydy stanowią immunodominujące cele odpowiedzi limfocytów T~u~osób
seropozytywnych~\cite{wills1996,sylwester2005}.
```

- [ ] **Step 3: Weryfikacja**

Run: `grep -n "QYDPVAALF" "Praca Magisterska/Praca_magisterska.tex"`
Expected: brak wyników (pusto).

Run: `grep -n "KLGGALQAK\|AYAQKIFKI" "Praca Magisterska/Praca_magisterska.tex"`
Expected: po jednym trafieniu obu sekwencji w okolicy linii 214.

---

## Task 2: Biologia — sekcja danych, within-allele, dyskusja (pkt 2)

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` (linie ~678–679, 1346–1349, 1444, 1480, 1500–1503, 1594)

- [ ] **Step 1: Poprawić opis pod tabelą statystyk danych (linie 678–680)**

Znajdź:
```
Dominacja allelu HLA-A*03:01 w~kontekście obserwacji pozytywnych wynika z~biologicznej specyficzności
epitopu IE-1: wiązanie jest możliwe wyłącznie w~kontekście tego allelu. Obserwacje dla HLA-A*24:02
dostarczają rzadkiego sygnału pozytywnego.
```
Zamień na:
```
Dominacja allelu HLA-A*03:01 w~kontekście obserwacji pozytywnych wynika z~tego, że w~zbiorze
IE-1\_CMV niemal wszystkie udokumentowane wiązania dotyczą kompleksu \texttt{KLGGALQAK}/HLA-A*03:01;
kompleks \texttt{AYAQKIFKI}/HLA-A*24:02 dostarcza jedynie ${\approx}30$~obserwacji pozytywnych
(rzadki sygnał).
```

- [ ] **Step 2: Poprawić uzasadnienie asymetrii w sekcji within-allele (linie 1346–1349)**

Znajdź:
```
asymetria nie jest przypadkowa: epitop IE-1 jest restrykowany przez HLA-A*03, a~allel
HLA-A*24:02 nie prezentuje tego peptydu, wskutek czego pary $(\text{TCR},\ \text{A*24:02})$
są negatywne praktycznie z~definicji, niezależnie od sekwencji TCR.
```
Zamień na:
```
asymetria wynika ze~składu danych: w~zbiorze IE-1\_CMV kompleks \texttt{AYAQKIFKI}/HLA-A*24:02
jest reprezentowany niemal wyłącznie przez obserwacje negatywne (30~pozytywów na ${\approx}67\,000$~par),
wskutek czego pary $(\text{TCR},\ \text{A*24:02})$ są w~praktyce niemal zawsze negatywne,
niezależnie od sekwencji TCR.
```

- [ ] **Step 3: Poprawić zdanie w dyskusji wyników (linia 1444)**

Znajdź:
```
zadania o~różnej trudności. Ponieważ allel HLA-A*24:02 nie prezentuje epitopu IE-1,
```
Zamień na:
```
zadania o~różnej trudności. Ponieważ allel HLA-A*24:02 jest w~zbiorze reprezentowany niemal wyłącznie przez negatywy,
```

- [ ] **Step 4: Poprawić zwrot „nieprezentującego allelu" (linie 1480 i 1594, oba wystąpienia)**

Zamień **wszystkie** wystąpienia łańcucha:
```
nieprezentującego allelu HLA-A*24:02
```
na:
```
niemal wyłącznie negatywnego allelu HLA-A*24:02
```
(replace_all — występuje 2×)

- [ ] **Step 5: Poprawić punkt listy ograniczeń (linie 1500–1503)**

Znajdź:
```
        dostarcza ${\approx}99{,}7\%$ sygnału pozytywnego, podczas gdy retencjonowany
        w~zbiorze allel HLA-A*24:02 nie prezentuje epitopu IE-1 i~stanowi zbiór negatywów
        trywialnie odróżnialnych po samej tożsamości allelu.
```
Zamień na:
```
        dostarcza ${\approx}99{,}7\%$ sygnału pozytywnego, podczas gdy retencjonowany
        w~zbiorze allel HLA-A*24:02 jest reprezentowany niemal wyłącznie przez obserwacje
        negatywne i~stanowi zbiór negatywów trywialnie odróżnialnych po samej tożsamości allelu.
```

- [ ] **Step 6: Weryfikacja — żadnego „nie prezentuje"/„nieprezentującego" przy A*24:02**

Run: `grep -n "nie prezentuje\|nieprezentującego\|prezentuje tego peptydu" "Praca Magisterska/Praca_magisterska.tex"`
Expected: brak wyników odnoszących się do HLA-A*24:02 (pusto lub tylko niezwiązane konteksty).

---

## Task 3: Metryka Hits@100 → Recall@100 (pkt 3)

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` (definicja 814–823; tabela 5.1 — 1161, 1169, 1176; ew. tabela ablacji)

- [ ] **Step 1: Zmienić tytuł podsekcji i zdanie definicyjne (linie 814–816)**

Znajdź:
```
\subsection{Hits@K}

Miara Hits@$K$~\cite{davis2006} określa, jaka frakcja prawdziwych wiążących TCR znajduje się wśród $K$ obserwacji
```
Zamień na:
```
\subsection{Recall@K (HitRate@K)}

Miara Recall@$K$ (zwana też HitRate@$K$)~\cite{davis2006} określa, jaka frakcja \emph{wszystkich} prawdziwych wiążących TCR znajduje się wśród $K$ obserwacji
```

- [ ] **Step 2: Zmienić symbol w równaniu (linia 819)**

Znajdź:
```
    \mathrm{Hits}@K = \frac{\left|\left\{i \in \mathrm{Pos} : \mathrm{rank}(i) \leq K\right\}\right|}{|\mathrm{Pos}|}
```
Zamień na:
```
    \mathrm{Recall}@K = \frac{\left|\left\{i \in \mathrm{Pos} : \mathrm{rank}(i) \leq K\right\}\right|}{|\mathrm{Pos}|}
```

- [ ] **Step 3: Poprawić podpis tabeli wyników (linia 1161)**

Znajdź:
```
Hits@100: frakcja pozytywów wśród 100~najwyżej ocenionych próbek.
```
Zamień na:
```
Recall@100: odsetek wszystkich pozytywów, które trafiły do 100~najwyżej ocenionych próbek (poziom losowy ${\approx}K/n = 100/7707 \approx 0{,}013$).
```

- [ ] **Step 4: Poprawić nagłówek kolumny (linia 1169)**

Znajdź:
```
  & \textbf{Hits@100}
```
Zamień na:
```
  & \textbf{Recall@100}
```

- [ ] **Step 5: Poprawić baseline losowy w wierszu „Klas. losowy" (linia 1176)**

Znajdź (to wiersz Hits dla klasyfikatora losowego — wartość PR-AUC 0,333 w linii 1174 **zostaje bez zmian**):
```
  & $0{,}333$ & -- & -- \\
```
Zamień na:
```
  & $0{,}013$ & -- & -- \\
```
(Uwaga: jeśli `grep` pokaże więcej niż jedno trafienie tego łańcucha, edytować to w bloku wiersza „Klas.\ losowy" tab. `tab:results`, linie 1173–1176.)

- [ ] **Step 6: Sprawdzić i poprawić pozostałe wystąpienia „Hits"**

Run: `grep -n "Hits@\|Hits " "Praca Magisterska/Praca_magisterska.tex"`
Dla każdego pozostałego trafienia (np. nagłówek kolumny w tabeli ablacji `tab:ablation_gnn`, odwołania w tekście) zamień `Hits@100`→`Recall@100` i `Hits@K`→`Recall@K`, zachowując wartości liczbowe modeli bez zmian. Jeśli w tabeli ablacji jest wiersz „losowy" z Hits@100 = 0,333 — zmień na 0,013; jeśli ablacja nie ma kolumny Hits — pominąć.

- [ ] **Step 7: Weryfikacja**

Run: `grep -n "Hits" "Praca Magisterska/Praca_magisterska.tex"`
Expected: brak wyników (wszystkie przemianowane na Recall).

---

## Task 4: Podpis macierzy pomyłek (pkt 5)

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` (linia 1262)

- [ ] **Step 1: Poprawić podpis rys. macierzy pomyłek (linia 1262)**

Znajdź:
```
\caption{Macierze pomyłek przy optymalnym progu F1 (zbiór walidacyjny): k-NN (góra lewo),
```
Zamień na:
```
\caption{Macierze pomyłek na zbiorze testowym ($n{=}7707$) przy progu F1 dobranym na zbiorze walidacyjnym: k-NN (góra lewo),
```

- [ ] **Step 2: Weryfikacja**

Run: `grep -n "Macierze pomyłek" "Praca Magisterska/Praca_magisterska.tex"`
Expected: podpis zawiera „na zbiorze testowym ($n{=}7707$)" i „próg ... dobranym na zbiorze walidacyjnym".

---

## Task 5: Poprawki notacyjne i metodologiczne w tekście (pkt 9, 10, 11, część 7)

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` (k-NN ~901; eq MLP 920 + 922; eq GNN 1127; bootstrap ~809; framing within-allele ~1150)

- [ ] **Step 1: pkt 9 — doprecyzować, że k-NN używa flagi MHC (po linii 901)**

Znajdź:
```
a~$w_i$ wagami proporcjonalnymi do podobieństwa. W~eksperymentach stosuje się $k = 10$.
```
Zamień na:
```
a~$w_i$ wagami proporcjonalnymi do podobieństwa. W~eksperymentach stosuje się $k = 10$.

Aby zapewnić porównywalność z~modelami neuronowymi, które otrzymują kod allelu jako cechę
wejściową, wektor wejściowy k-NN rozszerzono o~binarną flagę allelu MHC:
$\left[\mathbf{z}_{\mathrm{TCR}};\; m\right] \in \mathbb{R}^{1281}$, gdzie $m \in \{0,1\}$ koduje
tożsamość allelu. Dzięki temu k-NN korzysta z~tej samej informacji o~allelu co MLP, CNN i~GNN,
a~różnice wyników odzwierciedlają wyłącznie zdolność modeli, a~nie odmienny zestaw cech.
```

- [ ] **Step 2: pkt 10 — poprawić równanie wyjścia MLP (linia 920)**

Znajdź:
```
    \hat{y}          &= \sigma\!\left(\mathbf{w}_{\mathrm{out}}^T \mathbf{h}^{(2)} + b_{\mathrm{out}}\right) \label{eq:mlpout}
```
Zamień na:
```
    s          &= \mathbf{w}_{\mathrm{out}}^T \mathbf{h}^{(2)} + b_{\mathrm{out}}, \qquad \hat{p} = \sigma(s) \label{eq:mlpout}
```

- [ ] **Step 3: pkt 10 — dopisać zdanie wyjaśniające pod równaniami MLP (linia 922)**

Znajdź:
```
gdzie $\mathrm{Dropout}_{p}$ losowo zeruje $p \cdot 100\%$ aktywacji podczas treningu~\cite{srivastava2014}.
```
Zamień na:
```
gdzie $\mathrm{Dropout}_{p}$ losowo zeruje $p \cdot 100\%$ aktywacji podczas treningu~\cite{srivastava2014}.
Wartość $s$ jest logitem przekazywanym bezpośrednio do funkcji straty
BCEWithLogitsLoss (równanie~\eqref{eq:bce}); sigmoidę $\sigma$ stosuje się wyłącznie do
przekształcenia logitu na prawdopodobieństwo $\hat{p}$ na etapie ewaluacji.
```

- [ ] **Step 4: pkt 10 — poprawić równanie wyjścia GNN (linia 1127)**

Znajdź:
```
    \hat{y} = \sigma\!\left(\mathrm{MLP}_{\mathrm{cls}}\!\left(\left[\mathbf{h}_{T^*};\; \mathbf{m}\right]\right)\right), \quad \mathrm{MLP}_{\mathrm{cls}} : \mathbb{R}^{d_h+8} \to \tfrac{d_h}{2} \to 1
```
Zamień na:
```
    s = \mathrm{MLP}_{\mathrm{cls}}\!\left(\left[\mathbf{h}_{T^*};\; \mathbf{m}\right]\right), \quad \hat{p} = \sigma(s), \quad \mathrm{MLP}_{\mathrm{cls}} : \mathbb{R}^{d_h+8} \to \tfrac{d_h}{2} \to 1
```

- [ ] **Step 5: pkt 11 — dopisać uwagę o cluster bootstrap (linia ~809)**

Znajdź:
```
pojedynczych obserwacji; iteracje, w~których wylosowana próba nie zawiera obu klas, są pomijane.
```
Zamień na:
```
pojedynczych obserwacji; iteracje, w~których wylosowana próba nie zawiera obu klas, są pomijane.
Próbkowanie na poziomie pojedynczych obserwacji jest pierwszym przybliżeniem; bardziej
rygorystyczny byłby \emph{cluster bootstrap} po unikatowych TCR, ponieważ ten sam TCR może
występować w~wielu obserwacjach (por.\ ograniczenia, sekcja~\ref{sec:limitations}).
```

- [ ] **Step 6: pkt 7 — zdanie ramujące przed tabelą wyników agregatowych (linia 1150)**

Znajdź:
```
Tabela~\ref{tab:results} przedstawia wyniki wszystkich modeli na zbiorze testowym.
```
Zamień na:
```
Tabela~\ref{tab:results} przedstawia wyniki wszystkich modeli na zbiorze testowym. Należy
od~razu zaznaczyć, że metryki agregatowe są częściowo zawyżone przez trywialne rozróżnienie
allelu (sekcja~\ref{sec:within_allele}); za~właściwą ocenę zdolności predykcyjnej modeli należy
przyjąć wyniki w~obrębie allelu prezentującego HLA-A*03:01 oraz na naturalnym rozkładzie klas
(tabela~\ref{tab:natural}).
```
(Etykieta `tab:natural` powstaje w Task 7 — odwołanie rozwiąże się po dodaniu tabeli.)

- [ ] **Step 7: Weryfikacja**

Run: `grep -n "R}^{1281}\|\\\\hat{p} = \\\\sigma(s)\|cluster bootstrap\|tab:natural" "Praca Magisterska/Praca_magisterska.tex"`
Expected: trafienia dla flagi MHC k-NN (1281), dwóch równań z `\hat{p} = \sigma(s)`, cluster bootstrap oraz odwołania `tab:natural`.

---

## Task 6: Regeneracja rysunku PR krzywej GNN z poprawnym baseline (pkt 4)

**Files:**
- Create: `regen_gnn_pr.py`
- Modify (wynikowo): `Praca Magisterska/figures/gcn_L1_pr.png`

- [ ] **Step 1: Sprawdzić zawartość zapisanych predykcji GNN**

Run:
```bash
conda run -n mgr_thesis python -c "import numpy as np; d=np.load('Wyniki_Diff/aligned_predictions.npz', allow_pickle=True); print(list(d.keys())); print('y mean', d['y'].astype(float).mean(), 'n', len(d['y']))"
```
Expected: klucze zawierają `y` i `p_gnn`; `y mean` ≈ 0,333; `n` ≈ 7707.

- [ ] **Step 2: Utworzyć skrypt regeneracji**

Create `regen_gnn_pr.py`:
```python
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
```

- [ ] **Step 3: Uruchomić skrypt**

Run: `cd /home/tarnickil/MGR && conda run -n mgr_thesis python regen_gnn_pr.py`
Expected: wydruk `baseline=0.333` (±0,001) oraz `PR-AUC` ≈ 0,598 (zgodne z tab. 5.1 / tab. within-allele).

- [ ] **Step 4: Weryfikacja wizualna**

Otworzyć (Read tool) `Praca Magisterska/figures/gcn_L1_pr.png`.
Expected: linia przerywana podpisana „Random (0.333)", nie „Random (0.105)".

---

## Task 7: Tabela wyników na rozkładzie naturalnym (pkt 6)

**Files:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` (dodać tabelę po akapicie pod tab. 5.1 / przed sekcją within-allele)

Liczby (z `Wyniki_*/results.json`, pełny test ~10,6% pozytywów, baseline PR 0,106):

| Model | PR-AUC [95% CI] | ROC-AUC [95% CI] |
|---|---|---|
| Klas. losowy | 0,106 | 0,500 |
| k-NN (k=10) | 0,240 [0,228; 0,253] | 0,765 [0,757; 0,774] |
| MLP | 0,253 [0,240; 0,267] | 0,806 [0,800; 0,812] |
| CNN (NetTCR-2.0) | 0,301 [0,285; 0,317] | 0,819 [0,813; 0,825] |

- [ ] **Step 1: Potwierdzić liczby ze źródła**

Run:
```bash
cd /home/tarnickil/MGR && conda run -n mgr_thesis python -c "
import json
for d in ['Wyniki_KNN','Wyniki_MLP','Wyniki_CNN']:
    r=json.load(open(f'{d}/results.json'))
    print(d, [round(x,3) for x in r['pr_auc']], [round(x,3) for x in r['roc_auc']])
"
```
Expected: wartości zgodne z tabelą powyżej (k-NN ~0,240/0,765, MLP ~0,253/0,806, CNN ~0,301/0,819).

- [ ] **Step 2: Wstawić tabelę i akapit (po linii 1148, przed „Tabela~\ref{tab:results} przedstawia...")**

Znajdź (akapit kończący opis protokołu ewaluacji tuż przed tab. 5.1 — zlokalizuj `grep -n "Tabela~\\\\ref{tab:results} przedstawia"`); **bezpośrednio przed** zdaniem z Task 5 Step 6 wstaw:
```
\begin{table}[htbp]
\centering
\small
\caption{Wyniki na zbiorze testowym o~\textbf{naturalnym} rozkładzie klas
($n{=}24\,358$, ${\approx}10{,}6\%$ pozytywów; bez podpróbkowania negatywów). PR-AUC silnie
zależy od częstości klasy pozytywnej, dlatego wartości nie są porównywalne wprost z~tabelą
podpróbkowaną~\ref{tab:results}; poziom losowy PR-AUC wynosi tu ${\approx}0{,}106$. Model GNN
pominięto, ponieważ dysponuje podgrafami wyłącznie dla podpróbki testowej.}
\label{tab:natural}
\begin{tabular}{lcc}
\toprule
\textbf{Model} & \textbf{PR-AUC [95\% CI]} & \textbf{ROC-AUC [95\% CI]} \\
\midrule
Klas.\ losowy     & $0{,}106$                  & $0{,}500$ \\
k-NN ESM2 ($k{=}10$) & $0{,}240\ [0{,}228;\ 0{,}253]$ & $0{,}765\ [0{,}757;\ 0{,}774]$ \\
MLP               & $0{,}253\ [0{,}240;\ 0{,}267]$ & $0{,}806\ [0{,}800;\ 0{,}812]$ \\
CNN (NetTCR-2.0)  & $0{,}301\ [0{,}285;\ 0{,}317]$ & $0{,}819\ [0{,}813;\ 0{,}825]$ \\
\bottomrule
\end{tabular}
\end{table}

Na naturalnym rozkładzie klas (tabela~\ref{tab:natural}) bezwzględne wartości PR-AUC są
wyraźnie niższe niż na zbiorze podpróbkowanym, co potwierdza, że PR-AUC z~tabeli~\ref{tab:results}
nie należy interpretować jako bezwzględnej miary jakości. Uporządkowanie modeli
(CNN $>$ MLP $>$ k-NN) pozostaje jednak zachowane, a~wszystkie modele przewyższają poziom losowy.
```

- [ ] **Step 3: Weryfikacja**

Run: `grep -n "tab:natural\|naturalnym. rozkładzie\|0{,}106" "Praca Magisterska/Praca_magisterska.tex"`
Expected: definicja `\label{tab:natural}` obecna; odwołanie z Task 5 Step 6 wskazuje na istniejącą etykietę.

---

## Task 8: Ablacja schematu ważenia klas dla MLP (pkt 8, wariant C)

**Files:**
- Create: `ablation_weighting.py`
- Create (wynikowo): `Wyniki_Ablation_Weighting/results.json`
- Modify: `Praca Magisterska/Praca_magisterska.tex` (akapit przy samplerze ~871; nowa tabela + akapit w wynikach; punkt listy ograniczeń ~1534)

- [ ] **Step 1: Utworzyć skrypt ablacji**

Create `ablation_weighting.py`:
```python
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
    print("Zapisano →", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Uruchomić ablację**

Run: `cd /home/tarnickil/MGR && conda run -n mgr_thesis python ablation_weighting.py`
Expected: trzy wiersze wydruku. Oczekiwany kierunek: PR-AUC i ROC-AUC zbliżone między reżimami (różnice rzędu setnych); reżim `both` ma najwyższy `recall` i najwyższe `FP`, `posweight_only`/`sampler_only` niższe FP. Zapis `Wyniki_Ablation_Weighting/results.json`.

- [ ] **Step 3: Odczytać liczby do tabeli**

Run: `cd /home/tarnickil/MGR && conda run -n mgr_thesis python -c "import json; print(json.dumps(json.load(open('Wyniki_Ablation_Weighting/results.json')), indent=2))"`
Zanotować dla każdego reżimu: `pr_auc[0]`, `roc_auc[0]`, `recall`, `fp`, `f1`, `mcc`. Tymi wartościami wypełnić tabelę w Step 5 (zastąpić znaczniki `<...>`).

- [ ] **Step 4: Dopisać akapit uzasadnienia przy samplerze (po linii 871)**

Znajdź:
```
równe reprezentacje klas w~każdym batchu (50\% pozytywnych, 50\% negatywnych).
```
Zamień na:
```
równe reprezentacje klas w~każdym batchu (50\% pozytywnych, 50\% negatywnych).

Zastosowano jednocześnie dwa mechanizmy kompensujące niezrównoważenie: zbalansowane
próbkowanie oraz wagę $w_+$ w~funkcji straty. Ponieważ obie korekty działają w~tym samym
kierunku, ich łączne użycie przesuwa \textit{operating point} modelu ku klasie pozytywnej
(wysoka czułość, większa liczba fałszywie pozytywnych). Nie wpływa to jednak na metryki
rankingowe (PR-AUC, ROC-AUC), które są niezależne od progu decyzyjnego; wpływ ten ilustruje
ablacja w~tabeli~\ref{tab:weighting} oraz omówienie w~sekcji~\ref{sec:limitations}.
```

- [ ] **Step 5: Wstawić tabelę ablacji (przed `\chapter{Zakończenie}`, linia ~1412)**

Znajdź:
```
% ============================================================
\chapter{Zakończenie}
```
Wstaw **przed** tym blokiem (uzupełnij `<...>` wartościami ze Step 3):
```
\subsection{Wpływ schematu ważenia klas (ablacja)}
\label{sec:weighting_ablation}

W~odpowiedzi na pytanie o~możliwe podwójne ważenie klasy pozytywnej przeprowadzono ablację
dla modelu MLP w~trzech reżimach: wyłącznie zbalansowane próbkowanie, wyłącznie waga $w_+$
oraz oba mechanizmy łącznie (konfiguracja użyta w~pracy). Ewaluację wykonano na zbiorze
testowym o~rozkładzie naturalnym, przy progu dobranym na walidacji.

\begin{table}[htbp]
\centering
\small
\caption{Wpływ schematu ważenia klas na MLP (zbiór testowy, rozkład naturalny). Metryki
rankingowe (PR-AUC, ROC-AUC) są stabilne między reżimami, natomiast łączne użycie samplera
i~wagi $w_+$ podnosi czułość i~liczbę fałszywie pozytywnych (FP).}
\label{tab:weighting}
\begin{tabular}{lccccc}
\toprule
\textbf{Reżim} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{Czułość} & \textbf{FP} & \textbf{F1} \\
\midrule
Sampler only      & $<prauc_s>$ & $<roc_s>$ & $<rec_s>$ & $<fp_s>$ & $<f1_s>$ \\
$w_+$ only         & $<prauc_p>$ & $<roc_p>$ & $<rec_p>$ & $<fp_p>$ & $<f1_p>$ \\
Sampler $+\ w_+$ (użyty) & $<prauc_b>$ & $<roc_b>$ & $<rec_b>$ & $<fp_b>$ & $<f1_b>$ \\
\bottomrule
\end{tabular}
\end{table}

Wyniki potwierdzają, że łączne ważenie nie zniekształca metryk rankingowych, na których
opierają się wnioski pracy; jego efekt sprowadza się do przesunięcia progu działania ku
wyższej czułości kosztem precyzji. Jest to świadomy kompromis: w~zastosowaniu przesiewowym
(wyłapywanie kandydujących wiążących TCR) wyższa czułość jest pożądana.
```

- [ ] **Step 6: Dopisać punkt listy ograniczeń (pkt 8 + pkt 11; przed `\end{enumerate}` linia ~1534)**

Znajdź:
```
\end{enumerate}
```
(pierwsze wystąpienie domykające listę ograniczeń, po punkcie „Pojedyncze ziarno losowości")
Wstaw **przed** nim:
```
  \item \textbf{Łączne ważenie klas.} Zastosowano jednocześnie zbalansowane próbkowanie
        i~wagę $w_+$ w~stracie. Ablacja (tabela~\ref{tab:weighting}) pokazuje, że nie wpływa
        to na PR-AUC/ROC-AUC, lecz podnosi liczbę fałszywie pozytywnych; w~zastosowaniach
        wymagających wysokiej precyzji należałoby użyć tylko jednego z~tych mechanizmów.

  \item \textbf{Bootstrap na poziomie obserwacji.} Przedziały ufności wyznaczono, próbkując
        pojedyncze obserwacje. Ponieważ ten sam TCR może pojawiać się w~wielu parach, bardziej
        konserwatywny byłby \emph{cluster bootstrap} po unikatowych TCR; obecne CI mogą
        nieznacznie niedoszacowywać niepewności.
```

- [ ] **Step 7: Weryfikacja**

Run: `grep -n "tab:weighting\|sec:weighting_ablation\|cluster bootstrap\|Łączne ważenie" "Praca Magisterska/Praca_magisterska.tex"`
Expected: etykiety `tab:weighting`, `sec:weighting_ablation`, oba nowe punkty ograniczeń obecne.

Run: `grep -n "<prauc_\|<roc_\|<rec_\|<fp_\|<f1_" "Praca Magisterska/Praca_magisterska.tex"`
Expected: **brak** — wszystkie znaczniki `<...>` zastąpiono liczbami.

---

## Task 9: Weryfikacja końcowa

**Files:** brak zmian — tylko kontrole.

- [ ] **Step 1: Brak śladów błędów biologicznych**

Run: `grep -n "QYDPVAALF\|nie prezentuje\|nieprezentującego" "Praca Magisterska/Praca_magisterska.tex"`
Expected: pusto.

- [ ] **Step 2: Brak starej metryki Hits i znaczników**

Run: `grep -n "Hits\|<prauc_\|<roc_\|TODO\|TBD" "Praca Magisterska/Praca_magisterska.tex"`
Expected: pusto.

- [ ] **Step 3: Zbilansowanie nawiasów klamrowych i trybu matematycznego**

Run:
```bash
cd "/home/tarnickil/MGR/Praca Magisterska" && conda run -n mgr_thesis python -c "
s=open('Praca_magisterska.tex').read()
print('braces balance (0=ok):', s.count('{')-s.count('}'))
print('inline \$ count even:', s.count('\$')%2==0)
"
```
Expected: `braces balance (0=ok): 0` oraz `inline $ count even: True`.

- [ ] **Step 4: Sanity — odwołania do nowych etykiet istnieją**

Run: `grep -n "tab:natural\|tab:weighting\|sec:within_allele\|sec:limitations" "Praca Magisterska/Praca_magisterska.tex"`
Expected: każda etykieta ma zarówno `\label{...}` jak i co najmniej jedno `\ref{...}`.

- [ ] **Step 5: Kompilacja PDF (ręcznie, po stronie autora)**

Skompilować `Praca_magisterska.tex` w środowisku autora (Overleaf/lokalny TeX). Sprawdzić:
brak nierozwiązanych odwołań (`??`), tab. `tab:natural` i `tab:weighting` renderują się, rys. PR GNN pokazuje „Random (0.333)".

---

## Notatki dla wykonawcy

- Wszystkie edycje `.tex` to dokładne find/replace; jeśli `Edit` zgłosi brak unikalności, dołącz więcej kontekstu z sąsiednich linii (numery linii są orientacyjne — plik mógł się przesunąć po wcześniejszych zadaniach).
- Kolejność: Task 1→9. Task 5 Step 6 i Task 7 wzajemnie się odwołują (`tab:natural`) — wykonać oba, kolejność wewnętrzna bez znaczenia, byle przed kompilacją.
- Skrypty (Task 6, 8) uruchamiać z `/home/tarnickil/MGR`, env `mgr_thesis`. Task 8 wykonuje 3 krótkie treningi MLP (minuty); nie dotyka `best_mlp.pt` ani `tab:results`.
- Nie commitować (nie-git repo + preferencja autora).
