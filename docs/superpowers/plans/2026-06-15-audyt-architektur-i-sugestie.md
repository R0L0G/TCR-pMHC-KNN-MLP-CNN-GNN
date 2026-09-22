# Korekta opisów architektur + sugestie pkt 6–7 — plan wdrożenia

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dopasować opisy architektur MLP/CNN/GNN w pracy do faktycznie wytrenowanych modeli (checkpointów, które dały raportowane wyniki) oraz wpleść sugestie pkt 6–7 (rozkład naturalny, within-allele) do Podsumowania i Wniosków.

**Architecture:** Wyłącznie edycje LaTeX (find/replace dokładnych łańcuchów) w `Praca Magisterska/Praca_magisterska.tex`. Bez retreningu, bez zmiany kodu — checkpointy są źródłem prawdy. Spec: `docs/superpowers/specs/2026-06-15-audyt-architektur-i-sugestie-design.md`.

**Tech Stack:** LaTeX. Weryfikacja: `grep` + balans nawiasów (Python, env `mgr_thesis`). Brak lokalnego `pdflatex` — finalny PDF kompiluje autor.

**Zasady:** NIE git, NIE commit (repo nie-git, preferencja autora). Numery linii orientacyjne — dopasowuj po treści łańcucha. `PLIK = "/home/tarnickil/MGR/Praca Magisterska/Praca_magisterska.tex"`.

**Fakty z checkpointów (źródło korekt):**
- MLP `best_mlp.pt`: 1282→256→128→64→1.
- CNN `best_cnn.pt`: 16 filtrów/jądro (4×16=64/łańcuch), conv-aktywacja GELU, MHC `nn.Embedding(2,16)`→16-d, R^144, klasyfikator 144→64→32→1.
- GNN `gcn_L1`: proj 1280→256, node_fusion 262→256 (Linear+GELU+Dropout), GCN 256→256 ×L (główny wynik L=1), klasyfikator 256→128→1, bez ponownej konkatenacji MHC.

---

## Task 1: MLP — wymiary warstw ukrytych

**Files:** Modify `Praca Magisterska/Praca_magisterska.tex` (~937, ~939)

- [ ] **Step 1: Wymiary ukryte [512,256,128] → [256,128,64]**

Znajdź:
```
Dla wymiarów ukrytych $[512, 256, 128]$ sieć oblicza:
```
Zamień na:
```
Dla wymiarów ukrytych $[256, 128, 64]$ sieć oblicza:
```

- [ ] **Step 2: Wymiar W_0 w eq:mlp0**

Znajdź:
```
\quad \mathbf{W}_0 \in \mathbb{R}^{512 \times 1282} \label{eq:mlp0} \\
```
Zamień na:
```
\quad \mathbf{W}_0 \in \mathbb{R}^{256 \times 1282} \label{eq:mlp0} \\
```

- [ ] **Step 3: Weryfikacja**

Run: `grep -n 'R^{512\|512, 256, 128' "PLIK"`
Expected: PUSTO.
Run: `grep -n 'R^{256 \|256, 128, 64' "PLIK"`
Expected: trafienia (W_0 oraz lista wymiarów ukrytych).

---

## Task 2: CNN — filtry, aktywacja, embedding MHC, wymiary

**Files:** Modify `Praca Magisterska/Praca_magisterska.tex` (~1009, ~1019, ~1031, ~1036–1038, ~1041)

- [ ] **Step 1: Aktywacja konwolucji ReLU → GELU (eq:conv1d)**

Znajdź:
```
    c_{f,i} = \mathrm{ReLU}\!\left(\sum_{j=0}^{k-1} \mathbf{E}^\alpha_{:,i+j} \cdot \mathbf{f}_{:,j} + b_f\right), \quad i = 1, \ldots, L_\alpha - k + 1
```
Zamień na:
```
    c_{f,i} = \mathrm{GELU}\!\left(\sum_{j=0}^{k-1} \mathbf{E}^\alpha_{:,i+j} \cdot \mathbf{f}_{:,j} + b_f\right), \quad i = 1, \ldots, L_\alpha - k + 1
```

- [ ] **Step 2: Liczba filtrów 64 → 16**

Znajdź:
```
Dla każdego rozmiaru jądra stosuje się 64~filtry, łącznie $4 \times 64 = 256$~filtrów na
łańcuch.
```
Zamień na:
```
Dla każdego rozmiaru jądra stosuje się 16~filtrów, łącznie $4 \times 16 = 64$~filtry na
łańcuch.
```

- [ ] **Step 3: Wymiar wektorów łańcuchów R^256 → R^64**

Znajdź:
```
$\mathbf{p}^\alpha \in \mathbb{R}^{256}$ (analogicznie $\mathbf{p}^\beta \in \mathbb{R}^{256}$
dla łańcucha $\beta$).
```
Zamień na:
```
$\mathbf{p}^\alpha \in \mathbb{R}^{64}$ (analogicznie $\mathbf{p}^\beta \in \mathbb{R}^{64}$
dla łańcucha $\beta$).
```

- [ ] **Step 4: Embedding MHC (16-d, nie one-hot) + konkatenacja R^514 → R^144**

Znajdź:
```
Ostateczna reprezentacja TCR formowana jest przez konkatenację wektorów obu łańcuchów i~kodu MHC:
\begin{equation}
    \mathbf{r}_{\mathrm{CNN}} = \left[\mathbf{p}^\alpha;\; \mathbf{p}^\beta;\; \mathbf{m}\right] \in \mathbb{R}^{514}
```
Zamień na:
```
Ostateczna reprezentacja TCR formowana jest przez konkatenację wektorów obu łańcuchów oraz
osadzenia allelu MHC. W~odróżnieniu od MLP (gdzie MHC kodowane jest jako wektor one-hot), model
CNN stosuje uczone osadzenie (\texttt{nn.Embedding}) allelu w~przestrzeni $\mathbf{m} \in \mathbb{R}^{16}$:
\begin{equation}
    \mathbf{r}_{\mathrm{CNN}} = \left[\mathbf{p}^\alpha;\; \mathbf{p}^\beta;\; \mathbf{m}\right] \in \mathbb{R}^{144}
```

- [ ] **Step 5: Klasyfikator 514→256→128→1 → 144→64→32→1**

Znajdź:
```
Następnie stosowany jest klasyfikator MLP: $\mathbb{R}^{514} \to 256 \to 128 \to 1$, analogiczny
```
Zamień na:
```
Następnie stosowany jest klasyfikator MLP: $\mathbb{R}^{144} \to 64 \to 32 \to 1$, analogiczny
```

- [ ] **Step 6: Weryfikacja**

Run: `grep -nF -e 'R^{514}' -e '64~filtry' -e 'mathrm{ReLU}' -e '4 \times 64' "PLIK"`
Expected: PUSTO (żadnych starych wartości CNN; `mathrm{ReLU}` ma zniknąć z eq:conv1d). Uwaga: NIE szukaj samego `R^{256}` — `256` występuje też legalnie w innych modelach.
Run: `grep -nF -e 'R^{144}' -e '16~filtrów' -e 'nn.Embedding' -e 'R^{16}' "PLIK"`
Expected: trafienia (R^144 dwa razy — concat i klasyfikator; 16 filtrów; nn.Embedding; R^16).

---

## Task 3: GNN — warstwa fuzji, wymiary, liczba warstw, klasyfikator

**Files:** Modify `Praca Magisterska/Praca_magisterska.tex` (~1122–1126, ~1128–1129, ~1131, ~1137, ~1151)

- [ ] **Step 1: Krok 2 — dodać warstwę fuzji (262→256), h^(0) ∈ R^{d_h}**

Znajdź:
```
\textbf{Krok 2: Fuzja węzłów.}
\begin{equation}
    \mathbf{h}_v^{(0)} = \left[\mathbf{h}_v^{(0)\prime};\; \mathbf{c}_v^{(2)};\; \mathbf{c}_v^{(3)}\right] \in \mathbb{R}^{d_h + 6}
    \label{eq:gnn_fusion}
\end{equation}
```
Zamień na:
```
\textbf{Krok 2: Fuzja węzłów.} Wektor projekcji konkatenuje się z~kodem MHC $\mathbf{c}_v^{(2)}$
(one-hot, 2-d) oraz flagą pozycyjną $\mathbf{c}_v^{(3)}$ (4-d), a~następnie poddaje warstwie fuzji
(liniowa $+$ GELU $+$ Dropout) redukującej wymiar z~powrotem do $d_h$:
\begin{equation}
    \mathbf{h}_v^{(0)} = \mathrm{NodeFusion}\!\left(\left[\mathbf{h}_v^{(0)\prime};\; \mathbf{c}_v^{(2)};\; \mathbf{c}_v^{(3)}\right]\right) \in \mathbb{R}^{d_h}, \qquad \mathrm{NodeFusion} : \mathbb{R}^{d_h + 6} \to \mathbb{R}^{d_h}
    \label{eq:gnn_fusion}
\end{equation}
```

- [ ] **Step 2: Krok 3 — liczba warstw L (ogólnie), główny wynik L=1**

Znajdź:
```
\textbf{Krok 3: Stos warstw GCNConv.} $L = 3$~warstwy GCN z~biblioteki DHG~\cite{feng2019}, ze~wzoru~\eqref{eq:gcn},
z~aktywacją $\sigma$ i~połączeniami rezydualnymi:
```
Zamień na:
```
\textbf{Krok 3: Stos warstw GCNConv.} $L$~warstw GCN z~biblioteki DHG~\cite{feng2019}, ze~wzoru~\eqref{eq:gcn},
z~aktywacją $\sigma$ i~połączeniami rezydualnymi (liczbę warstw $L$ dobrano w~badaniu ablacyjnym,
tabela~\ref{tab:ablation_gnn}; w~wynikach głównych i~na rysunkach używany jest najlepszy wariant $L=1$):
```

- [ ] **Step 3: Krok 3 — uogólnić zakres warstw w równaniu (l=0,1,2 → l=0,…,L-1)**

Znajdź:
```
    \mathbf{h}_v^{(l+1)} = \mathbf{h}_v^{(l)} + \sigma\!\left(\mathrm{GCNConv}^{(l)}\!\left(\mathbf{H}^{(l)}, \tilde{\mathbf{A}}\right)_v\right), \quad l = 0,1,2
```
Zamień na:
```
    \mathbf{h}_v^{(l+1)} = \mathbf{h}_v^{(l)} + \sigma\!\left(\mathrm{GCNConv}^{(l)}\!\left(\mathbf{H}^{(l)}, \tilde{\mathbf{A}}\right)_v\right), \quad l = 0,\ldots,L-1
```

- [ ] **Step 4: Krok 4 — JK po wszystkich warstwach, wymiar R^{d_h}**

Znajdź:
```
    \mathbf{h}_v^{\mathrm{JK}} = \max\!\left(\mathbf{h}_v^{(0)},\, \mathbf{h}_v^{(1)},\, \mathbf{h}_v^{(2)},\, \mathbf{h}_v^{(3)}\right) \in \mathbb{R}^{d_h+6}
```
Zamień na:
```
    \mathbf{h}_v^{\mathrm{JK}} = \max\!\left(\mathbf{h}_v^{(0)},\, \mathbf{h}_v^{(1)},\, \ldots,\, \mathbf{h}_v^{(L)}\right) \in \mathbb{R}^{d_h}
```

- [ ] **Step 5: Krok 6 — klasyfikator: wejście h_{T*} ∈ R^{d_h} (bez ponownej konkatenacji MHC), R^{d_h+8}→R^{d_h}**

Znajdź:
```
    s = \mathrm{MLP}_{\mathrm{cls}}\!\left(\left[\mathbf{h}_{T^*};\; \mathbf{m}\right]\right), \quad \hat{p} = \sigma(s), \quad \mathrm{MLP}_{\mathrm{cls}} : \mathbb{R}^{d_h+8} \to \tfrac{d_h}{2} \to 1
```
Zamień na:
```
    s = \mathrm{MLP}_{\mathrm{cls}}\!\left(\mathbf{h}_{T^*}\right), \quad \hat{p} = \sigma(s), \quad \mathrm{MLP}_{\mathrm{cls}} : \mathbb{R}^{d_h} \to \tfrac{d_h}{2} \to 1
```

- [ ] **Step 6: Weryfikacja**

Run: `grep -nF -e 'd_h + 6' -e 'd_h+6' -e 'd_h+8' -e 'L = 3' -e 'l = 0,1,2' "PLIK"`
Expected: PUSTO.
Run: `grep -nF -e 'NodeFusion' -e 'L-1' -e 'in \mathbb{R}^{d_h}' "PLIK"`
Expected: trafienia (NodeFusion; uogólniony zakres l=0,…,L-1; R^{d_h} w JK i klasyfikatorze).

---

## Task 4: pkt 6–7 — wplecenie do Podsumowania i Wniosków

**Files:** Modify `Praca Magisterska/Praca_magisterska.tex` (~1544 Podsumowanie, ~1704 Wnioski)

- [ ] **Step 1: Podsumowanie — dodać zdanie o rozkładzie naturalnym + within-allele jako właściwa miara**

Znajdź:
```
artefakt doboru danych.

Przebiegi ablacyjne GNN z~głębokościami
```
Zamień na:
```
artefakt doboru danych. Dla pełnego obrazu w~tabeli~\ref{tab:natural} podano wyniki na
naturalnym rozkładzie klas (${\approx}10{,}6\%$ pozytywów): bezwzględne PR-AUC spada wówczas do
${\approx}0{,}24$--$0{,}30$ (poziom losowy 0{,}106), co potwierdza, że praktyczna jakość modeli
jest skromna, choć ranking CNN $>$ MLP $>$ k-NN pozostaje zachowany. Za właściwą miarę zdolności
rozpoznawania TCR należy zatem przyjąć ewaluację w~obrębie allelu prezentującego
(sekcja~\ref{sec:within_allele}), a~nie metryki agregatowe.

Przebiegi ablacyjne GNN z~głębokościami
```

- [ ] **Step 2: Wnioski — dodać zdanie o rozkładzie naturalnym jako podstawie ostrożnej oceny**

Znajdź:
```
hierarchia modeli pozostaje zachowana. Wniosek porównawczy pracy~--- przewaga modeli
neuronowych nad k-NN oraz brak przewagi GNN nad CNN~--- jest zatem odporny na ten
artefakt doboru danych.
```
Zamień na:
```
hierarchia modeli pozostaje zachowana. Wniosek porównawczy pracy~--- przewaga modeli
neuronowych nad k-NN oraz brak przewagi GNN nad CNN~--- jest zatem odporny na ten
artefakt doboru danych. Co więcej, na naturalnym rozkładzie klas (tabela~\ref{tab:natural})
bezwzględne PR-AUC modeli jest niskie (${\approx}0{,}24$--$0{,}30$ przy poziomie losowym 0{,}106),
dlatego praktyczną wartość modeli należy oceniać ostrożnie, opierając wnioski przede wszystkim
na ewaluacji w~obrębie allelu HLA-A*03:01.
```

- [ ] **Step 3: Weryfikacja**

Run: `grep -c 'tab:natural' "PLIK"`
Expected: ≥4 (definicja + wcześniejsze odwołania + 2 nowe).
Run: `grep -n 'Za właściwą miarę zdolności\|praktyczną wartość modeli należy oceniać ostrożnie' "PLIK"`
Expected: po jednym trafieniu (Podsumowanie i Wnioski).

---

## Task 5: Weryfikacja końcowa

**Files:** brak zmian — kontrole.

- [ ] **Step 1: Brak starych wymiarów architektur**

Run: `grep -nF -e 'R^{512' -e 'R^{514}' -e '64~filtry' -e '4 \times 64' -e 'd_h + 6' -e 'd_h+6' -e 'd_h+8' -e 'L = 3' "PLIK"`
Expected: PUSTO.

- [ ] **Step 2: Spójność wymiarów nowych (sanity)**

Run: `grep -nF -e 'R^{256 \times 1282}' -e 'R^{144}' -e '4 \times 16 = 64' -e 'NodeFusion' -e 'R^{16}' "PLIK"`
Expected: wszystkie obecne (MLP W_0; CNN R^144 ×2; 16 filtrów; GNN fuzja; MHC embedding 16-d).

- [ ] **Step 3: Balans nawiasów i trybu math**

Run:
```bash
cd "/home/tarnickil/MGR/Praca Magisterska" && conda run -n mgr_thesis python -c "
s=open('Praca_magisterska.tex').read()
print('braces (0=ok):', s.count('{')-s.count('}'))
print('inline \$ parzyste:', s.count('\$')%2==0)
"
```
Expected: `braces (0=ok): 0`, `inline $ parzyste: True`.

- [ ] **Step 4: Odwołania nietknięte**

Run: `grep -c 'label{eq:mlp0}\|label{eq:cnn_concat}\|label{eq:gnn_fusion}\|label{eq:gnn_jk}\|label{eq:gnn_cls}' "PLIK"`
Expected: 5 (wszystkie etykiety równań zachowane).

- [ ] **Step 5: Kompilacja PDF (ręcznie, autor)**

Skompilować w środowisku autora; sprawdzić brak `??`, równania MLP/CNN/GNN renderują się, wymiary spójne.

---

## Notatki dla wykonawcy
- Kolejność T1→T5. Zadania niezależne (różne sekcje), ale T5 po wszystkich.
- Jeśli `Edit` zgłosi brak unikalności łańcucha — dołącz sąsiednią linię z kontekstem.
- MHC: MLP one-hot 2-d, CNN embedding 16-d, GNN one-hot 2-d (w cechach węzła) — NIE ujednolicać.
- Po zakończeniu warto (osobno) zaktualizować notatkę w `CLAUDE.md` o rozbieżności CNN — teraz rozwiązana.
- Bez commitów (repo nie-git).
