# Audyt: zgodność opisów architektur z modelami + sugestie pkt 6–7 — design

**Data:** 2026-06-15
**Plik docelowy:** `Praca Magisterska/Praca_magisterska.tex`
**Geneza:** Ponowny audyt całej pracy vs `Poprawki 2.txt` (opcja: audyt pliku + szukanie nowych błędów). Stwierdzono: 9/11 punktów w pełni zrealizowanych, 2 częściowo (6, 7), oraz NOWĄ klasę błędów — opisy architektur MLP/CNN/GNN nie zgadzają się z wytrenowanymi modelami (`best_mlp.pt`, `best_cnn.pt`, `gcn_L1/checkpoint.pt`), które wyprodukowały raportowane wyniki.

## Ustalenia z autorem (decyzje)
1. **Architektury (B):** poprawić opisy MLP/CNN/GNN tak, by dokładnie odpowiadały checkpointom. **Bez retreningu** — checkpointy są źródłem prawdy i dały raportowane liczby; zmieniamy wyłącznie tekst/wzory pracy.
2. **Sugestie pkt 6–7:** wpleść do Podsumowania i Wniosków zdania opierające ocenę na rozkładzie naturalnym (tab:natural) i within-allele (ROC 0,55–0,59). **Bez przenoszenia** sekcji.

## Dowody (kształty warstw z checkpointów)
- `best_mlp.pt`: projekcja (256,1282); predykcja 256→128→64→1. → MLP: **1282→256→128→64→1**.
- `best_cnn.pt`: conv α/β po (16,640,k) dla k∈{3,5,7,9}; `mhc_embedding` (2,16); classifier (64,144)→(32,64)→(1,32). → CNN: **16 filtrów/jądro (4×16=64/łańcuch), MHC-embedding 16-d, R^144, 144→64→32→1**.
- `gcn_L1/checkpoint.pt`: `tcr_projection` (256,1280); `node_fusion` (256,262); `gcn_layers.theta` (256,256); classifier (128,256)→(1,128). → GNN: proj 1280→256, **fusion 262→256**, GCN 256→256 ×L, classifier **256→128→1**.
- Kod potwierdza: `model_MLP(in_dim=1282, hidden_dim=256)`; `TCRCMVConvNet(n_filters=16, n_mhc=2, mhc_emb_dim=16)`; `TCRCMVGraphNet` (node_fusion 262→256, classifier d_h→d_h/2→1).

## Zakres zmian

### Część A — MLP (sekcja „Model 2", ~linie 915–920)
- „Dla wymiarów ukrytych $[512, 256, 128]$" → **$[256, 128, 64]$**.
- eq:mlp0: $\mathbf{W}_0 \in \mathbb{R}^{512 \times 1282}$ → **$\mathbb{R}^{256 \times 1282}$**.
- eq:mlpl/eq:mlpout: struktura (halving, logit $s$, $\hat p=\sigma(s)$) bez zmian; wymiary wynikają z [256,128,64].
- Wejście $\mathbf{x}_{\mathrm{MLP}} \in \mathbb{R}^{1282}$ (1280 + one-hot 2-d MHC) — **bez zmian** (zgodne z kodem).

### Część B — CNN (sekcja „Model 3", ~linie 1018–1041)
- „Dla każdego rozmiaru jądra stosuje się 64~filtry, łącznie $4 \times 64 = 256$~filtrów na łańcuch." → **16~filtrów, $4 \times 16 = 64$ filtry na łańcuch.**
- $\mathbf{p}^\alpha \in \mathbb{R}^{256}$, $\mathbf{p}^\beta \in \mathbb{R}^{256}$ → **$\mathbb{R}^{64}$** (oba).
- Kodowanie MHC: opis musi mówić, że CNN używa **uczonego embeddingu** `nn.Embedding(2, 16)`, $\mathbf{m} \in \mathbb{R}^{16}$ (NIE one-hot 2-d jak MLP). Dodać/zmienić zdanie o MHC w sekcji CNN.
- eq:cnn_concat: $\mathbf{r}_{\mathrm{CNN}} = [\mathbf{p}^\alpha; \mathbf{p}^\beta; \mathbf{m}] \in \mathbb{R}^{514}$ → **$\mathbb{R}^{144}$** (64+64+16).
- Klasyfikator: $\mathbb{R}^{514} \to 256 \to 128 \to 1$ → **$\mathbb{R}^{144} \to 64 \to 32 \to 1$**.

### Część C — GNN (sekcja „Model 4", ~linie 1107–1151)
- Cechy węzła $\mathbf{x}_v \in \mathbb{R}^{1286}$ (1280+2+4) — bez zmian.
- Po projekcji $\mathbf{h}_v^{(0)\prime} = \mathbf{W}_{\mathrm{proj}} \mathbf{c}_v^{(1)} \in \mathbb{R}^{d_h}$ ($d_h=256$) — bez zmian.
- **DODAĆ brakującą warstwę fuzji:** po konkatenacji $[\mathbf{h}_v^{(0)\prime}; \mathbf{c}_v^{(2)}; \mathbf{c}_v^{(3)}] \in \mathbb{R}^{d_h+6}$ (=262) stosowany jest `node\_fusion`: $\mathbb{R}^{d_h+6} \to \mathbb{R}^{d_h}$ (Linear + GELU + Dropout). Dopiero wynik (256-d) wchodzi do warstw GCN. Obecny wzór (1124) sugeruje, że do GCN trafia 262-d — to błąd.
- Warstwy GCN operują na $\mathbb{R}^{d_h}$ (256), nie $\mathbb{R}^{d_h+6}$. JK-agregacja: $\mathbf{h}_v^{\mathrm{JK}} \in \mathbb{R}^{d_h}$ (256), nie $\mathbb{R}^{d_h+6}$ (eq w ~1137).
- Klasyfikator (eq ~1151): obecnie $\mathrm{MLP}_{\mathrm{cls}}: \mathbb{R}^{d_h+8} \to \tfrac{d_h}{2} \to 1$ i wejście $[\mathbf{h}_{T^*}; \mathbf{m}]$. Poprawić na: wejście **$\mathbf{h}_{T^*} \in \mathbb{R}^{d_h}$** (MHC jest już wfuzowane w cechy węzłów, brak ponownej konkatenacji $\mathbf{m}$), $\mathrm{MLP}_{\mathrm{cls}}: \mathbb{R}^{d_h} \to \tfrac{d_h}{2} \to 1$ (256→128→1). Usunąć „+8".
- Liczba warstw: opis mówi „$L=3$" (1128), ale raportowany model główny i rysunki to **$L=1$** (gcn\_L1, najlepszy wariant ablacji). Doprecyzować: architektura ogólna ma $L$ warstw, a w wynikach głównych (tab:results, rysunki) używany jest $L=1$; JK-agregacja wtedy po $\{\mathbf{h}^{(0)}, \mathbf{h}^{(1)}\}$. (Nie przepisywać całej sekcji na L=1 — dodać zdanie spinające z raportowanym wariantem.)

### Część D — pkt 6 i 7 (Podsumowanie + Wnioski)
- W „Podsumowanie wyników" (~1511) i „Wnioski" (~1681) dodać zdania:
  - **pkt 6:** odwołać się do tab:natural — na rozkładzie naturalnym (~10,6%) PR-AUC spada do ~0,24–0,30 (baseline 0,106); bezwzględna jakość jest skromna, ranking CNN>MLP>k-NN zachowany.
  - **pkt 7:** jawnie wskazać within-allele (ROC 0,55–0,59) jako właściwą miarę zdolności predykcji TCR, a wartości agregatowe jako częściowo artefaktowe; konkluzja porównawcza oparta na within-allele + natural.
- Nie przenosić sekcji within-allele; wzmocnić jej rolę przez prozę i odwołania.

## Poza zakresem
- Retrening jakiegokolwiek modelu.
- Zmiana KODU modeli (poprawiamy wyłącznie opis w pracy).
- Strukturalne przenoszenie sekcji.

## Ryzyka / uwagi
- CLAUDE.md zawiera notatkę „Do not reconcile [CNN 64 vs 16] without re-running training" — dotyczyła zmiany KODU; tu zmieniamy OPIS pod istniejące checkpointy, więc retrening zbędny. Po zmianie warto zaktualizować tę notatkę w CLAUDE.md (osobno).
- MHC w CNN to embedding 16-d (uczony), w MLP one-hot 2-d, w GNN one-hot 2-d (w cechach węzła) — opisy muszą to rozróżniać, nie ujednolicać.
- Po edycjach: kontrola spójności wymiarów (każdy wektor/Linear), balans nawiasów/`$`, brak osieroconych odwołań. Kompilacja PDF po stronie autora.
