# Predykcja wiązania TCR–CMV (k-NN / MLP / CNN / GNN)

Praca magisterska: binarna klasyfikacja wiązania receptora limfocytu T (TCR) z kompleksem
peptyd–MHC wirusa cytomegalii. Dane są silnie niezbalansowane (~10% klasy pozytywnej), więc
metryką główną jest **PR-AUC** z 95% przedziałami ufności (bootstrap, 1000 iteracji), a nie accuracy.

Porównywane są cztery modele: **k-NN**, **MLP**, **CNN** (à la NetTCR-2.0) i **GNN** (GCNConv z biblioteki DHG).

---

## 1. Środowisko

```bash
conda create -n mgr_thesis python=3.14
conda activate mgr_thesis
pip install -r requirements.txt
```

Zweryfikowane wersje: Python 3.14.2, `torch 2.11.0+cu126`, `dhg 0.9.6`, `scikit-learn 1.7.2`,
`pandas 2.3.3`, `numpy 2.3.5`. GPU użyte w eksperymentach: NVIDIA TITAN RTX (CUDA 12.6).

**Model ESM2** (`esm2_t30_150M_UR50D`) pobierany jest przy pierwszym uruchomieniu przez
`torch.hub.load("facebookresearch/esm", ...)` — wymaga połączenia z internetem. Pakiet `fair-esm`
nie musi być instalowany przez pip.

**Wszystkie skrypty uruchamiaj z katalogu głównego repozytorium** — ścieżki w `config.py` są względne:

```bash
conda run -n mgr_thesis python <skrypt>.py
```

---

## 2. Czego NIE ma w repozytorium

Repozytorium zawiera **kod, checkpointy modeli, splity i wyniki**. Nie zawiera dwóch grup plików,
które są zbyt duże dla Gita, ale w całości **odtwarzalne z kodu**:

| Brakujące | Rozmiar | Czym odtworzyć | Etap |
|---|---|---|---|
| `TRAIT/Neg_obs/`, `TRAIT/Pos_obs/` | ~400 MB | dane źródłowe — pobierz osobno (patrz §3, krok 0) | wejście |
| `TRAIT/*.pkl` | ~270 MB | `Data_prep.py` | 1 |
| `graph_ready_data/` | ~840 MB | `Data_prep.py` | 1 |
| `CNN_embedings_data/tcr_embeddings_CNN.pkl` | ~GB | `CNN_embeddings.py` | 3 |
| `Graphs/Training_graphs/`, `Validation_graphs/`, `Test_graphs/` | ~41 700 plików `.pt` | `graph.py` **po odkomentowaniu** (patrz §4) | 4 |

Wpisy te są w `.gitignore`. Jeśli chcesz tylko **powtórzyć ewaluację** bez trenowania od zera,
przejdź od razu do §5.6 — checkpointy i splity są w repo.

---

## 3. Pipeline danych (kolejność obowiązkowa)

```
krok 0  TRAIT/Neg_obs/*.txt + TRAIT/Pos_obs/*.txt + TRAIT/MHC_pseudo_seqs.txt
           |
krok 1     |  python Data_prep.py                       (wolne, GPU — embeddingi ESM2)
           v
        graph_ready_data/{tcr,mhc,peptide}_embeddings.pkl, all_relations.pkl
           |
krok 2     |  jupyter: train_test_split.ipynb           (jedyny krok bez wersji .py)
           v
        training_ready_data/{CMV_Dataset,Train_data,Val_data,Test.data}.pkl
           |
krok 3     |  python CNN_embeddings.py                  (embeddingi per-residue dla CNN)
           v
        CNN_embedings_data/{tcr,mhc}_embeddings_CNN.pkl
           |
krok 4     |  python graph.py                           (podgrafy dla GNN — WYMAGA EDYCJI, §4)
           v
        Graphs/{Training,Validation,Test}_graphs/edge_*.pt  +  indexy_graphs/*.pkl
           |
krok 5     |  python model_base.py                      (trening GNN -> best_gnn2.pt)
```

Kroki 1, 3 i 4 są kosztowne (godziny, GPU). Kroki 1–3 wykonują się „same” — kod zapisujący
wyniki jest w nich aktywny. Krok 4 wymaga jednorazowej edycji opisanej niżej.

### Uwaga o embeddingach CNN (`tcr_embeddings_CNN.pkl`)

Plik jest w `.gitignore` z powodu rozmiaru, ale **kod, który go tworzy, jest aktywny** —
`CNN_embeddings.py:56`:

```python
MHC_embeddings_df.to_pickle(CNN_mhc_embeddings_path)
TCR_embeddings_CNN.to_pickle(CNN_tcr_embeddings_path)   # <- aktywne, nie wymaga edycji
```

Wystarczy uruchomić `python CNN_embeddings.py`. **Bez tego pliku `import model_base` kończy się
`FileNotFoundError`** (`model_base.py:44` ładuje go na poziomie modułu), co blokuje wszystkie
skrypty ewaluacyjne. To pierwsza rzecz do wykonania po sklonowaniu repozytorium.

> Ten sam plik `.py` zawiera zapis, którego **nie ma** w `CNN_embeddings.ipynb` — notebook kończy się
> na wyliczeniu embeddingów. W tym kroku `.py` jest wersją nowszą i to jego należy uruchamiać.

---

## 4. Generowanie podgrafów GNN — zakomentowane wywołania w `graph.py`

`graph.py` zawiera **kompletny** kod budujący podgrafy, ale trzy wywołania zapisujące je na dysk
zostały zakomentowane po wygenerowaniu zbiorów (żeby ponowne uruchomienie pliku nie nadpisywało
41 700 plików `.pt`). Aby odtworzyć katalog `Graphs/`, odkomentuj **sześć linii**:

| Linie | Zbiór | Wyjście |
|---|---|---|
| `graph.py:487–488` | treningowy | `Graphs/Training_graphs/` (26 419 plików) |
| `graph.py:500–501` | walidacyjny | `Graphs/Validation_graphs/` (7 594 pliki) |
| `graph.py:513–514` | testowy | `Graphs/Test_graphs/` (7 708 plików) |

Przykład (linie 487–488):

```python
training_graphs_struct = graph_edge_chooser_cl(cmv_dataset, target_edges=training_indexs, knn_data=alpha_beta_pd, X_TCR=tcr_embbedings, X_MHC=mhc_embbedings
                                               , knn_graph=knn_graph, norm_embeddings=norm_embeddings, output_dir="Graphs/Training_graphs", n_jobs=16)
```

Wszystkie argumenty tych wywołań są zdefiniowane w aktywnym kodzie powyżej, więc odkomentowanie
niczego więcej nie wymaga:

- `alpha_beta_pd` — `graph.py:66`
- `knn_graph`, `norm_embeddings` — `graph.py:454` (`graph_knn_cl(alpha_beta_pd, n_jobs=1)`)
- `cmv_dataset`, `train_data`, `val_data`, `test_data` — `graph.py:44–53`
- `training_indexs` / `validation_indexs` / `test_indexs` — `graph.py:485 / 497 / 510`
- katalog wyjściowy tworzy sama funkcja (`os.makedirs(output_dir, exist_ok=True)`, `graph.py:156`)

### Czego NIE odkomentowywać

Linii **517–519**:

```python
# training_graphs_struct.to_pickle(training_graphs_struct_path)
```

W trybie dyskowym (`output_dir` podany) `graph_edge_chooser_cl` zwraca **listę ścieżek `str`**,
a nie `DataFrame` — `.to_pickle()` skończy się `AttributeError`. Ścieżki
`*_graphs_struct_path` w `config.py` są pozostałością po wcześniejszym trybie in-memory.
Tak samo zakomentowane wywołania `r_1` / `r_2` (`graph.py:456`, `468`) to testy pojedynczego
podgrafu w pamięci — nie są częścią pipeline'u.

### Koszt

`graph.py` przetwarza 41 721 celów z `ThreadPoolExecutor` (`n_jobs=16` dla treningu, `8` dla
val/test). Każdy plik `.pt` zawiera macierz cech `X` o wymiarach *(liczba węzłów × 1286)* w `float32`
plus listę krawędzi, więc łączny rozmiar katalogu `Graphs/` idzie w **gigabajty** — zaplanuj miejsce
na dysku. Plik ustawia też `OPENBLAS_NUM_THREADS=1` (i pokrewne) na starcie, żeby BLAS nie walczył
o rdzenie z pulą wątków.

Podgrafy budowane są wyłącznie dla **podpróbki 1 pozytyw : 2 negatywy** (`random_state=42`),
dlatego zbiór testowy GNN ma ~33% klasy pozytywnej, a nie ~10% jak pełny split. To jest powód
istnienia skryptów z §5.2.

---

## 5. Pełne odtworzenie od zera — runbook

Kolejność jest obowiązkowa: każdy krok czyta wyjście poprzedniego. Kroki 1, 3 i 4 to godziny
pracy GPU. Jeśli chcesz tylko powtórzyć ewaluację z gotowych checkpointów, patrz §5.6.

```bash
conda activate mgr_thesis
cd /ścieżka/do/MGR          # wszystkie komendy z katalogu głównego repo
git lfs pull                # pobierz dane spod wskaźników LFS
```

### Krok 0 — dane źródłowe

Umieść w repo (nie są wersjonowane, patrz §2):

```
TRAIT/Neg_obs/*.txt        50 plików, ~394 MB
TRAIT/Pos_obs/*.txt        50 plików, ~2,5 MB
TRAIT/MHC_pseudo_seqs.txt  (jest w repo)
```

Sprawdzenie:

```bash
ls TRAIT/Neg_obs | wc -l && ls TRAIT/Pos_obs | wc -l     # oczekiwane: 50 i 50
```

### Krok 1 — embeddingi ESM2 (wolne, GPU, wymaga internetu)

```bash
python Data_prep.py
```

Pobiera `esm2_t30_150M_UR50D` przez `torch.hub` i zapisuje:

```
TRAIT/{negative,positive}_interactions.pkl
graph_ready_data/{tcr,mhc,peptide}_embeddings.pkl
graph_ready_data/all_relations.pkl
```

Sprawdzenie:

```bash
ls -la graph_ready_data/     # 4 pliki, razem ~840 MB
```

### Krok 2 — podział na zbiory

```bash
jupyter notebook train_test_split.ipynb     # uruchom wszystkie komórki
```

Jedyny krok pipeline'u bez wersji `.py`. Zapisuje:

```
training_ready_data/{CMV_Dataset,Train_data,Val_data,Test.data}.pkl
```

> Repozytorium zawiera te pliki w LFS. Jeśli chcesz **dokładnie** powtórzyć liczby z pracy,
> pomiń ten krok i użyj splitów z repo — ponowny podział da inne zbiory.

### Krok 3 — embeddingi per-residue dla CNN (wolne, GPU)

```bash
python CNN_embeddings.py
```

Zapisuje `CNN_embedings_data/{tcr,mhc}_embeddings_CNN.pkl`. Kod zapisujący jest aktywny —
żadnej edycji nie trzeba (szczegóły w §3).

Sprawdzenie — **bez tego pliku nie zadziała nic, co importuje `model_base`**:

```bash
python -c "import model_base" && echo "import OK"
```

### Krok 4 — podgrafy GNN (wolne, wymaga edycji)

Odkomentuj sześć linii opisanych w §4:

```bash
sed -i '487,488s/^# //; 500,501s/^# //; 513,514s/^# //' graph.py
sed -n '487,488p;500,501p;513,514p' graph.py     # sprawdź, czy '#' zniknął
```

Uruchom:

```bash
python graph.py
```

Oczekiwany wynik (~41 700 plików `.pt`, kilka GB):

```
Graphs/Training_graphs/      26 419 plików
Graphs/Validation_graphs/     7 594 pliki
Graphs/Test_graphs/           7 708 plików
indexy_graphs/{train,val,test}_index.pkl
```

Sprawdzenie:

```bash
for d in Training Validation Test; do echo "$d: $(ls Graphs/${d}_graphs | wc -l)"; done
```

Po wygenerowaniu warto zakomentować te linie z powrotem, żeby ponowne uruchomienie
`graph.py` nie nadpisywało zbioru:

```bash
sed -i '487,488s/^/# /; 500,501s/^/# /; 513,514s/^/# /' graph.py
```

### Krok 5 — trening GNN

```bash
python model_base.py        # main() -> trenuje TCRCMVGraphNet, zapisuje best_gnn2.pt
```

Trening MLP i CNN jest w `model_base.py` zakomentowany — checkpointy `best_mlp.pt`
i `best_cnn.pt` są w repo (przetrenowane 2026-06-10).

### 5.6 Skrót: sama ewaluacja z gotowych checkpointów

Jeden skrypt działa **natychmiast po sklonowaniu**, bez żadnego z kroków powyżej:

```bash
python regen_gnn_pr.py
# -> Zapisano Praca Magisterska/figures/gcn_L1_pr.png: PR-AUC=0.598, baseline=0.333
```

Wynik musi się zgadzać z `Wyniki_Ablation/summary_table.csv` (gcn_L1 = 0,5976) — to najszybszy
test, czy klon jest kompletny.

Pozostałe skrypty ewaluacyjne mają twarde zależności:

| Skrypt | Wymaga dodatkowo |
|---|---|
| `within_a0301.py` | krok 1 (`graph_ready_data/tcr_embeddings.pkl`) |
| `audit_knn.py` | kroki 1 i 4 (`Graphs/Test_graphs`) |
| `evaluate_models.py`, `evaluate_on_gnn_testset.py`, `eval_diff_ci.py`, `ablation_*.py` | kroki 1–4 (importują `model_base`) |

Ostatni wiersz wynika z tego, że `model_base.py` ładuje dane na poziomie modułu
(linie 34–58) i buduje `train_loader_gnn` z `Graphs/Training_graphs` (linia 649) —
sam `import model_base` czyta ~880 MB pickli.

---

## 6. Ewaluacja

### 6.1 Pełne splity

```bash
conda run -n mgr_thesis python evaluate_models.py        # CNN + k-NN -> Wyniki_CNN/, Wyniki_KNN/
```

### 6.2 Porównanie na tej samej podpróbce co GNN

GNN widzi tylko podpróbkę 1:2 (~33% pozytywów), a MLP/CNN/k-NN domyślnie liczone są na pełnym
splicie (~10%). Porównywanie PR-AUC między tymi zbiorami jest bezsensowne, dlatego:

```bash
conda run -n mgr_thesis python evaluate_on_gnn_testset.py  # MLP/CNN/k-NN na zbiorze GNN
conda run -n mgr_thesis python eval_diff_ci.py             # sparowany bootstrap różnic (CNN-MLP, CNN-GNN)
```

`eval_diff_ci.py` łączy predykcje po unikalnym `Name`, więc wynik nie zależy od kolejności
iteracji loaderów. Zapisuje `Wyniki_Diff/aligned_predictions.npz` — **wejście** dla dwóch
skryptów poniżej.

### 6.3 Audyt (druga runda recenzji)

```bash
conda run -n mgr_thesis python audit_knn.py      # kontrole k-NN: permutacja etykiet, ablacja flagi MHC, pojedyncze łańcuchy
conda run -n mgr_thesis python within_a0301.py   # ewaluacja w obrębie HLA-A*03:01 (flaga MHC stała)
```

`audit_knn.py` wymaga katalogu `Graphs/Test_graphs` (czyta z niego `target_relation`).
`within_a0301.py` wymaga wcześniejszego `eval_diff_ci.py`.

### 6.4 Ablacje

```bash
conda run -n mgr_thesis python ablation_gnn.py        # GCN L=1..4 + GAT -> Wyniki_Ablation/
conda run -n mgr_thesis python ablation_weighting.py  # sampler vs pos_weight vs oba -> Wyniki_Ablation_Weighting/
conda run -n mgr_thesis python regen_gnn_pr.py        # krzywa PR z poprawnym baseline 0,333 (bez retreningu)
```

---

## 7. Pliki `.py` a notebooki

Część kodu istnieje w dwóch wersjach — notebooki służyły do bieżącego podglądu wyników przy
pisaniu pracy. **Źródłem prawdy dla opublikowanych wyników są pliki `.py`**, bo to one są
importowane przez skrypty ewaluacyjne (`from model_base import ...`).

| `.py` | notebook | uwaga |
|---|---|---|
| `model_base.py` | `model_base.ipynb` | `.py` zawiera aktualną `step()` z disjoint-batchem; notebook ma też starszą pętlę per-graf |
| `graph.py` | `graph.ipynb` | notebook modyfikowany **później** niż konwersja `.py` — sprawdź go, zanim zmienisz logikę budowy grafu |
| `CNN_embeddings.py` | `CNN_embeddings.ipynb` | `.py` nowszy — ma zapis `to_pickle`, którego notebook nie ma |
| `train_test_split.ipynb` | — | **bez wersji `.py`**, obowiązkowy krok 2 pipeline'u |
| `UMAP.ipynb` | — | wizualizacja przestrzeni embeddingów, poza pipeline'em |

Nagłówek `# Converted at:` w pliku `.py` podaje datę konwersji — zmiany wprowadzone w notebooku
po tej dacie istnieją tylko w `.ipynb`.

---

## 8. Pułapki, które łatwo przeoczyć

- **`class_balanced=False` na val/test.** Użycie `True` daje `y_test.mean() ≈ 0.5` zamiast realnych
  ~10% i czyni PR-AUC nieporównywalnym. `Evaluator.evaluate()` ostrzega (`UserWarning`), gdy
  `y_test.mean()` wpada w 0,4–0,6, a nie podano `pos_rate`.
- **Format krawędzi DHG.** `dhg.Graph()` oczekuje `List[Tuple[int, int]]`. Przekazanie tensora
  `[2, E]` tworzy 2 hiperkrawędzie zamiast `E` krawędzi — błąd cichy i krytyczny. W `step()`
  zawsze `.tolist()` na tensorze `[E, 2]`.
- **`model_base.py` ładuje dane przy imporcie**, nie w funkcjach. Sam `import model_base` czyta
  ~880 MB pickli i buduje `train_loader_gnn` z `Graphs/Training_graphs` — bez tego katalogu
  i bez `tcr_embeddings_CNN.pkl` import się nie powiedzie.
- **`mhc_id` budowany wyłącznie z danych treningowych.** Allel w val/test nieobecny w treningu
  daje `KeyError`.
- **`best_mlp.pt`** został przetrenowany 2026-06-10 na poprawionej architekturze (wcześniejsza
  wersja miała `nn.Sigmoid()` przed `BCEWithLogitsLoss`). Skrypty konstruują go jako
  `model_MLP(in_dim=1282, hidden_dim=256)`.
- **Git LFS.** W LFS są wyłącznie `*.pkl` (splity, indeksy) i `*.pt` (checkpointy) — razem
  17 plików, ~59 MB. Po sklonowaniu wykonaj `git lfs pull`, inaczej dostaniesz pliki-wskaźniki
  zamiast danych. Wykresy `*.png`, `*.tex` i `*.npz` są **poza** LFS celowo: praca kompiluje
  się i `regen_gnn_pr.py` działa zaraz po zwykłym `git clone`, bez pobierania LFS.

---

## 9. Struktura wyników

| Katalog | Zawartość |
|---|---|
| `Wyniki_KNN/`, `Wyniki_MLP/`, `Wyniki_CNN/`, `Wyniki_GNN/` | `results.json` + wykresy; `results_gnnset.json` dla podpróbki GNN |
| `Wyniki_Ablation/` | warianty GCN L=1..4 i GAT, `summary_table.csv` (`gcn_L1` = wariant główny w pracy) |
| `Wyniki_Ablation_Weighting/` | ablacja ważenia klas dla MLP |
| `Wyniki_Diff/` | `diff_ci.json`, `aligned_predictions.npz/.csv` |
| `Wyniki_Audyt/` | `knn_controls.json`, `within_a0301.json` |
| `Praca Magisterska/` | źródła LaTeX; `figures/` to kopie wykresów z `Wyniki_*/` |

Wykres w `Wyniki_*/` **nie** aktualizuje pracy automatycznie — trzeba go skopiować do
`Praca Magisterska/figures/` (wyjątek: `regen_gnn_pr.py` zapisuje tam bezpośrednio).
