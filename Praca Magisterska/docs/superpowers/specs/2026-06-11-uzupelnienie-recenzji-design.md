# Uzupełnienie recenzji + rozszerzenie pracy — specyfikacja

**Cel:** domknąć wszystkie punkty `poprawki_recenzja.md`, rozszerzyć pracę do ~50 stron treścią
merytoryczną (nie „laniem wody"), zweryfikować bibliografię w sieci.

## Decyzje (uzgodnione z autorem)

1. **Punkt 7 (wiele seedów):** pomijamy eksperymenty; dodajemy jedno uczciwe zdanie w „Ograniczeniach".
2. **Notacja aktywacji:** w formułach modeli zamiana jawnego `GELU(·)` na generyczne `σ(·)`;
   zachowujemy podrozdział „Funkcja aktywacji GELU" + nota: „σ(·) oznacza aktywację, realizowaną
   jako GELU". Rozróżnienie od sigmoidy wyjściowej notą notacyjną (kolizja symbolu).
3. **Rozszerzenia (50 stron):** (a) nowy rozdział „Przegląd literatury" (NetTCR, TITAN, ERGO,
   pMTnet, DLpTCR); (b) głębsza teoria ML/metod (masked LM ESM2, over-smoothing, bootstrap,
   uzasadnienie hiperparametrów).
4. **Bibliografia:** weryfikacja w sieci; dodanie kilku zweryfikowanych pozycji do przeglądu.
5. **Wniosek o CNN (pkt 3+9):** raportujemy CI różnic; przyznajemy istotność (CI wyklucza zero),
   ale utrzymujemy **ostrożny ton** — eksponujemy mały rozmiar efektu i pojedyncze ziarno.

## Wyniki obliczeń (gotowe)

Sparowany bootstrap różnic (zbiór 7707, ten sam zestaw indeksów per iteracja), `Wyniki_Diff/`:

| Różnica | PR-AUC [95% CI] | ROC-AUC [95% CI] |
|---|---|---|
| CNN − MLP | +0,038 [+0,021; +0,054] | +0,010 [+0,003; +0,017] |
| CNN − GNN | +0,022 [+0,006; +0,037] | +0,008 [+0,001; +0,015] |

Wartości punktowe (zgodne z tabelą): MLP 0,581/0,806; CNN 0,619/0,816; GNN(L1) 0,598/0,808.
Progi F1-optymalne (walidacja): MLP 0,77; CNN 0,90; GNN 0,59; k-NN 0,01.

## Tabela przepływu danych (pkt 4)

| Etap | Obserwacje | Pozytywy | pos% | pos:neg |
|---|---|---|---|---|
| TRAIT (wiele epitopów) | ~3,9 mln | — | — | — |
| Po filtracji do IE-1_CMV | ~267 000 | 13 906 | ~5% | 1:18 |
| Po wykluczeniu HLA-A\*01:01 | 133 914 | 13 906 | 10,4% | 1:8,6 |
| Train | 85 588 | 8 806 | 10,3% | 1:8,7 |
| Walidacja | 23 968 | 2 531 | 10,6% | 1:8,5 |
| Test | 24 358 | 2 569 | 10,6% | 1:8,5 |

Usuwa konflikty 1:18 vs 1:9 oraz 267k vs 140k (267k = przed wykluczeniem allelu).

## Bibliografia — błędy znalezione (4/34), reszta poprawna

- `santangelo2002`: rok/wol/strony → Immunity 9(2):179–186, **1998**; inicjał „P.G. Waterbury".
- `montemurro2022`: → Communications Biology **4:1060, 2021**.
- `moris2021`: autorzy zmyśleni → Moris, De Pauw, Postovskaya, Gielis, De Neuter, Bittremieux,
  Ogunjimi, Laukens, Meysman.
- `pedregosa2011`: „E. Duchesneau" → „É. Duchesnay".

Klucze pozostają (styl numeryczny — nazwy kluczy niewidoczne w PDF).

## Zakres edycji `Praca_magisterska.tex`

1. [zrobione] 4 poprawki bibliografii.
2. Tabela przepływu danych + korekta tekstu 267k/140k.
3. Sekcja/akapit z CI różnic + rewizja zdań o „nakładających się CI" (ostrożny ton).
4. Podrozdział „Analiza progów decyzyjnych" (pkt 8).
5. Zdanie o pojedynczym ziarnie w „Ograniczeniach" (pkt 7).
6. Notacja σ w formułach MLP/GNN + nota.
7. Nowy rozdział „Przegląd literatury".
8. Rozszerzenie teorii (ESM2 masked LM, over-smoothing, bootstrap, hiperparametry).

**Styl:** bez zmian w `SGGW-thesis.cls`; spójność językowa i brak błędów LaTeX (walidacja
nawiasów/środowisk skryptem, bez kompilacji — brak `pdflatex` na serwerze).
