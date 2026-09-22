# Design: LaTeX Formatting Fixes — Praca_magisterska.tex

**Data:** 2026-06-03  
**Plik docelowy:** `Praca Magisterska/Praca_magisterska.tex`  
**Klasa:** `SGGW-thesis.cls` (nie modyfikowana)  
**Zasada:** Wyłącznie zmiany formatowania — treść merytoryczna pozostaje bez zmian.

---

## Kontekst

Trzy problemy formatowania zidentyfikowane w skompilowanym PDF:

1. Po słowach kluczowych streszczenia pojawia się "Summary, Keywords--" z pustą treścią.
2. Cały tekst dokumentu (od spisu treści) wygląda jak pogrubiony.
3. Tabele zajmują całe strony bez otaczającego tekstu; tekst łamie się nieestetycznie między stronami.

---

## Zmiany

### Sekcja 1 — Preambuła

#### 1a. Usuń `\usepackage{bm}`

- **Linia:** 19 (`\usepackage{bm}`)
- **Powód:** Pakiet `bm` załadowany ale nieużywany (brak wywołań `\bm{}` w dokumencie). Wchodzi w znany konflikt z `mathptmx` (font klasy SGGW) — podmienia font tekstu na wariant bold dla całego dokumentu.
- **Zmiana:** Usunąć linię.

#### 1b. Dodaj `\usepackage[stretch=10,shrink=10]{microtype}`

- **Pozycja:** Po `\usepackage{algpseudocode}` (koniec bloku pakietów)
- **Powód:** Optical font expansion i protrusion — LaTeX może nieznacznie skalować znaki by dopasować wiersz bez łamania. Eliminuje znaczną część nieestetycznych łamań automatycznie.
- **Zmiana:** Dodać linię po ostatnim `\usepackage`.

#### 1c. Dodaj penalizacje sierot i wdów

- **Pozycja:** Po bloku `\usepackage`, przed `\renewcommand{\bibname}`
- **Powód:** Zapobiega zostawianiu pojedynczej linii akapitu samotnie na początku lub końcu strony.
- **Zmiana:** Dodać:
  ```latex
  \widowpenalty=10000
  \clubpenalty=10000
  ```

#### 1d. Przedefiniuj `\abstractpage` — warunkowe streszczenie angielskie

- **Pozycja:** Po penalizacjach, przed `\begin{document}`
- **Powód:** Komenda `\abstractpage` z `.cls` zawsze renderuje sekcję angielską (Summary/Keywords) nawet gdy argumenty `#4`, `#5`, `#6` są puste `{}`. Powoduje to pojawienie się "Summary" i "Keywords --" z pustą treścią. Nie modyfikujemy `.cls` — zamiast tego redefiniujemy komendę lokalnie przez `\renewcommand`.
- **Zmiana:** Dodać przed `\begin{document}`:
  ```latex
  \renewcommand{\abstractpage}[6]{
      \setstretch{1.4}
      \null
      \vfill
      \begin{center}
          \textbf{Streszczenie}\\
      \end{center}
      \noindent
      \textbf{#1}\\[1.5ex]
      {#2}
      \\[4ex]
      Słowa kluczowe -- {#3}
      \ifthenelse{\equal{#4}{}}{}{%
          \vfill
          \begin{center}
              \textbf{Summary}\\
          \end{center}
          \noindent
          \textbf{#4}\\[1.5ex]
          {\selecthyphenation{english}#5}
          \\[4ex]
          Keywords -- {\selecthyphenation{english}#6}
      }
      \vfill
      \pagestyle{empty}
      \newpage
      \null
      \pagestyle{empty}
      \newpage
      \pagestyle{plain}
  }
  ```
  Sekcja angielska pojawia się tylko gdy `#4` jest niepuste. Gdy `#4 = {}` (jak w obecnym wywołaniu), Summary/Keywords są całkowicie pomijane.

---

### Sekcja 2 — Treść dokumentu

#### 2a. Usuń `\doublespacing` ze spisu treści

- **Linia:** 52–55
- **Obecny kod:**
  ```latex
  {
    \doublespacing
    \tableofcontents
  }
  ```
- **Powód:** Podwójna interlinia w TOC jest niespójna z resztą dokumentu (1.4×). Może powodować niespójne przejście spacingu po zamknięciu grupy (zależnie od wersji pakietu `setspace`).
- **Zmiana:** Zastąpić przez `\tableofcontents` bez grupy.

#### 2b. Tabela 1 — `tab:dataset_stats` (linia 464)

- **Obecny placement:** `[ht]`
- **Zmiana placement:** `[htbp]` — LaTeX może użyć dolnej strefy strony lub następnej, zamiast wymuszać tabelę w miejscu lub na dedykowanej stronie float.
- **Dodanie `\small`:** Wstawić `\small` po `\centering` (przed `\caption`) — redukcja rozmiaru fontu tabeli i podpisu.

#### 2c. Tabela 2 — `tab:results` (linia 920)

- **Obecny placement:** `[ht]`
- **Zmiana placement:** `[htbp]`
- **Dodanie `\small`:** Po `\centering`. Ta tabela ma wyjątkowo długi podpis (6 linii + przypis `†`); `\small` redukuje rozmiar obu elementów i pozwala na współistnienie tabeli z otaczającym tekstem na tej samej stronie.

---

## Podsumowanie zmian

| # | Plik | Miejsce | Zmiana |
|---|------|---------|--------|
| 1a | `.tex` preambuła | linia 19 | Usuń `\usepackage{bm}` |
| 1b | `.tex` preambuła | po ostatnim `\usepackage` | Dodaj `\usepackage[stretch=10,shrink=10]{microtype}` |
| 1c | `.tex` preambuła | przed `\renewcommand{\bibname}` | Dodaj `\widowpenalty=10000`, `\clubpenalty=10000` |
| 1d | `.tex` preambuła | przed `\begin{document}` | Dodaj `\renewcommand{\abstractpage}` z warunkową sekcją angielską |
| 2a | `.tex` dokument | linie 52–55 | Zastąp `{\doublespacing \tableofcontents}` przez `\tableofcontents` |
| 2b | `.tex` dokument | linia 464 | `[ht]` → `[htbp]`, dodaj `\small` |
| 2c | `.tex` dokument | linia 920 | `[ht]` → `[htbp]`, dodaj `\small` |

**Pliki nie modyfikowane:** `SGGW-thesis.cls` (żadne zmiany w klasie)
