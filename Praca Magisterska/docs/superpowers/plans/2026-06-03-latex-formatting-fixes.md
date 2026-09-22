# LaTeX Formatting Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Naprawić 3 problemy formatowania w `Praca_magisterska.tex` bez zmiany treści merytorycznej ani klasy `SGGW-thesis.cls`.

**Architecture:** 7 punktowych edycji jednego pliku `.tex`. Preambuła dostaje 4 zmiany (font, microtype, penalizacje, redef abstractpage). Treść dokumentu dostaje 3 zmiany (TOC, dwie tabele).

**Tech Stack:** LaTeX (pdflatex), klasa SGGW-thesis v1.071a, pakiety setspace/xifthen (już w klasie).

> **Uwaga:** Na serwerze brak pdflatex. Kroki weryfikacji należy wykonać lokalnie w swoim środowisku LaTeX (TeXstudio, Overleaf, VSCode + LaTeX Workshop itp.).

---

### Task 1: Usuń `\usepackage{bm}` i dodaj `\usepackage{microtype}`

**Plik:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` — linie 19 i 26 (po `\usepackage{algpseudocode}`)

- [ ] **Krok 1: Usuń linię `\usepackage{bm}`**

  Znajdź w preambule (okolice linii 19):
  ```latex
  \usepackage{bm}
  ```
  Usuń tę linię całkowicie. Pakiet jest załadowany ale nieużywany — wchodzi w konflikt z `mathptmx` z klasy SGGW i podmienia font tekstu na bold.

- [ ] **Krok 2: Dodaj `\usepackage{microtype}` na końcu bloku pakietów**

  Znajdź ostatni `\usepackage` w preambule (obecnie `\usepackage{algpseudocode}`):
  ```latex
  \usepackage{algpseudocode}
  ```
  Zastąp przez:
  ```latex
  \usepackage{algpseudocode}
  \usepackage[stretch=10,shrink=10]{microtype}
  ```

- [ ] **Krok 3: Skompiluj i zweryfikuj**

  Skompiluj `Praca_magisterska.tex` (dwa przejścia pdflatex dla referencji).
  Oczekiwany efekt: tekst główny w akapitach ma normalną (nie pogrubioną) wagę fontu Times Roman. Brak nowych błędów kompilacji.

---

### Task 2: Dodaj penalizacje sierot i wdów

**Plik:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` — przed `\renewcommand{\bibname}`

- [ ] **Krok 1: Wstaw penalizacje przed `\renewcommand{\bibname}`**

  Znajdź w preambule:
  ```latex
  \renewcommand{\bibname}{Bibliografia}
  ```
  Wstaw bezpośrednio przed tą linią:
  ```latex
  \widowpenalty=10000
  \clubpenalty=10000

  ```

  Pełny fragment po zmianie:
  ```latex
  \widowpenalty=10000
  \clubpenalty=10000

  \renewcommand{\bibname}{Bibliografia}
  ```

- [ ] **Krok 2: Skompiluj i zweryfikuj**

  Skompiluj. Oczekiwany efekt: brak nowych błędów. Efekt penalizacji widoczny przy przeglądaniu akapitów — żaden nie powinien kończyć się pojedynczą linią na nowej stronie.

---

### Task 3: Przedefiniuj `\abstractpage` — usuń pustą sekcję angielską

**Plik:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` — przed `\begin{document}`

- [ ] **Krok 1: Dodaj `\renewcommand{\abstractpage}` przed `\begin{document}`**

  Znajdź linię:
  ```latex
  \begin{document}
  ```
  Wstaw bezpośrednio przed nią:
  ```latex
  \renewcommand{\abstractpage}[6]{%
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
      }%
      \vfill
      \pagestyle{empty}
      \newpage
      \null
      \pagestyle{empty}
      \newpage
      \pagestyle{plain}
  }

  ```

  Logika: `\ifthenelse{\equal{#4}{}}` sprawdza czy angielski tytuł (arg #4) jest pusty. Przy obecnym wywołaniu `\abstractpage{...}{...}{...}{}{}{}` sekcja angielska jest całkowicie pomijana.

- [ ] **Krok 2: Skompiluj i zweryfikuj**

  Skompiluj. Otwórz PDF na stronie streszczenia.
  Oczekiwany efekt:
  - Widoczne: "Streszczenie", tytuł po polsku, treść streszczenia, "Słowa kluczowe -- ..."
  - Niewidoczne: "Summary", "Keywords --", żadne puste linie po słowach kluczowych
  - Następna strona po streszczeniu: pusta (jak w klasie), potem spis treści

---

### Task 4: Napraw spis treści i obie tabele

**Plik:**
- Modify: `Praca Magisterska/Praca_magisterska.tex` — linie ~52–55, ~464, ~920

- [ ] **Krok 1: Usuń `\doublespacing` ze spisu treści**

  Znajdź blok (okolice linii 52–55):
  ```latex
  {
    \doublespacing
    \tableofcontents
  }
  ```
  Zastąp przez:
  ```latex
  \tableofcontents
  ```

- [ ] **Krok 2: Napraw Tabelę 1 (`tab:dataset_stats`)**

  Znajdź (okolice linii 464):
  ```latex
  \begin{table}[ht]
  \centering
  \caption{Statystyki zbioru danych po filtracji do epitopu IE-1\_CMV (allele uwzględnione w~zbiorach uczących).}
  ```
  Zastąp przez:
  ```latex
  \begin{table}[htbp]
  \centering
  \small
  \caption{Statystyki zbioru danych po filtracji do epitopu IE-1\_CMV (allele uwzględnione w~zbiorach uczących).}
  ```

- [ ] **Krok 3: Napraw Tabelę 2 (`tab:results`)**

  Znajdź (okolice linii 920):
  ```latex
  \begin{table}[ht]
  \centering
  \caption{Wyniki modeli na zbiorze testowym.
  ```
  Zastąp `[ht]` na `[htbp]` i dodaj `\small` po `\centering`:
  ```latex
  \begin{table}[htbp]
  \centering
  \small
  \caption{Wyniki modeli na zbiorze testowym.
  ```
  (Reszta caption i zawartość tabeli bez zmian.)

- [ ] **Krok 4: Skompiluj (dwa przejścia) i zweryfikuj całość**

  Wykonaj dwie kompilacje pdflatex (dla poprawnych referencji i numeracji stron).
  Sprawdź w PDF:
  - [ ] Spis treści: interlinia spójna z resztą dokumentu (1.4×), nie podwójna
  - [ ] Tabela 1 (Rozdział 3): pojawia się blisko tekstu który ją przywołuje, nie na oddzielnej pustej stronie
  - [ ] Tabela 2 (Rozdział 4, Wyniki): podobnie — nie zajmuje całej pustej strony
  - [ ] Font w tabelach jest nieznacznie mniejszy niż tekst główny (efekt `\small`) — czytelny, ale kompaktowy
  - [ ] Tekst akapitów: żadna sierocia linia na początku/końcu strony
  - [ ] Brak błędów kompilacji (`.log` bez `ERROR`)

---

## Checklist końcowa

- [ ] Tekst główny nie jest pogrubiony
- [ ] Strona streszczenia: widoczne tylko Streszczenie / Słowa kluczowe (bez Summary / Keywords)
- [ ] Spis treści: spójna interlinia
- [ ] Obie tabele: umieszczone blisko tekstu, nie na pustych stronach
- [ ] Klasa `SGGW-thesis.cls`: niezmieniona
- [ ] Treść merytoryczna: niezmieniona
