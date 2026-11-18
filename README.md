# Grobe Merkmale – wenn ein Symptom nur als Oberbegriff in der Akte steht – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-grobe-merkmale-demo.streamlit.app/)** -->

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Entscheidungsanalyse und Machine Learning). Aus den **dokumentierten Merkmalen** (Symptomen) eines Patienten wird eine Rangliste möglicher **seltener Krankheiten** erstellt, wenn ein Teil der Merkmale in der Akte nur als **Oberbegriff** steht (zum Beispiel „Augenauffälligkeit“ statt des genauen Befunds). Die Demo vergleicht fünf Ranker: den **exakten** Ranker, der die Regeln des Erzeugers dieser Welt kennt (das **Orakel**), einen Ranker, der die Oberbegriffe **überliest**, die Heuristik **„Oberbegriff = alle Kinder“**, einen **Resnik-artigen** Ähnlichkeitswert (eigene Fassung) und ein **naives Überlesen**, das u nicht kennt. Gemessen wird, was das Überlesen kostet und wie viel davon die Heuristik und der Ähnlichkeitswert zurückholen.

**Lehrmodell mit erfundenen Daten, keine medizinische Beratung und keine Diagnoseempfehlung.** Welt, Krankheiten, Merkmale und alle Zahlen sind erzeugt; es gibt keine echten Patientendaten und keine echte Wissensbasis. Die Aussagen gelten für diese Welt, bei diesen Einstellungen und für die gemessene Regel jedes Rankers. **Voraussetzung für alle Vergleiche mit „exakt“:** der exakte Ranker ist das Orakel des Erzeugers: er kennt den Anteil grober Angaben u und alle Häufigkeiten der Welt; in echten Daten gibt es dieses Orakel nicht, u müsste geschätzt werden.

**Bezug zu OR:** eine Rangliste aus unvollständigen Beobachtungen ist eine Entscheidung unter Unsicherheit; wie grob die Beobachtung ist, bestimmt, wie viel von der Information ankommt. Die Demo zeigt den Lernteil für sich und baut auf derselben Welt auf wie `phaenotyp-ranking-demo`.

## Warum dieses Problem

Klinische Akten nennen ein Symptom oft nur auf der Ebene eines Oberbegriffs. Ein Ranker, der solche Angaben nicht kennt, überliest sie oder rät. **Kernfrage 1:** Was kostet es, grobe Angaben zu überlesen? **Kernfrage 2:** Holt die einfache Heuristik „Oberbegriff = alle Kinder“ den Verlust zurück? **Kernfrage 3:** Hilft ein Resnik-artiger Ähnlichkeitswert?

## Befunde und Korrekturen gegenüber der früheren Messung

Eine frühere Messung vor dem Bau (NumPy-Zufall, fünf Welten × 1500 Patienten, 200 Krankheiten, q = 0,3; sie ist nicht Teil dieses Repos, ihre Zahlen in der Spalte „früher“ sind hier nicht nachprüfbar) ging dem Bau voraus. Im Bau wurde mit dem **SplitMix64** der Demo in **zwei Mengen von je fünf Welten** (Zufalls-Seed 1000 und 2000) neu gemessen, in 24 Zellen. Die Kernbefunde bestehen der Richtung nach; gegenüber den (hier nicht nachprüfbaren) Zahlen der früheren Messung verschieben sich die neuen Recall-Zahlen um höchstens 2,2 Punkte. Die Spalte „neu“ ist das Mittel der beiden Mengen (zehn Welten); der Resnik-artige Ranker war in der früheren Messung nicht dabei.

| Kernbeleg (200 Krankheiten, q = 0,3; Recall in %) | Früher | Neu gemessen (zehn Welten) |
|---|---|---|
| u = 0: exakt, überlesen und alle Kinder (gleich), Recall@1 / Recall@5 | 52,7 / 74,4 | 54,9 / 76,5 |
| u = 0,3: exakt / überlesen / alle Kinder, Recall@1 | 45,6 / 36,7 / 36,0 | 47,1 / 37,8 / 37,6 |
| u = 0,3: exakt / überlesen / alle Kinder, Recall@5 | 68,4 / 59,0 / 60,6 | 70,3 / 60,8 / 62,8 |
| u = 0,6: exakt / überlesen / alle Kinder, Recall@1 | 36,4 / 19,6 / 23,7 | 37,9 / 20,2 / 24,6 |
| u = 0,6: exakt / überlesen / alle Kinder, Recall@5 | 60,5 / 38,3 / 48,4 | 62,5 / 40,0 / 50,3 |
| u = 0,9: exakt / überlesen / alle Kinder, Recall@1 | 25,2 / 3,6 / 15,3 | 26,4 / 4,0 / 15,9 |
| u = 0,9: exakt / überlesen / alle Kinder, Recall@5 | 49,9 / 12,2 / 37,2 | 52,1 / 12,6 / 39,4 |
| Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff stehen, bei u = 0,3 / 0,6 / 0,9 (Prozent) | 30,2 / 59,9 / 90,2 | 30,0 / 60,0 / 89,9 |

**Korrekturen und Abweichungen gegenüber der früheren Messung:**

1. **Der Resnik-artige Ranker liegt hinter der Heuristik und hinter dem exakten Ranker, und schon ohne grobe Angaben weit hinter exakt.** Bei u = 0 liegt er in den sechs Zellen mit 50 und 200 Krankheiten und q = 0 / 0,3 / 0,6 11,3 bis 17,6 Punkte Recall@5 hinter exakt. Gegenüber dem Überlesen liegt er in 11 der 18 Zellen mit u > 0 mehr als zwei Standardfehler vorn, in 6 dahinter und in 1 ohne auflösbaren Unterschied; gegenüber der Heuristik liegt er in 18 von 18 Zellen dahinter (Befund 5). Die Fassung ist eine eigene, einfache (Summe über die Begriffe der Akte; keine Häufigkeiten, kein Vorab, keine Abwesenheit); Aussagen über sie gelten für diese Fassung, nicht für Ähnlichkeitsmaße im Allgemeinen. Warum der Abstand schon bei u = 0 so groß ist, ist nicht einzeln gemessen; plausibel ist, dass ihr die Häufigkeiten der Profile fehlen (Deutung, nicht geprüft).
2. **Gleichstände werden nach dem Runden auf neun Stellen gezählt.** Krankheiten, die die Akte nicht unterscheidet, haben mathematisch gleiche Punktzahlen; die Summen weichen in den letzten Bits ab, wenn sie in anderer Reihenfolge gebildet werden. Ohne Rundung unterscheiden sich bei u = 0 „exakt“ und „überlesen“ um wenige Hundertstel Punkte Recall@5, obwohl sie dieselbe Rangliste meinen (Voreinstellung, Menge A, gepaart: weniger als 0,05 Punkte, von `test_gleichstaende_ohne_rundung_*` nachgerechnet; der genaue Wert hängt vom Rundungsrauschen der Plattform ab und steht deshalb hier nicht); mit Rundung ist der gepaarte Unterschied bei u = 0 in allen sechs Zellen und beiden Mengen genau 0 (Befund 7; `tests/test_live.py`). Die Messreihe wurde nach der Rundung gerechnet.
3. **„Überlesen“ ist der starke Gegner, nicht der naive:** der Ranker rechnet mit den richtigen Randwahrscheinlichkeiten (ein genaues Merkmal steht mit Wahrscheinlichkeit g(1 − u) in der Akte), er kennt also u. Der naive Ranker, der die groben Angaben ebenfalls überliest, aber mit den ungeteilten Häufigkeiten g rechnet, ist als fünfter Ranker mitgemessen: in 17 der 18 Zellen mit u > 0 liegt „überlesen“ mehr als zwei Standardfehler vor ihm, die Differenzen der Zellen liegen zwischen +0,28 und +10,59 Punkten Recall@5 (Befund 10).
4. **Die Schwellen der Verdikte sind gesetzt, nicht abgeleitet:** Verlust (exakt minus überlesen, Recall@5) unter 5 Punkten, bis 15 Punkte, darüber. Die Verdikte laufen live aus der aktuellen Messung (gepaarte Differenz über die Welten, Standardfehler dabei); die Abbildungen und Tabellen der Kernfragen stammen aus der vorgerechneten Messreihe.
5. **Abnahmekriterium des Presets „Meist grob“ geändert:** vorgesehen war „überlesen höchstens 40 Prozent Recall@5“; gemessen sind 40,3 und 39,7 (Mengen A und B), die Schwelle läge auf der Zellgrenze. Ersetzt durch: exakt liegt mehr als 15 Punkte vor überlesen, und alle Kinder liegt mehr als zwei Standardfehler vor überlesen und mehr als zwei Standardfehler hinter exakt (`gmk_stories.py`).
6. **Nicht gerechnet:** eine Hierarchie mit drei Ebenen und ein Anteil grober Angaben, der je Merkmal verschieden ist. Die Demo zeigt eine Ebene (20 Oberbegriffe zu je fünf Merkmalen) und ein einziges u für alle Merkmale.

## Modell

**Welt.** Dieselbe wie in `phaenotyp-ranking-demo`: 100 Merkmale mit Zipf-verteilter Beliebtheit $w_s \propto \mathrm{Rang}^{-0{,}8}$; jede Krankheit hat 4 bis 12 Profilmerkmale (gleichverteilt, gezogen ohne Zurücklegen proportional zur Beliebtheit) mit Häufigkeit 0,95 / 0,6 / 0,3 / 0,1 (je Profilmerkmal mit den Wahrscheinlichkeiten 15 / 25 / 35 / 25 Prozent gezogen); Hintergrundrate $b_s = 0{,}01 + 0{,}2\,(w_s / w_{\max})^2$. Ein vorhandenes Merkmal wird mit Wahrscheinlichkeit $1 - q$ aufgeschrieben ($q$ = 0 / 0,3 / 0,6): $g_{dc} = 1 - (1 - f_{dc}(1 - q))(1 - b_c)$; Merkmale unabhängig gegeben die Krankheit. Vorab gleichverteilt, alle Merkmale geprüft, keine falschen Einträge, Wissensbasis = wahre Häufigkeiten. Patienten ohne jede Angabe in der Akte werden nicht gerankt: in den vorgerechneten Zellen mit u = 0,3 bleiben bei 200 Krankheiten 99,2 / 97,8 / 93,9 Prozent der Patienten bei q = 0 / 0,3 / 0,6 (bei 50 Krankheiten 99,4 / 98,0 / 93,9); **die Recall-Werte gelten für diese gerankten Patienten.**

**Hierarchie und grobe Dokumentation.** Die Merkmale $5G$ bis $5G + 4$ bilden die Gruppe $G$ (20 Gruppen, je ein Oberbegriff, fünf Kinder). Ein aufgeschriebenes Merkmal steht mit Wahrscheinlichkeit $1 - u$ genau in der Akte und mit Wahrscheinlichkeit $u$ nur als Oberbegriff seiner Gruppe ($u$ = 0 / 0,3 / 0,6 / 0,9). Mehrere grobe Angaben derselben Gruppe ergeben **einen** Oberbegriff: aus ihm ist nicht zu erkennen, wie viele Merkmale dahinterstehen. Die Akte einer Gruppe besteht aus der Menge $L_G$ der genau genannten Kinder und dem Zeichen $F_G$ („Oberbegriff steht in der Akte“).

**Exakter Ranker (Orakel des Erzeugers).** Wahrscheinlichkeit der Akte unter dem Erzeuger, Produkt über die Gruppen: $\prod_{c \in L_G} g_{dc}(1 - u)$ mal $\prod_{c \notin L_G}(1 - g_{dc})$ bei $F_G = 0$ beziehungsweise $\prod_{c \notin L_G}(1 - g_{dc} + g_{dc} u) - \prod_{c \notin L_G}(1 - g_{dc})$ bei $F_G = 1$ (mindestens ein Kind außerhalb von $L_G$ steht grob in der Akte). **Das Orakel gehört zum Erzeuger und ist nur in diesem Modell eine Obergrenze;** in allen 768 Vergleichen (24 Zellen × 2 Mengen × 4 andere Ranker × 4 Kennzahlen) liegt kein anderer Ranker über ihm (größter Vorsprung 0,0 Punkte).

## Methodik

Fünf Ranker: **exakt** (Gruppen-Wahrscheinlichkeit der Akte, kennt $u$ und alle $g$); **nur genaue Begriffe** („überlesen“: das Zeichen $F_G$ wird nicht gelesen, jedes Merkmal steht mit $g(1-u)$ genau in der Akte oder nicht; das ist die Randverteilung des exakten Rankers über $F_G$); **Oberbegriff = alle Kinder** (ein Oberbegriff zählt, als stünden alle fünf Kinder in der Akte; Naive Bayes mit den ungeteilten $g$, ohne Wissen um $u$); **Resnik-artig** (eigene Fassung, nicht Phenomizer: Annotation einer Krankheit = Profilhäufigkeit über 0,05; Informationsgehalt $-\ln$ des Anteils der Krankheiten, die den Begriff annotieren, Wurzel 0; ein genaues Merkmal trifft eine Krankheit mit seinem Informationsgehalt, wenn sie es annotiert, sonst mit dem seines Oberbegriffs, wenn sie den Oberbegriff annotiert, sonst mit 0; ein Oberbegriff trifft sie mit seinem Informationsgehalt, wenn sie ihn annotiert; Punktzahl = Summe über die Begriffe der Akte, kein Vorab); **naives Überlesen** (nur die genauen Begriffe, Naive Bayes mit den ungeteilten $g$, kennt $u$ nicht).

**Kennzahlen.** Recall@1, @5, @10: Anteil der **gerankten** Patienten, bei denen die wahre Krankheit unter den ersten $k$ steht; Gleichstände werden als Wahrscheinlichkeit gezählt (zufällige Auflösung, $\min(1, \max(0, (k - r + 1)/(t + 1)))$ mit dem kleinsten Rang $r$ und $t$ Gleichständen), nicht als Treffer. Dazu der **reziproke mittlere Rang** $1/(r + t/2)$. **Einheit:** Patienten (Recall) beziehungsweise Welten (Streuung).

**Messung.** Je Einstellung 5 Welten × 1500 Patienten (SplitMix64, `gmk_stream.py` ist die vektorisierte Fassung der Skalarklasse in `gmk_rng.py`, bitgleich geprüft). Mittel ± Streuung (n − 1) über die Welten. **Gepaarte Vergleiche:** Ob ein Ranker vor einem anderen liegt, entscheidet die Differenz je Welt (Mittel ± Standardfehler = Streuung der Differenzen / √5); Unterschiede unter zwei Standardfehlern werden nicht gedeutet (ein Richtungskriterium bei fünf Welten, kein 95-Prozent-Test: bei vier Freiheitsgraden entspricht ein t von 2 einem zweiseitigen p von etwa 0,12). **Vorgerechnet** (`data/gmk_results.json`, `tools/sweep.py`): 24 Zellen (2 Zahlen von Krankheiten × 3 Dokumentationsverluste × 4 Anteile grober Angaben), je zwei Mengen von fünf Welten (Zufalls-Seed 1000 und 2000). Der gepoolte Standardfehler der zehn Welten ist aus den beiden Standardfehlern zusammengesetzt: $\sqrt{s_A^2 + s_B^2}/2$.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 200 Krankheiten, 100 Merkmale, Dokumentationsverlust q = 0,3, Vorab gleichverteilt, wenn nicht anders genannt; Recall@5 in Prozent der gerankten Patienten, **Mittel der zwei Mengen zu je 5 Welten × 1500 Patienten (zehn Welten)**; „gepaart“ = Mittel der Differenzen je Welt in Punkten, ± gepoolter Standardfehler. Jede Zeile nennt den Test in `tests/test_claims.py`, der die Zahl aus `data/gmk_results.json` unabhängig nachrechnet.

| # | Frage | Befund | Test |
|---|---|---|---|
| 1 | Voreinstellung (u = 0,3, q = 0,3, 200 Krankheiten): wie gut sind die fünf Ranker? | Recall@5: exakt 70,3, überlesen 60,8, alle Kinder 62,8, Resnik-artig 52,1, naives Überlesen 60,0. Recall@1: 47,1 / 37,8 / 37,6 / 29,8 / 37,6. Recall@10: 78,7 / 69,8 / 73,1 / 62,2 / 69,2. Reziproker mittlerer Rang: 57,8 / 48,7 / 49,4 / 38,4 / 48,2. Gepaart: exakt − überlesen +9,5 ± 0,3; exakt − alle Kinder +7,5 ± 0,3; alle Kinder − überlesen +2,0 ± 0,3; Resnik-artig − überlesen −8,7 ± 0,2. | `test_befund_1_*` |
| 2 | **Was kostet es, grobe Angaben zu überlesen?** (Kernfrage 1; Verlust = exakt − überlesen, Recall@5 in Punkten) | u = 0: 0,00 (kein Unterschied, bei u = 0 keine groben Angaben). u = 0,3 / 0,6 / 0,9: +9,5 / +22,5 / +39,5 Punkte (Standardfehler 0,3 / 0,5 / 0,6). Recall@5 überlesen: 60,8 / 40,0 / 12,6; exakt: 70,3 / 62,5 / 52,1. Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff stehen: 30,0 / 60,0 / 89,9 Prozent. | `test_befund_2_*` |
| 3 | **Holt „Oberbegriff = alle Kinder“ den Verlust zurück?** (Kernfrage 2; gepaart alle Kinder − überlesen) | u = 0,3 / 0,6 / 0,9: +2,0 / +10,3 / +26,8 Punkte (Standardfehler 0,3 / 0,5 / 0,6), das sind 21 / 46 / 68 Prozent des Verlusts; der Rest bleibt verloren (exakt − alle Kinder +7,5 / +12,1 / +12,7). In 17 von 18 Zellen mit u > 0 liegt die Heuristik mehr als zwei Standardfehler vor dem Überlesen, in 1 dahinter, in 0 ohne auflösbaren Unterschied. Die Ausnahme: 200 Krankheiten, q = 0,6, u = 0,3: −0,54 ± 0,26. | `test_befund_3_*` |
| 4 | Wie hängt der Verlust an der Zahl der Krankheiten und am Dokumentationsverlust? | Verlust (exakt − überlesen, Recall@5) bei u = 0,3 / 0,6 / 0,9: 50 Krankheiten, q = 0,0: +7,3 / +23,5 / +59,5; 50 Krankheiten, q = 0,3: +7,9 / +22,0 / +48,5; 50 Krankheiten, q = 0,6: +6,5 / +15,9 / +31,1; 200 Krankheiten, q = 0,0: +10,8 / +29,7 / +58,8; 200 Krankheiten, q = 0,3: +9,5 / +22,5 / +39,5; 200 Krankheiten, q = 0,6: +5,4 / +11,7 / +19,6. Der Verlust ist in allen sechs (Krankheiten, q)-Paaren bei u = 0,3 kleiner als bei u = 0,6 und dort kleiner als bei u = 0,9. | `test_befund_4_*` |
| 5 | **Hilft ein Resnik-artiger Ähnlichkeitswert?** (Kernfrage 3; eigene Fassung) | Voreinstellung: Resnik-artig − überlesen −8,7 / +3,0 / +18,6 bei u = 0,3 / 0,6 / 0,9 (Standardfehler 0,2 / 0,3 / 0,4); Resnik-artig − alle Kinder −10,7 / −7,3 / −8,1; exakt − Resnik-artig +18,2 bei u = 0,3. Über alle 18 Zellen mit u > 0: gegenüber dem Überlesen vorn in 11, hinten in 6, ohne auflösbaren Unterschied in 1 Zellen (bei u = 0,3 / 0,6 / 0,9 vorn in 0 / 5 / 6 von je sechs); gegenüber alle Kinder hinten in 18 von 18. Bei u = 0: Recall@5 exakt 76,5, Resnik-artig 59,4; über die sechs Zellen mit u = 0 liegt der Abstand zwischen 11,3 und 17,6 Punkten. | `test_befund_5_*` |
| 6 | Viele Lücken (q = 0,6): ändert sich das Bild? | 200 Krankheiten, q = 0,6, u = 0,3: exakt 43,8, überlesen 38,4, alle Kinder 37,9; Verlust +5,4 ± 0,1; alle Kinder − überlesen −0,54 ± 0,26 (hier holt die Heuristik nichts zurück: gepoolt liegt sie knapp mehr als zwei Standardfehler hinter dem Überlesen, in jeder der beiden Mengen einzeln innerhalb von zwei Standardfehlern; gedeutet wird das nicht). Zählung über alle sechs (Krankheiten, u)-Paare mit u > 0: der Verlust nimmt mit steigendem q in 5 Paaren in beiden Schritten ab (Ausnahme: 50 Krankheiten, u = 0,3: 7,3 / 7,9 / 6,5 bei q = 0 / 0,3 / 0,6), die Rückholung der Heuristik in 6 Paaren. | `test_befund_6_*` |
| 7 | Ohne grobe Angaben (u = 0): gleich? | In allen sechs Zellen mit u = 0 ist der gepaarte Unterschied zwischen exakt, überlesen und alle Kinder in beiden Mengen genau 0,0 Punkte; Anteil grober Angaben 0 (das folgt aus der Rechnung: ohne grobe Angaben sind die Formeln gleich; gemessen ist, dass die Umsetzung es auch ist, seit Gleichstände nach dem Runden gezählt werden, Korrektur 2). | `test_befund_7_*` |
| 8 | Kann ein anderer Ranker den exakten übertreffen? | In 768 Vergleichen (24 Zellen × 2 Mengen × 4 andere Ranker × 4 Kennzahlen) liegt kein Ranker über dem exakten; größter Vorsprung 0,0 Punkte (das ist die Optimalität des Posteriors im Erwartungswert, in einer Stichprobe nicht selbstverständlich). | `test_befund_8_*` |
| 9 | Stabilität | Die beiden Mengen von Welten unterscheiden sich im Recall@5 um höchstens 3,04 Punkte (geprüft über 24 Zellen × 5 Ranker); die Streuung über die Welten beträgt höchstens 3,7 Punkte (über alle Zellen, Ranker und beide Mengen). | `test_befund_9_*` |
| 10 | Wie viel schlechter ist das naive Überlesen (ohne Wissen um u)? | Voreinstellung: naives Überlesen Recall@5 60,0; „überlesen“ minus naiv +0,8 ± 0,2; exakt minus naiv +10,3 ± 0,3. Über alle 18 Zellen mit u > 0 liegt „überlesen“ in 17 mehr als zwei Standardfehler vor dem naiven; die Differenzen der Zellen liegen zwischen +0,28 und +10,59 Punkten. Bei u = 0 sind beide gleich. | `test_befund_10_*` |

Die Zahlen sind in der Datei auf Rundungsstellen genau gespeichert und im Test auf die hier gezeigte Stelle geprüft; `test_neu_rechnen_*` rechnet die Voreinstellung (Menge A) neu und vergleicht im Band von 0,6 Punkten (Linux rechnet an der letzten Binärstelle anders); der Zufallsstrom selbst ist bitgleich (`tests/test_stream.py`). Was **nicht** im Test steht: die Zahlen der Spalte „früher“ (Messung vor dem Bau, nicht Teil dieses Repos), die Rechenzeiten und die Zahl der Tests selbst.

## Ehrliche Grenzen

- **Das Orakel gehört zum Erzeuger.** Dass „exakt“ nirgends hinter einem anderen Ranker liegt, ist die Optimalität des Posteriors im Erwartungswert (nur für Patienten aus dem angenommenen Erzeuger), kein Beleg für echte Akten. In echten Daten kennt man $u$ und die Häufigkeiten nicht.
- **Eine Ebene Oberbegriffe, fünf Kinder je Oberbegriff, ein einziges $u$.** Echte Merkmalshierarchien sind tief, verzweigt und ungleich; ob grob oder genau dokumentiert wird, hängt von Person, Merkmal und Kontext ab. Drei Ebenen und ungleiche Anteile sind nicht gerechnet.
- **Der Resnik-artige Ranker ist eine eigene einfache Fassung.** Aussagen über ihn gelten für diese Fassung (Befund 5); wie ein ausgebautes Verfahren mit Profilhäufigkeiten abschneidet, zeigt die Demo nicht.
- **Der Ranker „überlesen“ kennt $u$** (starker Gegner); der naive Ranker ohne dieses Wissen ist mitgemessen (Korrektur 3, Befund 10).
- **Merkmale unabhängig gegeben die Krankheit**, nur an oder aus; echte Merkmale sind abhängig, es gibt Schweregrad, Alter und Verlauf.
- **Größe:** 50 und 200 Krankheiten, 100 Merkmale, nicht Tausende; zwei Mengen von je fünf Welten. Die Streuung über die Welten und gepaarte Standardfehler stehen bei den Zahlen; Unterschiede unter zwei Standardfehlern werden nicht gedeutet.
- **Gerankt werden nur Patienten mit mindestens einer Angabe in der Akte;** die Recall-Werte gelten für sie (Abschnitt Modell).
- **Stand der Technik** (Ähnlichkeitsmaße auf Ontologien wie Phenomizer, Likelihood-Ratio-Ranker, Sprachmodelle) wurde hier nicht nachgebaut und nicht verglichen.

## Tests

165 Tests (`pytest tests/`, mit `-n 8` je nach Rechnerlast 25 bis 90 Sekunden, zweimal gemessen): Zufallsstrom gegen die Skalarklasse (bitgleich) und Kopien der Hilfsmodule aus `phaenotyp-ranking-demo`; Welt und Kennzahlen (Randfälle, Gleichstände); **Orakel** (Aufzählung aller $3^3$ Kind-Zustände einer Mini-Gruppe gegen die Formel des exakten Rankers für jede Beobachtung; Randverteilung für „überlesen“; Handrechnung für „alle Kinder“; Zufallsziehungen mit eigener Konstruktion gegen die Aufzählung; schleifenweise Rechnung in reinem Python gegen die vektorisierte Rechnung über 20 Gruppen; Resnik-Tabellen von Hand); Hierarchie (Form, u = 0 und u = 1, Anteil grober Angaben); Live-Messung (Wiederholbarkeit, **Zweig-Test je Regler**: jede Stufe ändert mindestens einen Ranker und u jeden der fünf, exakt = überlesen = alle Kinder = naiv bei u = 0, Worst-Case ohne Zeit-Assertion); Verdikte (Schwellen an handgesetzten Werten, jede Zahl im Text); Abnahmekriterien der Presets an der Messreihe, jedes an künstlichen Zellen einzeln kippend; Messreihe (Form, Plausibilität, Stabilität); Zahlen dieser README (mit Neurechnung der Voreinstellung als Band) (`test_claims.py`, README gegen `tools/make_readme.py`); AppTest (Standard, jedes Preset, jede Stufe jedes Reglers am Minimum und Maximum, Kombination aller Extremwerte, PDF erst nach Klick); Layout, Footer, Laufzeit-Importe. Keine Mutationstests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | Git- und Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `gmk_constants.py` | Annahmen der Welt, Hierarchie, Regler-Stufen, Ranker, Farben, Presets |
| `gmk_rng.py` | SplitMix64 (unveränderte Kopie der Vorlage aus dem Website-Repo) |
| `gmk_stream.py`, `gmk_world.py`, `gmk_metrics.py` | Kopien aus `phaenotyp-ranking-demo` mit umbenanntem Präfix: vektorisierter Strom, Erzeuger der Welt, Kennzahlen (Kopier-Test in `tests/test_copies.py`) |
| `gmk_hierarchy.py` | Gruppen der Merkmale, grobe Dokumentation der Patienten, Annotationen |
| `gmk_rankers.py` | exakter Ranker, Überlesen, alle Kinder, Resnik-artig |
| `gmk_live.py` | Live-Messung, Auswertung über die Welten, gepaarte Differenz |
| `gmk_evaluation.py` | Verdikte (reine Funktionen), Beispielpatient, Zahlenformat |
| `gmk_results.py`, `data/gmk_results.json` | Zugriff auf die vorgerechnete Messreihe (24 Zellen × 2 Mengen) |
| `gmk_stories.py` | Abnahmekriterien der Presets |
| `gmk_presets.py`, `gmk_layout.py` | Stufen-Regler, Permalink (rastet auf Stufen ein), Schnellstart, Hervorhebung, fester Kopf |
| `gmk_visualization.py`, `gmk_ui_panel.py`, `gmk_pdf_export.py` | Abbildungen (Achsen gesperrt), Darstellung, PDF (fpdf2, erst nach Klick) |
| `tools/sweep.py` | reproduziert `data/gmk_results.json` (rund eineinhalb Minuten, nicht in der CI) |
| `tools/make_readme.py` | erzeugt diese README aus der Messreihe |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit beziehungsweise plus pytest |
| `tests/` | siehe „Tests“ |

## Bewusst nicht umgesetzt

Hierarchie mit drei Ebenen, ungleiche Anteile grober Angaben je Merkmal, Abhängigkeiten zwischen Merkmalen, Schätzung von $u$ aus Daten, ein Ranker mit Profilhäufigkeiten nach dem Vorbild von Phenomizer, echte Wissensbasen (HPO, Orphanet), Sprachmodell-Ranker, mehr als 200 Krankheiten.

## Verwandte Demos mit demselben mathematischen Modell

- [`phaenotyp-ranking-demo`](https://github.com/sebastian-hanisch/phaenotyp-ranking-demo): dieselbe Welt (Profile, Hintergrund, Dokumentationsverlust) und das Orakel als Maßstab; dort sind die Merkmale flach, hier gibt es Oberbegriffe.
- [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo): Kalibrierung und Schwelle aus dem Kostenverhältnis; hier wird eine Rangliste gemessen, dort die Entscheidung aus einer Wahrscheinlichkeit.
- [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo): Bayes-Update und Wert eines Tests bei zwei Zuständen; hier sind es viele Krankheiten und ein Ranking.
- `konforme-ursachen-demo`: Vorhersagemengen mit Abdeckungsgarantie für Ursachen; anderes Modell, die Demos sind nicht verbunden.

Zahlen aus diesen Demos werden hier weder nachgerechnet noch verglichen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Python-Version der Streamlit Cloud ist in der App-Einstellung zu wählen (die CI läuft mit Python 3.12); die Demo nutzt nur Sprachmerkmale ab Python 3.10. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
