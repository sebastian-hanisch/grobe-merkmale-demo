"""Erzeugt README.md aus der Messreihe (data/gmk_results.json): jede Zahl der Befunde kommt aus der Datei, nie aus der Hand.

Aufruf (im Demo-Ordner): python tools/make_readme.py          schreibt README.md
                         python tools/make_readme.py --check  prüft, ob README.md dem Erzeugnis entspricht (tests/test_claims.py tut dasselbe)

Die Zahlen der Spalte „früher“ (Messung vor dem Bau, NumPy-Zufall, fünf Welten × 1500 Patienten, 200 Krankheiten, q = 0,3) stehen als Konstanten in ALT:
sie sind nicht Teil dieses Repos und hier nicht nachprüfbar. Die Zahl der Tests steht in TESTS_TOTAL und wird gegen die gesammelten Tests geprüft (tests/test_claims.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gmk_constants as C  # noqa: E402
import gmk_results as RES  # noqa: E402

TESTS_TOTAL = 165
TESTS_BY_FILE = {}   # Teilzahlen werden nicht genannt (nur die Gesamtzahl)

# Frühere Messung (200 Krankheiten, q = 0,3, fünf Welten × 1500 Patienten, NumPy-Zufall): Recall@1 und Recall@5 in Prozent je u und Ranker; Anteil grober Angaben in Prozent.
ALT = {
    0.0: {"X": (52.7, 74.4), "I": (52.7, 74.4), "A": (52.7, 74.4), "share": 0.0},
    0.3: {"X": (45.6, 68.4), "I": (36.7, 59.0), "A": (36.0, 60.6), "share": 30.2},
    0.6: {"X": (36.4, 60.5), "I": (19.6, 38.3), "A": (23.7, 48.4), "share": 59.9},
    0.9: {"X": (25.2, 49.9), "I": (3.6, 12.2), "A": (15.3, 37.2), "share": 90.2},
}


def f(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def sf(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def cell(nd=200, q=0.3, u=0.3):
    return RES.cell(RES.Settings(nd=nd, q=q, u=u))


def r(c, ranker, k=5):
    return RES.pooled(c, ranker, k)


def pair(c, p):
    return RES.pooled_pair(c, p)


def share(c, key):
    return (c["A"][key] + c["B"][key]) / 2.0


def zahl(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def zwei_se(m, se):
    return abs(m) > C.SD_FACTOR * se


def befunde() -> dict:
    """Alle Zahlen der README, einmal berechnet."""
    z = {}
    base = cell()
    z["base"] = {rk: [r(base, rk, k) for k in (1, 5, 10)] + [r(base, rk, "mrr")] for rk in C.RANKERS}
    z["base_pair"] = {p: pair(base, p) for p in ("X-I", "X-A", "A-I", "R-I", "R-A", "X-R")}
    z["u_rows"] = []
    for u in C.U_OPTIONS:
        c = cell(u=u)
        xi, ai = pair(c, "X-I"), pair(c, "A-I")
        z["u_rows"].append({"u": u, "xi": xi, "ai": ai, "ri": pair(c, "R-I"), "ra": pair(c, "R-A"), "x": r(c, "X"), "i": r(c, "I"), "a": r(c, "A"), "rr": r(c, "R"),
                            "x1": r(c, "X", 1), "i1": r(c, "I", 1), "a1": r(c, "A", 1), "back_share": (100.0 * ai[0] / xi[0]) if xi[0] > 0.1 else None,
                            "coarse": share(c, "share_coarse"), "ranked": share(c, "share_ranked")})
    z["grid"] = {(nd, q): [pair(cell(nd, q, u), "X-I") for u in C.U_OPTIONS] for nd in C.ND_OPTIONS for q in C.Q_OPTIONS}
    z["grid_back"] = {(nd, q): [pair(cell(nd, q, u), "A-I") for u in C.U_OPTIONS] for nd in C.ND_OPTIONS for q in C.Q_OPTIONS}
    z["ranked"] = {q: share(cell(200, q, 0.3), "share_ranked") for q in C.Q_OPTIONS}
    z["ranked50"] = {q: share(cell(50, q, 0.3), "share_ranked") for q in C.Q_OPTIONS}
    # Heuristik: Zellen mit u > 0, in denen sie mehr als zwei Standardfehler vor dem Überlesen liegt
    count = {"A-I": [0, 0, 0], "R-I": [0, 0, 0], "R-A": [0, 0, 0]}
    by_u = {u: {"R-I": [0, 0, 0]} for u in C.U_OPTIONS if u > 0}
    odd = []
    for nd in C.ND_OPTIONS:
        for q in C.Q_OPTIONS:
            for u in C.U_OPTIONS:
                if u == 0:
                    continue
                c = cell(nd, q, u)
                for p in count:
                    m, se = pair(c, p)
                    i = 0 if m > C.SD_FACTOR * se else (1 if m < -C.SD_FACTOR * se else 2)
                    count[p][i] += 1
                    if p == "R-I":
                        by_u[u]["R-I"][i] += 1
                    if p == "A-I" and i != 0:
                        odd.append((nd, q, u, m, se))
    z["count"], z["by_u"], z["odd"] = count, by_u, odd
    z["xr0"] = {(nd, q): (pair(cell(nd, q, 0.0), "X-R")[0], r(cell(nd, q, 0.0), "X"), r(cell(nd, q, 0.0), "R")) for nd in C.ND_OPTIONS for q in C.Q_OPTIONS}
    # Obergrenze: größter Vorsprung eines anderen Rankers vor exakt über alle Zellen, Mengen und Kennzahlen
    best, n = -1e9, 0
    for s in RES.sweep_settings():
        c = RES.cell(s)
        for w in ("A", "B"):
            for rk in ("I", "A", "R", "N"):
                for k in ("1", "5", "10", "mrr"):
                    n += 1
                    best = max(best, c[w]["summary"][rk][k][0] - c[w]["summary"]["X"][k][0])
    z["over"] = (best, n)
    # Verlust nach Dokumentationsverlust q: in wie vielen (Krankheiten, u)-Paaren nimmt er mit q in beiden Schritten ab (und die Rückholung der Heuristik)?
    z["q_mono"] = {"X-I": [], "A-I": []}
    for p_ in z["q_mono"]:
        for nd in C.ND_OPTIONS:
            for u in (0.3, 0.6, 0.9):
                vals = [pair(cell(nd, q, u), p_)[0] for q in C.Q_OPTIONS]
                z["q_mono"][p_].append((nd, u, vals, vals[0] > vals[1] > vals[2]))
    # naives Überlesen: überlesen (kennt u) minus naiv, Zellen mit u > 0
    ni = [(nd, q, u) + pair(cell(nd, q, u), "I-N") for nd in C.ND_OPTIONS for q in C.Q_OPTIONS for u in C.U_OPTIONS if u > 0]
    z["naive"] = {"min": min(v[3] for v in ni), "max": max(v[3] for v in ni), "vor": sum(v[3] > C.SD_FACTOR * v[4] for v in ni), "n": len(ni),
                  "base": pair(cell(), "I-N"), "xn": pair(cell(), "X-N")}
    z["stab"] = (max(RES.sets_agree(RES.cell(s), rk) for s in RES.sweep_settings() for rk in C.RANKERS),
                 max(c[w]["summary"][rk]["5"][1] for c in RES.load()["cells"].values() for w in ("A", "B") for rk in C.RANKERS))
    # Vergleich früher gegen neu
    shifts = []
    for u in C.U_OPTIONS:
        row = next(x for x in z["u_rows"] if x["u"] == u)
        for rk, k1, k5 in (("X", row["x1"], row["x"]), ("I", row["i1"], row["i"]), ("A", row["a1"], row["a"])):
            shifts += [abs(k1 - ALT[u][rk][0]), abs(k5 - ALT[u][rk][1])]
    z["shift_max"] = max(shifts)
    return z


def build() -> str:
    z = befunde()
    meta = RES.load()["meta"]
    b = z["base"]
    bp = z["base_pair"]
    rows_u = {row["u"]: row for row in z["u_rows"]}
    cnt = z["count"]
    over, n_over = z["over"]
    stab, sd_max = z["stab"]
    L = []
    add = L.append

    add("# Grobe Merkmale – wenn ein Symptom nur als Oberbegriff in der Akte steht – Streamlit-Demo")
    add("")
    add("<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):")
    add("**[→ Demo live ausprobieren](https://sebastianhanisch-grobe-merkmale-demo.streamlit.app/)** -->")
    add("")
    add("Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Entscheidungsanalyse und Machine Learning). Aus den **dokumentierten Merkmalen** (Symptomen) eines Patienten wird eine Rangliste "
        "möglicher **seltener Krankheiten** erstellt, wenn ein Teil der Merkmale in der Akte nur als **Oberbegriff** steht (zum Beispiel „Augenauffälligkeit“ statt des genauen Befunds). "
        "Die Demo vergleicht fünf Ranker: den **exakten** Ranker, der die Regeln des Erzeugers dieser Welt kennt (das **Orakel**), einen Ranker, der die Oberbegriffe **überliest**, "
        "die Heuristik **„Oberbegriff = alle Kinder“**, einen **Resnik-artigen** Ähnlichkeitswert (eigene Fassung) und ein **naives Überlesen**, das u nicht kennt. Gemessen wird, was das Überlesen kostet und wie viel davon die Heuristik und der Ähnlichkeitswert zurückholen.")
    add("")
    add("**Lehrmodell mit erfundenen Daten, keine medizinische Beratung und keine Diagnoseempfehlung.** Welt, Krankheiten, Merkmale und alle Zahlen sind erzeugt; es gibt keine echten Patientendaten und keine echte Wissensbasis. "
        "Die Aussagen gelten für diese Welt, bei diesen Einstellungen und für die gemessene Regel jedes Rankers. **Voraussetzung für alle Vergleiche mit „exakt“:** der exakte Ranker ist das Orakel des Erzeugers: "
        "er kennt den Anteil grober Angaben u und alle Häufigkeiten der Welt; in echten Daten gibt es dieses Orakel nicht, u müsste geschätzt werden.")
    add("")
    add("**Bezug zu OR:** eine Rangliste aus unvollständigen Beobachtungen ist eine Entscheidung unter Unsicherheit; wie grob die Beobachtung ist, bestimmt, wie viel von der Information ankommt. Die Demo zeigt den Lernteil für sich "
        "und baut auf derselben Welt auf wie `phaenotyp-ranking-demo`.")
    add("")
    add("## Warum dieses Problem")
    add("")
    add("Klinische Akten nennen ein Symptom oft nur auf der Ebene eines Oberbegriffs. Ein Ranker, der solche Angaben nicht kennt, überliest sie oder rät. **Kernfrage 1:** Was kostet es, grobe Angaben zu überlesen? "
        "**Kernfrage 2:** Holt die einfache Heuristik „Oberbegriff = alle Kinder“ den Verlust zurück? **Kernfrage 3:** Hilft ein Resnik-artiger Ähnlichkeitswert?")
    add("")
    add("## Befunde und Korrekturen gegenüber der früheren Messung")
    add("")
    add(f"Eine frühere Messung vor dem Bau (NumPy-Zufall, fünf Welten × 1500 Patienten, 200 Krankheiten, q = 0,3; sie ist nicht Teil dieses Repos, ihre Zahlen in der Spalte „früher“ sind hier nicht nachprüfbar) "
        f"ging dem Bau voraus. Im Bau wurde mit dem **SplitMix64** der Demo in **zwei Mengen von je fünf Welten** (Zufalls-Seed {meta['sets']['A']} und {meta['sets']['B']}) neu gemessen, in {meta['cells']} Zellen. "
        f"Die Kernbefunde bestehen der Richtung nach; gegenüber den (hier nicht nachprüfbaren) Zahlen der früheren Messung verschieben sich die neuen Recall-Zahlen um höchstens {f(z['shift_max'])} Punkte. Die Spalte „neu“ ist das Mittel der beiden Mengen (zehn Welten); der Resnik-artige Ranker war in der früheren Messung nicht dabei.")
    add("")
    add("| Kernbeleg (200 Krankheiten, q = 0,3; Recall in %) | Früher | Neu gemessen (zehn Welten) |")
    add("|---|---|---|")
    add(f"| u = 0: exakt, überlesen und alle Kinder (gleich), Recall@1 / Recall@5 | {f(ALT[0.0]['X'][0])} / {f(ALT[0.0]['X'][1])} | {f(rows_u[0.0]['x1'])} / {f(rows_u[0.0]['x'])} |")
    for u in (0.3, 0.6, 0.9):
        a, n = ALT[u], rows_u[u]
        add(f"| u = {f(u)}: exakt / überlesen / alle Kinder, Recall@1 | {f(a['X'][0])} / {f(a['I'][0])} / {f(a['A'][0])} | {f(n['x1'])} / {f(n['i1'])} / {f(n['a1'])} |")
        add(f"| u = {f(u)}: exakt / überlesen / alle Kinder, Recall@5 | {f(a['X'][1])} / {f(a['I'][1])} / {f(a['A'][1])} | {f(n['x'])} / {f(n['i'])} / {f(n['a'])} |")
    add(f"| Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff stehen, bei u = 0,3 / 0,6 / 0,9 (Prozent) | {f(ALT[0.3]['share'])} / {f(ALT[0.6]['share'])} / {f(ALT[0.9]['share'])} | "
        f"{f(rows_u[0.3]['coarse'])} / {f(rows_u[0.6]['coarse'])} / {f(rows_u[0.9]['coarse'])} |")
    add("")
    add("**Korrekturen und Abweichungen gegenüber der früheren Messung:**")
    add("")
    add(f"1. **Der Resnik-artige Ranker liegt hinter der Heuristik und hinter dem exakten Ranker, und schon ohne grobe Angaben weit hinter exakt.** Bei u = 0 liegt er in den sechs Zellen mit 50 und 200 Krankheiten und q = 0 / 0,3 / 0,6 "
        f"{f(min(v[0] for v in z['xr0'].values()))} bis {f(max(v[0] for v in z['xr0'].values()))} Punkte Recall@5 hinter exakt. Gegenüber dem Überlesen liegt er in {cnt['R-I'][0]} der 18 Zellen mit u > 0 mehr als zwei Standardfehler vorn, "
        f"in {cnt['R-I'][1]} dahinter und in {cnt['R-I'][2]} ohne auflösbaren Unterschied; gegenüber der Heuristik liegt er in {cnt['R-A'][1]} von 18 Zellen dahinter (Befund 5). "
        "Die Fassung ist eine eigene, einfache (Summe über die Begriffe der Akte; keine Häufigkeiten, kein Vorab, keine Abwesenheit); Aussagen über sie gelten für diese Fassung, nicht für Ähnlichkeitsmaße im Allgemeinen. "
        "Warum der Abstand schon bei u = 0 so groß ist, ist nicht einzeln gemessen; plausibel ist, dass ihr die Häufigkeiten der Profile fehlen (Deutung, nicht geprüft).")
    add("2. **Gleichstände werden nach dem Runden auf neun Stellen gezählt.** Krankheiten, die die Akte nicht unterscheidet, haben mathematisch gleiche Punktzahlen; die Summen weichen in den letzten Bits ab, wenn sie in anderer Reihenfolge "
        "gebildet werden. Ohne Rundung unterscheiden sich bei u = 0 „exakt“ und „überlesen“ um wenige Hundertstel Punkte Recall@5, obwohl sie dieselbe Rangliste meinen (Voreinstellung, Menge A, gepaart: weniger als 0,05 Punkte, "
        "von `test_gleichstaende_ohne_rundung_*` nachgerechnet; der genaue Wert hängt vom Rundungsrauschen der Plattform ab und steht deshalb hier nicht); mit Rundung ist der gepaarte Unterschied "
        "bei u = 0 in allen sechs Zellen und beiden Mengen genau 0 (Befund 7; `tests/test_live.py`). Die Messreihe wurde nach der Rundung gerechnet.")
    nv = z["naive"]
    add("3. **„Überlesen“ ist der starke Gegner, nicht der naive:** der Ranker rechnet mit den richtigen Randwahrscheinlichkeiten (ein genaues Merkmal steht mit Wahrscheinlichkeit g(1 − u) in der Akte), er kennt also u. "
        "Der naive Ranker, der die groben Angaben ebenfalls überliest, aber mit den ungeteilten Häufigkeiten g rechnet, ist als fünfter Ranker mitgemessen: "
        f"in {nv['vor']} der {nv['n']} Zellen mit u > 0 liegt „überlesen“ mehr als zwei Standardfehler vor ihm, die Differenzen der Zellen liegen zwischen {sf(nv['min'], 2)} und {sf(nv['max'], 2)} Punkten Recall@5 (Befund 10).")
    add("4. **Die Schwellen der Verdikte sind gesetzt, nicht abgeleitet:** Verlust (exakt minus überlesen, Recall@5) unter 5 Punkten, bis 15 Punkte, darüber. Die Verdikte laufen live aus der aktuellen Messung (gepaarte Differenz über die Welten, "
        "Standardfehler dabei); die Abbildungen und Tabellen der Kernfragen stammen aus der vorgerechneten Messreihe.")
    add(f"5. **Abnahmekriterium des Presets „Meist grob“ geändert:** vorgesehen war „überlesen höchstens 40 Prozent Recall@5“; gemessen sind {f(RES.stat(cell(200, 0.3, 0.6), 'A', 'I')[0])} und {f(RES.stat(cell(200, 0.3, 0.6), 'B', 'I')[0])} (Mengen A und B), die Schwelle läge auf der Zellgrenze. "
        "Ersetzt durch: exakt liegt mehr als 15 Punkte vor überlesen, und alle Kinder liegt mehr als zwei Standardfehler vor überlesen und mehr als zwei Standardfehler hinter exakt (`gmk_stories.py`).")
    add("6. **Nicht gerechnet:** eine Hierarchie mit drei Ebenen und ein Anteil grober Angaben, der je Merkmal verschieden ist. Die Demo zeigt eine Ebene (20 Oberbegriffe zu je fünf Merkmalen) und ein einziges u für alle Merkmale.")
    add("")
    add("## Modell")
    add("")
    add("**Welt.** Dieselbe wie in `phaenotyp-ranking-demo`: 100 Merkmale mit Zipf-verteilter Beliebtheit $w_s \\propto \\mathrm{Rang}^{-0{,}8}$; jede Krankheit hat 4 bis 12 Profilmerkmale (gleichverteilt, gezogen ohne Zurücklegen "
        "proportional zur Beliebtheit) mit Häufigkeit 0,95 / 0,6 / 0,3 / 0,1 (je Profilmerkmal mit den Wahrscheinlichkeiten 15 / 25 / 35 / 25 Prozent gezogen); Hintergrundrate $b_s = 0{,}01 + 0{,}2\\,(w_s / w_{\\max})^2$. "
        "Ein vorhandenes Merkmal wird mit Wahrscheinlichkeit $1 - q$ aufgeschrieben ($q$ = 0 / 0,3 / 0,6): $g_{dc} = 1 - (1 - f_{dc}(1 - q))(1 - b_c)$; Merkmale unabhängig gegeben die Krankheit. "
        "Vorab gleichverteilt, alle Merkmale geprüft, keine falschen Einträge, Wissensbasis = wahre Häufigkeiten. Patienten ohne jede Angabe in der Akte werden nicht gerankt: in den vorgerechneten Zellen mit u = 0,3 bleiben bei 200 Krankheiten "
        f"{f(z['ranked'][0.0])} / {f(z['ranked'][0.3])} / {f(z['ranked'][0.6])} Prozent der Patienten bei q = 0 / 0,3 / 0,6 (bei 50 Krankheiten {f(z['ranked50'][0.0])} / {f(z['ranked50'][0.3])} / {f(z['ranked50'][0.6])}); "
        "**die Recall-Werte gelten für diese gerankten Patienten.**")
    add("")
    add("**Hierarchie und grobe Dokumentation.** Die Merkmale $5G$ bis $5G + 4$ bilden die Gruppe $G$ (20 Gruppen, je ein Oberbegriff, fünf Kinder). Ein aufgeschriebenes Merkmal steht mit Wahrscheinlichkeit $1 - u$ genau in der Akte "
        "und mit Wahrscheinlichkeit $u$ nur als Oberbegriff seiner Gruppe ($u$ = 0 / 0,3 / 0,6 / 0,9). Mehrere grobe Angaben derselben Gruppe ergeben **einen** Oberbegriff: aus ihm ist nicht zu erkennen, wie viele Merkmale dahinterstehen. "
        "Die Akte einer Gruppe besteht aus der Menge $L_G$ der genau genannten Kinder und dem Zeichen $F_G$ („Oberbegriff steht in der Akte“).")
    add("")
    add("**Exakter Ranker (Orakel des Erzeugers).** Wahrscheinlichkeit der Akte unter dem Erzeuger, Produkt über die Gruppen: $\\prod_{c \\in L_G} g_{dc}(1 - u)$ mal $\\prod_{c \\notin L_G}(1 - g_{dc})$ bei $F_G = 0$ "
        "beziehungsweise $\\prod_{c \\notin L_G}(1 - g_{dc} + g_{dc} u) - \\prod_{c \\notin L_G}(1 - g_{dc})$ bei $F_G = 1$ (mindestens ein Kind außerhalb von $L_G$ steht grob in der Akte). "
        "**Das Orakel gehört zum Erzeuger und ist nur in diesem Modell eine Obergrenze;** in allen "
        f"{zahl(n_over)} Vergleichen (24 Zellen × 2 Mengen × 4 andere Ranker × 4 Kennzahlen) liegt kein anderer Ranker über ihm (größter Vorsprung {f(max(over, 0.0))} Punkte).")
    add("")
    add("## Methodik")
    add("")
    add("Fünf Ranker: **exakt** (Gruppen-Wahrscheinlichkeit der Akte, kennt $u$ und alle $g$); **nur genaue Begriffe** („überlesen“: das Zeichen $F_G$ wird nicht gelesen, jedes Merkmal steht mit $g(1-u)$ genau in der Akte oder nicht; "
        "das ist die Randverteilung des exakten Rankers über $F_G$); **Oberbegriff = alle Kinder** (ein Oberbegriff zählt, als stünden alle fünf Kinder in der Akte; Naive Bayes mit den ungeteilten $g$, ohne Wissen um $u$); "
        "**Resnik-artig** (eigene Fassung, nicht Phenomizer: Annotation einer Krankheit = Profilhäufigkeit über 0,05; Informationsgehalt $-\\ln$ des Anteils der Krankheiten, die den Begriff annotieren, Wurzel 0; ein genaues Merkmal trifft "
        "eine Krankheit mit seinem Informationsgehalt, wenn sie es annotiert, sonst mit dem seines Oberbegriffs, wenn sie den Oberbegriff annotiert, sonst mit 0; ein Oberbegriff trifft sie mit seinem Informationsgehalt, wenn sie ihn annotiert; "
        "Punktzahl = Summe über die Begriffe der Akte, kein Vorab); **naives Überlesen** (nur die genauen Begriffe, Naive Bayes mit den ungeteilten $g$, kennt $u$ nicht).")
    add("")
    add("**Kennzahlen.** Recall@1, @5, @10: Anteil der **gerankten** Patienten, bei denen die wahre Krankheit unter den ersten $k$ steht; Gleichstände werden als Wahrscheinlichkeit gezählt (zufällige Auflösung, "
        "$\\min(1, \\max(0, (k - r + 1)/(t + 1)))$ mit dem kleinsten Rang $r$ und $t$ Gleichständen), nicht als Treffer. Dazu der **reziproke mittlere Rang** $1/(r + t/2)$. **Einheit:** Patienten (Recall) beziehungsweise Welten (Streuung).")
    add("")
    add(f"**Messung.** Je Einstellung 5 Welten × 1500 Patienten (SplitMix64, `gmk_stream.py` ist die vektorisierte Fassung der Skalarklasse in `gmk_rng.py`, bitgleich geprüft). Mittel ± Streuung (n − 1) über die Welten. "
        "**Gepaarte Vergleiche:** Ob ein Ranker vor einem anderen liegt, entscheidet die Differenz je Welt (Mittel ± Standardfehler = Streuung der Differenzen / √5); Unterschiede unter zwei Standardfehlern werden nicht gedeutet "
        "(ein Richtungskriterium bei fünf Welten, kein 95-Prozent-Test: bei vier Freiheitsgraden entspricht ein t von 2 einem zweiseitigen p von etwa 0,12). "
        f"**Vorgerechnet** (`data/gmk_results.json`, `tools/sweep.py`): {meta['cells']} Zellen (2 Zahlen von Krankheiten × 3 Dokumentationsverluste × 4 Anteile grober Angaben), je zwei Mengen von fünf Welten "
        f"(Zufalls-Seed {meta['sets']['A']} und {meta['sets']['B']}). Der gepoolte Standardfehler der zehn Welten ist aus den beiden Standardfehlern zusammengesetzt: $\\sqrt{{s_A^2 + s_B^2}}/2$.")
    add("")
    add("## Befunde (gemessen, keine Behauptungen)")
    add("")
    add("Alle Zahlen: 200 Krankheiten, 100 Merkmale, Dokumentationsverlust q = 0,3, Vorab gleichverteilt, wenn nicht anders genannt; Recall@5 in Prozent der gerankten Patienten, **Mittel der zwei Mengen zu je 5 Welten × 1500 Patienten "
        "(zehn Welten)**; „gepaart“ = Mittel der Differenzen je Welt in Punkten, ± gepoolter Standardfehler. Jede Zeile nennt den Test in `tests/test_claims.py`, der die Zahl aus `data/gmk_results.json` unabhängig nachrechnet.")
    add("")
    add("| # | Frage | Befund | Test |")
    add("|---|---|---|---|")
    add("| 1 | Voreinstellung (u = 0,3, q = 0,3, 200 Krankheiten): wie gut sind die fünf Ranker? | Recall@5: exakt "
        f"{f(b['X'][1])}, überlesen {f(b['I'][1])}, alle Kinder {f(b['A'][1])}, Resnik-artig {f(b['R'][1])}, naives Überlesen {f(b['N'][1])}. Recall@1: {f(b['X'][0])} / {f(b['I'][0])} / {f(b['A'][0])} / {f(b['R'][0])} / {f(b['N'][0])}. "
        f"Recall@10: {f(b['X'][2])} / {f(b['I'][2])} / {f(b['A'][2])} / {f(b['R'][2])} / {f(b['N'][2])}. Reziproker mittlerer Rang: {f(b['X'][3])} / {f(b['I'][3])} / {f(b['A'][3])} / {f(b['R'][3])} / {f(b['N'][3])}. "
        f"Gepaart: exakt − überlesen {sf(bp['X-I'][0])} ± {f(bp['X-I'][1])}; exakt − alle Kinder {sf(bp['X-A'][0])} ± {f(bp['X-A'][1])}; alle Kinder − überlesen {sf(bp['A-I'][0])} ± {f(bp['A-I'][1])}; "
        f"Resnik-artig − überlesen {sf(bp['R-I'][0])} ± {f(bp['R-I'][1])}. | `test_befund_1_*` |")
    ur = [rows_u[u] for u in (0.3, 0.6, 0.9)]
    add("| 2 | **Was kostet es, grobe Angaben zu überlesen?** (Kernfrage 1; Verlust = exakt − überlesen, Recall@5 in Punkten) | "
        f"u = 0: {f(rows_u[0.0]['xi'][0], 2)} (kein Unterschied, bei u = 0 keine groben Angaben). u = 0,3 / 0,6 / 0,9: {sf(ur[0]['xi'][0])} / {sf(ur[1]['xi'][0])} / {sf(ur[2]['xi'][0])} Punkte "
        f"(Standardfehler {f(ur[0]['xi'][1])} / {f(ur[1]['xi'][1])} / {f(ur[2]['xi'][1])}). Recall@5 überlesen: {f(ur[0]['i'])} / {f(ur[1]['i'])} / {f(ur[2]['i'])}; exakt: {f(ur[0]['x'])} / {f(ur[1]['x'])} / {f(ur[2]['x'])}. "
        f"Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff stehen: {f(ur[0]['coarse'])} / {f(ur[1]['coarse'])} / {f(ur[2]['coarse'])} Prozent. | `test_befund_2_*` |")
    add("| 3 | **Holt „Oberbegriff = alle Kinder“ den Verlust zurück?** (Kernfrage 2; gepaart alle Kinder − überlesen) | "
        f"u = 0,3 / 0,6 / 0,9: {sf(ur[0]['ai'][0])} / {sf(ur[1]['ai'][0])} / {sf(ur[2]['ai'][0])} Punkte (Standardfehler {f(ur[0]['ai'][1])} / {f(ur[1]['ai'][1])} / {f(ur[2]['ai'][1])}), "
        f"das sind {f(ur[0]['back_share'], 0)} / {f(ur[1]['back_share'], 0)} / {f(ur[2]['back_share'], 0)} Prozent des Verlusts; der Rest bleibt verloren (exakt − alle Kinder {sf(pair(cell(u=0.3), 'X-A')[0])} / "
        f"{sf(pair(cell(u=0.6), 'X-A')[0])} / {sf(pair(cell(u=0.9), 'X-A')[0])}). In {cnt['A-I'][0]} von 18 Zellen mit u > 0 liegt die Heuristik mehr als zwei Standardfehler vor dem Überlesen, in {cnt['A-I'][1]} dahinter, "
        f"in {cnt['A-I'][2]} ohne auflösbaren Unterschied. Die Ausnahme: "
        + "; ".join(f"{nd} Krankheiten, q = {f(q)}, u = {f(u)}: {sf(m, 2)} ± {f(se, 2)}" for nd, q, u, m, se in z["odd"]) + ". | `test_befund_3_*` |")
    add("| 4 | Wie hängt der Verlust an der Zahl der Krankheiten und am Dokumentationsverlust? | Verlust (exakt − überlesen, Recall@5) bei u = 0,3 / 0,6 / 0,9: "
        + "; ".join(f"{nd} Krankheiten, q = {f(q)}: {' / '.join(sf(v[0]) for v in z['grid'][(nd, q)][1:])}" for nd in C.ND_OPTIONS for q in C.Q_OPTIONS)
        + ". Der Verlust ist in allen sechs (Krankheiten, q)-Paaren bei u = 0,3 kleiner als bei u = 0,6 und dort kleiner als bei u = 0,9. | `test_befund_4_*` |")
    add("| 5 | **Hilft ein Resnik-artiger Ähnlichkeitswert?** (Kernfrage 3; eigene Fassung) | "
        f"Voreinstellung: Resnik-artig − überlesen {sf(rows_u[0.3]['ri'][0])} / {sf(rows_u[0.6]['ri'][0])} / {sf(rows_u[0.9]['ri'][0])} bei u = 0,3 / 0,6 / 0,9 "
        f"(Standardfehler {f(rows_u[0.3]['ri'][1])} / {f(rows_u[0.6]['ri'][1])} / {f(rows_u[0.9]['ri'][1])}); Resnik-artig − alle Kinder {sf(rows_u[0.3]['ra'][0])} / {sf(rows_u[0.6]['ra'][0])} / {sf(rows_u[0.9]['ra'][0])}; "
        f"exakt − Resnik-artig {sf(bp['X-R'][0])} bei u = 0,3. Über alle 18 Zellen mit u > 0: gegenüber dem Überlesen vorn in {cnt['R-I'][0]}, hinten in {cnt['R-I'][1]}, ohne auflösbaren Unterschied in {cnt['R-I'][2]} Zellen "
        f"(bei u = 0,3 / 0,6 / 0,9 vorn in {z['by_u'][0.3]['R-I'][0]} / {z['by_u'][0.6]['R-I'][0]} / {z['by_u'][0.9]['R-I'][0]} von je sechs); gegenüber alle Kinder hinten in {cnt['R-A'][1]} von 18. "
        f"Bei u = 0: Recall@5 exakt {f(rows_u[0.0]['x'])}, Resnik-artig {f(rows_u[0.0]['rr'])}; über die sechs Zellen mit u = 0 liegt der Abstand zwischen {f(min(v[0] for v in z['xr0'].values()))} und "
        f"{f(max(v[0] for v in z['xr0'].values()))} Punkten. | `test_befund_5_*` |")
    qm = z["q_mono"]
    n_loss = sum(v[3] for v in qm["X-I"])
    n_back = sum(v[3] for v in qm["A-I"])
    ausnahmen = "; ".join(f"{nd} Krankheiten, u = {f(u)}: {' / '.join(f(x) for x in vals)} bei q = 0 / 0,3 / 0,6" for nd, u, vals, ok in qm["X-I"] if not ok)
    mono_text = (f"Zählung über alle sechs (Krankheiten, u)-Paare mit u > 0: der Verlust nimmt mit steigendem q in {n_loss} Paaren in beiden Schritten ab"
                 + (f" (Ausnahme: {ausnahmen})" if ausnahmen else "") + f", die Rückholung der Heuristik in {n_back} Paaren.")
    add("| 6 | Viele Lücken (q = 0,6): ändert sich das Bild? | "
        f"200 Krankheiten, q = 0,6, u = 0,3: exakt {f(r(cell(200, 0.6, 0.3), 'X'))}, überlesen {f(r(cell(200, 0.6, 0.3), 'I'))}, alle Kinder {f(r(cell(200, 0.6, 0.3), 'A'))}; "
        f"Verlust {sf(pair(cell(200, 0.6, 0.3), 'X-I')[0])} ± {f(pair(cell(200, 0.6, 0.3), 'X-I')[1])}; alle Kinder − überlesen {sf(pair(cell(200, 0.6, 0.3), 'A-I')[0], 2)} ± {f(pair(cell(200, 0.6, 0.3), 'A-I')[1], 2)} "
        "(hier holt die Heuristik nichts zurück: gepoolt liegt sie knapp mehr als zwei Standardfehler hinter dem Überlesen, in jeder der beiden Mengen einzeln innerhalb von zwei Standardfehlern; "
        "gedeutet wird das nicht). " + mono_text + " | `test_befund_6_*` |")
    add("| 7 | Ohne grobe Angaben (u = 0): gleich? | In allen sechs Zellen mit u = 0 ist der gepaarte Unterschied zwischen exakt, überlesen und alle Kinder in beiden Mengen genau 0,0 Punkte; Anteil grober Angaben 0 (das folgt aus der Rechnung: "
        "ohne grobe Angaben sind die Formeln gleich; gemessen ist, dass die Umsetzung es auch ist, seit Gleichstände nach dem Runden gezählt werden, Korrektur 2). | `test_befund_7_*` |")
    add(f"| 8 | Kann ein anderer Ranker den exakten übertreffen? | In {zahl(n_over)} Vergleichen (24 Zellen × 2 Mengen × 4 andere Ranker × 4 Kennzahlen) liegt kein Ranker über dem exakten; größter Vorsprung {f(max(over, 0.0))} Punkte "
        "(das ist die Optimalität des Posteriors im Erwartungswert, in einer Stichprobe nicht selbstverständlich). | `test_befund_8_*` |")
    add(f"| 9 | Stabilität | Die beiden Mengen von Welten unterscheiden sich im Recall@5 um höchstens {f(stab, 2)} Punkte (geprüft über {meta['cells']} Zellen × {len(C.RANKERS)} Ranker); die Streuung über die Welten beträgt höchstens {f(sd_max, 1)} Punkte "
        "(über alle Zellen, Ranker und beide Mengen). | `test_befund_9_*` |")
    add(f"| 10 | Wie viel schlechter ist das naive Überlesen (ohne Wissen um u)? | Voreinstellung: naives Überlesen Recall@5 {f(b['N'][1])}; „überlesen“ minus naiv {sf(nv['base'][0])} ± {f(nv['base'][1])}; exakt minus naiv {sf(nv['xn'][0])} ± {f(nv['xn'][1])}. "
        f"Über alle {nv['n']} Zellen mit u > 0 liegt „überlesen“ in {nv['vor']} mehr als zwei Standardfehler vor dem naiven; die Differenzen der Zellen liegen zwischen {sf(nv['min'], 2)} und {sf(nv['max'], 2)} Punkten. Bei u = 0 sind beide gleich. | `test_befund_10_*` |")
    add("")
    add("Die Zahlen sind in der Datei auf Rundungsstellen genau gespeichert und im Test auf die hier gezeigte Stelle geprüft; `test_neu_rechnen_*` rechnet die Voreinstellung (Menge A) neu und vergleicht im Band von 0,6 Punkten "
        "(Linux rechnet an der letzten Binärstelle anders); der Zufallsstrom selbst ist bitgleich (`tests/test_stream.py`). Was **nicht** im Test steht: die Zahlen der Spalte „früher“ (Messung vor dem Bau, nicht Teil dieses Repos), "
        "die Rechenzeiten und die Zahl der Tests selbst.")
    add("")
    add("## Ehrliche Grenzen")
    add("")
    add("- **Das Orakel gehört zum Erzeuger.** Dass „exakt“ nirgends hinter einem anderen Ranker liegt, ist die Optimalität des Posteriors im Erwartungswert (nur für Patienten aus dem angenommenen Erzeuger), kein Beleg für echte Akten. "
        "In echten Daten kennt man $u$ und die Häufigkeiten nicht.")
    add("- **Eine Ebene Oberbegriffe, fünf Kinder je Oberbegriff, ein einziges $u$.** Echte Merkmalshierarchien sind tief, verzweigt und ungleich; ob grob oder genau dokumentiert wird, hängt von Person, Merkmal und Kontext ab. "
        "Drei Ebenen und ungleiche Anteile sind nicht gerechnet.")
    add("- **Der Resnik-artige Ranker ist eine eigene einfache Fassung.** Aussagen über ihn gelten für diese Fassung (Befund 5); wie ein ausgebautes Verfahren mit Profilhäufigkeiten abschneidet, zeigt die Demo nicht.")
    add("- **Der Ranker „überlesen“ kennt $u$** (starker Gegner); der naive Ranker ohne dieses Wissen ist mitgemessen (Korrektur 3, Befund 10).")
    add("- **Merkmale unabhängig gegeben die Krankheit**, nur an oder aus; echte Merkmale sind abhängig, es gibt Schweregrad, Alter und Verlauf.")
    add("- **Größe:** 50 und 200 Krankheiten, 100 Merkmale, nicht Tausende; zwei Mengen von je fünf Welten. Die Streuung über die Welten und gepaarte Standardfehler stehen bei den Zahlen; Unterschiede unter zwei Standardfehlern werden nicht gedeutet.")
    add("- **Gerankt werden nur Patienten mit mindestens einer Angabe in der Akte;** die Recall-Werte gelten für sie (Abschnitt Modell).")
    add("- **Stand der Technik** (Ähnlichkeitsmaße auf Ontologien wie Phenomizer, Likelihood-Ratio-Ranker, Sprachmodelle) wurde hier nicht nachgebaut und nicht verglichen.")
    add("")
    add("## Tests")
    add("")
    add(f"{TESTS_TOTAL} Tests (`pytest tests/`, mit `-n 8` je nach Rechnerlast 25 bis 90 Sekunden, zweimal gemessen): Zufallsstrom gegen die Skalarklasse (bitgleich) und Kopien der Hilfsmodule aus `phaenotyp-ranking-demo`; Welt und Kennzahlen (Randfälle, Gleichstände); "
        "**Orakel** (Aufzählung aller $3^3$ Kind-Zustände einer Mini-Gruppe gegen die Formel des exakten Rankers für jede Beobachtung; Randverteilung für „überlesen“; Handrechnung für „alle Kinder“; Zufallsziehungen mit eigener Konstruktion "
        "gegen die Aufzählung; schleifenweise Rechnung in reinem Python gegen die vektorisierte Rechnung über 20 Gruppen; Resnik-Tabellen von Hand); Hierarchie (Form, u = 0 und u = 1, Anteil grober Angaben); Live-Messung "
        "(Wiederholbarkeit, **Zweig-Test je Regler**: jede Stufe ändert mindestens einen Ranker und u jeden der fünf, exakt = überlesen = alle Kinder = naiv bei u = 0, Worst-Case ohne Zeit-Assertion); Verdikte (Schwellen an handgesetzten Werten, "
        "jede Zahl im Text); Abnahmekriterien der Presets an der Messreihe, jedes an künstlichen Zellen einzeln kippend; Messreihe (Form, Plausibilität, Stabilität); Zahlen dieser README (mit Neurechnung der Voreinstellung als Band) (`test_claims.py`, README gegen `tools/make_readme.py`); "
        "AppTest (Standard, jedes Preset, jede Stufe jedes Reglers am Minimum und Maximum, Kombination aller Extremwerte, PDF erst nach Klick); Layout, Footer, Laufzeit-Importe. Keine Mutationstests.")
    add("")
    add("## Dateistruktur")
    add("")
    add("| Datei | Zweck |")
    add("|---|---|")
    for name, text in (
        ("`.github/workflows/tests.yml`", "CI: Push, Pull Request, wöchentlicher Canary"),
        ("`.gitignore`, `.streamlit/config.toml`", "Git- und Streamlit-Einstellungen (keine Nutzungsstatistik)"),
        ("`app.py`", "Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2)"),
        ("`gmk_constants.py`", "Annahmen der Welt, Hierarchie, Regler-Stufen, Ranker, Farben, Presets"),
        ("`gmk_rng.py`", "SplitMix64 (unveränderte Kopie der Vorlage aus dem Website-Repo)"),
        ("`gmk_stream.py`, `gmk_world.py`, `gmk_metrics.py`", "Kopien aus `phaenotyp-ranking-demo` mit umbenanntem Präfix: vektorisierter Strom, Erzeuger der Welt, Kennzahlen (Kopier-Test in `tests/test_copies.py`)"),
        ("`gmk_hierarchy.py`", "Gruppen der Merkmale, grobe Dokumentation der Patienten, Annotationen"),
        ("`gmk_rankers.py`", "exakter Ranker, Überlesen, alle Kinder, Resnik-artig"),
        ("`gmk_live.py`", "Live-Messung, Auswertung über die Welten, gepaarte Differenz"),
        ("`gmk_evaluation.py`", "Verdikte (reine Funktionen), Beispielpatient, Zahlenformat"),
        ("`gmk_results.py`, `data/gmk_results.json`", "Zugriff auf die vorgerechnete Messreihe (24 Zellen × 2 Mengen)"),
        ("`gmk_stories.py`", "Abnahmekriterien der Presets"),
        ("`gmk_presets.py`, `gmk_layout.py`", "Stufen-Regler, Permalink (rastet auf Stufen ein), Schnellstart, Hervorhebung, fester Kopf"),
        ("`gmk_visualization.py`, `gmk_ui_panel.py`, `gmk_pdf_export.py`", "Abbildungen (Achsen gesperrt), Darstellung, PDF (fpdf2, erst nach Klick)"),
        ("`tools/sweep.py`", "reproduziert `data/gmk_results.json` (rund eineinhalb Minuten, nicht in der CI)"),
        ("`tools/make_readme.py`", "erzeugt diese README aus der Messreihe"),
        ("`requirements.txt`, `requirements-dev.txt`", "Laufzeit beziehungsweise plus pytest"),
        ("`tests/`", "siehe „Tests“"),
    ):
        add(f"| {name} | {text} |")
    add("")
    add("## Bewusst nicht umgesetzt")
    add("")
    add("Hierarchie mit drei Ebenen, ungleiche Anteile grober Angaben je Merkmal, Abhängigkeiten zwischen Merkmalen, Schätzung von $u$ aus Daten, ein Ranker mit Profilhäufigkeiten nach dem Vorbild von Phenomizer, "
        "echte Wissensbasen (HPO, Orphanet), Sprachmodell-Ranker, mehr als 200 Krankheiten.")
    add("")
    add("## Verwandte Demos mit demselben mathematischen Modell")
    add("")
    add("- [`phaenotyp-ranking-demo`](https://github.com/sebastian-hanisch/phaenotyp-ranking-demo): dieselbe Welt (Profile, Hintergrund, Dokumentationsverlust) und das Orakel als Maßstab; dort sind die Merkmale flach, hier gibt es Oberbegriffe.")
    add("- [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo): Kalibrierung und Schwelle aus dem Kostenverhältnis; hier wird eine Rangliste gemessen, dort die Entscheidung aus einer Wahrscheinlichkeit.")
    add("- [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo): Bayes-Update und Wert eines Tests bei zwei Zuständen; hier sind es viele Krankheiten und ein Ranking.")
    add("- `konforme-ursachen-demo`: Vorhersagemengen mit Abdeckungsgarantie für Ursachen; anderes Modell, die Demos sind nicht verbunden.")
    add("")
    add("Zahlen aus diesen Demos werden hier weder nachgerechnet noch verglichen.")
    add("")
    add("## Lokal ausführen")
    add("")
    add("```bash")
    add("python3 -m venv venv")
    add("source venv/bin/activate          # Windows: venv\\Scripts\\activate")
    add("")
    add("pip install -r requirements.txt")
    add("streamlit run app.py")
    add("```")
    add("")
    add("Die Python-Version der Streamlit Cloud ist in der App-Einstellung zu wählen (die CI läuft mit Python 3.12); die Demo nutzt nur Sprachmerkmale ab Python 3.10. <!-- lint-ok: dezimalpunkt -->")
    add("")
    add("## Tests ausführen")
    add("")
    add("```bash")
    add("pip install -r requirements-dev.txt")
    add("pytest tests/ -v")
    add("```")
    add("")
    add("Gebaut mit Streamlit, Plotly, NumPy und fpdf2.")
    add("")
    add("---")
    add("")
    add("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
        "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).")
    return "\n".join(L) + "\n"


def main(check: bool = False):
    text = build()
    path = ROOT / "README.md"
    if check:
        current = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        print("README gleich dem Erzeugnis" if current == text else "ABWEICHUNG: README.md entspricht nicht tools/make_readme.py")
        sys.exit(0 if current == text else 1)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"geschrieben: {path} ({len(text)} Zeichen)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main("--check" in sys.argv)
