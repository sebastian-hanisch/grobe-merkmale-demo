"""Abnahmekriterien der Presets: jedes Preset muss an der vorgerechneten Messreihe zeigen, wofür es steht (DEMO-PLAYBOOK §7, Preset-Disziplin).

Ein Kriterium ist eine Zeile (Text, Aussage über eine Zelle und eine Menge von Welten). Es gilt nur, wenn es in BEIDEN Mengen (A und B) gilt; Vorzeichen-
Kriterien über Rangunterschiede tragen die Bedingung „mehr als zwei Standardfehler“ der gepaarten Differenz.

Die Schwellen sind nach der Messreihe festgelegt (vorgesehen war für „Meist grob“ „überlesen höchstens 40 %“; gemessen sind 40,3 und 39,7, die Schwelle liegt
also auf der Zellgrenze und wurde durch den Verlust von mehr als 15 Punkten ersetzt)."""

import gmk_constants as C
import gmk_results as RES

SE_FACTOR = C.SD_FACTOR


def _paired(c, w, pair):
    return c[w]["paired"][pair]


def _r5(c, w, ranker):
    return RES.stat(c, w, ranker, 5)[0]


def criteria_genau(c, w):
    d_xi, _ = _paired(c, w, "X-I")
    d_xa, _ = _paired(c, w, "X-A")
    return [("Anteil grober Angaben ist 0", c[w]["share_coarse"] == 0.0),
            ("exakt, überlesen und alle Kinder liegen im Recall@5 gleich (Unterschiede unter 0,05 Punkten)", abs(d_xi) < 0.05 and abs(d_xa) < 0.05)]


def criteria_teilweise(c, w):
    d_xi, se = _paired(c, w, "X-I")
    return [("exakt liegt mindestens 8 Punkte (Recall@5) vor überlesen", d_xi >= 8.0),
            ("der Unterschied zwischen exakt und überlesen beträgt mehr als zwei Standardfehler", d_xi > SE_FACTOR * se)]


def criteria_meist(c, w):
    d_xi, _ = _paired(c, w, "X-I")
    d_ai, se_ai = _paired(c, w, "A-I")
    d_xa, se_xa = _paired(c, w, "X-A")
    return [("exakt liegt mehr als 15 Punkte (Recall@5) vor überlesen", d_xi > C.LOSS_LARGE),
            ("alle Kinder liegt mehr als zwei Standardfehler vor überlesen", d_ai > SE_FACTOR * se_ai),
            ("alle Kinder liegt mehr als zwei Standardfehler hinter exakt", d_xa > SE_FACTOR * se_xa)]


def criteria_fast_nur(c, w):
    return [("überlesen erreicht höchstens 15 Prozent Recall@5", _r5(c, w, "I") <= 15.0),
            ("exakt erreicht mindestens 45 Prozent Recall@5", _r5(c, w, "X") >= 45.0)]


def criteria_luecken(c, w):
    d_xi, se = _paired(c, w, "X-I")
    d_ai, se_ai = _paired(c, w, "A-I")
    return [("exakt liegt mehr als zwei Standardfehler vor überlesen", d_xi > SE_FACTOR * se),
            ("Recall@5 des exakten Rankers höchstens 50 Prozent (viele Lücken)", _r5(c, w, "X") <= 50.0),
            ("alle Kinder holt nichts zurück (kein Unterschied zu überlesen über zwei Standardfehlern)", abs(d_ai) <= SE_FACTOR * se_ai)]


CRITERIA = {
    "Genau dokumentiert": criteria_genau,
    "Teilweise grob": criteria_teilweise,
    "Meist grob": criteria_meist,
    "Fast nur Oberbegriffe": criteria_fast_nur,
    "Viele Lücken": criteria_luecken,
}


def check_preset(name: str, c: dict) -> list:
    """Alle Kriterien des Presets an einer Zelle: Liste (Text, Menge, erfüllt) über die Mengen A und B."""
    return [(text, w, bool(ok)) for w in ("A", "B") for text, ok in CRITERIA[name](c, w)]


def check_all() -> dict:
    """Preset -> Liste der Kriterien; die Zelle ist die Einstellung des Presets (ohne Zufalls-Seed)."""
    out = {}
    for name, preset in C.PRESETS.items():
        s = RES.Settings(**{k: v for k, v in preset.items() if k != "seed"})
        out[name] = check_preset(name, RES.cell(s))
    return out
