"""Kennzahlen und Verdikte (reine Funktionen: Art, Text), dazu der Beispielpatient. Jede Zahl im Text kommt aus der Messung."""

import numpy as np

import gmk_constants as C
from gmk_live import Measurement, Settings, paired_diff, share_coarse, share_ranked, summarize  # noqa: F401  (Settings für die App)
from gmk_rankers import all_scores  # noqa: F401


def pct(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def spct(x: float) -> str:
    """Mit Vorzeichen (für Deltas: Streamlit färbt nach dem ersten Zeichen)."""
    return f"{x:+.1f}".replace(".", ",")


def num(x: float) -> str:
    """Kurze Zahl (Reglerstufe wie 0,3) mit Dezimalkomma; nur die Zahl wird umformatiert, nie ein ganzer Satz."""
    return f"{x:g}".replace(".", ",")


def dec(x: float, nd: int = 3) -> str:
    """Zahl mit Dezimalkomma; nur die Zahl wird umformatiert, nie ein ganzer Satz."""
    return f"{x:.{nd}f}".replace(".", ",")


def verdict_verlust(u: float, x5: float, i5: float, loss: float, se: float):
    """Verdikt in drei Zuständen. Verlust = Recall@5 des exakten Rankers minus Recall@5 des Rankers, der Oberbegriffe überliest (gepaart über die Welten);
    Schwellen: unter C.LOSS_SMALL Punkte „success“, bis C.LOSS_LARGE „info“, darüber „warning“."""
    teil = f"Recall@5 exakt {pct(x5)} %, Oberbegriff überlesen {pct(i5)} %: Unterschied {pct(loss)} Punkte (± {pct(se)} Standardfehler)."
    if u == 0.0:
        return "success", f"Ohne grobe Angaben (u = 0) gibt es nichts zu überlesen: {teil}"
    if loss < C.LOSS_SMALL:
        return "success", f"Die groben Angaben zu überlesen kostet hier weniger als {pct(C.LOSS_SMALL)} Punkte Recall@5. {teil}"
    if loss <= C.LOSS_LARGE:
        return "info", f"Die groben Angaben zu überlesen kostet hier {pct(C.LOSS_SMALL)} bis {pct(C.LOSS_LARGE)} Punkte Recall@5. {teil}"
    return "warning", f"Die groben Angaben zu überlesen kostet hier mehr als {pct(C.LOSS_LARGE)} Punkte Recall@5. {teil}"


def verdict_heuristik(loss: float, back: float, se: float):
    """Verdikt zur Heuristik „Oberbegriff = alle Kinder“: gepaarte Differenz zum Überlesen (Recall@5, Punkte) und ihr Anteil am Verlust."""
    if loss < 1.0:
        return "info", f"Es gibt kaum etwas zurückzuholen: der Unterschied zwischen exaktem Ranker und Überlesen liegt unter einem Punkt ({pct(loss)})."
    if back < -C.SD_FACTOR * se:
        return "warning", (f"Die Heuristik „Oberbegriff = alle Kinder“ liegt {pct(-back)} Punkte Recall@5 hinter dem Überlesen (mehr als zwei Standardfehler; "
                           f"Standardfehler {pct(se)}).")
    if back > C.SD_FACTOR * se:
        return "info", (f"Die Heuristik „Oberbegriff = alle Kinder“ holt {pct(back)} Punkte Recall@5 gegenüber dem Überlesen zurück (Standardfehler {pct(se)}): "
                        f"{100.0 * back / loss:.0f} % des Unterschieds zwischen exaktem Ranker und Überlesen ({pct(loss)} Punkte).")
    return "info", (f"Zwischen der Heuristik „Oberbegriff = alle Kinder“ und dem Überlesen liegt kein Unterschied über zwei Standardfehlern "
                    f"({spct(back)} Punkte, Standardfehler {pct(se)}).")


def verdict_resnik(x5: float, r5: float, r_minus_i: tuple):
    """Verdikt zum Resnik-artigen Ranker (eigene Fassung): Abstand zu exakt und gepaarte Differenz zum Überlesen."""
    mean, se = r_minus_i
    if mean > C.SD_FACTOR * se:
        lage = f"{pct(mean)} Punkte vor dem Überlesen"
    elif mean < -C.SD_FACTOR * se:
        lage = f"{pct(-mean)} Punkte hinter dem Überlesen"
    else:
        lage = f"ohne Unterschied über zwei Standardfehlern zum Überlesen ({spct(mean)} Punkte)"
    return "info", (f"Der Resnik-artige Ranker (eigene Fassung, kein Nachbau eines Verfahrens aus der Literatur) erreicht Recall@5 {pct(r5)} %: "
                    f"{pct(x5 - r5)} Punkte unter dem exakten Ranker und {lage} (Standardfehler {pct(se)}).")


def kennzahlen(m: Measurement) -> dict:
    """Alle Kennzahlen der Hauptansicht aus einer Messung."""
    s = summarize(m)
    return {"summary": s, "loss": paired_diff(m, "X", "I"), "back": paired_diff(m, "A", "I"), "resnik_i": paired_diff(m, "R", "I"),
            "share_coarse": share_coarse(m), "share_ranked": share_ranked(m)}


def verdicts(settings: Settings, k: dict):
    s = k["summary"]
    v1 = verdict_verlust(settings.u, s["X"][5][0], s["I"][5][0], *k["loss"])
    v2 = verdict_heuristik(k["loss"][0], *k["back"])
    v3 = verdict_resnik(s["X"][5][0], s["R"][5][0], k["resnik_i"])
    return v1, v2, v3


def term_labels(leaf_row: np.ndarray, flag_row: np.ndarray, gsz: int = C.GROUP_SIZE) -> tuple:
    """Begriffe in der Akte eines Patienten: genaue Merkmale (M1 ...) und Oberbegriffe (Gruppe 1 ... mit dem Bereich der Kinder)."""
    exact = [f"M{g * gsz + c + 1}" for g in range(leaf_row.shape[0]) for c in range(gsz) if leaf_row[g, c] > 0]
    coarse = [f"Gruppe {g + 1} (irgendeines von M{g * gsz + 1} bis M{g * gsz + gsz})" for g in range(flag_row.shape[0]) if flag_row[g]]
    return exact, coarse


def example_view(m: Measurement, idx: int, top: int = 5) -> dict:
    """Beispielpatient idx (0-basiert, aus Welt 0): wahre Krankheit, Akte, die obersten `top` Krankheiten je Ranker."""
    ex = m.runs[0].example
    n = len(ex["d"])
    idx = int(min(max(idx, 0), n - 1))
    exact, coarse = term_labels(ex["leaf"][idx], ex["flag"][idx])
    out = {"true": int(ex["d"][idx]), "exact": exact, "coarse": coarse, "top": {}}
    for r in C.RANKERS:
        sc = ex["scores"][r][idx]
        order = np.argsort(-sc, kind="stable")[:top]
        out["top"][r] = {"ids": [int(i) for i in order], "scores": [float(sc[i]) for i in order], "rank_true": int(1 + (sc > sc[out["true"]]).sum())}
    z = ex["scores"]["X"][idx] - ex["scores"]["X"][idx].max()
    p = np.exp(z) / np.exp(z).sum()
    out["posterior_x"] = {int(i): float(p[i]) for i in np.argsort(-p, kind="stable")[:top]}
    return out
