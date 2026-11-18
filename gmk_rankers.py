"""Ranker für grob dokumentierte Merkmale. Alle liefern eine Punktzahl je Patient und Krankheit (Zeilen = Patienten); größer = wahrscheinlicher.

Beobachtung je Gruppe (Oberbegriff) mit Kindern c: die Menge L der genau genannten Kinder und das Zeichen F, dass der Oberbegriff in der Akte steht.
Erzeuger: Ein Kind ist mit Wahrscheinlichkeit g_c aufgeschrieben; aufgeschrieben steht es mit 1 − u genau, mit u nur als Oberbegriff.

  exakt       Wahrscheinlichkeit der Beobachtung unter dem Erzeuger (Orakel des Erzeugers; u und alle Häufigkeiten bekannt)
  ignoriert   dasselbe ohne das Zeichen F: die Oberbegriffe in der Akte werden überlesen
  alle Kinder ein Oberbegriff zählt, als stünden alle Kinder der Gruppe in der Akte; Naive Bayes mit den ungeteilten Häufigkeiten
  Resnik-artig Summe über die Begriffe der Akte von Informationsgehalt des besten gemeinsamen Vorfahren mit den Annotationen der Krankheit (eigene Fassung)
  naiv        Oberbegriffe überlesen UND mit den ungeteilten Häufigkeiten g rechnen (kennt u nicht): ein Merkmal steht mit Wahrscheinlichkeit g genau in der Akte oder nicht"""

import numpy as np

import gmk_constants as C


def _prior_term(prior: np.ndarray) -> np.ndarray:
    return np.log(prior)[None, :]


def _mm(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """a (Patienten, Kinder) mal w (Krankheiten, Kinder) -> (Patienten, Krankheiten). Je Gruppe nur fünf Spalten: `einsum` ohne BLAS ist hier schneller und von der
    Zahl der Rechenkerne unabhängig (die Matrixmultiplikation der Bibliothek verlor bei so kleinen Matrizen gegen den Aufwand der Thread-Verteilung)."""
    return np.einsum("nc,dc->nd", a, w)


def score_exakt(leaf: np.ndarray, flag: np.ndarray, g: np.ndarray, u: float, prior: np.ndarray) -> np.ndarray:
    """Exakte Log-Wahrscheinlichkeit der Beobachtung (genaue Begriffe L, Oberbegriffszeichen F je Gruppe) für jede Krankheit, plus Log-Vorab.

    Je Gruppe: Π_{c in L} g_c (1 − u) · [F = 0: Π_{c not in L} (1 − g_c);  F = 1: Π_{c not in L} (1 − g_c + g_c u) − Π_{c not in L} (1 − g_c)].
    `leaf`: (n, Gruppen, Kinder); `flag`: (n, Gruppen); `g`: (Krankheiten, Gruppen, Kinder)."""
    n, ng, gsz = leaf.shape
    nd = g.shape[0]
    out = np.zeros((n, nd)) + _prior_term(prior)
    log_leaf = np.log(g * (1.0 - u))
    log_abs = np.log(1.0 - g)
    log_gen = np.log(1.0 - g + g * u)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(ng):
            lk = leaf[:, k, :]
            nl = 1.0 - lk
            out += _mm(lk, log_leaf[:, k, :])
            a = _mm(nl, log_abs[:, k, :])        # F = 0: kein Kind (außerhalb von L) ist grob aufgeschrieben
            out += a
            rows = np.flatnonzero(flag[:, k])    # nur die Patienten mit Oberbegriff in dieser Gruppe brauchen den zweiten Fall
            if rows.size:
                b = _mm(nl[rows], log_gen[:, k, :])      # Kinder außerhalb von L sind abwesend oder grob
                f1 = b + np.log1p(-np.exp(np.minimum(a[rows] - b, -1e-12)))   # F = 1: mindestens ein Kind ist grob aufgeschrieben
                out[rows] += f1 - a[rows]
    return out


def score_ignoriert(leaf: np.ndarray, g: np.ndarray, u: float, prior: np.ndarray) -> np.ndarray:
    """Wie exakt, aber der Oberbegriff in der Akte wird überlesen: jedes Kind steht mit Wahrscheinlichkeit g (1 − u) als genauer Begriff in der Akte oder nicht."""
    n, ng, gsz = leaf.shape
    pl = np.clip(g * (1.0 - u), C.PROB_FLOOR, 1.0 - C.PROB_FLOOR)
    out = np.zeros((n, g.shape[0])) + _prior_term(prior)
    for k in range(ng):
        lk = leaf[:, k, :]
        out += _mm(lk, np.log(pl[:, k, :])) + _mm(1.0 - lk, np.log(1.0 - pl[:, k, :]))
    return out


def score_alle_kinder(leaf: np.ndarray, flag: np.ndarray, g: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Ein Oberbegriff zählt, als stünden alle seine Kinder in der Akte (Naive Bayes mit den ungeteilten Häufigkeiten g, ohne Wissen um u)."""
    n, ng, gsz = leaf.shape
    expanded = np.clip(leaf + flag[:, :, None].astype(float), 0.0, 1.0)
    out = np.zeros((n, g.shape[0])) + _prior_term(prior)
    for k in range(ng):
        ek = expanded[:, k, :]
        out += _mm(ek, np.log(g[:, k, :])) + _mm(1.0 - ek, np.log(1.0 - g[:, k, :]))
    return out


def score_naiv(leaf: np.ndarray, g: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Naives Überlesen: nur die genauen Begriffe, Naive Bayes mit den ungeteilten Häufigkeiten g (kennt u nicht; die Wahrscheinlichkeit, genau aufgeschrieben zu sein, ist g(1 − u))."""
    n, ng, gsz = leaf.shape
    out = np.zeros((n, g.shape[0])) + _prior_term(prior)
    for k in range(ng):
        lk = leaf[:, k, :]
        out += _mm(lk, np.log(g[:, k, :])) + _mm(1.0 - lk, np.log(1.0 - g[:, k, :]))
    return out


def information_content(leaf_ann: np.ndarray, parent_ann: np.ndarray) -> tuple:
    """Informationsgehalt IC = −ln(Anteil der Krankheiten, die den Begriff annotieren) für genaue Begriffe (Form: Merkmale) und Oberbegriffe (Form: Gruppen);
    ein Begriff, den keine Krankheit annotiert, bekommt den größten vorkommenden Wert (er kann keine Krankheit stützen). Die Wurzel hat IC 0."""
    nd = leaf_ann.shape[0]
    frac_leaf = np.maximum(leaf_ann.sum(axis=0), 1.0) / nd
    frac_par = np.maximum(parent_ann.sum(axis=0), 1.0) / nd
    return -np.log(frac_leaf), -np.log(frac_par)


def resnik_tables(leaf_ann: np.ndarray, parent_ann: np.ndarray) -> tuple:
    """Ähnlichkeit je Begriff und Krankheit (Resnik): Ein genauer Begriff t trifft eine Krankheit mit IC(t), wenn sie ihn annotiert; sonst mit IC(Oberbegriff von t), wenn sie den
    Oberbegriff annotiert (gemeinsamer Vorfahre), sonst mit 0 (Wurzel). Ein Oberbegriff trifft sie mit IC(Oberbegriff), wenn sie ihn annotiert, sonst mit 0.
    Rückgabe: (sim_leaf[d, t], sim_parent[d, Gruppe])."""
    nd, ns = leaf_ann.shape
    ng = parent_ann.shape[1]
    gsz = ns // ng
    ic_leaf, ic_par = information_content(leaf_ann, parent_ann)
    par_of_leaf = np.repeat(np.arange(ng), gsz)
    sim_par = parent_ann * ic_par[None, :]
    via_parent = parent_ann[:, par_of_leaf] * ic_par[par_of_leaf][None, :]
    sim_leaf = np.where(leaf_ann > 0, ic_leaf[None, :], via_parent)
    return sim_leaf, sim_par


def score_resnik(leaf: np.ndarray, flag: np.ndarray, sim_leaf: np.ndarray, sim_par: np.ndarray) -> np.ndarray:
    """Resnik-artig (eigene Fassung, nicht Phenomizer): Summe über die Begriffe der Akte von der Ähnlichkeit zur Krankheit; kein Vorab, keine Abwesenheit."""
    n = leaf.shape[0]
    return leaf.reshape(n, -1) @ sim_leaf.T + flag.astype(float) @ sim_par.T


def all_scores(leaf, flag, world, u: float, hier) -> dict:
    """Punktzahlen aller Ranker (Schlüssel wie C.RANKERS) für dieselben Patienten. `hier`: (g in Gruppenform, sim_leaf, sim_par).

    Die Punktzahlen werden auf TIE_DIGITS Stellen gerundet: Krankheiten, die in der Akte nichts unterscheidet, haben mathematisch gleiche Punktzahlen; die Summen
    in verschiedener Reihenfolge weichen in den letzten Bits ab und würden Gleichstände je Ranker anders auflösen (bei u = 0 wären exakt, überlesen und alle Kinder sonst nicht gleich)."""
    g, sim_leaf, sim_par = hier
    out = {
        "X": score_exakt(leaf, flag, g, u, world.prior),
        "I": score_ignoriert(leaf, g, u, world.prior),
        "A": score_alle_kinder(leaf, flag, g, world.prior),
        "R": score_resnik(leaf, flag, sim_leaf, sim_par),
        "N": score_naiv(leaf, g, world.prior),
    }
    return {r: np.round(v, C.TIE_DIGITS) for r, v in out.items()}
