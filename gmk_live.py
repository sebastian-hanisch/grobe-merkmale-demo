"""Live-Messung: fünf Welten × Patienten für eine Einstellung, Auswertung über die Welten, Beispielpatienten.

`measure` ist deterministisch (Zufalls-Seed der Einstellung, SplitMix64) und hängt nicht von st.session_state ab."""

from dataclasses import dataclass, field

import numpy as np

import gmk_constants as C
import gmk_hierarchy as H
import gmk_metrics as M
import gmk_rankers as R
import gmk_world as W
from gmk_stream import derive


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Einstellung einer Messung. Nur mit Schlüsselwörtern erzeugen (ein eingefügtes Feld darf keine Werte still verschieben)."""

    nd: int = C.DEFAULT["nd"]
    q: float = C.DEFAULT["q"]
    u: float = C.DEFAULT["u"]
    seed: int = C.DEFAULT["seed"]

    def key(self) -> str:
        """Schlüssel der Zelle in data/gmk_results.json (ohne den Zufalls-Seed)."""
        return f"nd{self.nd}_q{self.q}_u{self.u}"


@dataclass(frozen=True, eq=False)
class WorldRun:
    metrics: dict           # Ranker -> {1, 5, 10, "mrr"} -> Feld über die Patienten
    n_ranked: int           # Zahl der gerankten Patienten (mit mindestens einer Angabe in der Akte)
    share_coarse: float     # Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff stehen
    example: dict = field(default_factory=dict)   # erste Patienten: wahre Krankheit, Akte, Punktzahlen je Ranker (nur Welt 0)


@dataclass(frozen=True, eq=False)
class Measurement:
    settings: Settings
    n_patients: int
    runs: tuple


def hierarchy_inputs(world: W.World):
    """Größen, die nur von der Welt abhängen: Häufigkeiten in Gruppenform und die Ähnlichkeitstabellen des Resnik-artigen Rankers."""
    leaf_ann, parent_ann = H.annotation(world)
    sim_leaf, sim_par = R.resnik_tables(leaf_ann, parent_ann)
    return H.group_g(world), sim_leaf, sim_par


def run_world(s: Settings, i: int, n_patients: int, n_example: int = 0) -> WorldRun:
    wseed = derive(s.seed, 10 + i)
    world = W.make_world(wseed, s.nd, s.q, 0)               # Vorab gleichverteilt
    pat = W.sample_patients(world, wseed, n_patients)
    cp = H.coarsen(pat, s.u, wseed)
    sc = R.all_scores(cp.leaf, cp.flag, world, s.u, hierarchy_inputs(world))
    metrics = {r: M.metric_arrays(sc[r], cp.d) for r in C.RANKERS}
    example = {}
    if n_example:
        example = {"d": cp.d[:n_example], "leaf": cp.leaf[:n_example], "flag": cp.flag[:n_example],
                   "scores": {r: sc[r][:n_example] for r in C.RANKERS}}
    return WorldRun(metrics, len(cp.d), cp.share_coarse, example)


def measure(s: Settings, n_worlds: int = C.N_WORLDS, n_patients: int = C.N_PATIENTS, n_example: int = 30) -> Measurement:
    runs = tuple(run_world(s, i, n_patients, n_example if i == 0 else 0) for i in range(n_worlds))
    return Measurement(s, n_patients, runs)


def summarize(m: Measurement) -> dict:
    """Ranker -> Kennzahl (1, 5, 10, "mrr") -> (Mittel, Streuung über die Welten) in Prozent."""
    out = {}
    for r in C.RANKERS:
        out[r] = {k: M.mean_sd([100.0 * run.metrics[r][k].mean() for run in m.runs]) for k in (1, 5, 10, "mrr")}
    return out


def paired_diff(m: Measurement, r1: str, r2: str, k: int = 5) -> tuple:
    """Gepaarte Differenz r1 minus r2 in Prozentpunkten Recall@k: Mittel über die Welten und Standardfehler (Streuung der Differenzen / Wurzel der Weltenzahl)."""
    d = [100.0 * (run.metrics[r1][k].mean() - run.metrics[r2][k].mean()) for run in m.runs]
    mean, sd = M.mean_sd(d)
    return mean, sd / float(len(d)) ** 0.5


def share_coarse(m: Measurement) -> float:
    """Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff in der Akte stehen, in % (Mittel über die Welten)."""
    return float(np.mean([run.share_coarse for run in m.runs]) * 100.0)


def share_ranked(m: Measurement) -> float:
    """Anteil der Patienten mit mindestens einer Angabe in der Akte (nur sie werden gerankt), in % der gezogenen Patienten."""
    return float(np.mean([run.n_ranked for run in m.runs]) / m.n_patients * 100.0)
