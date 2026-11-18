"""Hierarchie der Merkmale und grobe Dokumentation: Patienten, deren Merkmale manchmal nur als Oberbegriff in der Akte stehen.

Die Merkmale 5g bis 5g + 4 bilden die Gruppe g (Oberbegriff g). Ein aufgeschriebenes Merkmal steht mit Wahrscheinlichkeit 1 − u als genauer Begriff
in der Akte und mit Wahrscheinlichkeit u nur als Oberbegriff seiner Gruppe; mehrere grobe Angaben derselben Gruppe ergeben einen Oberbegriff.
Der Zufall kommt nur über den Strom aus gmk_stream (SplitMix64)."""

from dataclasses import dataclass

import numpy as np

import gmk_constants as C
from gmk_stream import Stream, derive
from gmk_world import Patients, World


@dataclass(frozen=True, eq=False)
class CoarsePatients:
    d: np.ndarray    # wahre Krankheit je Patient
    leaf: np.ndarray  # genaue Begriffe in der Akte, Form (Patienten, Gruppen, Kinder), 0/1
    flag: np.ndarray  # Oberbegriff in der Akte, Form (Patienten, Gruppen), Wahrheitswert
    x: np.ndarray    # alle aufgeschriebenen Merkmale (genau oder grob), Form (Patienten, Merkmale), 0/1
    share_coarse: float  # Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff in der Akte stehen


def coarsen(pat: Patients, u: float, seed: int, gsz: int = C.GROUP_SIZE) -> CoarsePatients:
    """Jedes aufgeschriebene Merkmal wird mit Wahrscheinlichkeit u zur groben Angabe. Zufälle aus dem eigenen Strom (Salz 4), unabhängig von der Welt."""
    n, ns = pat.x.shape
    _check_groups(ns, gsz)
    ng = ns // gsz
    st = Stream(derive(seed, 4))
    gen = (st.uniform((n, ns)) < u) & (pat.x > 0)
    leaf = ((pat.x > 0) & ~gen).reshape(n, ng, gsz).astype(float)
    flag = gen.reshape(n, ng, gsz).any(axis=2)
    written = float((pat.x > 0).sum())
    return CoarsePatients(pat.d, leaf, flag, pat.x, float(gen.sum()) / written if written else 0.0)


def _check_groups(ns: int, gsz: int) -> None:
    if ns % gsz:
        raise ValueError("Die Zahl der Merkmale muss ein Vielfaches der Gruppengröße sein.")


def group_g(world: World, gsz: int = C.GROUP_SIZE) -> np.ndarray:
    """Dokumentationswahrscheinlichkeiten g[d, Gruppe, Kind], auf (PROB_FLOOR, 1 − PROB_FLOOR) begrenzt (Logarithmen)."""
    g = np.clip(world.g, C.PROB_FLOOR, 1.0 - C.PROB_FLOOR)
    return g.reshape(world.nd, world.ns // gsz, gsz)


def annotation(world: World, gsz: int = C.GROUP_SIZE) -> tuple:
    """Annotationen des Profils: Merkmal t gehört zur Krankheit d, wenn die Profilhäufigkeit über C.SIMPLE_THRESHOLD liegt; der Oberbegriff, wenn eines seiner Kinder gehört.
    Rückgabe: (leaf[d, t] als 0/1, parent[d, Gruppe] als 0/1)."""
    leaf = (world.f > C.SIMPLE_THRESHOLD).astype(float)
    parent = leaf.reshape(world.nd, world.ns // gsz, gsz).max(axis=2)
    return leaf, parent
