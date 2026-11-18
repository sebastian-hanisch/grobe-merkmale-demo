"""Orakel-Tests (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

- Exakter Ranker (Gruppen-Wahrscheinlichkeit der Akte): Aufzählung aller 3^3 Zustände der Kinder einer Mini-Gruppe (nicht aufgeschrieben, genau, nur als Oberbegriff) gegen die Formel der Demo,
  für jede mögliche Beobachtung (L, F).
- Überlesen: die Randverteilung über F derselben Aufzählung.
- Alle Kinder: Naive Bayes von Hand an der Mini-Gruppe.
- Erzeuger: Zufallsziehungen mit einer eigenen Konstruktion (eine Gleichverteilte je Kind) gegen die Aufzählung.
- Vektorisierte Rechnung über 20 Gruppen: schleifenweise Rechnung in reinem Python gegen `score_exakt`.
- Resnik-artig: Handrechnung an einer Mini-Hierarchie."""

import math

import numpy as np
import pytest

import gmk_constants as C
import gmk_hierarchy as H
import gmk_rankers as R
import gmk_world as W
from conftest import MINI_G, MINI_PRIOR, MINI_U
from gmk_stream import Stream


def _akte(obs):
    """Eine Beobachtung (L, F) als Eingabe der Ranker: leaf (1, 1, 3), flag (1, 1)."""
    leaf, f = obs
    return np.array([[list(map(float, leaf))]]), np.array([[bool(f)]])


def test_aufzaehlung_summiert_zu_eins(beobachtungen):
    for d in range(2):
        assert sum(p[d] for p in beobachtungen.values()) == pytest.approx(1.0, rel=1e-12)


def test_exakt_gleich_aufzaehlung_fuer_jede_beobachtung(beobachtungen, mini_g, mini_prior, mini_u):
    for obs, p in beobachtungen.items():
        leaf, flag = _akte(obs)
        sc = R.score_exakt(leaf, flag, mini_g, mini_u, mini_prior)[0]
        for d in range(2):
            assert sc[d] == pytest.approx(math.log(MINI_PRIOR[d] * p[d]), rel=1e-10, abs=1e-10), (obs, d)


def test_ueberlesen_gleich_randverteilung_ueber_das_oberbegriffszeichen(beobachtungen, mini_g, mini_prior, mini_u):
    for leaf_t in {obs[0] for obs in beobachtungen}:
        marg = [sum(p[d] for (lt, f), p in beobachtungen.items() if lt == leaf_t) for d in range(2)]
        leaf, flag = _akte((leaf_t, 0))
        sc = R.score_ignoriert(leaf, mini_g, mini_u, mini_prior)[0]
        for d in range(2):
            assert sc[d] == pytest.approx(math.log(MINI_PRIOR[d] * marg[d]), rel=1e-10, abs=1e-10), (leaf_t, d)


def test_alle_kinder_gleich_naive_bayes_von_hand(mini_g, mini_prior):
    # Akte: Kind 1 genau, Oberbegriff vorhanden -> Kinder 1, 2, 3 zählen als aufgeschrieben (alle Kinder), g ungeteilt
    leaf, flag = _akte(((1, 0, 0), 1))
    sc = R.score_alle_kinder(leaf, flag, mini_g, mini_prior)[0]
    for d in range(2):
        assert sc[d] == pytest.approx(math.log(MINI_PRIOR[d] * MINI_G[d][0] * MINI_G[d][1] * MINI_G[d][2]))
    # Akte: nur Kind 3 genau, kein Oberbegriff -> Kinder 1 und 2 abwesend
    leaf, flag = _akte(((0, 0, 1), 0))
    sc = R.score_alle_kinder(leaf, flag, mini_g, mini_prior)[0]
    for d in range(2):
        assert sc[d] == pytest.approx(math.log(MINI_PRIOR[d] * (1 - MINI_G[d][0]) * (1 - MINI_G[d][1]) * MINI_G[d][2]))


def test_naiv_gleich_naive_bayes_auf_den_genauen_begriffen_von_hand(mini_g, mini_prior):
    # Akte: Kinder 1 und 3 genau, Oberbegriff vorhanden (wird nicht gelesen): Kind 2 abwesend, ungeteilte g
    leaf, flag = _akte(((1, 0, 1), 1))
    sc = R.score_naiv(leaf, mini_g, mini_prior)[0]
    for d in range(2):
        assert sc[d] == pytest.approx(math.log(MINI_PRIOR[d] * MINI_G[d][0] * (1 - MINI_G[d][1]) * MINI_G[d][2]))


def test_ohne_grobe_angaben_rechnen_exakt_ueberlesen_alle_kinder_und_naiv_gleich(mini_g, mini_prior):
    for leaf_t in ((0, 0, 0), (1, 0, 1), (1, 1, 1)):
        leaf, flag = _akte((leaf_t, 0))
        x = R.score_exakt(leaf, flag, mini_g, 0.0, mini_prior)
        i = R.score_ignoriert(leaf, mini_g, 0.0, mini_prior)
        a = R.score_alle_kinder(leaf, flag, mini_g, mini_prior)
        n = R.score_naiv(leaf, mini_g, mini_prior)
        assert np.allclose(x, i, atol=1e-12) and np.allclose(x, a, atol=1e-12) and np.allclose(x, n, atol=1e-12)


def test_erzeuger_ziehungen_gegen_aufzaehlung(beobachtungen):
    """Eigene Konstruktion: je Kind eine Gleichverteilte r; r < g(1 − u) genau, r < g nur als Oberbegriff, sonst abwesend (Krankheit 0, 60 000 Ziehungen, Zufalls-Seed 17)."""
    n = 60000
    r = Stream(17).uniform((n, 3))
    g = MINI_G[0]
    exakt = r < g * (1 - MINI_U)
    grob = (r >= g * (1 - MINI_U)) & (r < g)
    flag = grob.any(axis=1)
    codes = exakt.astype(int) @ np.array([4, 2, 1]) * 2 + flag
    for (lt, f), p in beobachtungen.items():
        code = (lt[0] * 4 + lt[1] * 2 + lt[2]) * 2 + f
        freq = float((codes == code).mean())
        assert abs(freq - p[0]) < 5 * math.sqrt(p[0] * (1 - p[0]) / n) + 1e-9, ((lt, f), freq, p[0])


def test_gruppenrechnung_vektorisiert_gleich_schleife_in_reinem_python():
    world = W.make_world(41, 12, 0.3, 0)
    pat = W.sample_patients(world, 41, 40)
    cp = H.coarsen(pat, 0.5, 41)
    g = H.group_g(world)
    u = 0.5
    sc = R.score_exakt(cp.leaf, cp.flag, g, u, world.prior)
    for i in (0, 7, 23, len(cp.d) - 1):
        for d in range(world.nd):
            logp = math.log(world.prior[d])
            for k in range(cp.leaf.shape[1]):
                kids = range(cp.leaf.shape[2])
                inl = [bool(cp.leaf[i, k, c]) for c in kids]
                pl = math.prod(g[d, k, c] * (1 - u) for c in kids if inl[c])
                ab = math.prod(1 - g[d, k, c] for c in kids if not inl[c])
                gen = math.prod(1 - g[d, k, c] + g[d, k, c] * u for c in kids if not inl[c])
                logp += math.log(pl * ((gen - ab) if cp.flag[i, k] else ab))
            assert sc[i, d] == pytest.approx(logp, rel=1e-9, abs=1e-9), (i, d)


def test_resnik_tabellen_von_hand():
    # 3 Krankheiten, 2 Oberbegriffe mit je 2 Kindern (Merkmale 0, 1 | 2, 3)
    leaf_ann = np.array([[1, 0, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 0]], dtype=float)
    parent_ann = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    ic_leaf, ic_par = R.information_content(leaf_ann, parent_ann)
    # Merkmal 0: 2 von 3 Krankheiten; Merkmal 1: 1 von 3; Merkmal 2: 1 von 3; Merkmal 3: keine (zählt als 1 von 3); Oberbegriff 0: 2 von 3; Oberbegriff 1: 1 von 3
    assert ic_leaf == pytest.approx([-math.log(2 / 3), math.log(3), math.log(3), math.log(3)])
    assert ic_par == pytest.approx([-math.log(2 / 3), math.log(3)])
    sim_leaf, sim_par = R.resnik_tables(leaf_ann, parent_ann)
    # Merkmal 1 trifft Krankheit 1 mit IC(Merkmal 1), Krankheit 0 über den gemeinsamen Oberbegriff 0 (IC 0,405), Krankheit 2 gar nicht (Wurzel: 0)
    assert sim_leaf[1, 1] == pytest.approx(math.log(3))
    assert sim_leaf[0, 1] == pytest.approx(-math.log(2 / 3))
    assert sim_leaf[2, 1] == 0.0
    assert sim_par[:, 0] == pytest.approx([-math.log(2 / 3), -math.log(2 / 3), 0.0])
    # Akte mit Merkmal 1 genau und Oberbegriff 1: Punktzahl = Summe der beiden Treffer
    leaf = np.array([[[0, 1], [0, 0]]], dtype=float)
    flag = np.array([[False, True]])
    sc = R.score_resnik(leaf, flag, sim_leaf, sim_par)[0]
    assert sc == pytest.approx([-math.log(2 / 3), math.log(3), math.log(3)])


def test_alle_ranker_werden_von_all_scores_geliefert():
    world = W.make_world(3, 10, 0.3, 0)
    pat = W.sample_patients(world, 3, 20)
    cp = H.coarsen(pat, 0.3, 3)
    ann_leaf, ann_par = H.annotation(world)
    hier = (H.group_g(world), *R.resnik_tables(ann_leaf, ann_par))
    sc = R.all_scores(cp.leaf, cp.flag, world, 0.3, hier)
    assert set(sc) == set(C.RANKERS)
    for r in C.RANKERS:
        assert sc[r].shape == (len(cp.d), 10)
