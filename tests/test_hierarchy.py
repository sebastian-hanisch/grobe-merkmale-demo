"""Hierarchie und grobe Dokumentation: Form, Grenzfälle u = 0 und u = 1, Anteil grober Angaben, Wiederholbarkeit, Annotation."""

import numpy as np
import pytest

import gmk_constants as C
import gmk_hierarchy as H
import gmk_world as W


@pytest.fixture(scope="module")
def welt_und_patienten():
    w = W.make_world(7, 60, 0.3, 0)
    return w, W.sample_patients(w, 7, 1500)


def test_form_der_akte(welt_und_patienten):
    _, p = welt_und_patienten
    cp = H.coarsen(p, 0.3, 1)
    n = len(p.d)
    assert cp.leaf.shape == (n, C.N_GROUPS, C.GROUP_SIZE) and cp.flag.shape == (n, C.N_GROUPS)
    assert set(np.unique(cp.leaf)) <= {0.0, 1.0}


def test_ohne_grobe_angaben_steht_jedes_merkmal_genau(welt_und_patienten):
    _, p = welt_und_patienten
    cp = H.coarsen(p, 0.0, 1)
    assert np.array_equal(cp.leaf.reshape(len(p.d), -1), (p.x > 0).astype(float))
    assert not cp.flag.any() and cp.share_coarse == 0.0


def test_mit_u_eins_steht_kein_merkmal_genau_und_der_oberbegriff_folgt_der_gruppe(welt_und_patienten):
    _, p = welt_und_patienten
    cp = H.coarsen(p, 1.0, 1)
    assert not cp.leaf.any()
    assert np.array_equal(cp.flag, (p.x > 0).reshape(len(p.d), C.N_GROUPS, C.GROUP_SIZE).any(axis=2))
    assert cp.share_coarse == 1.0


@pytest.mark.parametrize("u", [0.3, 0.6, 0.9])
def test_anteil_grober_angaben_liegt_bei_u(welt_und_patienten, u):
    _, p = welt_und_patienten
    cp = H.coarsen(p, u, 5)
    assert abs(cp.share_coarse - u) < 0.01     # über 1500 Patienten mit je einigen aufgeschriebenen Merkmalen: Standardfehler unter 0,005


def test_jedes_aufgeschriebene_merkmal_steht_genau_oder_als_oberbegriff(welt_und_patienten):
    _, p = welt_und_patienten
    cp = H.coarsen(p, 0.5, 2)
    n = len(p.d)
    genau = cp.leaf.reshape(n, -1) > 0
    grup = np.repeat(cp.flag, C.GROUP_SIZE, axis=1)
    assert (genau <= (p.x > 0)).all()                     # genau steht nur, was aufgeschrieben ist
    assert (((p.x > 0) & ~genau) <= grup).all()           # was nicht genau steht, hinterlässt seinen Oberbegriff
    assert (grup <= np.repeat((p.x > 0).reshape(n, C.N_GROUPS, C.GROUP_SIZE).any(axis=2), C.GROUP_SIZE, axis=1)).all()   # kein Oberbegriff ohne aufgeschriebenes Kind


def test_gleicher_zufalls_seed_gleiche_akte_anderer_seed_andere(welt_und_patienten):
    _, p = welt_und_patienten
    a, b, c = H.coarsen(p, 0.5, 3), H.coarsen(p, 0.5, 3), H.coarsen(p, 0.5, 4)
    assert np.array_equal(a.leaf, b.leaf) and np.array_equal(a.flag, b.flag)
    assert not np.array_equal(a.flag, c.flag)


def test_gruppengroesse_muss_die_merkmalszahl_teilen(welt_und_patienten):
    _, p = welt_und_patienten
    with pytest.raises(ValueError):
        H.coarsen(p, 0.3, 1, gsz=7)


def test_gruppen_haeufigkeiten_sind_begrenzt_und_haben_die_gruppenform(welt_und_patienten):
    w, _ = welt_und_patienten
    g = H.group_g(w)
    assert g.shape == (60, C.N_GROUPS, C.GROUP_SIZE)
    assert g.min() >= C.PROB_FLOOR and g.max() <= 1.0 - C.PROB_FLOOR


def test_annotation_ist_haeufigkeit_ueber_der_schwelle_und_oberbegriff_ist_das_maximum_der_kinder(welt_und_patienten):
    w, _ = welt_und_patienten
    leaf, par = H.annotation(w)
    assert np.array_equal(leaf > 0, w.f > C.SIMPLE_THRESHOLD)
    for k in range(C.N_GROUPS):
        assert np.array_equal(par[:, k], leaf[:, k * C.GROUP_SIZE:(k + 1) * C.GROUP_SIZE].max(axis=1))
    assert (leaf.sum(axis=1) >= C.K_MIN_FEAT).all()        # jede Krankheit hat mindestens vier annotierte Merkmale
