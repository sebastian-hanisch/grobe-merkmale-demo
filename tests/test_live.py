"""Live-Messung: Wiederholbarkeit, Gleichheit von exakt, überlesen, alle Kinder und naiv bei u = 0, Zweig-Tests je Regler (jede Stufe ändert mindestens einen Ranker), Auswertungsfunktionen."""

import numpy as np
import pytest

import gmk_constants as C
import gmk_live as L

KLEIN = dict(n_worlds=2, n_patients=400, n_example=5)   # kleine Messungen: schnell, nur für Zweig- und Formtests
BASIS = L.Settings(seed=11)


def _r5(s, **kw):
    sm = L.summarize(L.measure(s, **{**KLEIN, **kw}))
    return {r: sm[r][5][0] for r in C.RANKERS}


def test_gleicher_zufalls_seed_gleiche_messung_anderer_seed_andere():
    a, b = _r5(BASIS), _r5(BASIS)
    assert a == b
    assert a != _r5(L.Settings(seed=12))


def test_settings_nur_mit_schluesselwoertern():
    with pytest.raises(TypeError):
        L.Settings(200, 0.3, 0.3, 35)   # positional: nicht erlaubt (ein eingefügtes Feld würde Werte still verschieben)
    assert L.Settings(seed=3).key() == "nd200_q0.3_u0.3"


def test_ohne_grobe_angaben_sind_exakt_ueberlesen_alle_kinder_und_naiv_dieselbe_rangliste():
    m = L.measure(L.Settings(seed=11, u=0.0), **KLEIN)
    for run in m.runs:
        for k in (1, 5, 10, "mrr"):
            assert np.array_equal(run.metrics["X"][k], run.metrics["I"][k])
            assert np.array_equal(run.metrics["X"][k], run.metrics["A"][k])
            assert np.array_equal(run.metrics["X"][k], run.metrics["N"][k])


@pytest.mark.parametrize("aenderung", [dict(u=0.0), dict(u=0.6), dict(u=0.9), dict(q=0.6), dict(q=0.0), dict(nd=50)])
def test_jeder_regler_aendert_mindestens_einen_ranker(aenderung):
    """Zweig-Test (Nullspalten-Falle): jede Stufe weicht von der Voreinstellung auf mindestens einem Ranker ab."""
    base = _r5(BASIS)
    neu = _r5(L.Settings(seed=11, **aenderung))
    assert any(abs(neu[r] - base[r]) > 1e-9 for r in C.RANKERS), aenderung


def test_u_aendert_jeden_ranker_ausser_dem_gleichstand_bei_u_null():
    """Zweig-Test je Ranker: bei u = 0,6 weicht jeder der fünf Ranker von u = 0 ab (kein Ranker ist von u unabhängig)."""
    a, b = _r5(L.Settings(seed=11, u=0.0)), _r5(L.Settings(seed=11, u=0.6))
    for r in C.RANKERS:
        assert abs(a[r] - b[r]) > 1e-9, r


def test_jede_regelart_weicht_auf_mindestens_einer_einstellung_von_jeder_anderen_ab():
    werte = {r: [] for r in C.RANKERS}
    for s in (L.Settings(seed=11, u=0.3), L.Settings(seed=11, u=0.6), L.Settings(seed=11, u=0.9)):
        for r, v in _r5(s).items():
            werte[r].append(v)
    for i, r1 in enumerate(C.RANKERS):
        for r2 in C.RANKERS[i + 1:]:
            assert werte[r1] != werte[r2], (r1, r2)


def test_exakt_liegt_bei_grober_dokumentation_vor_dem_ueberlesen():
    for u in (0.3, 0.6, 0.9):
        r = _r5(L.Settings(seed=11, u=u), n_worlds=2, n_patients=800)
        assert r["X"] > r["I"], u


def test_gleichstaende_ohne_rundung_weichen_bei_u_null_nur_um_rundungsrauschen_ab(monkeypatch):
    """Ohne die Rundung der Punktzahlen (TIE_DIGITS) lösen die Summen in anderer Reihenfolge Gleichstände je Ranker anders auf: der gepaarte Unterschied bei u = 0 ist klein, aber nicht garantiert 0."""
    monkeypatch.setattr(C, "TIE_DIGITS", 30)
    m = L.measure(L.Settings(seed=C.SWEEP_SEEDS["A"], u=0.0), n_example=0)
    assert abs(L.paired_diff(m, "X", "I")[0]) < 0.05
    monkeypatch.setattr(C, "TIE_DIGITS", 9)
    assert L.paired_diff(L.measure(L.Settings(seed=C.SWEEP_SEEDS["A"], u=0.0), n_example=0), "X", "I") == (0.0, 0.0)


def test_summarize_form_und_nichtnegative_streuung():
    m = L.measure(BASIS, **KLEIN)
    s = L.summarize(m)
    assert set(s) == set(C.RANKERS)
    for r in C.RANKERS:
        assert set(s[r]) == {1, 5, 10, "mrr"}
        assert s[r][1][0] <= s[r][5][0] <= s[r][10][0] <= 100.0 and s[r][5][1] >= 0.0


def test_paired_diff_gegen_handrechnung():
    m = L.measure(BASIS, **KLEIN)
    diffs = [100 * (run.metrics["X"][5].mean() - run.metrics["I"][5].mean()) for run in m.runs]
    mean, se = L.paired_diff(m, "X", "I")
    assert mean == pytest.approx(np.mean(diffs))
    assert se == pytest.approx(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    assert L.paired_diff(m, "I", "I") == (0.0, 0.0)


def test_anteil_grober_angaben_der_messung_liegt_bei_u_und_gerankte_patienten_bei_unter_100():
    m = L.measure(BASIS, n_worlds=2, n_patients=1500, n_example=0)
    assert abs(L.share_coarse(m) - 30.0) < 1.0
    assert 90.0 < L.share_ranked(m) < 100.0
    assert L.share_ranked(L.measure(L.Settings(seed=11, q=0.6), **KLEIN)) < L.share_ranked(L.measure(L.Settings(seed=11, q=0.0), **KLEIN))


def test_beispielpatienten_nur_in_der_ersten_welt():
    m = L.measure(BASIS, **KLEIN)
    ex = m.runs[0].example
    assert len(ex["d"]) == 5 and m.runs[1].example == {}
    assert ex["leaf"].shape == (5, C.N_GROUPS, C.GROUP_SIZE) and ex["flag"].shape == (5, C.N_GROUPS)
    assert set(ex["scores"]) == set(C.RANKERS)


def test_worst_case_am_regler_maximum_laeuft_durch():
    """Slider-Maximum (200 Krankheiten, q = 0,6, u = 0,9): Hänger oder Aufblähen fallen hier auf; keine Zeitassertion."""
    m = L.measure(L.Settings(seed=3, nd=200, q=0.6, u=0.9), n_worlds=1, n_patients=1500, n_example=3)
    assert np.isfinite(L.summarize(m)["X"][5][0])
