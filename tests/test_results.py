"""Vorgerechnete Messreihe: Form, Plausibilität (Reihenfolge der Ranker, Monotonie in u), Stabilität der beiden Mengen, Plausibilität."""

import pytest

import gmk_constants as C
import gmk_results as RES


def _zellen():
    return [(s, RES.cell(s)) for s in RES.sweep_settings()]


def test_meta_stimmt_mit_den_konstanten_ueberein():
    meta = RES.load()["meta"]
    assert meta["worlds_per_set"] == C.N_WORLDS and meta["patients_per_world"] == C.N_PATIENTS and meta["sets"] == C.SWEEP_SEEDS
    assert meta["rankers"] == list(C.RANKERS) and set(meta["pairs"]) == {"X-I", "X-A", "X-R", "A-I", "R-I", "R-A", "I-N", "X-N"}


def test_jede_zelle_hat_beide_mengen_mit_allen_rankern_und_paaren():
    for s, c in _zellen():
        assert c is not None and c["settings"] == {"nd": s.nd, "q": s.q, "u": s.u}
        for w in ("A", "B"):
            assert set(c[w]["summary"]) == set(C.RANKERS) and set(c[w]["paired"]) == set(RES.load()["meta"]["pairs"])
            for r in C.RANKERS:
                assert set(c[w]["summary"][r]) == {"1", "5", "10", "mrr"}
                assert c[w]["summary"][r]["1"][0] <= c[w]["summary"][r]["5"][0] <= c[w]["summary"][r]["10"][0] <= 100.0


def test_ohne_grobe_angaben_sind_die_ersten_drei_ranker_gleich_in_jeder_zelle():
    for s, c in _zellen():
        if s.u == 0.0:
            for w in ("A", "B"):
                assert c[w]["share_coarse"] == 0.0
                assert abs(c[w]["paired"]["X-I"][0]) < 0.05 and abs(c[w]["paired"]["X-A"][0]) < 0.05


def test_exakt_liegt_in_jeder_zelle_mit_grober_angabe_vor_dem_ueberlesen_und_der_verlust_waechst_mit_u():
    for nd in C.ND_OPTIONS:
        for q in C.Q_OPTIONS:
            loss = [RES.pooled_pair(RES.cell(RES.Settings(nd=nd, q=q, u=u)), "X-I")[0] for u in C.U_OPTIONS]
            assert abs(loss[0]) < 0.05 and loss == sorted(loss) and loss[1] > 0, (nd, q, loss)


def test_anteil_grober_angaben_der_zellen_liegt_bei_u_in_prozent():
    for s, c in _zellen():
        for w in ("A", "B"):
            assert abs(c[w]["share_coarse"] - 100.0 * s.u) < 1.5, s.key()


def test_beide_mengen_stimmen_im_recall_auf_wenige_punkte_ueberein():
    for s, c in _zellen():
        for r in C.RANKERS:
            assert RES.sets_agree(c, r) < 6.0, (s.key(), r)    # fünf Welten je Menge streuen; Zahl nur als Warnung gegen Ausreißer


def test_gepoolte_differenz_und_standardfehler_gegen_handrechnung():
    c = {"A": {"paired": {"X-I": [10.0, 1.0]}}, "B": {"paired": {"X-I": [12.0, 2.0]}}}
    mean, se = RES.pooled_pair(c, "X-I")
    assert mean == 11.0 and se == pytest.approx((1.0 + 4.0) ** 0.5 / 2.0)
