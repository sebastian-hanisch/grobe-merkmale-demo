"""Abnahmekriterien der Presets an der vorgerechneten Messreihe; jedes Kriterium kippt einzeln an einer künstlichen Zelle (Preset-Disziplin)."""

import gmk_constants as C
import gmk_stories as ST


def test_alle_kriterien_aller_presets_gelten_in_beiden_mengen():
    for name, rows in ST.check_all().items():
        assert rows and {w for _, w, _ in rows} == {"A", "B"}, name
        assert all(ok for _, _, ok in rows), [(t, w) for t, w, ok in rows if not ok]


def test_jedes_preset_hat_kriterien():
    assert set(ST.CRITERIA) == set(C.PRESETS)


def _zelle(x=60.0, i=40.0, d_xi=20.0, d_ai=10.0, d_xa=10.0, se=1.0, grob=0.3):
    summary = {r: {"5": [v, 1.0]} for r, v in (("X", x), ("I", i), ("A", 0.0), ("R", 0.0))}
    paired = {"X-I": [d_xi, se], "A-I": [d_ai, se], "X-A": [d_xa, se]}
    return {"A": {"summary": summary, "paired": paired, "share_coarse": grob}, "B": {"summary": summary, "paired": paired, "share_coarse": grob}}


def test_genau_kippt_einzeln():
    ok = _zelle(d_xi=0.04, d_xa=-0.04, grob=0.0)
    assert all(v for _, _, v in ST.check_preset("Genau dokumentiert", ok))
    assert not ST.check_preset("Genau dokumentiert", _zelle(d_xi=0.04, d_xa=0.0, grob=0.01))[0][2]
    assert not ST.check_preset("Genau dokumentiert", _zelle(d_xi=0.05, d_xa=0.0, grob=0.0))[1][2]
    assert not ST.check_preset("Genau dokumentiert", _zelle(d_xi=0.0, d_xa=0.05, grob=0.0))[1][2]


def test_teilweise_kippt_einzeln():
    assert all(v for _, _, v in ST.check_preset("Teilweise grob", _zelle(d_xi=8.0, se=1.0)))
    assert not ST.check_preset("Teilweise grob", _zelle(d_xi=7.9, se=1.0))[0][2]
    assert not ST.check_preset("Teilweise grob", _zelle(d_xi=8.0, se=4.0))[1][2]     # genau zwei Standardfehler: nicht deutbar


def test_meist_kippt_einzeln():
    assert all(v for _, _, v in ST.check_preset("Meist grob", _zelle(d_xi=15.1, d_ai=2.1, d_xa=2.1, se=1.0)))
    assert not ST.check_preset("Meist grob", _zelle(d_xi=15.0, d_ai=2.1, d_xa=2.1))[0][2]
    assert not ST.check_preset("Meist grob", _zelle(d_xi=20.0, d_ai=2.0, d_xa=2.1))[1][2]
    assert not ST.check_preset("Meist grob", _zelle(d_xi=20.0, d_ai=2.1, d_xa=2.0))[2][2]


def test_fast_nur_kippt_einzeln():
    assert all(v for _, _, v in ST.check_preset("Fast nur Oberbegriffe", _zelle(x=45.0, i=15.0)))
    assert not ST.check_preset("Fast nur Oberbegriffe", _zelle(x=45.0, i=15.1))[0][2]
    assert not ST.check_preset("Fast nur Oberbegriffe", _zelle(x=44.9, i=15.0))[1][2]


def test_luecken_kippt_einzeln():
    assert all(v for _, _, v in ST.check_preset("Viele Lücken", _zelle(x=50.0, d_xi=2.1, d_ai=-2.0, se=1.0)))
    assert not ST.check_preset("Viele Lücken", _zelle(x=50.0, d_xi=2.0, d_ai=0.0, se=1.0))[0][2]
    assert not ST.check_preset("Viele Lücken", _zelle(x=50.1, d_xi=2.1, d_ai=0.0, se=1.0))[1][2]
    assert not ST.check_preset("Viele Lücken", _zelle(x=50.0, d_xi=2.1, d_ai=2.1, se=1.0))[2][2]


def test_ein_kriterium_gilt_nur_wenn_es_in_beiden_mengen_gilt():
    c = {"A": _zelle(x=50.0, i=14.0)["A"], "B": _zelle(x=50.0, i=16.0)["B"]}
    rows = ST.check_preset("Fast nur Oberbegriffe", c)
    assert [ok for _, w, ok in rows] == [True, True, False, True]
