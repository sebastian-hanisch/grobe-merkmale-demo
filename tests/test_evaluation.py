"""Verdikte (reine Funktionen, Schwellen und jede Zahl im Text an handgesetzten Werten), Beispielpatient, Zahlenformat."""

import pytest

import gmk_constants as C
import gmk_evaluation as ev
import gmk_live as L


def test_zahlenformat_mit_komma_und_vorzeichen():
    assert ev.pct(12.34) == "12,3" and ev.spct(2.0) == "+2,0" and ev.spct(-0.04) == "-0,0" and ev.dec(0.95) == "0,950"
    assert ev.num(0.3) == "0,3" and ev.num(200) == "200" and ev.num(0.0) == "0"


@pytest.mark.parametrize("loss,kind", [(0.0, "success"), (4.9, "success"), (5.0, "info"), (15.0, "info"), (15.1, "warning"), (39.5, "warning")])
def test_verdikt_verlust_schwellen(loss, kind):
    assert ev.verdict_verlust(0.6, 60.0, 60.0 - loss, loss, 1.0)[0] == kind


def test_verdikt_verlust_nennt_zahlen_und_schwelle():
    _, text = ev.verdict_verlust(0.6, 62.5, 40.0, 22.5, 0.8)
    assert "62,5" in text and "40,0" in text and "22,5" in text and "0,8 Standardfehler" in text and "mehr als 15,0 Punkte" in text
    _, text = ev.verdict_verlust(0.3, 70.0, 63.0, 7.0, 0.4)
    assert "5,0 bis 15,0 Punkte" in text
    _, text = ev.verdict_verlust(0.3, 70.0, 67.0, 3.0, 0.4)
    assert "weniger als 5,0 Punkte" in text


def test_verdikt_verlust_ohne_grobe_angaben_sagt_dass_es_nichts_zu_ueberlesen_gibt():
    kind, text = ev.verdict_verlust(0.0, 76.5, 76.5, 0.0, 0.0)
    assert kind == "success" and "nichts zu überlesen" in text


def test_verdikt_heuristik_drei_faelle_und_anteil():
    kind, text = ev.verdict_heuristik(22.5, 10.0, 1.0)             # 10 > 2 · 1: deutbar
    assert kind == "info" and "10,0 Punkte" in text and "44 %" in text and "22,5 Punkte" in text
    kind, text = ev.verdict_heuristik(22.5, -3.0, 1.0)
    assert kind == "warning" and "3,0 Punkte Recall@5 hinter dem Überlesen" in text
    kind, text = ev.verdict_heuristik(22.5, 2.0, 1.0)              # genau zwei Standardfehler: nicht deutbar
    assert kind == "info" and "kein Unterschied über zwei Standardfehlern" in text and "+2,0" in text
    kind, text = ev.verdict_heuristik(0.5, 0.0, 0.1)
    assert "unter einem Punkt" in text


def test_verdikt_resnik_drei_faelle():
    assert "3,0 Punkte vor dem Überlesen" in ev.verdict_resnik(62.5, 43.0, (3.0, 0.5))[1]
    assert "9,0 Punkte hinter dem Überlesen" in ev.verdict_resnik(70.0, 52.0, (-9.0, 0.4))[1]
    kind, text = ev.verdict_resnik(70.0, 52.0, (0.9, 0.5))
    assert kind == "info" and "ohne Unterschied über zwei Standardfehlern" in text and "18,0 Punkte unter dem exakten Ranker" in text
    assert "eigene Fassung" in text


def test_kennzahlen_und_verdikte_einer_messung():
    s = L.Settings(seed=11)
    m = L.measure(s, n_worlds=2, n_patients=500, n_example=5)
    k = ev.kennzahlen(m)
    assert set(k) == {"summary", "loss", "back", "resnik_i", "share_coarse", "share_ranked"}
    v1, v2, v3 = ev.verdicts(s, k)
    assert all(kind in ("success", "info", "warning") and text for kind, text in (v1, v2, v3))
    assert k["loss"] == L.paired_diff(m, "X", "I") and k["back"] == L.paired_diff(m, "A", "I")


def test_beispielpatient_akte_und_rangliste():
    m = L.measure(L.Settings(seed=11, u=0.6), n_worlds=1, n_patients=300, n_example=8)
    for idx in (0, 3, 7, 99):          # 99 wird auf den letzten Beispielpatienten begrenzt
        v = ev.example_view(m, idx)
        assert 0 <= v["true"] < 200
        assert set(v["top"]) == set(C.RANKERS) and all(len(v["top"][r]["ids"]) == 5 for r in C.RANKERS)
        assert all(v["top"][r]["rank_true"] >= 1 for r in C.RANKERS)
        assert sum(v["posterior_x"].values()) <= 1.0 + 1e-9
    ex = m.runs[0].example
    exact, coarse = ev.term_labels(ex["leaf"][0], ex["flag"][0])
    assert len(exact) == int(ex["leaf"][0].sum()) and len(coarse) == int(ex["flag"][0].sum())


def test_begriffsnamen_von_hand():
    import numpy as np
    leaf = np.zeros((2, 5))
    leaf[0, 1] = 1.0
    leaf[1, 4] = 1.0
    flag = np.array([False, True])
    exact, coarse = ev.term_labels(leaf, flag)
    assert exact == ["M2", "M10"] and coarse == ["Gruppe 2 (irgendeines von M6 bis M10)"]
