"""Zahlen der README: jede Befundzeile wird aus data/gmk_results.json UNABHÄNGIG nachgerechnet (eigene Rechnung hier, nicht die Funktionen der Demo) und im Text der README gesucht.

Dazu: die README entspricht dem Erzeugnis von tools/make_readme.py, die Testzahl in der README ist die Zahl der gesammelten Tests, die Voreinstellung lässt sich neu rechnen."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import gmk_constants as C  # noqa: E402
import gmk_live as L  # noqa: E402
import make_readme as MK  # noqa: E402

DATA = json.loads((ROOT / "data" / "gmk_results.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def _c(nd=200, q=0.3, u=0.3):
    return DATA["cells"][f"nd{nd}_q{q}_u{u}"]


def _v(c, ranker, k="5"):
    """Mittel der beiden Mengen (je fünf Welten), in Prozent."""
    return (c["A"]["summary"][ranker][k][0] + c["B"]["summary"][ranker][k][0]) / 2


def _p(c, pair):
    """Gepaarte Differenz: Mittel der beiden Mittel, Standardfehler aus den beiden Standardfehlern."""
    a, b = c["A"]["paired"][pair], c["B"]["paired"][pair]
    return (a[0] + b[0]) / 2, math.sqrt(a[1] ** 2 + b[1] ** 2) / 2


def _f(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def _sf(x, nd=1):
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def _row(n):
    for line in README.splitlines():
        if line.startswith(f"| {n} | "):
            return line
    raise AssertionError(f"Befundzeile {n} fehlt in der README")


def test_readme_entspricht_dem_erzeugnis_des_generators():
    assert MK.build() == README


def test_befund_1_voreinstellung():
    c, row = _c(), _row(1)
    for ranker in C.RANKERS:
        assert _f(_v(c, ranker)) in row
    assert f"exakt {_f(_v(c, 'X'))}, überlesen {_f(_v(c, 'I'))}, alle Kinder {_f(_v(c, 'A'))}, Resnik-artig {_f(_v(c, 'R'))}" in row
    for pair, label in (("X-I", "exakt − überlesen"), ("X-A", "exakt − alle Kinder"), ("A-I", "alle Kinder − überlesen"), ("R-I", "Resnik-artig − überlesen")):
        m, se = _p(c, pair)
        assert f"{label} {_sf(m)} ± {_f(se)}" in row


def test_befund_2_verlust_waechst_mit_u():
    row = _row(2)
    loss = [_p(_c(u=u), "X-I") for u in (0.3, 0.6, 0.9)]
    assert f"{_sf(loss[0][0])} / {_sf(loss[1][0])} / {_sf(loss[2][0])} Punkte" in row
    assert f"{_f(loss[0][1])} / {_f(loss[1][1])} / {_f(loss[2][1])}" in row
    assert loss[0][0] < loss[1][0] < loss[2][0]
    for u in (0.3, 0.6, 0.9):       # Anteil grober Angaben liegt bei u
        assert abs((_c(u=u)["A"]["share_coarse"] + _c(u=u)["B"]["share_coarse"]) / 2 - 100 * u) < 0.5
    assert _f(_v(_c(u=0.6), "I")) in row and _f(_v(_c(u=0.9), "X")) in row


def test_befund_3_heuristik_holt_einen_teil_zurueck():
    row = _row(3)
    ai = [_p(_c(u=u), "A-I") for u in (0.3, 0.6, 0.9)]
    xi = [_p(_c(u=u), "X-I") for u in (0.3, 0.6, 0.9)]
    assert f"{_sf(ai[0][0])} / {_sf(ai[1][0])} / {_sf(ai[2][0])} Punkte" in row
    shares = [100 * a[0] / x[0] for a, x in zip(ai, xi)]
    assert f"{shares[0]:.0f} / {shares[1]:.0f} / {shares[2]:.0f} Prozent des Verlusts" in row
    assert all(0 < s < 100 for s in shares)             # holt einen Teil, nicht alles zurück
    vor = hinten = gleich = 0
    for nd in C.ND_OPTIONS:
        for q in C.Q_OPTIONS:
            for u in (0.3, 0.6, 0.9):
                m, se = _p(_c(nd, q, u), "A-I")
                vor += m > 2 * se
                hinten += m < -2 * se
                gleich += abs(m) <= 2 * se
    assert f"In {vor} von 18 Zellen mit u > 0 liegt die Heuristik mehr als zwei Standardfehler vor dem Überlesen, in {hinten} dahinter, in {gleich} ohne auflösbaren Unterschied" in row


def test_befund_4_verlust_nach_krankheiten_und_dokumentationsverlust():
    row = _row(4)
    for nd in C.ND_OPTIONS:
        for q in C.Q_OPTIONS:
            loss = [_p(_c(nd, q, u), "X-I")[0] for u in (0.3, 0.6, 0.9)]
            assert f"{nd} Krankheiten, q = {_f(q)}: {' / '.join(_sf(v) for v in loss)}" in row
            assert loss[0] < loss[1] < loss[2], (nd, q)


def test_befund_5_resnik_artig():
    row = _row(5)
    ri = [_p(_c(u=u), "R-I") for u in (0.3, 0.6, 0.9)]
    ra = [_p(_c(u=u), "R-A") for u in (0.3, 0.6, 0.9)]
    assert f"{_sf(ri[0][0])} / {_sf(ri[1][0])} / {_sf(ri[2][0])} bei u = 0,3 / 0,6 / 0,9" in row
    assert f"{_sf(ra[0][0])} / {_sf(ra[1][0])} / {_sf(ra[2][0])}" in row
    cnt = [0, 0, 0]
    behind_a = 0
    for nd in C.ND_OPTIONS:
        for q in C.Q_OPTIONS:
            for u in (0.3, 0.6, 0.9):
                m, se = _p(_c(nd, q, u), "R-I")
                cnt[0 if m > 2 * se else (1 if m < -2 * se else 2)] += 1
                m, se = _p(_c(nd, q, u), "R-A")
                behind_a += m < -2 * se
    assert f"vorn in {cnt[0]}, hinten in {cnt[1]}, ohne auflösbaren Unterschied in {cnt[2]} Zellen" in row
    assert f"gegenüber alle Kinder hinten in {behind_a} von 18" in row
    abstand = [_p(_c(nd, q, 0.0), "X-R")[0] for nd in C.ND_OPTIONS for q in C.Q_OPTIONS]
    assert f"zwischen {_f(min(abstand))} und {_f(max(abstand))} Punkten" in row
    assert all(a > 10 for a in abstand)                  # schon ohne grobe Angaben liegt er weit hinter exakt


def test_befund_6_viele_luecken():
    row, c = _row(6), _c(200, 0.6, 0.3)
    assert f"exakt {_f(_v(c, 'X'))}, überlesen {_f(_v(c, 'I'))}, alle Kinder {_f(_v(c, 'A'))}" in row
    assert f"Verlust {_sf(_p(c, 'X-I')[0])} ± {_f(_p(c, 'X-I')[1])}" in row
    m, se = _p(c, "A-I")
    assert f"{_sf(m, 2)} ± {_f(se, 2)}" in row and -m > 2 * se          # gepoolt knapp mehr als zwei Standardfehler hinter dem Überlesen
    assert all(abs(c[w]["paired"]["A-I"][0]) <= 2 * c[w]["paired"]["A-I"][1] for w in "AB")      # in jeder Menge einzeln innerhalb von zwei Standardfehlern
    assert _p(c, "X-I")[0] < _p(_c(200, 0.3, 0.3), "X-I")[0]              # weniger zu verlieren als bei q = 0,3


def test_befund_6_zaehlung_ueber_alle_zellen_nicht_nur_die_zelle_des_satzes():
    row = _row(6)
    n_loss = n_back = 0
    for nd in C.ND_OPTIONS:
        for u in (0.3, 0.6, 0.9):
            loss = [_p(_c(nd, q, u), "X-I")[0] for q in C.Q_OPTIONS]
            back = [_p(_c(nd, q, u), "A-I")[0] for q in C.Q_OPTIONS]
            n_loss += loss[0] > loss[1] > loss[2]
            n_back += back[0] > back[1] > back[2]
            if not loss[0] > loss[1] > loss[2]:
                assert f"{nd} Krankheiten, u = {_f(u)}: {' / '.join(_f(x) for x in loss)} bei q = 0 / 0,3 / 0,6" in row
    assert f"der Verlust nimmt mit steigendem q in {n_loss} Paaren in beiden Schritten ab" in row and f"die Rückholung der Heuristik in {n_back} Paaren" in row
    assert (n_loss, n_back) == (5, 6)


def test_befund_7_ohne_grobe_angaben_gleich():
    for nd in C.ND_OPTIONS:
        for q in C.Q_OPTIONS:
            c = _c(nd, q, 0.0)
            for w in ("A", "B"):
                assert c[w]["share_coarse"] == 0.0
                assert all(abs(c[w]["paired"][p][0]) < 1e-9 for p in ("X-I", "X-A", "A-I"))
    assert "genau 0,0 Punkte" in _row(7)


def test_befund_8_kein_ranker_ueber_exakt():
    vorsprung, n = -1e9, 0
    for c in DATA["cells"].values():
        for w in ("A", "B"):
            for ranker in ("I", "A", "R", "N"):
                for k in ("1", "5", "10", "mrr"):
                    n += 1
                    vorsprung = max(vorsprung, c[w]["summary"][ranker][k][0] - c[w]["summary"]["X"][k][0])
    assert n == 768 and vorsprung <= 1e-9
    assert "In 768 Vergleichen" in _row(8) and "größter Vorsprung 0,0 Punkte" in _row(8)


def test_befund_10_naives_ueberlesen():
    row = _row(10)
    c = _c()
    m, se = _p(c, "I-N")
    assert f"naives Überlesen Recall@5 {_f(_v(c, 'N'))}" in row and f"„überlesen“ minus naiv {_sf(m)} ± {_f(se)}" in row
    diffs = [_p(_c(nd, q, u), "I-N") for nd in C.ND_OPTIONS for q in C.Q_OPTIONS for u in (0.3, 0.6, 0.9)]
    vor = sum(d > 2 * s for d, s in diffs)
    assert f"in {vor} mehr als zwei Standardfehler vor dem naiven" in row
    assert f"zwischen {_sf(min(d for d, _ in diffs), 2)} und {_sf(max(d for d, _ in diffs), 2)} Punkten" in row
    assert min(d for d, _ in diffs) > 0                              # in jeder Zelle mit grober Angabe liegt das naive Überlesen hinter dem Überlesen mit Wissen um u
    for nd in C.ND_OPTIONS:
        for q in C.Q_OPTIONS:
            cc = _c(nd, q, 0.0)
            assert all(abs(cc[w]["paired"]["I-N"][0]) < 1e-9 for w in "AB")      # bei u = 0 gleich


def test_befund_9_stabilitaet():
    diff = max(abs(c["A"]["summary"][r]["5"][0] - c["B"]["summary"][r]["5"][0]) for c in DATA["cells"].values() for r in C.RANKERS)
    sd = max(c[w]["summary"][r]["5"][1] for c in DATA["cells"].values() for w in "AB" for r in C.RANKERS)
    assert f"um höchstens {_f(diff, 2)} Punkte" in _row(9) and f"höchstens {_f(sd, 1)} Punkte" in _row(9)


def test_vergleich_frueher_gegen_neu_nennt_die_gemessenen_zahlen():
    for u in (0.3, 0.6, 0.9):
        c = _c(u=u)
        zeile = [ln for ln in README.splitlines() if ln.startswith(f"| u = {_f(u)}: exakt / überlesen / alle Kinder, Recall@5")][0]
        assert f"{_f(_v(c, 'X'))} / {_f(_v(c, 'I'))} / {_f(_v(c, 'A'))}" in zeile
        alt = MK.ALT[u]
        assert f"{_f(alt['X'][1])} / {_f(alt['I'][1])} / {_f(alt['A'][1])}" in zeile
    neu = max(abs(_v(_c(u=u), r, k) - MK.ALT[u][r][i]) for u in (0.3, 0.6, 0.9) for r in "XIA" for i, k in ((0, "1"), (1, "5")))
    assert f"um höchstens {_f(neu)} Punkte" in README


def test_neu_rechnen_voreinstellung_im_band():
    """Menge A, Voreinstellung: jede Zahl Recall@5 liegt im Band von 0,6 Punkten um den gespeicherten Wert (Linux rechnet an der letzten Binärstelle anders)."""
    s = L.Settings(seed=C.SWEEP_SEEDS["A"])
    sm = L.summarize(L.measure(s, n_example=0))
    for r in C.RANKERS:
        assert sm[r][5][0] == pytest.approx(DATA["cells"][s.key()]["A"]["summary"][r]["5"][0], abs=0.6), r


def test_testzahl_der_readme_ist_die_zahl_der_gesammelten_tests():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=600).stdout
    m = re.search(r"(\d+) tests? collected", out) or re.search(r"^(\d+) tests", out, re.M)
    gesammelt = int(m.group(1)) if m else len([ln for ln in out.splitlines() if "::" in ln])
    assert re.findall(r"(\d+) Tests \(`pytest", README) == [str(gesammelt)], gesammelt
