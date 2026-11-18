"""AppTest-Rauchtests (DEMO-PLAYBOOK §1d Eckfall-Matrix): Standard, jedes Preset, jede Stufe jedes Reglers am Minimum und Maximum, Kombination aller Extremwerte, Beispielpatient, PDF."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gmk_constants as C
import gmk_presets as P

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
STUFEN = [k for k, s in P.SETTING_SPECS.items() if s.options is not None]


def _fresh() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_laedt_ohne_ausnahme():
    at = _fresh()
    assert len(at.metric) >= 4


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_durch(name):
    at = _fresh()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == preset[key]


@pytest.mark.parametrize("state_key", STUFEN)
@pytest.mark.parametrize("kante", ["lo", "hi"])
def test_jede_stufe_am_rand_laeuft_durch(state_key, kante):
    at = _fresh()
    spec = P.SETTING_SPECS[state_key]
    at.sidebar.select_slider(key=state_key).set_value(spec.lo if kante == "lo" else spec.hi).run(timeout=TIMEOUT)
    assert not at.exception, (state_key, kante, [str(e) for e in at.exception])


def test_kombination_aller_minima_und_aller_maxima():
    for kante in ("lo", "hi"):
        at = _fresh()
        for state_key in STUFEN:
            spec = P.SETTING_SPECS[state_key]
            at.sidebar.select_slider(key=state_key).set_value(spec.lo if kante == "lo" else spec.hi)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (kante, [str(e) for e in at.exception])


def test_jede_stufe_von_u_laeuft_durch_und_aendert_das_verdikt_der_ersten_kernfrage():
    texte = {}
    for u in C.U_OPTIONS:
        at = _fresh()
        at.sidebar.select_slider(key="u_select").set_value(u).run(timeout=TIMEOUT)
        assert not at.exception, (u, [str(e) for e in at.exception])
        texte[u] = " ".join(s.value for s in at.success) + " ".join(s.value for s in at.info) + " ".join(s.value for s in at.warning)
    assert "nichts zu überlesen" in texte[0.0] and "mehr als 15,0 Punkte" in texte[0.9]


def test_seed_und_patient_regler_am_rand():
    for wert in (0, C.SEED_MAX):
        at = _fresh()
        at.sidebar.number_input(key="seed_input").set_value(wert).run(timeout=TIMEOUT)
        assert not at.exception
    for wert in (1, 30):
        at = _fresh()
        at.slider(key="patient_idx").set_value(wert).run(timeout=TIMEOUT)
        assert not at.exception


def test_beispielpatient_aufloesen():
    at = _fresh()
    at.toggle(key="patient_reveal").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Wahre Krankheit: K" in c.value for c in at.caption)


def test_pdf_wird_erst_nach_dem_klick_erzeugt():
    at = _fresh()
    assert not [b for b in at.get("download_button")]
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.session_state["pdf_bytes"][:4] == b"%PDF"


def test_kernfragen_und_hinweise_stehen_auf_der_seite():
    at = _fresh()
    texte = [m.value for m in at.markdown]
    assert all(any(f"Kernfrage {i}" in t for t in texte) for i in (1, 2, 3))
    assert any("keine medizinische Beratung" in t for t in texte)
    assert any("eigene Fassung" in t for t in texte)
