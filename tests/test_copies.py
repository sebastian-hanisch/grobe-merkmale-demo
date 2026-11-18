"""Die Hilfsmodule gmk_stream, gmk_world und gmk_metrics sind Kopien aus phaenotyp-ranking-demo (nur das Präfix ist umbenannt), gmk_rng.py die der Vorlage.

Die Kopien werden nicht verändert, auch nicht in Docstrings: ein Unterschied hier heißt, dass jemand an der Kopie statt an der Vorlage gearbeitet hat.
Fehlt die Nachbar-Demo (Repo einzeln geklont), werden die Tests übersprungen."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NACHBAR = ROOT.parent / "phaenotyp-ranking-demo"
VORLAGE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


@pytest.mark.skipif(not VORLAGE.exists(), reason="Vorlage nicht vorhanden (Repo einzeln geklont)")
def test_rng_ist_die_unveraenderte_kopie_der_vorlage():
    assert (ROOT / "gmk_rng.py").read_bytes() == VORLAGE.read_bytes()


@pytest.mark.skipif(not NACHBAR.exists(), reason="phaenotyp-ranking-demo nicht vorhanden (Repo einzeln geklont)")
@pytest.mark.parametrize("name", ["stream", "world", "metrics"])
def test_hilfsmodul_ist_kopie_mit_umbenanntem_praefix(name):
    quelle = (NACHBAR / f"phn_{name}.py").read_text(encoding="utf-8").replace("phn_", "gmk_")
    kopie = (ROOT / f"gmk_{name}.py").read_text(encoding="utf-8")
    assert kopie.replace("\r\n", "\n") == quelle.replace("\r\n", "\n")
