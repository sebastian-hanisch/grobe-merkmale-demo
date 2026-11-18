"""Vorgerechnete Messreihe (data/gmk_results.json, erzeugt von tools/sweep.py): Zellen, Zugriff, gepoolte Kennzahlen.

Eine Zelle = eine Einstellung (ohne Zufalls-Seed); je Zelle zwei Mengen von je fünf Welten (A: Zufalls-Seed 1000, B: 2000). Kennzahlen sind
Mittel ± Streuung über die fünf Welten in Prozent, Einheit: Patienten (Recall@k) beziehungsweise Welten (Streuung)."""

import json
from functools import lru_cache
from pathlib import Path

import gmk_constants as C
from gmk_live import Settings

DATA = Path(__file__).resolve().parent / "data" / "gmk_results.json"
BASE = Settings()


def sweep_settings() -> list:
    """Alle gemessenen Zellen: jede Kombination der Stufen von Krankheiten, Dokumentationsverlust und Anteil grober Angaben."""
    return [Settings(nd=nd, q=q, u=u) for nd in C.ND_OPTIONS for q in C.Q_OPTIONS for u in C.U_OPTIONS]


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def cell(settings: Settings) -> dict | None:
    """Die gemessene Zelle zur Einstellung oder None, wenn sie nicht gemessen ist."""
    return load()["cells"].get(settings.key())


def stat(c: dict, which: str, ranker: str, k=5) -> tuple:
    """(Mittel, Streuung) einer Kennzahl aus Menge `which` ("A" oder "B") in Prozent."""
    m, sd = c[which]["summary"][ranker][str(k)]
    return m, sd


def pooled(c: dict, ranker: str, k=5) -> float:
    """Mittel der beiden Mengen (je fünf Welten): zehn Welten insgesamt."""
    return (stat(c, "A", ranker, k)[0] + stat(c, "B", ranker, k)[0]) / 2.0


def pooled_pair(c: dict, pair: str) -> tuple:
    """Gepaarte Differenz (Recall@5, Punkte) aus beiden Mengen: Mittel der beiden Mittel und Standardfehler über die zehn Welten (aus den beiden Standardfehlern)."""
    a, b = c["A"]["paired"][pair], c["B"]["paired"][pair]
    return (a[0] + b[0]) / 2.0, ((a[1] ** 2 + b[1] ** 2) ** 0.5) / 2.0


def sets_agree(c: dict, ranker: str, k=5) -> float:
    """Unterschied der beiden Mengen in Punkten (Betrag), ein Maß für die Stabilität der Zahl."""
    return abs(stat(c, "A", ranker, k)[0] - stat(c, "B", ranker, k)[0])
