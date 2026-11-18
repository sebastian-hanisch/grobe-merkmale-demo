"""Gemeinsame Fixtures: die Mini-Gruppe, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mini-Welt: 2 Krankheiten, EINE Gruppe (Oberbegriff) mit 3 Kindern. g[d][c] = Wahrscheinlichkeit, dass Kind c bei Krankheit d aufgeschrieben wird.
MINI_G = np.array([[0.90, 0.50, 0.10],
                   [0.10, 0.40, 0.80]])
MINI_PRIOR = np.array([0.6, 0.4])
MINI_U = 0.4


def enumerate_observations(g_row, u):
    """Alle 3^3 Zustände der Kinder (0 = nicht aufgeschrieben, 1 = genau, 2 = nur als Oberbegriff) mit ihrer Wahrscheinlichkeit, zusammengefasst zur Beobachtung (L, F):
    L = Tupel der genau aufgeschriebenen Kinder, F = mindestens ein Kind steht als Oberbegriff. Reine Aufzählung ohne die Formel der Demo."""
    out = {}
    for zustand in itertools.product((0, 1, 2), repeat=len(g_row)):
        p = 1.0
        for z, g in zip(zustand, g_row):
            p *= {0: 1.0 - g, 1: g * (1.0 - u), 2: g * u}[z]
        obs = (tuple(int(z == 1) for z in zustand), int(any(z == 2 for z in zustand)))
        out[obs] = out.get(obs, 0.0) + p
    return out


@pytest.fixture
def mini_g():
    """g in der Form der Demo: (Krankheiten, Gruppen, Kinder)."""
    return MINI_G[:, None, :].copy()


@pytest.fixture
def mini_prior():
    return MINI_PRIOR.copy()


@pytest.fixture
def mini_u():
    return MINI_U


@pytest.fixture
def beobachtungen():
    """Alle möglichen Beobachtungen (L, F) der Mini-Gruppe: (L, F) -> Wahrscheinlichkeit je Krankheit (Aufzählung)."""
    per_d = [enumerate_observations(MINI_G[d], MINI_U) for d in range(MINI_G.shape[0])]
    keys = sorted(set().union(*[set(p) for p in per_d]))
    return {k: [p.get(k, 0.0) for p in per_d] for k in keys}
