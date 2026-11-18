"""Reproduziert data/gmk_results.json: Messreihe über alle Zellen (zwei Mengen von je fünf Welten).

Aufruf (im Demo-Ordner): python tools/sweep.py [--check]   (nicht in der CI; einige Minuten)
Mit --check wird nichts geschrieben, sondern die Datei gegen eine Neurechnung der Voreinstellung verglichen."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import gmk_constants as C  # noqa: E402
import gmk_live as L  # noqa: E402
import gmk_results as RES  # noqa: E402

PAIRS = (("X", "I"), ("X", "A"), ("X", "R"), ("A", "I"), ("R", "I"), ("R", "A"), ("I", "N"), ("X", "N"))


def cell_result(s: L.Settings) -> dict:
    m = L.measure(s, n_example=0)
    summ = L.summarize(m)
    return {"summary": {r: {str(k): list(v) for k, v in summ[r].items()} for r in C.RANKERS},
            "share_coarse": L.share_coarse(m), "share_ranked": L.share_ranked(m),
            "paired": {f"{a}-{b}": list(L.paired_diff(m, a, b)) for a, b in PAIRS}}


def main(check: bool = False):
    t0 = time.perf_counter()
    cells = {}
    for s in RES.sweep_settings():
        row = {"settings": {"nd": s.nd, "q": s.q, "u": s.u}}
        for name, seed in C.SWEEP_SEEDS.items():
            row[name] = cell_result(L.Settings(nd=s.nd, q=s.q, u=s.u, seed=seed))
        cells[s.key()] = row
        print(f"{s.key():<22} R@5 exakt {RES.pooled(row, 'X'):5.1f} überlesen {RES.pooled(row, 'I'):5.1f} alle Kinder {RES.pooled(row, 'A'):5.1f} "
              f"Resnik {RES.pooled(row, 'R'):5.1f}", flush=True)
    data = {"meta": {"worlds_per_set": C.N_WORLDS, "patients_per_world": C.N_PATIENTS, "sets": C.SWEEP_SEEDS, "cells": len(cells),
                     "rankers": list(C.RANKERS), "pairs": [f"{a}-{b}" for a, b in PAIRS], "seconds": round(time.perf_counter() - t0, 1)},
            "cells": cells}
    if check:
        old = json.loads(RES.DATA.read_text(encoding="utf-8"))
        key = L.Settings().key()
        a_old = old["cells"][key]["A"]["summary"]["X"]["5"][0]
        a_new = cells[key]["A"]["summary"]["X"]["5"][0]
        print("Prüfung Voreinstellung Menge A, exakt, Recall@5:", a_old, a_new, "gleich" if abs(a_old - a_new) < 1e-9 else "ABWEICHUNG")
        return
    RES.DATA.parent.mkdir(exist_ok=True)
    RES.DATA.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {RES.DATA} ({len(cells)} Zellen, {data['meta']['seconds']} s)")


if __name__ == "__main__":
    main("--check" in sys.argv)
