"""Darstellung einzelner Bausteine der Seite (Streamlit): Kennzahlen, Beispielpatient, Tabelle der Ranker. Rechnet nichts selbst."""

import streamlit as st

import gmk_constants as C
import gmk_evaluation as ev
import gmk_visualization as V


def metrics_row(k: dict):
    """Vier Kennzahlen im 2 x 2-Raster; Deltas immer „mein Wert minus Referenz“, höher ist besser."""
    s = k["summary"]
    x5, i5, a5 = s["X"][5][0], s["I"][5][0], s["A"][5][0]
    c1, c2 = st.columns(2)
    c1.metric("Recall@5 exakt", f"{ev.pct(x5)} %", delta="kennt die Hierarchie (Orakel des Erzeugers)", delta_color="off")
    c2.metric("Recall@5 Oberbegriff überlesen", f"{ev.pct(i5)} %", delta=f"{ev.spct(i5 - x5)} Punkte gegen exakt", delta_color="normal")
    c3, c4 = st.columns(2)
    c3.metric("Recall@5 Oberbegriff = alle Kinder", f"{ev.pct(a5)} %", delta=f"{ev.spct(a5 - i5)} Punkte gegen Überlesen", delta_color="normal")
    c4.metric("Grob aufgeschriebene Merkmale", f"{ev.pct(k['share_coarse'])} %", delta="Anteil der aufgeschriebenen Merkmale", delta_color="off")


def ranker_table(summary: dict) -> str:
    """Markdown-Tabelle: Recall@1, @5, @10 und reziproker mittlerer Rang je Ranker (Mittel ± Streuung über die Welten, in Prozent)."""
    rows = ["| Ranker | Recall@1 | Recall@5 | Recall@10 | reziproker mittlerer Rang |", "|---|---|---|---|---|"]
    for r in C.RANKERS:
        cells = [f"{ev.pct(summary[r][k][0])} ± {ev.pct(summary[r][k][1])}" for k in (1, 5, 10, "mrr")]
        rows.append(f"| {C.RANKER_NAMES[r]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def patient_panel(m, n_example: int):
    """Beispielpatient: Akte (genaue Begriffe und Oberbegriffe), oberste fünf Krankheiten je Ranker, Posterior des exakten Rankers; die wahre Krankheit erst nach „Auflösen“."""
    idx = st.slider("Patient", 1, n_example, 1, key="patient_idx", help="Die ersten Patienten der ersten Welt.") - 1
    reveal = st.toggle("Wahre Krankheit zeigen", key="patient_reveal", value=False)
    v = ev.example_view(m, idx)
    exact = ", ".join(v["exact"]) or "keine"
    coarse = "; ".join(v["coarse"]) or "keine"
    st.markdown(f"**Genaue Merkmale in der Akte ({len(v['exact'])}):** {exact}")
    st.markdown(f"**Nur als Oberbegriff in der Akte ({len(v['coarse'])}):** {coarse}")
    true_id = v["true"] if reveal else None
    rows = ["| Ranker | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|"]
    for r in C.RANKERS:
        ids = v["top"][r]["ids"]
        cells = [f"K{i + 1}" + (" ✅" if true_id is not None and i == true_id else "") for i in ids]
        rows.append(f"| {C.RANKER_SHORT[r]} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(rows))
    if reveal:
        ranks = ", ".join(f"{C.RANKER_SHORT[r]}: Rang {v['top'][r]['rank_true']}" for r in C.RANKERS)
        st.caption(f"Wahre Krankheit: K{v['true'] + 1}. Ihr Rang je Ranker: {ranks}.")
    st.plotly_chart(V.fig_patient(v["posterior_x"], true_id), width="stretch", key=f"patient_chart_{idx}_{reveal}", config=V.CONFIG)
