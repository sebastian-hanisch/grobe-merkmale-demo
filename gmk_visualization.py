"""Plotly-Abbildungen (alle Achsen gesperrt, transparenter Hintergrund). Zahlen kommen aus der Messung, nie aus dem Text."""

import plotly.graph_objects as go

import gmk_constants as C
import gmk_results as RES
from gmk_evaluation import num

CONFIG = {"displayModeBar": False}
_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=50, r=20, t=40, b=60), legend=dict(orientation="h", y=-0.25))

PAIR_NAMES = {
    "X-I": "exakt minus überlesen", "A-I": "alle Kinder minus überlesen", "X-A": "exakt minus alle Kinder",
    "I-N": "überlesen minus naives Überlesen", "X-N": "exakt minus naives Überlesen",
    "R-I": "Resnik-artig minus überlesen", "R-A": "Resnik-artig minus alle Kinder", "X-R": "exakt minus Resnik-artig",
}
PAIR_COLORS = {"X-I": C.COLOR_X, "A-I": C.COLOR_A, "X-A": "#7a5cc7", "R-I": C.COLOR_R, "R-A": "#a0508c", "X-R": "#888888", "I-N": "#c77700", "X-N": "#5b6b8c"}


def _lock(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def fig_recall(summary: dict) -> go.Figure:
    """Recall@1, @5, @10 je Ranker (Balken) mit Streuung über die Welten (Fehlerbalken)."""
    fig = go.Figure()
    xs = ["Recall@1", "Recall@5", "Recall@10"]
    for r in C.RANKERS:
        ys = [summary[r][k][0] for k in (1, 5, 10)]
        err = [summary[r][k][1] for k in (1, 5, 10)]
        fig.add_bar(name=C.RANKER_SHORT[r], x=xs, y=ys, error_y=dict(type="data", array=err, visible=True), marker_color=C.RANKER_COLORS[r],
                    hovertemplate="%{y:.1f} %<extra>" + C.RANKER_SHORT[r] + "</extra>")
    fig.update_layout(barmode="group", yaxis_title="Anteil der Patienten in %", yaxis_range=[0, 105], title="Recall der Ranker (Mittel ± Streuung über die Welten)", **_LAYOUT)
    return _lock(fig)


def fig_u_curve(nd: int, q: float, current_u: float) -> go.Figure:
    """Recall@5 über den Anteil grober Angaben u (vorgerechnet, Mittel der beiden Mengen zu je fünf Welten) für die gewählten Krankheiten und den gewählten Verlust."""
    cells = [RES.cell(RES.Settings(nd=nd, q=q, u=u)) for u in C.U_OPTIONS]
    labels = [num(u) for u in C.U_OPTIONS]
    fig = go.Figure()
    for r in C.RANKERS:
        fig.add_scatter(x=labels, y=[RES.pooled(c, r) for c in cells], mode="lines+markers", name=C.RANKER_SHORT[r],
                        line=dict(color=C.RANKER_COLORS[r], dash="dot" if r == "R" else "solid"))
    fig.add_vline(x=labels.index(num(current_u)), line_dash="dash", line_color="rgba(128,128,128,0.7)", annotation_text="gewählt", annotation_position="top")
    fig.update_layout(xaxis_title="Anteil u der aufgeschriebenen Merkmale, die nur als Oberbegriff stehen", yaxis_title="Recall@5 in %", yaxis_range=[0, 100],
                      title=f"Recall@5 nach Anteil grober Angaben ({nd} Krankheiten, q = {num(q)}; vorgerechnet, 10 Welten)", **_LAYOUT)
    return _lock(fig)


def fig_pairs(nd: int, q: float, pairs: tuple, title: str) -> go.Figure:
    """Gepaarte Differenzen (Recall@5, Punkte) über u, mit zwei Standardfehlern als Fehlerbalken (vorgerechnet, zehn Welten); u = 0 entfällt (alle Differenzen zwischen exakt, überlesen und alle Kinder sind 0)."""
    us = [u for u in C.U_OPTIONS if u > 0]
    cells = [RES.cell(RES.Settings(nd=nd, q=q, u=u)) for u in us]
    labels = [f"u = {num(u)}" for u in us]
    fig = go.Figure()
    for p in pairs:
        vals = [RES.pooled_pair(c, p) for c in cells]
        fig.add_bar(name=PAIR_NAMES[p], x=labels, y=[v[0] for v in vals], error_y=dict(type="data", array=[C.SD_FACTOR * v[1] for v in vals], visible=True),
                    marker_color=PAIR_COLORS[p], hovertemplate="%{y:.1f} Punkte<extra>" + PAIR_NAMES[p] + "</extra>")
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.7)")
    fig.update_layout(barmode="group", yaxis_title="Unterschied im Recall@5 in Punkten", title=title, **_LAYOUT)
    return _lock(fig)


def fig_patient(posterior: dict, true_id: int | None) -> go.Figure:
    """Posterior des exakten Rankers für die obersten Krankheiten eines Beispielpatienten; die wahre Krankheit ist hervorgehoben, wenn sie aufgelöst ist."""
    ids = list(posterior)
    colors = ["#c0392b" if (true_id is not None and i == true_id) else C.COLOR_X for i in ids]
    fig = go.Figure(go.Bar(x=[posterior[i] for i in ids], y=[f"Krankheit {i + 1}" for i in ids], orientation="h", marker_color=colors,
                           hovertemplate="%{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="Posterior des exakten Rankers", xaxis_range=[0, 1], yaxis_autorange="reversed", title="Die obersten fünf Krankheiten (exakter Ranker)",
                      **{**_LAYOUT, "legend": None})
    return _lock(fig)
