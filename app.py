"""Grobe Merkmale – wenn ein Symptom nur als Oberbegriff in der Akte steht - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Lehrmodell mit erfundenen Daten, keine medizinische Beratung. Aufbau nach DEMO-PLAYBOOK §2 (Fall-Demo).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import gmk_constants as C
import gmk_evaluation as ev
import gmk_live as L
import gmk_pdf_export as PDF
import gmk_presets as P
import gmk_results as RES
import gmk_ui_panel as UI
import gmk_visualization as V
from gmk_layout import dunkles_design, kopf_css, tabellen_css

st.set_page_config(page_title="Grobe Merkmale – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Messe fünf Welten ...")
def _measure(settings):
    return L.measure(settings)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in gmk_layout.py).
st.html(kopf_css(dunkles_design()))
st.html(tabellen_css())
with st.container(key="kopf"):
    st.title("🌳 Grobe Merkmale – wenn ein Symptom nur als Oberbegriff in der Akte steht")
st.markdown(
    """
Aus den **dokumentierten Merkmalen** (Symptomen) eine Rangliste möglicher **seltener Krankheiten** erstellen, wenn ein Teil der Merkmale nur als **Oberbegriff** in der Akte steht
(„Augenauffälligkeit“ statt des genauen Befunds). Die Demo misst, was es kostet, solche groben Angaben zu **überlesen**, ob die einfache Heuristik **„Oberbegriff = alle Kinder“** den Verlust zurückholt
und wie ein **Resnik-artiger** Ähnlichkeitswert (eigene Fassung) abschneidet.

Der Vergleichsmaßstab ist der **exakte Ranker**: das **Orakel** des Erzeugers dieser Welt, also der Ranker, der die Regeln kennt, nach denen die Akten erzeugt wurden (die Häufigkeiten und den Anteil grober Angaben).
Er ist nur in diesem Modell eine Obergrenze.

**Lehrmodell mit erfundenen Daten, keine medizinische Beratung.** Alle Aussagen gelten für diese Welt, nicht für echte Krankheiten.

Weiter unten: drei **Kernfragen** und der Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich**.
"""
)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

P.load_permalink_settings()
P.init_session_state_defaults()


def _stufe(label_text, state_key, help_text, **kw):
    return st.select_slider(label_text, options=P.options(state_key), key=state_key, format_func=lambda v, k=state_key: P.label(k, v), help=help_text, **kw)


with st.sidebar:
    P.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    nd = _stufe("Zahl der Krankheiten", "nd_select", "Wie viele Krankheiten im Ranking stehen. Mehr Krankheiten heißt mehr Verwechslungen.")
    q = _stufe("Dokumentationsverlust q", "q_select", "Anteil der vorhandenen Merkmale, der gar nicht aufgeschrieben wird.")
    st.markdown("**Grobe Angaben**")
    u = _stufe("Anteil grober Angaben u", "u_select", "Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff ihrer Gruppe (fünf Merkmale je Oberbegriff) in der Akte stehen.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed der Messung", *P.bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welten würfeln", width="stretch", on_click=P.randomize_seed)

P.sync_query_params({"nd_select": nd, "q_select": q, "u_select": u, "seed_input": int(seed)})

settings = L.Settings(nd=int(nd), q=float(q), u=float(u), seed=int(seed))
m = _measure(settings)
kz = ev.kennzahlen(m)
(v1_kind, v1_text), (v2_kind, v2_text), (v3_kind, v3_text) = ev.verdicts(settings, kz)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Eine Messung (live)")
st.caption(f"Diese Messung: {len(m.runs)} Welten × {m.n_patients} Patienten (Zufalls-Seed {settings.seed}), Mittel ± Streuung über die Welten; gerankt werden "
           f"{ev.pct(kz['share_ranked'])} % der Patienten (die mit mindestens einer Angabe in der Akte). Ein einzelner Patient belegt nichts. Die Verdikte unter den Kernfragen rechnen live aus dieser Messung "
           "(gepaart über die Welten); die Abbildungen der Kernfragen stammen aus der vorgerechneten Messreihe.")
UI.metrics_row(kz)
st.plotly_chart(V.fig_recall(kz["summary"]), width="stretch", key="main_chart", config=V.CONFIG)
st.caption("Recall@k: Anteil der gerankten Patienten, bei denen die wahre Krankheit unter den ersten k der Rangliste steht (Gleichstände zufällig aufgelöst). "
           "Exakt ist das Orakel dieser Welt, eine Obergrenze nur innerhalb dieses Modells.")

if st.button("📄 PDF der Messung erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = PDF.build_pdf(settings, kz, ((v1_kind, v1_text), (v2_kind, v2_text), (v3_kind, v3_text)), len(m.runs), m.n_patients)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="grobe-merkmale-messung.pdf", mime="application/pdf", key="pdf_download")

st.markdown("### 🧍 Ein Beispielpatient")
UI.patient_panel(m, len(m.runs[0].example["d"]))

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

meta = RES.load()["meta"]
vorgerechnet = (f"Vorgerechnet: {meta['cells']} Zellen, je zwei Mengen von {meta['worlds_per_set']} Welten (Zufalls-Seed {meta['sets']['A']} und {meta['sets']['B']}), "
                f"{meta['patients_per_world']} Patienten je Welt; gewählt sind {settings.nd} Krankheiten und q = {ev.num(settings.q)}.")

st.markdown("---")
st.markdown("## ❓ Kernfrage 1: Was kostet es, grobe Angaben zu überlesen?")
getattr(st, v1_kind)(v1_text)
st.plotly_chart(V.fig_u_curve(settings.nd, settings.q, settings.u), width="stretch", key="u_chart", config=V.CONFIG)
st.caption(vorgerechnet + f" Schwellen des Verdikts (Recall@5, exakt minus überlesen): unter {ev.pct(C.LOSS_SMALL)} Punkte, "
           f"von {ev.pct(C.LOSS_SMALL)} bis {ev.pct(C.LOSS_LARGE)} Punkte und mehr als {ev.pct(C.LOSS_LARGE)} Punkte; die Schwellen sind gesetzt, nicht abgeleitet.")

st.markdown("## ❓ Kernfrage 2: Holt „Oberbegriff = alle Kinder“ den Verlust zurück?")
getattr(st, v2_kind)(v2_text)
st.plotly_chart(V.fig_pairs(settings.nd, settings.q, ("X-I", "A-I", "X-A"), f"Wie viel die Heuristik zurückholt ({settings.nd} Krankheiten, q = {ev.num(settings.q)}; vorgerechnet, 10 Welten)"),
                width="stretch", key="pairs_chart_heur", config=V.CONFIG)
st.caption(vorgerechnet + " Gepaarte Differenzen über die Welten; die Fehlerbalken sind zwei Standardfehler. Die Heuristik kennt u nicht und rechnet mit den ungeteilten Häufigkeiten.")

st.markdown("## ❓ Kernfrage 3: Hilft ein Resnik-artiger Ähnlichkeitswert?")
getattr(st, v3_kind)(v3_text)
st.plotly_chart(V.fig_pairs(settings.nd, settings.q, ("R-I", "R-A", "X-R"), f"Resnik-artig im Vergleich ({settings.nd} Krankheiten, q = {ev.num(settings.q)}; vorgerechnet, 10 Welten)"),
                width="stretch", key="pairs_chart_resnik", config=V.CONFIG)
st.caption(vorgerechnet + " Der Resnik-artige Ranker ist eine eigene, einfache Fassung (Definition im Tab „🔎 Resnik-artig“), kein Nachbau von Phenomizer oder einem anderen Verfahren aus der Literatur.")

st.markdown("---")

# --- Methodenvergleich ---------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    tab_r, tab_h, tab_s = st.tabs(["📏 Ranker", "🌳 Hierarchie und Akte", "🔎 Resnik-artig"])
    with tab_r:
        st.markdown(UI.ranker_table(kz["summary"]))
        st.markdown(
            "- **exakt (Orakel des Erzeugers):** Wahrscheinlichkeit der Akte unter dem Erzeuger dieser Welt, je Gruppe aus genauen Merkmalen und Oberbegriffszeichen; kennt die Häufigkeiten und den Anteil u.\n"
            "- **nur genaue Begriffe (Oberbegriff überlesen):** dieselbe Rechnung, aber das Oberbegriffszeichen wird nicht gelesen: jedes Merkmal steht mit Wahrscheinlichkeit g (1 − u) als genauer Begriff in der Akte oder nicht.\n"
            "- **Oberbegriff = alle Kinder (Heuristik):** ein Oberbegriff zählt, als stünden alle fünf Merkmale der Gruppe in der Akte; Naive Bayes mit den ungeteilten Häufigkeiten g, ohne Wissen um u.\n"
            "- **Resnik-artig (eigene Fassung):** Summe über die Begriffe der Akte von einem Ähnlichkeitswert zur Krankheit (siehe Tab „🔎 Resnik-artig“); kein Vorab, keine Häufigkeiten, keine Abwesenheit.\n"
            "- **naives Überlesen:** nur die genauen Begriffe, Naive Bayes mit den ungeteilten Häufigkeiten g; kennt u nicht (das Gegenstück zum Überlesen, das mit g (1 − u) rechnet)."
        )
        st.caption("Streuung = Standardabweichung der Mittel der einzelnen Welten (n − 1). Gleichstände werden als Wahrscheinlichkeit gezählt, nicht als Treffer oder Fehlschlag.")
    with tab_h:
        st.markdown(
            f"Die {C.NS} Merkmale bilden {C.N_GROUPS} **Gruppen** zu je {C.GROUP_SIZE} Merkmalen (Merkmale M1 bis M{C.GROUP_SIZE} sind Gruppe 1 und so weiter); jede Gruppe hat einen **Oberbegriff**. "
            "Ein aufgeschriebenes Merkmal steht mit Wahrscheinlichkeit 1 − u genau in der Akte und mit Wahrscheinlichkeit u nur als Oberbegriff seiner Gruppe. "
            "Mehrere grobe Angaben derselben Gruppe ergeben einen einzigen Oberbegriff in der Akte: aus ihm ist nicht zu erkennen, wie viele Merkmale dahinterstehen."
        )
        rows = ["| Anteil grober Angaben u | " + " | ".join(C.RANKER_SHORT[r] for r in C.RANKERS) + " | grob aufgeschrieben (Anteil der aufgeschriebenen Merkmale) | gerankte Patienten |",
                "|---|" + "---|" * (len(C.RANKERS) + 2)]
        for u_v in C.U_OPTIONS:
            c = RES.cell(RES.Settings(nd=settings.nd, q=settings.q, u=u_v))
            rows.append(f"| {P.label('u_select', u_v)} | " + " | ".join(ev.pct(RES.pooled(c, r)) for r in C.RANKERS) + f" | {ev.pct((c['A']['share_coarse'] + c['B']['share_coarse']) / 2)} % | "
                        f"{ev.pct((c['A']['share_ranked'] + c['B']['share_ranked']) / 2)} % |")
        st.markdown(f"Recall@5 in %, vorgerechnet für {settings.nd} Krankheiten und q = {ev.num(settings.q)} (Mittel der beiden Mengen, zehn Welten):\n\n" +"\n".join(rows))
        st.caption("Patienten ohne jede Angabe in der Akte werden nicht gerankt; die Recall-Werte gelten für die gerankten Patienten. "
                   "Bei u = 0 stehen exakt, überlesen, alle Kinder und naives Überlesen auf demselben Wert, weil es keine groben Angaben gibt.")
    with tab_s:
        st.markdown(
            "Der **Resnik-artige** Ranker ist hier eine **eigene, einfache Fassung**, kein Nachbau eines Verfahrens aus der Literatur:\n\n"
            f"- Ein Merkmal gehört zur **Annotation** einer Krankheit, wenn seine Häufigkeit im Profil über {ev.num(C.SIMPLE_THRESHOLD)} liegt; der Oberbegriff gehört dazu, wenn eines seiner fünf Merkmale dazugehört.\n"
            "- Der **Informationsgehalt** eines Begriffs ist der negative natürliche Logarithmus des Anteils der Krankheiten, die ihn annotieren (seltene Begriffe zählen mehr); die Wurzel (alle Merkmale zusammen) hat 0.\n"
            "- Ein genaues Merkmal der Akte trifft eine Krankheit mit seinem eigenen Informationsgehalt, wenn sie es annotiert; sonst mit dem seines Oberbegriffs, wenn sie den Oberbegriff annotiert (gemeinsamer Vorfahre), sonst mit 0.\n"
            "- Ein Oberbegriff der Akte trifft eine Krankheit mit dem Informationsgehalt des Oberbegriffs, wenn sie ihn annotiert, sonst mit 0.\n"
            "- Die Punktzahl einer Krankheit ist die **Summe** dieser Treffer über alle Begriffe der Akte. Kein Vorab, keine Häufigkeiten, keine Abwesenheit als Hinweis."
        )
        cells = [RES.cell(RES.Settings(nd=settings.nd, q=settings.q, u=u_v)) for u_v in C.U_OPTIONS]
        r0 = RES.pooled(cells[0], "R")
        x0 = RES.pooled(cells[0], "X")
        st.markdown(f"Schon ohne grobe Angaben (u = 0) liegt der Resnik-artige Ranker bei {ev.pct(r0)} % Recall@5, der exakte bei {ev.pct(x0)} % (vorgerechnet für {settings.nd} Krankheiten und q = {ev.num(settings.q)}; zehn Welten).")
        st.caption("Deutung, nicht gemessen: Die Fassung benutzt weder die Häufigkeiten der Profile noch das Vorab noch nicht aufgeschriebene Merkmale als Hinweis; das kann den Abstand schon bei u = 0 erklären. "
                   "Wie stark ein ausgebautes Verfahren mit Profilhäufigkeiten abschneidet, zeigt diese Messung nicht.")

with st.expander("Wie funktioniert diese Demo?"):
    c_def = RES.cell(RES.BASE)
    c_q6 = RES.cell(RES.Settings(q=0.6))
    st.markdown(
        f"""
**Die Welt.** {C.NS} Merkmale mit Zipf-verteilter Beliebtheit: beliebte Merkmale kommen in vielen Krankheitsprofilen und im Hintergrund vor und tragen wenig Information. Jede Krankheit hat
{C.K_MIN_FEAT} bis {C.K_MAX_FEAT} Profilmerkmale mit Häufigkeit 0,95, 0,6, 0,3 oder 0,1. Ein vorhandenes Merkmal wird mit Wahrscheinlichkeit 1 − q aufgeschrieben, dazu kommt eine Hintergrundrate je Merkmal.

**Grobe Angaben.** Die {C.NS} Merkmale bilden {C.N_GROUPS} Gruppen zu je {C.GROUP_SIZE}. Jedes aufgeschriebene Merkmal steht mit Wahrscheinlichkeit u nur als Oberbegriff seiner Gruppe in der Akte;
mehrere grobe Angaben derselben Gruppe ergeben einen Oberbegriff.

**Patienten.** Krankheit gleichverteilt, Merkmale unabhängig gegeben die Krankheit. Patienten ohne jede Angabe in der Akte werden nicht gerankt: in den vorgerechneten Zellen bleiben
{ev.pct((c_def['A']['share_ranked'] + c_def['B']['share_ranked']) / 2)} Prozent der Patienten in der Voreinstellung und {ev.pct((c_q6['A']['share_ranked'] + c_q6['B']['share_ranked']) / 2)} Prozent bei Dokumentationsverlust 0,6; die Recall-Werte gelten für die gerankten Patienten.

**Messung.** Je Einstellung {C.N_WORLDS} Welten mit je {C.N_PATIENTS} Patienten (SplitMix64: der Zufallsstrom ist auf jeder Plattform derselbe; Gleitkomma-Rundung kann die Recall-Werte minimal verschieben). Der Recall ist ein Erwartungswert über die zufällige Auflösung von Gleichständen.

**Auswertung gepaart.** Ob ein Ranker vor einem anderen liegt, entscheidet der Mittelwert der Differenzen über die Welten gegen seinen Standardfehler (zwei Standardfehler als Schwelle).
"""
    )
    st.markdown("#### 🚧 Wo die Annahmen enden")
    st.markdown(
        "| Annahme der Demo | Was in der Wirklichkeit anders ist |\n|---|---|\n"
        "| Eine Ebene Oberbegriffe, fünf Kinder je Oberbegriff, alle gleich groß | echte Merkmalshierarchien (HPO) sind tief, verzweigt und ungleich |\n"
        "| Jedes Merkmal wird mit derselben Wahrscheinlichkeit u grob aufgeschrieben | ob grob oder genau dokumentiert wird, hängt von Person, Merkmal und Kontext ab |\n"
        "| Der exakte Ranker kennt u und alle Häufigkeiten | in echten Daten gibt es kein Orakel; u müsste geschätzt werden |\n"
        "| Merkmale unabhängig gegeben die Krankheit | echte Merkmale sind abhängig |\n"
        f"| {C.NS} Merkmale, 50 oder 200 Krankheiten | echte Wissensbasen haben mehrere Tausend Krankheiten und Merkmale |"
    )

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Erfundene Welt mit Naive-Bayes-Struktur (Merkmale unabhängig gegeben die Krankheit), ein Oberbegriff je fünf Merkmale, derselbe Anteil u für alle Merkmale. Parameter (Häufigkeiten, Hintergrund,
Dokumentationsverlust) sind gesetzt, nicht aus Daten geschätzt.

**Grenzen.** Der exakte Ranker gehört zum Erzeuger und ist nur in diesem Modell eine Obergrenze. Fünf Welten je Messung: die Streuung über die Welten ist ausgewiesen, Unterschiede unter zwei Standardfehlern
werden nicht gedeutet. Der Resnik-artige Ranker ist eine eigene einfache Fassung; Aussagen über ihn gelten für diese Fassung, nicht für Ähnlichkeitsmaße im Allgemeinen. Drei Ebenen der Hierarchie und
ungleiche Anteile grober Angaben sind nicht gerechnet.

**Bekannte Probleme und Unsicherheiten.** Nur an erfundenen Welten geprüft; keine echten Patientendaten, keine echte Wissensbasis (HPO, Orphanet). Die Aussagen gelten für die gemessenen Zellen und die
gemessene Regel jedes Rankers, nicht für die Methodenklasse. **Keine medizinische Beratung, keine Diagnoseempfehlung.**
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben: die Akte eines Patienten (genaue Merkmale und Oberbegriffe), die Häufigkeiten $g$ und der Anteil $u$ grober Angaben. Gesucht: eine Rangliste der Krankheiten $d$.

| Symbol | Bedeutung |
|---|---|
| $d$ | Krankheit |
| $G$, $c$ | Gruppe (Oberbegriff) und eines ihrer Merkmale (Kind) |
| $g_{dc}$ | Wahrscheinlichkeit, dass das Merkmal $c$ bei Krankheit $d$ aufgeschrieben wird |
| $q$ | Dokumentationsverlust |
| $u$ | Anteil der aufgeschriebenen Merkmale, die nur als Oberbegriff stehen |
| $L_G$ | Menge der Kinder von $G$, die genau in der Akte stehen |
| $F_G \in \{0,1\}$ | der Oberbegriff von $G$ steht in der Akte |
| $\pi_d$ | Vorab der Krankheit $d$ (hier gleichverteilt) |
        """
    )
    st.markdown("Erzeuger (Dokumentationswahrscheinlichkeit, mit Hintergrundrate $b_c = 0{,}01 + 0{,}2\\,(w_c / w_{\\max})^2$, $w_c \\propto \\mathrm{Rang}^{-0{,}8}$, und Profilhäufigkeit $f_{dc}$):")
    st.latex(r"g_{dc} = 1 - \bigl(1 - f_{dc}(1 - q)\bigr)\bigl(1 - b_c\bigr)")
    st.markdown("Ein aufgeschriebenes Merkmal steht genau in der Akte mit Wahrscheinlichkeit $1 - u$, nur als Oberbegriff mit $u$. Exakter Ranker (Wahrscheinlichkeit der Akte, Produkt über die Gruppen):")
    st.latex(r"P(\text{Akte} \mid d) = \prod_{G} \Bigl[\, \prod_{c \in L_G} g_{dc}(1 - u) \cdot \Phi_{dG} \Bigr]")
    st.markdown("mit dem Faktor für die Kinder außerhalb von $L_G$ (er hängt davon ab, ob der Oberbegriff in der Akte steht):")
    st.latex(r"\Phi_{dG} = \begin{cases} \prod_{c \notin L_G} (1 - g_{dc}) & F_G = 0 \\[4pt] \prod_{c \notin L_G} (1 - g_{dc} + g_{dc}\,u) \;-\; \prod_{c \notin L_G} (1 - g_{dc}) & F_G = 1 \end{cases}")
    st.markdown("Oberbegriff überlesen (das Zeichen $F_G$ wird nicht gelesen; jedes Merkmal steht mit $g(1-u)$ genau in der Akte oder nicht):")
    st.latex(r"\log P(d \mid \text{Akte}) = \log \pi_d + \sum_{G}\sum_{c} \bigl[ \mathbf 1_{c \in L_G} \log\bigl(g_{dc}(1-u)\bigr) + \mathbf 1_{c \notin L_G} \log\bigl(1 - g_{dc}(1-u)\bigr) \bigr] + \text{const.}")
    st.markdown("Oberbegriff = alle Kinder (mit $\\tilde x_c = 1$, wenn $c \\in L_G$ oder $F_G = 1$):")
    st.latex(r"\log P(d \mid \text{Akte}) = \log \pi_d + \sum_{G}\sum_{c} \bigl[ \tilde x_c \log g_{dc} + (1 - \tilde x_c) \log(1 - g_{dc}) \bigr] + \text{const.}")
    st.markdown("Resnik-artig (Informationsgehalt $\\mathrm{IC}(t) = -\\ln\\bigl(\\text{Anteil der Krankheiten, die } t \\text{ annotieren}\\bigr)$; $\\operatorname{sim}(t, d)$ ist $\\mathrm{IC}$ des besten gemeinsamen Vorfahren von $t$ mit den Annotationen von $d$, 0 für die Wurzel):")
    st.latex(r"\mathrm{score}(d) = \sum_{t \,\in\, \text{Akte}} \operatorname{sim}(t, d)")
    st.markdown("Recall@$k$ mit zufälliger Auflösung von Gleichständen ($r_i$ kleinster Rang der wahren Krankheit, $t_i$ Zahl der Gleichstände mit ihr):")
    st.latex(r"\mathrm{Recall@}k = \frac{1}{N}\sum_{i=1}^{N} \min\Bigl(1,\; \max\Bigl(0,\; \frac{k - r_i + 1}{t_i + 1}\Bigr)\Bigr)")
    st.caption("Der Formelblock zeigt Erzeuger, exakten Ranker, Überlesen, Heuristik, Resnik-artigen Ranker und Recall (das naive Überlesen rechnet wie die Heuristik, nur mit den genauen Begriffen). Nicht in den Gleichungen stehen: die Untergrenze 10⁻⁶ der Häufigkeiten in den Logarithmen "
               "und die Zufallsziehungen (Profilmerkmale ohne Zurücklegen proportional zur Beliebtheit, Häufigkeiten 0,95 / 0,6 / 0,3 / 0,1 je Profilmerkmal mit den Wahrscheinlichkeiten 15 / 25 / 35 / 25 Prozent gezogen); sie stehen im README und im Code.")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
