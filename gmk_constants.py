"""Konstanten: Regler-Stufen, feste Annahmen der Welt, Farben und Presets (Grobe Angaben, Lehrmodell mit erfundenen Daten).

Die Welt (Krankheitsprofile, Hintergrund, Dokumentationsverlust) ist dieselbe wie im Phänotyp-Ranking; die Konstanten bis `SIMPLE_THRESHOLD` und
`CALIB_EDGES` brauchen die kopierten Module gmk_world.py und gmk_metrics.py unter denselben Namen."""

# --- Welt (Annahmen des Erzeugers) -------------------------------------------------------------------------------------------
NS = 100                      # Zahl der Merkmale (Symptome)
POP_EXP = 0.8                 # Zipf-Exponent der Beliebtheit der Merkmale: beliebte Merkmale kommen in vielen Profilen und im Hintergrund vor
FREQ = (0.95, 0.6, 0.3, 0.1)  # Häufigkeit eines Profilmerkmals bei einer Person mit der Krankheit
FREQ_P = (0.15, 0.25, 0.35, 0.25)  # Anteile dieser Häufigkeiten an den Profilmerkmalen
K_MIN_FEAT, K_MAX_FEAT = 4, 12     # Zahl der Profilmerkmale je Krankheit (gleichverteilt, beide Grenzen inklusive)
BG_BASE, BG_SLOPE, BG_POWER = 0.01, 0.2, 2   # Hintergrundrate eines Merkmals = 0,01 + 0,2 * (Beliebtheit / größte Beliebtheit)^2
PRIOR_ZIPF_EXP = 1.0          # (von gmk_world benutzt; die App rechnet nur mit gleichverteiltem Vorab)
LAPLACE_A = 0.5               # (von gmk_world benutzt; die App schätzt keine Häufigkeiten)
SHRINK_A = (3, 10)            # (von gmk_world benutzt)
PROB_FLOOR = 1e-6             # Untergrenze der Häufigkeiten im Logarithmus
SIMPLE_THRESHOLD = 0.05       # Ein Merkmal gehört zum Profil einer Krankheit (Annotation), wenn seine Häufigkeit im Profil über dieser Schwelle liegt
CALIB_EDGES = (0.0, 0.5, 0.9, 0.99, 1.0000001)

# --- Hierarchie -------------------------------------------------------------------------------------------------------------------
GROUP_SIZE = 5                # Kinder je Oberbegriff (Merkmale 5g bis 5g + 4 gehören zur Gruppe g)
N_GROUPS = NS // GROUP_SIZE   # 20 Oberbegriffe
IC_MIN = 0.0                  # Informationsgehalt der Wurzel (alle Merkmale zusammen)
TIE_DIGITS = 9                # Punktzahlen werden auf so viele Stellen gerundet, bevor Gleichstände gezählt werden (Rundungsrauschen der Summen ist etwa 1e-14)

# --- Messung ------------------------------------------------------------------------------------------------------------------------------
N_WORLDS = 5                  # Welten je Messung
N_PATIENTS = 1500             # Patienten je Welt (vor dem Entfernen von Patienten ohne jede dokumentierte Angabe)
SWEEP_SEEDS = {"A": 1000, "B": 2000}   # zwei Mengen von je N_WORLDS Welten (Zufalls-Seed der Messung); nie ein einzelner Seed als Befund
RANKERS = ("X", "I", "A", "R", "N")
RANKER_NAMES = {
    "X": "exakt (kennt die Hierarchie)",
    "I": "nur genaue Begriffe (Oberbegriff überlesen)",
    "A": "Oberbegriff = alle Kinder (Heuristik)",
    "R": "Resnik-artig (eigene Fassung)",
    "N": "naives Überlesen (ohne Wissen um u)",
}
RANKER_SHORT = {"X": "exakt", "I": "überlesen", "A": "alle Kinder", "R": "Resnik-artig", "N": "naives Überlesen"}
SD_FACTOR = 2.0               # Unterschiede unter dieser Zahl von Standardfehlern werden nicht gedeutet
LOSS_SMALL, LOSS_LARGE = 5.0, 15.0   # Verdikt: Verlust (Recall@5 in Punkten, exakt minus überlesen) unter 5 Punkte, bis 15 Punkte, darüber

# --- Regler (Stufen = gemessene Zellen) ---------------------------------------------------------------------------------------------------
ND_OPTIONS = (50, 200)
Q_OPTIONS = (0.0, 0.3, 0.6)
U_OPTIONS = (0.0, 0.3, 0.6, 0.9)
SEED_MAX, DEFAULT_SEED = 999999, 35
DEFAULT = {"nd": 200, "q": 0.3, "u": 0.3, "seed": DEFAULT_SEED}

COLOR_X = "#5b6b8c"
COLOR_I = "#c77700"
COLOR_A = "#2a7de1"
COLOR_R = "#34a89c"
COLOR_N = "#8c8c8c"
RANKER_COLORS = {"X": COLOR_X, "I": COLOR_I, "A": COLOR_A, "R": COLOR_R, "N": COLOR_N}

# --- Presets (je eine Änderung gegenüber der Voreinstellung; die Zellen stehen in data/gmk_results.json) ---------------------------------------
PRESETS = {
    "Teilweise grob": {**DEFAULT},
    "Genau dokumentiert": {**DEFAULT, "u": 0.0},
    "Meist grob": {**DEFAULT, "u": 0.6},
    "Fast nur Oberbegriffe": {**DEFAULT, "u": 0.9},
    "Viele Lücken": {**DEFAULT, "q": 0.6},
}
PRESET_HELP = {
    "Teilweise grob": "30 Prozent der aufgeschriebenen Merkmale stehen nur als Oberbegriff in der Akte (u = 0,3), 200 Krankheiten, 30 Prozent der vorhandenen Merkmale werden gar nicht aufgeschrieben (q = 0,3).",
    "Genau dokumentiert": "Kein Merkmal steht nur als Oberbegriff in der Akte (u = 0). Exakt, Überlesen und alle Kinder rechnen dann gleich.",
    "Meist grob": "60 Prozent der aufgeschriebenen Merkmale stehen nur als Oberbegriff in der Akte (u = 0,6).",
    "Fast nur Oberbegriffe": "90 Prozent der aufgeschriebenen Merkmale stehen nur als Oberbegriff in der Akte (u = 0,9).",
    "Viele Lücken": "Zusätzlich werden 60 Prozent der vorhandenen Merkmale gar nicht aufgeschrieben (q = 0,6); 30 Prozent der aufgeschriebenen stehen nur als Oberbegriff.",
}
