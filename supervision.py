"""
Lancement : streamlit run supervision.py
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────── Configuration ────────────────────────────────────
st.set_page_config(
    page_title="🚁 Drone Agricole — Supervision DRL",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Couleurs ─────────────────────────────────────────
VERT   = "#2ECC71"
ORANGE = "#E67E22"
ROUGE  = "#E74C3C"
BLEU   = "#2980B9"
FOND2  = "#1A1F2E"
TEXTE  = "#ECF0F1"

# ─────────────────────────── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #ECF0F1; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1F2E 0%, #16213E 100%);
    border: 1px solid #2ECC71; border-radius: 12px; padding: 16px;
    box-shadow: 0 4px 15px rgba(46,204,113,0.15);
}
div[data-testid="metric-container"] label {
    color: #95A5A6 !important; font-size: 0.78rem !important;
    font-weight: 600 !important; text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #2ECC71 !important; font-size: 1.8rem !important; font-weight: 700 !important;
}
.hero-title {
    background: linear-gradient(135deg,#2ECC71 0%,#27AE60 50%,#1ABC9C 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 800; margin-bottom: 0;
}
.hero-sub { color: #7F8C8D; font-size: 0.95rem; margin-top: 4px; }
.section-title {
    color: #2ECC71; font-size: 1.1rem; font-weight: 700;
    border-left: 4px solid #2ECC71; padding-left: 10px; margin: 16px 0 10px 0;
}
.badge-ok   { background:#2ECC71;color:#000;padding:3px 10px;border-radius:20px;font-size:.78rem;font-weight:700; }
.badge-warn { background:#E67E22;color:#000;padding:3px 10px;border-radius:20px;font-size:.78rem;font-weight:700; }
.badge-err  { background:#E74C3C;color:#fff;padding:3px 10px;border-radius:20px;font-size:.78rem;font-weight:700; }
.row-bar  { display:flex;gap:6px;align-items:center;margin:8px 0; }
.row-done { background:linear-gradient(90deg,#2ECC71,#27AE60);height:22px;border-radius:4px;flex:1;
            display:flex;align-items:center;justify-content:center;color:#000;font-weight:700;font-size:.75rem; }
.row-todo { background:#2C3E50;height:22px;border-radius:4px;flex:1;
            display:flex;align-items:center;justify-content:center;color:#7F8C8D;font-size:.75rem; }
.info-box {
    background: #1A1F2E; border: 1px solid #2ECC71; border-radius: 10px;
    padding: 18px; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Chemins ──────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
STATS_FILE  = BASE_DIR / "training_stats.json"
CSV_MISSION = BASE_DIR / "data" / "rapport_mission_agricole.csv"
CSV_ALT     = BASE_DIR / "mission_agricole_rangees.csv"
MODEL_DIR   = BASE_DIR / "models"
CKPT_DIR    = BASE_DIR / "checkpoints"
LOG_DIR     = BASE_DIR / "logs"

# ─────────────────────────── Chargement données ───────────────────────────────
def load_stats():
    """Charge training_stats.json sans cache (données fraîches à chaque appel)."""
    try:
        if STATS_FILE.exists():
            with open(STATS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return None

def load_csv_mission():
    """Charge le CSV de mission (rapport_mission_agricole.csv ou mission_agricole_rangees.csv)."""
    for path in [CSV_MISSION, CSV_ALT]:
        try:
            if path.exists():
                return pd.read_csv(path)
        except Exception:
            pass
    return None

# ─────────────────────────── Layout Plotly de base ────────────────────────────
def base_layout(**overrides):
    layout = dict(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(26,31,46,0.8)",
        font          = dict(color=TEXTE, family="Inter, sans-serif"),
        margin        = dict(l=40, r=20, t=40, b=40),
        xaxis         = dict(gridcolor="#2C3E50", showgrid=True),
        yaxis         = dict(gridcolor="#2C3E50", showgrid=True),
        legend        = dict(bgcolor="rgba(0,0,0,0)"),
    )
    layout.update(overrides)   # ← overrides écrase proprement sans conflit
    return layout

# ─────────────────────────── Graphiques ───────────────────────────────────────
def fig_recompense(rewards):
    smoothed = pd.Series(rewards).rolling(20, min_periods=1).mean().tolist()
    x = list(range(len(rewards)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rewards, name="Brute",
                             line=dict(color=BLEU, width=1), opacity=0.35, mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=smoothed, name="Moy. mobile 20 ep.",
                             line=dict(color=VERT, width=2.5), mode="lines"))
    fig.update_layout(**base_layout(title="📈 Récompense par épisode", height=320))
    return fig

def fig_rangees(rangees):
    x   = list(range(len(rangees)))
    avg = pd.Series(rangees).rolling(20, min_periods=1).mean().tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=rangees, name="Rangées/ep",
                         marker_color=VERT, opacity=0.45))
    fig.add_trace(go.Scatter(x=x, y=avg, name="Moy. mobile",
                             line=dict(color=ORANGE, width=2), mode="lines"))
    # ✅ yaxis passé via base_layout() → pas de conflit
    fig.update_layout(**base_layout(
        title  = "🌾 Rangées traitées par épisode (max=4)",
        height = 280,
        yaxis  = dict(range=[0, 4.5], gridcolor="#2C3E50"),
    ))
    return fig

def fig_carte_champ(drone_x=None, drone_y=None, r_finies=0, r_active=0):
    RANGEES_Y = [-4.5, -1.5, 1.5, 4.5]
    fig = go.Figure()
    for i, ry in enumerate(RANGEES_Y):
        col = VERT if i < r_finies else (ORANGE if i == r_active else "#2C3E50")
        for pi in range(30):
            px_ = -20.0 + pi * 1.4
            mc  = "#E2C700" if (pi % 5 == 0) else col
            fig.add_trace(go.Scatter(x=[px_], y=[ry], mode="markers",
                                     marker=dict(size=6, color=mc, symbol="square"),
                                     showlegend=False,
                                     hovertemplate=f"({px_:.1f}, {ry})<extra></extra>"))
        stat = "✓ Traitée" if i < r_finies else ("▶ Active" if i == r_active else "○ En attente")
        fig.add_annotation(x=-22.5, y=ry, text=f"R{i+1} {stat}",
                           showarrow=False, font=dict(color=TEXTE, size=10), xanchor="right")
    if drone_x is not None:
        fig.add_trace(go.Scatter(x=[drone_x], y=[drone_y], mode="markers+text",
                                 marker=dict(size=24, color=ROUGE, symbol="star",
                                             line=dict(color="#fff", width=1.5)),
                                 text=["🚁"], textposition="top center", name="Drone"))
    fig.update_layout(**base_layout(
        title  = "🗺️ Vue aérienne du champ",
        height = 360,
        xaxis  = dict(range=[-25, 23], title="X (m)", gridcolor="#2C3E50"),
        yaxis  = dict(range=[-8, 8],   title="Y (m)", gridcolor="#2C3E50",
                      scaleanchor="x", scaleratio=0.8),
    ))
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="hero-title">🌿 DRL Drone</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Drone — Supervision Agricole</p>', unsafe_allow_html=True)
    st.divider()

    # Bouton refresh manuel — ne recharge QUE quand l'utilisateur clique
    if st.button("🔄 Actualiser les données", use_container_width=True, type="primary"):
        st.cache_data.clear()

    st.divider()

    # Statut
    stats = load_stats()
    if stats:
        ts = stats.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts).strftime("%d/%m %H:%M:%S")
        except Exception:
            pass
        st.markdown(f'<span class="badge-ok">● EN COURS</span>', unsafe_allow_html=True)
        st.caption(f"Dernière mise à jour : {ts}")
        st.metric("Timesteps",       f"{stats.get('total_timesteps', 0):,}")
        st.metric("Épisodes",        f"{stats.get('n_episodes', 0):,}")
        st.metric("Récompense moy.", f"{stats.get('mean_reward', 0):.2f}")
        st.metric("Rangées moy./ep", f"{stats.get('mean_rangees', 0):.2f} / 4")
        crash = stats.get("crash_rate", 0)
        level = "err" if crash > 0.3 else ("warn" if crash > 0.1 else "ok")
        st.markdown(f'<span class="badge-{level}">💀 {crash:.0%} crash</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">○ EN ATTENTE</span>', unsafe_allow_html=True)
        st.caption("Lance `python train_drl.py` pour démarrer.")

    st.divider()
    st.caption("💡 Clique sur **Actualiser** après chaque mise à jour.")

# ═════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ═════════════════════════════════════════════════════════════════════════════
tab_train, tab_mission, tab_analyse, tab_config = st.tabs([
    "📈 Entraînement",
    "🗺️ Mission",
    "📊 Analyse CSV",
    "⚙️ Configuration",
])


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 1 — ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
with tab_train:
    st.markdown('<p class="section-title">Métriques d\'entraînement</p>',
                unsafe_allow_html=True)

    stats = load_stats()

    if stats is None:
        st.markdown("""
        <div class="info-box">
        <b>💡 Aucun entraînement détecté</b><br><br>
        Lance cette commande dans un terminal :<br>
        <code>python train_drl.py --steps 500000</code><br><br>
        Puis clique sur <b>🔄 Actualiser</b> dans la sidebar toutes les 30 secondes
        pour voir les courbes apparaître.
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── KPIs ──────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🕹️ Timesteps",      f"{stats.get('total_timesteps', 0):,}")
        k2.metric("🏆 Récompense moy.", f"{stats.get('mean_reward', 0):.2f}",
                  delta=f"±{stats.get('std_reward', 0):.2f}")
        k3.metric("🌾 Rangées moy.",    f"{stats.get('mean_rangees', 0):.2f} / 4")
        k4.metric("💀 Taux crash",      f"{stats.get('crash_rate', 0):.1%}")
        k5.metric("📏 Longueur ep.",    f"{stats.get('mean_ep_length', 0):.0f} steps")

        st.divider()

        # ── Courbes ───────────────────────────────────────────────────────
        rewards = stats.get("rewards_history", [])
        rangees = stats.get("rangees_history", [])

        if not rewards:
            st.info("⏳ Entraînement démarré — les courbes apparaîtront après les premiers épisodes.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_recompense(rewards), use_container_width=True)
            with c2:
                st.plotly_chart(fig_rangees(rangees), use_container_width=True)

            # Progression rangées
            st.markdown('<p class="section-title">Progression moyenne des rangées</p>',
                        unsafe_allow_html=True)
            mean_r   = stats.get("mean_rangees", 0)
            row_html = '<div class="row-bar">'
            for i in range(4):
                if i + 1 <= int(mean_r):
                    row_html += f'<div class="row-done">R{i+1} ✓</div>'
                elif i < mean_r:
                    pct = int((mean_r - i) * 100)
                    row_html += f'<div class="row-done" style="opacity:.6">R{i+1} ~{pct}%</div>'
                else:
                    row_html += f'<div class="row-todo">R{i+1}</div>'
            row_html += "</div>"
            st.markdown(row_html, unsafe_allow_html=True)

            # Interprétation automatique
            st.divider()
            st.markdown('<p class="section-title">Interprétation</p>', unsafe_allow_html=True)
            mr = stats.get("mean_reward", 0)
            mr_rangees = stats.get("mean_rangees", 0)
            crash = stats.get("crash_rate", 0)

            if mr < 0:
                st.error(f"⚠️ Récompense négative ({mr:.1f}) — l'agent crashe encore souvent. "
                         f"C'est normal en début d'entraînement.")
            elif mr < 20:
                st.warning(f"🔄 Récompense faible ({mr:.1f}) — l'agent apprend à voler "
                           f"mais ne maîtrise pas encore les rangées.")
            elif mr_rangees < 2:
                st.warning(f"✈️ L'agent vole bien (récompense {mr:.1f}) mais traite peu "
                           f"de rangées ({mr_rangees:.1f}/4). Continue l'entraînement.")
            else:
                st.success(f"🎉 Excellent ! Récompense {mr:.1f}, "
                           f"{mr_rangees:.1f}/4 rangées traitées en moyenne.")

            # Détails
            last = stats.get("last_info", {})
            if last:
                with st.expander("🔍 Dernier épisode — détails"):
                    cols = st.columns(3)
                    for idx, (k, v) in enumerate(last.items()):
                        cols[idx % 3].metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 2 — MISSION
# ─────────────────────────────────────────────────────────────────────────────
with tab_mission:
    st.markdown('<p class="section-title">Supervision de la mission</p>',
                unsafe_allow_html=True)

    df_csv   = load_csv_mission()
    drone_x  = drone_y = None
    r_finies = r_active = 0
    alt      = 0.0
    masse    = 15.0

    if df_csv is not None and not df_csv.empty:
        last_row = df_csv.iloc[-1]

        # Colonnes possibles selon ton CSV
        for col_x in ["Position_X", "x", "pos_x"]:
            if col_x in df_csv.columns:
                drone_x = float(last_row[col_x])
                break
        for col_y in ["Position_Y", "y", "pos_y"]:
            if col_y in df_csv.columns:
                drone_y = float(last_row[col_y])
                break
        for col_z in ["Altitude_m", "z", "altitude"]:
            if col_z in df_csv.columns:
                alt = float(last_row[col_z])
                break
        for col_m in ["Masse_kg", "masse", "mass"]:
            if col_m in df_csv.columns:
                masse = float(last_row[col_m])
                break

        # Déduire la rangée depuis l'état mission si dispo
        for col_e in ["Etat_Mission", "etat", "state"]:
            if col_e in df_csv.columns:
                etat = str(last_row[col_e])
                if   "ROW1" in etat or "SHIFT_TO_ROW2" in etat: r_active, r_finies = 0, 0
                elif "ROW2" in etat or "SHIFT_TO_ROW3" in etat: r_active, r_finies = 1, 1
                elif "ROW3" in etat or "SHIFT_TO_ROW4" in etat: r_active, r_finies = 2, 2
                elif "ROW4" in etat:                             r_active, r_finies = 3, 3
                elif "LANDING" in etat:                          r_finies = 4
                break

        # Métriques
        i1, i2, i3, i4 = st.columns(4)
        pos_txt = f"({drone_x:.1f}, {drone_y:.1f}) m" if drone_x is not None else "—"
        i1.metric("📍 Position XY",   pos_txt)
        i2.metric("🔼 Altitude",      f"{alt:.1f} m")
        pct_res = max(0, min(100, (masse - 10) / 5 * 100))
        i3.metric("💧 Réservoir",     f"{masse:.1f} kg ({pct_res:.0f}%)")
        i4.metric("🌾 Rangées",       f"{r_finies} / 4")

        # Carte champ
        st.plotly_chart(fig_carte_champ(drone_x, drone_y, r_finies, r_active),
                        use_container_width=True)

        # Jauges
        g1, g2 = st.columns(2)
        with g1:
            fig_alt = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=alt,
                title={"text": "Altitude (m)", "font": {"color": TEXTE}},
                delta={"reference": 15.0},
                gauge={
                    "axis":  {"range": [0, 30], "tickcolor": TEXTE},
                    "bar":   {"color": BLEU}, "bgcolor": FOND2,
                    "steps": [{"range": [0, 5],   "color": ROUGE},
                               {"range": [5, 13],  "color": ORANGE},
                               {"range": [13, 17], "color": VERT}],
                    "threshold": {"line": {"color": VERT, "width": 3}, "value": 15},
                }
            ))
            fig_alt.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  font_color=TEXTE, height=220)
            st.plotly_chart(fig_alt, use_container_width=True)

        with g2:
            fig_res = go.Figure(go.Indicator(
                mode="gauge+number", value=masse,
                title={"text": "Réservoir (kg)", "font": {"color": TEXTE}},
                gauge={
                    "axis":  {"range": [10, 15], "tickcolor": TEXTE},
                    "bar":   {"color": VERT}, "bgcolor": FOND2,
                    "steps": [{"range": [10, 11],   "color": ROUGE},
                               {"range": [11, 12.5], "color": ORANGE}],
                }
            ))
            fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  font_color=TEXTE, height=220)
            st.plotly_chart(fig_res, use_container_width=True)

    else:
        st.markdown("""
        <div class="info-box">
        <b>💡 Aucune donnée de mission</b><br><br>
        Lance d'abord la simulation mission fixe :<br>
        <code>python environnement_agricole.py</code><br><br>
        Le CSV sera généré automatiquement dans <code>data/rapport_mission_agricole.csv</code>
        </div>
        """, unsafe_allow_html=True)
        # Afficher quand même la carte vide
        st.plotly_chart(fig_carte_champ(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3 — ANALYSE CSV
# ─────────────────────────────────────────────────────────────────────────────
with tab_analyse:
    st.markdown('<p class="section-title">Analyse de la mission enregistrée</p>',
                unsafe_allow_html=True)

    df = load_csv_mission()

    if df is None:
        st.markdown("""
        <div class="info-box">
        <b>💡 Fichier CSV introuvable</b><br><br>
        Lance <code>python environnement_agricole.py</code> pour générer les données.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅  {len(df)} lignes — colonnes : {', '.join(df.columns.tolist())}")

        # Colonnes numériques pour les métriques
        col_temps  = next((c for c in ["Temps","temps","time","t"] if c in df.columns), None)
        col_alt    = next((c for c in ["Altitude_m","altitude","z"] if c in df.columns), None)
        col_masse  = next((c for c in ["Masse_kg","masse","mass"] if c in df.columns), None)
        col_x      = next((c for c in ["Position_X","x","pos_x"] if c in df.columns), None)
        col_y      = next((c for c in ["Position_Y","y","pos_y"] if c in df.columns), None)
        col_etat   = next((c for c in ["Etat_Mission","etat","state"] if c in df.columns), None)

        # KPIs
        s1, s2, s3, s4 = st.columns(4)
        if col_temps:
            s1.metric("⏱ Durée", f"{df[col_temps].max():.1f} s")
        if col_alt:
            s2.metric("📏 Alt. moy.", f"{df[col_alt].mean():.1f} m")
        if col_masse:
            conso = df[col_masse].max() - df[col_masse].min()
            s3.metric("💧 Pulvérisé", f"{conso:.2f} kg")
        if col_x and col_y:
            dist = np.sqrt(df[col_x].diff()**2 + df[col_y].diff()**2).sum()
            s4.metric("📐 Distance", f"{dist:.0f} m")

        st.divider()

        # Graphique altitude
        if col_temps and col_alt:
            fig_a = px.line(df, x=col_temps, y=col_alt,
                            color_discrete_sequence=[BLEU], title="Profil d'altitude")
            fig_a.add_hline(y=15, line_dash="dash", line_color=VERT,
                            annotation_text="Cible 15 m")
            fig_a.update_layout(**base_layout(height=260))
            st.plotly_chart(fig_a, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if col_temps and col_masse:
                fig_m = px.line(df, x=col_temps, y=col_masse,
                                color_discrete_sequence=[ORANGE], title="Vidage réservoir")
                fig_m.update_layout(**base_layout(height=250))
                st.plotly_chart(fig_m, use_container_width=True)
        with c2:
            if col_x and col_y and col_etat:
                fig_t = px.scatter(df, x=col_x, y=col_y, color=col_etat,
                                   title="Trajectoire XY",
                                   color_discrete_sequence=px.colors.qualitative.Vivid)
                fig_t.update_layout(**base_layout(height=250))
                st.plotly_chart(fig_t, use_container_width=True)

        if col_etat:
            etat_c = df[col_etat].value_counts().reset_index()
            etat_c.columns = ["État", "Pas"]
            fig_s = px.bar(etat_c, x="État", y="Pas", color="État",
                           title="Répartition des états",
                           color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_s.update_layout(**base_layout(height=250, showlegend=False))
            st.plotly_chart(fig_s, use_container_width=True)

        with st.expander("📋 Données brutes"):
            st.dataframe(df.head(500), use_container_width=True, height=300)

        st.download_button("⬇️ Télécharger CSV",
                           data=df.to_csv(index=False).encode(),
                           file_name="mission_export.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 4 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_config:
    st.markdown('<p class="section-title">Lancer l\'entraînement</p>',
                unsafe_allow_html=True)

    with st.form("config_form"):
        c1, c2 = st.columns(2)
        with c1:
            total_steps = st.number_input("Timesteps cibles", 10_000,
                                          10_000_000, 500_000, 50_000)
            n_envs = st.number_input("Environnements parallèles", 1, 8, 1)
        with c2:
            urdf_path = st.text_input("Chemin URDF", "models/agri_hexacopter.urdf")
            resume    = st.checkbox("Reprendre depuis le dernier checkpoint")
        submitted = st.form_submit_button("📋 Générer la commande", type="primary")

    if submitted:
        cmd = f"python train_drl.py --steps {int(total_steps)} --n-envs {n_envs} --urdf-path {urdf_path}"
        if resume:
            cmd += " --resume"
        st.code(cmd, language="bash")
        st.success("Copie cette commande et lance-la dans ton terminal (venv activé).")

    st.divider()
    st.markdown('<p class="section-title">Toutes les commandes</p>', unsafe_allow_html=True)

    commandes = {
        "🧪 Test physique (GUI)":              "python test_env.py --gui --steps 300",
        "🚁 Simulation mission fixe":          "python environnement_agricole.py",
        "🧠 Entraînement DRL (500k steps)":   "python train_drl.py --steps 500000",
        "🔄 Reprendre l'entraînement":         "python train_drl.py --resume",
        "🎬 Évaluation (GUI PyBullet)":        "python train_drl.py --eval",
        "📊 Dashboard":                        "streamlit run supervision.py",
        "📉 TensorBoard":                      "tensorboard --logdir logs/",
    }
    for label, cmd in commandes.items():
        col1, col2 = st.columns([1, 2])
        col1.markdown(f"**{label}**")
        col2.code(cmd, language="bash")

    st.divider()
    st.markdown('<p class="section-title">Structure du projet</p>', unsafe_allow_html=True)
    st.code("""
DRL/
├── environnement_agricole.py   ← simulation mission fixe
├── drone_env.py                ← environnement Gymnasium (RL)
├── train_drl.py                ← entraînement PPO
├── test_env.py                 ← test rapide physique
├── supervision.py              ← ce dashboard
├── models/
│   ├── agri_hexacopter.urdf
│   ├── best_model.zip          ← meilleur modèle (auto)
│   └── drone_ppo_final.zip     ← modèle final (auto)
├── data/
│   └── rapport_mission_agricole.csv
├── checkpoints/                ← sauvegardes intermédiaires
├── logs/                       ← TensorBoard
└── training_stats.json         ← métriques live (auto)
    """, language="text")

    st.info(
        "**🌿 DRL Drone Agricole** — DIT Dakar | L2 Big Data\n\n"
        "Agent PPO entraîné dans PyBullet pour optimiser la couverture "
        "d'arrosage sur 4 rangées agricoles avec un hexacoptère intelligent."
    )
