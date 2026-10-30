"""
Shared look & feel for the dashboard: theme CSS, line icons, cards, and a
Plotly template so every chart matches. Colors follow the portfolio site
(deep plum + purple accent, Inter / Playfair Display).
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ============================================================================
# TOKENS
# ============================================================================

BG = '#120a18'
SURFACE = '#1c1124'
SURFACE_2 = '#251831'
BORDER = 'rgba(190, 150, 227, 0.14)'
TEXT = '#f3eef6'
TEXT_2 = '#c9b8d6'
MUTED = '#8f7f9c'
ACCENT = '#9e59de'
ACCENT_SOFT = 'rgba(158, 89, 222, 0.16)'
GRID = 'rgba(243, 238, 246, 0.07)'

# Categorical slots (validated for the dark plum surface), used in fixed order
SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']

# Status colors are reserved for risk levels and always ship with an icon + label
STATUS = {
    'LOW': ('#0ca30c', 'check'),
    'MODERATE': ('#ec835a', 'alert'),
    'HIGH': ('#d03b3b', 'x'),
}

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["resetScale2d"],
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "responsive": True
}

# ============================================================================
# ICONS (Lucide, ISC license) - stroke-based so they inherit currentColor
# ============================================================================

_ICON_PATHS = {
    'activity': '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    'alert': '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    'brain': '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>',
    'check': '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    'database': '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    'droplet': '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>',
    'heart-pulse': '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    'info': '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    'layers': '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    'leaf': '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    'map-pin': '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    'moon': '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    'scale': '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    'ban': '<circle cx="12" cy="12" r="10"/><path d="m4.9 4.9 14.2 14.2"/>',
    'stethoscope': '<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6 6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/>',
    'sun': '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    'thermometer': '<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>',
    'trending-down': '<path d="m22 17-8.5-8.5-5 5L2 7"/><path d="M16 17h6v-6"/>',
    'trending-up': '<path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>',
    'users': '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    'x': '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
}


def icon(name, size=18, color='currentColor', stroke=2):
    """Inline SVG icon."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="flex-shrink:0;vertical-align:middle">{_ICON_PATHS[name]}</svg>')


def icon_badge(name, color, size=40):
    """Icon inside a softly tinted rounded square - used for personas and stat tiles."""
    return (f'<span class="hr-badge" style="width:{size}px;height:{size}px;'
            f'background:{rgba(color, 0.16)};color:{color};">{icon(name, int(size * 0.5))}</span>')


def rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


# ============================================================================
# CSS
# ============================================================================

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
button, input, select, textarea, label {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(135, 71, 194, 0.22), transparent 60%),
        radial-gradient(900px 400px at -10% 10%, rgba(57, 135, 229, 0.08), transparent 60%),
        {BG};
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }}

h1, h2, h3 {{ font-family: 'Playfair Display', Georgia, serif !important; letter-spacing: -0.01em; color: {TEXT}; }}
h2 {{ font-size: 1.9rem !important; }}
h3 {{ font-size: 1.3rem !important; }}
p, li {{ color: {TEXT_2}; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; }}

/* Tabs: quiet pills with an accent underline */
[data-testid="stTabs"] [role="tablist"], [data-baseweb="tab-list"] {{ gap: 6px !important; }}
[data-testid="stTabs"] [role="tab"] {{
    padding: 10px 16px !important; border-radius: 10px 10px 0 0; color: {TEXT_2};
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ color: {TEXT}; background: {ACCENT_SOFT}; }}
[data-testid="stTabs"] [role="tab"] p {{ font-size: 15px; font-weight: 500; }}

/* Expanders, forms, metrics as cards */
[data-testid="stExpander"] details, [data-testid="stForm"] {{
    background: {SURFACE}; border: 1px solid {BORDER} !important; border-radius: 14px;
}}
[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px;
}}
[data-testid="stMetricValue"] {{ font-weight: 600; }}
[data-testid="stAlert"] {{ border-radius: 12px; }}
hr {{ border-color: {BORDER} !important; }}

/* Custom components */
.hr-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 20px;
    height: 100%;
}}
.hr-badge {{
    display: inline-flex; align-items: center; justify-content: center; border-radius: 12px; flex-shrink: 0;
}}
.hr-eyebrow {{
    font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: {MUTED}; font-weight: 600;
}}
.hr-section {{ display: flex; gap: 14px; align-items: center; margin: 8px 0 6px; }}
.hr-section h2 {{ margin: 0 !important; padding: 0 !important; }}
.hr-section p {{ margin: 2px 0 0; color: {MUTED}; font-size: 15px; }}
.hr-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}

.hr-hero {{
    position: relative; overflow: hidden; border-radius: 22px; padding: 36px 36px 28px;
    background: linear-gradient(135deg, rgba(135, 71, 194, 0.28), rgba(28, 17, 36, 0.9) 55%);
    border: 1px solid {BORDER}; margin-bottom: 18px;
}}
.hr-hero h1 {{ font-size: clamp(2rem, 4.5vw, 3.1rem) !important; margin: 6px 0 8px !important; padding: 0 !important; max-width: 640px; line-height: 1.1 !important; }}
.hr-hero .lede {{ font-size: 17px; color: {TEXT_2}; max-width: 520px; line-height: 1.55; margin: 0; }}
.hr-hero .art {{ position: absolute; right: 0; top: 0; height: 100%; width: 42%; opacity: 0.95; pointer-events: none; }}
.hr-hero .stats {{ display: flex; flex-wrap: wrap; gap: 28px; margin-top: 26px; position: relative; }}
.hr-hero .stat b {{ display: block; font-size: 22px; color: {TEXT}; font-weight: 600; }}
.hr-hero .stat span {{ font-size: 13px; color: {MUTED}; }}
@media (max-width: 760px) {{
    .hr-hero {{ padding: 26px 20px 22px; }}
    .hr-hero .art {{ opacity: 0.22; width: 100%; }}
}}

.hr-meter {{ background: rgba(243, 238, 246, 0.08); height: 8px; border-radius: 4px; overflow: hidden; }}
.hr-meter > div {{ height: 100%; border-radius: 4px; }}
.hr-pill {{
    display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600;
    padding: 4px 10px; border-radius: 999px;
}}
.hr-row {{ display: flex; gap: 14px; align-items: flex-start; }}
.hr-footer {{ color: {MUTED}; font-size: 13.5px; line-height: 1.6; padding: 8px 0; }}
.hr-footer b {{ color: {TEXT_2}; }}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


# ============================================================================
# COMPONENTS
# ============================================================================

def _cluster_art():
    """Decorative SVG: seven soft clouds of dots, one per persona color."""
    rng = np.random.default_rng(7)
    centers = [(120, 70), (230, 50), (330, 95), (90, 175), (205, 160), (315, 205), (175, 250)]
    dots = []
    for (cx, cy), color in zip(centers, SERIES):
        for x, y in rng.normal([cx, cy], [26, 22], size=(38, 2)):
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rng.uniform(1.6, 3.4):.1f}" '
                        f'fill="{color}" opacity="{rng.uniform(0.35, 0.85):.2f}"/>')
        dots.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{color}" stroke="{TEXT}" stroke-width="1.5"/>')
    return (f'<svg class="art" viewBox="0 0 420 300" preserveAspectRatio="xMaxYMid meet" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(dots)}</svg>')


def hero(eyebrow, title, lede, stats):
    stats_html = ''.join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for k, v in stats)
    st.markdown(f"""
    <div class="hr-hero">
        {_cluster_art()}
        <div style="position: relative;">
            <div class="hr-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p class="lede">{lede}</p>
            <div class="stats">{stats_html}</div>
        </div>
    </div>""", unsafe_allow_html=True)


def section(title, subtitle="", icon_name=None, color=ACCENT):
    badge = icon_badge(icon_name, color, 44) if icon_name else ''
    sub = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f'<div class="hr-section">{badge}<div><h2>{title}</h2>{sub}</div></div>',
                unsafe_allow_html=True)


def stat_card(label, value, pct=None, color=ACCENT, icon_name=None, note=""):
    """A number with an optional 0-100 meter."""
    badge = icon_badge(icon_name, color, 36) if icon_name else ''
    meter = (f'<div class="hr-meter" style="margin-top:14px;"><div style="width:{min(max(pct, 0), 100):.1f}%;'
             f'background:{color};"></div></div>') if pct is not None else ''
    note_html = f'<div style="font-size:12.5px;color:{MUTED};margin-top:8px;">{note}</div>' if note else ''
    return f"""
    <div class="hr-card">
        <div class="hr-row" style="align-items:center;justify-content:space-between;">
            <span style="font-size:14px;color:{TEXT_2};font-weight:500;">{label}</span>{badge}
        </div>
        <div style="font-size:30px;font-weight:600;color:{TEXT};margin-top:6px;">{value}</div>
        {meter}{note_html}
    </div>"""


def status_pill(level):
    color, icon_name = STATUS[level]
    return (f'<span class="hr-pill" style="background:{rgba(color, 0.16)};color:{color};">'
            f'{icon(icon_name, 14)}{level.title()}</span>')


def callout(text, icon_name='info', color=ACCENT):
    st.markdown(f"""
    <div class="hr-card" style="padding:14px 16px;margin:6px 0 12px;">
        <div class="hr-row" style="align-items:center;">{icon_badge(icon_name, color, 32)}
            <div style="font-size:14.5px;color:{TEXT_2};line-height:1.5;">{text}</div></div>
    </div>""", unsafe_allow_html=True)


def footer(html):
    st.markdown(f'<hr/><div class="hr-footer">{html}</div>', unsafe_allow_html=True)


# ============================================================================
# CHARTS
# ============================================================================

pio.templates['plum'] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT_2, size=13),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    colorway=SERIES,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickcolor=BORDER,
               title_font=dict(color=MUTED), automargin=True),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickcolor=BORDER,
               title_font=dict(color=MUTED), automargin=True),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(color=TEXT, family="Inter")),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_2)),
    bargap=0.35,
    modebar=dict(bgcolor='rgba(0,0,0,0)', color=MUTED, activecolor=TEXT),
    margin=dict(l=20, r=20, t=30, b=20),
))
pio.templates.default = 'plum'


def chart(fig):
    """Render with our template, set per figure so Streamlit's chart theme can't override it.
    The top margin keeps the toolbar off the data."""
    fig.update_layout(template='plum')
    # Streamlit's frontend fills these from its own theme unless the layout sets them directly
    if fig.layout.font.family is None:
        fig.update_layout(font=dict(family="Inter, system-ui, sans-serif", color=TEXT_2, size=13))
    if fig.layout.plot_bgcolor is None:
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    if fig.layout.paper_bgcolor is None:
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    if fig.layout.margin.t is None or fig.layout.margin.t < 40:
        fig.update_layout(margin_t=40)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
