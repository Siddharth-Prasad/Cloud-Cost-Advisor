"""
Cloud Cost Optimization Advisor
Run:  streamlit run app.py
"""
import html
import io
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src import config as C
from src.anomaly import detect_anomalies
from src.data_preprocessing import (DataValidationError, clean_data, daily_totals,
                                    load_and_clean_data)
from src.forecasting import forecast_total_cost, monthly_projection, trend_per_day
from src.optimizer import (analyze_resource_efficiency, efficiency_score,
                           savings_by_category, what_if)
from src.reports import html_report, to_csv_bytes, to_excel_bytes

st.set_page_config(page_title="Cloud Cost Advisor", page_icon="☁️", layout="wide")

# ================================================================ identity
# One idea runs through the whole interface: INFRARED. Spend is heat.
# Every day of spend is a stripe coloured like a thermal-camera image,
# from cold indigo (cheap day) to white-hot yellow (expensive day), and the
# hero lets that heat bloom out of the stripes into the dark.
INK, PAPER, LINE, MUTED = "#F4F0FF", "#0C0724", "#33288A", "#A59DDB"
ACCENT, GOOD, HOT, AMBER = "#8F84FF", "#3DDCAA", "#FF6A3D", "#FFB020"
RAMP = ["#2A1A9E", "#6B1FA8", "#C2276B", "#FF5A36", "#FFA62B", "#FFE08A"]
STATUS_COLORS = {
    "Idle": "#FF5A3C", "Underutilized": "#FFB020", "Off-hours waste": "#A78BFA",
    "Commitment candidate": "#5AA9FF", "Overloaded": "#FF4F9A", "Optimal": "#7C75B8",
}
FONT_BODY = "Instrument Sans, Segoe UI, Arial, sans-serif"
FONT_HEAD = "Bricolage Grotesque, Arial Black, Segoe UI, sans-serif"


def _rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def ramp_color(t, stops=RAMP):
    t = min(max(float(t), 0.0), 1.0) * (len(stops) - 1)
    i = int(t); j = min(i + 1, len(stops) - 1); f = t - i
    a, b = _rgb(stops[i]), _rgb(stops[j])
    return "#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


RAMP_SCALE = [[i / (len(RAMP) - 1), c] for i, c in enumerate(RAMP)]
CHART_RAMP = ["#8F84FF", "#C26BE8", "#FF6FA0", "#FF8A4C", "#FFB020", "#FFE08A"]
CHART_SCALE = [[i / (len(CHART_RAMP) - 1), c] for i, c in enumerate(CHART_RAMP)]
RAMP_CSS = "linear-gradient(90deg," + ",".join(RAMP) + ")"

_GRID = "rgba(190,182,240,0.24)"
pio.templates["advisor"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT_BODY, color="#EDE9FF", size=14),
    colorway=["#FF7A2F", ACCENT, GOOD, AMBER, "#FF4F9A", "#5AA9FF", "#C7C0F5"],
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(family=FONT_HEAD, size=16, color=INK), x=0.02, xanchor="left"),
    xaxis=dict(gridcolor=_GRID, linecolor="rgba(190,182,240,0.6)", zeroline=False,
               ticks="outside", tickcolor="rgba(190,182,240,0.6)", tickfont=dict(size=13, color="#EDE9FF")),
    yaxis=dict(gridcolor=_GRID, linecolor="rgba(0,0,0,0)", zeroline=False, tickfont=dict(size=13, color="#EDE9FF")),
    legend=dict(orientation="h", y=-0.22, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#1E1560", bordercolor="#FF7A2F", font=dict(family=FONT_BODY, color=INK)),
    margin=dict(l=54, r=24, t=60, b=54),
))
pio.templates.default = "advisor"
px.defaults.template = "advisor"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,400..800&family=Instrument+Sans:wght@400;500;600;700&display=swap');
:root { --void:#0C0724; --deep:#07041A; --panel:#150E45; --panel2:#1E1560; --line:#33288A; --line2:#271D6B;
        --ink:#F4F0FF; --muted:#A59DDB; --heat:#FF7A2F; --good:#3DDCAA; --hot:#FF6A3D; }
html, body, .stApp, .stMarkdown, label, p, input, textarea, button { font-family:'Instrument Sans','Segoe UI',Arial,sans-serif; }
h1, h2, h3, .hd { font-family:'Bricolage Grotesque','Arial Black','Segoe UI',sans-serif !important; font-stretch:100%; letter-spacing:-0.008em; color:var(--ink); font-optical-sizing:auto; }
.stApp { background:
    radial-gradient(900px 520px at 100% -8%, rgba(255,90,54,.20), transparent 62%),
    radial-gradient(1000px 640px at 8% -12%, rgba(107,31,168,.34), transparent 62%),
    radial-gradient(800px 500px at 50% 112%, rgba(194,39,107,.16), transparent 65%),
    #0C0724; background-attachment:fixed; }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1260px; }
.stDeployButton, [data-testid="stAppDeployButton"], #MainMenu, footer { display:none !important; }
header[data-testid="stHeader"] { background:transparent; }
::selection { background:#FF7A2F; color:#0C0724; }
:focus-visible { outline:2px solid #FFA62B !important; outline-offset:2px; }

/* ---- tabs, buttons, inputs */
button[data-baseweb="tab"] { font-weight:600; color:var(--muted); }
button[data-baseweb="tab"][aria-selected="true"] { color:#FFB27A; }
div[data-baseweb="tab-highlight"] { background:linear-gradient(90deg,#FF5A36,#FFA62B) !important; height:3px !important; }
div[data-baseweb="tab-border"] { background:var(--line2) !important; }
.stButton > button, .stDownloadButton > button { background:transparent; color:#FFB27A; border:1px solid #FF7A2F; border-radius:8px; font-weight:600; transition:background .15s, color .15s; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#FF7A2F; color:#0C0724; border-color:#FF7A2F; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] { background:var(--panel) !important; border-color:var(--line) !important; border-radius:8px; }
[data-testid="stAlert"] { border-radius:10px; border:1px solid var(--line); }

/* ---- sidebar: the deepest layer, so the page reads as lit from inside */
section[data-testid="stSidebar"] { border-right:1px solid var(--line2); background:linear-gradient(180deg,#07041A 0%,#0A0622 60%,#13062E 100%) !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top:.4rem; }
[data-testid="stSidebarNavLink"] { border-radius:8px; margin:1px 0; padding:7px 12px; border-left:3px solid transparent; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background:linear-gradient(90deg,rgba(255,90,54,.22),rgba(255,90,54,0) 85%) !important; border-left-color:#FF7A2F; }
[data-testid="stSidebarNavLink"] span { font-weight:500; }
.side-h { font-size:12px; color:#8F86CC; margin:20px 0 6px 2px; font-weight:600; letter-spacing:.02em; }
.side-note { font-size:11.5px; color:#8F86CC; line-height:1.55; margin-top:18px; padding-top:14px; border-top:1px solid var(--line2); }
.health { background:linear-gradient(160deg,#1E1560,#150E45); border:1px solid var(--line); border-radius:12px; padding:14px 16px 15px; margin-top:6px; }
.health .hl { font-size:12px; color:#A59DDB; font-weight:500; }
.health .hrow { display:flex; align-items:baseline; gap:6px; margin-top:4px; }
.health .hscore { font-family:'Bricolage Grotesque',sans-serif; font-stretch:100%; font-size:40px; font-weight:800; color:#FFFFFF; line-height:1; }
.health .hof { font-size:12.5px; color:#A59DDB; }
.gauge { position:relative; height:8px; border-radius:99px; margin-top:13px; background:linear-gradient(90deg,#FF5A36,#FFA62B,#3DDCAA); }
.gauge i { position:absolute; top:-4px; width:4px; height:16px; border-radius:2px; background:#FFFFFF; box-shadow:0 0 0 2px #1E1560; }
.health .hnote { font-size:12.5px; color:#D6D0F7; margin-top:12px; line-height:1.45; }
.health .hnote b { color:#FFFFFF; font-weight:600; }
section[data-testid="stSidebar"] div[role="radiogroup"] { background:#150E45; border:1px solid var(--line); border-radius:10px; padding:3px; gap:2px; width:100%; flex-wrap:nowrap; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label { flex:1 1 0; justify-content:center; padding:5px 6px; border-radius:7px; margin:0; text-align:center; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display:none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background:linear-gradient(90deg,#FF5A36,#FFA62B); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color:#0C0724; font-weight:700; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-size:13px; }

/* ---- page header */
.ph { margin:2px 0 22px 0; }
.pbar { display:flex; width:96px; height:6px; border-radius:3px; margin-bottom:16px; box-shadow:0 6px 22px rgba(255,90,54,.55); }
.pbar i { flex:1; display:block; }
.pbar i:first-child { border-radius:3px 0 0 3px; } .pbar i:last-child { border-radius:0 3px 3px 0; }
.ph h1 { font-size:38px; font-weight:800; font-stretch:100%; margin:0; padding:0; line-height:1.02; }
.ph p { color:var(--muted); margin:9px 0 0 0; font-size:15.5px; }
.sec { margin:32px 0 10px 0; }
.sec h3 { font-size:22px; font-weight:800; font-stretch:100%; margin:0; padding:0; }
.sec p { color:var(--muted); margin:3px 0 0 0; font-size:14px; }

/* ---- hero: the memorable element. Stripes with a blurred heat bloom beneath. */
.hero { position:relative; overflow:hidden; border-radius:18px; padding:38px 40px 52px; margin-bottom:22px;
        border:1px solid #3A2C9A;
        background:radial-gradient(700px 300px at 85% 0%, rgba(255,90,54,.22), transparent 70%),
                   linear-gradient(165deg,#241680 0%,#150E45 48%,#0E0830 100%); }
.hero-top { position:relative; display:flex; justify-content:space-between; align-items:flex-start; gap:36px; }
.hero h1 { color:#FFFFFF !important; font-size:54px; font-weight:800; font-stretch:100%; line-height:.98; margin:0; padding:0; letter-spacing:-0.012em; }
.hero p { color:#CFC8F5; font-size:16px; line-height:1.6; margin:18px 0 0 0; max-width:600px; }
.hero p b { color:#FFFFFF; font-weight:600; }
.hero-score { text-align:right; flex:0 0 auto; }
.hero-score .sc { font-family:'Bricolage Grotesque',sans-serif; font-stretch:100%; font-weight:800; font-size:112px; line-height:.85; letter-spacing:-0.008em; }
.hero-score .sl { color:#A59DDB; font-size:12.5px; margin-top:12px; line-height:1.4; }
.stripes-wrap { position:relative; margin-top:34px; }
.stripes { display:flex; height:104px; border-radius:8px; overflow:hidden; position:relative; animation:wipe 1s cubic-bezier(.2,.7,.2,1) both; }
.stripes i { flex:1 1 0; display:block; }
.stripes.bloom { position:absolute; inset:0; filter:blur(24px) saturate(1.6); opacity:.95; transform:translateY(34px) scale(.96,1.05); animation:none; border-radius:0; pointer-events:none; }
.stripe-axis { position:relative; z-index:1; display:flex; justify-content:space-between; align-items:center; color:#A59DDB; font-size:12px; margin-top:12px; gap:12px; }
.lgbar { display:inline-block; width:110px; height:8px; border-radius:4px; vertical-align:middle; margin:0 8px; }
@keyframes wipe { from { clip-path:inset(0 100% 0 0); } to { clip-path:inset(0 0 0 0); } }
@media (prefers-reduced-motion: reduce) { .stripes { animation:none; } }

/* ---- KPI strip */
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); background:rgba(21,14,69,.78); border:1px solid var(--line); border-radius:12px; margin:0 0 18px 0; overflow:hidden; }
.kpi { padding:16px 22px 15px; border-right:1px solid var(--line2); border-top:3px solid #6B4FE0; }
.kpi:last-child { border-right:0; }
.kpi.good { border-top-color:#3DDCAA; } .kpi.bad { border-top-color:#FF6A3D; }
.kl { font-size:12.5px; color:var(--muted); font-weight:500; }
.kv { font-family:'Bricolage Grotesque',sans-serif; font-stretch:100%; font-size:32px; font-weight:800; color:var(--ink); margin-top:4px; font-variant-numeric:tabular-nums; letter-spacing:-0.008em; }
.kv.good { color:#3DDCAA; } .kv.bad { color:#FF8A62; }
.kn { font-size:12px; color:var(--muted); margin-top:3px; }

/* ---- statement (ledger) */
.stmt { background:rgba(21,14,69,.78); border:1px solid var(--line); border-radius:12px; padding:22px 24px 20px; }
.stmt h3 { font-size:19px; font-weight:800; font-stretch:100%; margin:0 0 10px 0; padding:0; }
.ln { display:flex; align-items:baseline; padding:7px 0; font-size:14px; color:#DAD4F8; }
.ln i { flex:1; border-bottom:2px dotted #43379F; margin:0 9px; transform:translateY(-3px); }
.ln b { font-family:'Bricolage Grotesque',sans-serif; font-stretch:100%; font-weight:700; font-variant-numeric:tabular-nums; color:var(--ink); }
.ln.cat { padding:3px 0 3px 18px; font-size:13px; color:var(--muted); }
.ln.cat b { font-weight:600; font-size:13px; color:#DAD4F8; }
.ln .sq { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:9px; }
.ln.hot b { color:#FF8A62; }
.ln.total { border-top:1px solid #6B5FD0; margin-top:8px; padding-top:13px; font-weight:600; color:var(--ink); }
.ln.total b { font-size:26px; font-weight:800; border-bottom:4px double #FFA62B; }
.stmt .foot { color:var(--muted); font-size:12.5px; margin-top:14px; line-height:1.5; }
.stmt .foot b { color:#3DDCAA; }

/* ---- insights */
.insights { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); margin:6px 0 18px 0; }
.ins { padding:4px 26px 8px 0; margin-right:26px; border-right:1px solid var(--line2); }
.ins:last-child { border-right:0; margin-right:0; }
.ins .in { font-family:'Bricolage Grotesque',sans-serif; font-stretch:100%; font-weight:800; font-size:56px; line-height:1; letter-spacing:-0.008em; }
.ins .it { font-size:14px; color:#D6D0F7; line-height:1.55; margin-top:8px; }

/* ---- charts and tables */
[data-testid="stPlotlyChart"] { background:linear-gradient(90deg,#2A1A9E,#6B1FA8,#C2276B,#FF5A36,#FFA62B,#FFE08A) top/100% 3px no-repeat, linear-gradient(180deg,rgba(38,26,120,.9),rgba(21,14,69,.9)); border:1px solid var(--line); border-radius:12px; padding:10px 6px 6px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }

.cap { color:var(--muted); font-size:13.5px; line-height:1.5; margin:-6px 0 20px 2px; padding-left:10px; border-left:3px solid #FF7A2F; }

/* ---- opportunity line items */
.opps { background:rgba(21,14,69,.78); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.opp { display:grid; grid-template-columns:5px 1fr auto; gap:18px; padding:16px 22px 16px 0; border-bottom:1px solid var(--line2); }
.opp:last-child { border-bottom:0; }
.tick { display:block; }
.oh b { font-family:'Bricolage Grotesque',sans-serif; font-stretch:100%; font-weight:700; font-size:16px; color:var(--ink); }
.oh .os { font-size:13px; font-weight:600; margin-left:12px; }
.ometa { color:var(--muted); font-size:12.5px; margin-top:2px; }
.oa { color:#D6D0F7; font-size:13.5px; margin-top:5px; line-height:1.5; }
.ov { text-align:right; font-family:'Bricolage Grotesque',sans-serif; font-stretch:100%; font-weight:800; font-size:30px; color:var(--ink); font-variant-numeric:tabular-nums; white-space:nowrap; }
.ov small { display:block; font-family:'Instrument Sans',sans-serif; font-stretch:100%; font-weight:400; font-size:11.5px; color:var(--muted); }
@media (max-width:760px) { .hero { padding:26px 22px 20px; } .hero-top { flex-direction:column; gap:20px; } .hero h1 { font-size:40px; } .hero-score { text-align:left; } .hero-score .sc { font-size:88px; } .ins { border-right:0; margin-right:0; } }
</style>""", unsafe_allow_html=True)


def lollipop(names, values, colors, title, fmt="${:,.0f}"):
    """Horizontal stems ending in a dot, biggest on top."""
    fig = go.Figure()
    for n, v, c in zip(names, values, colors):
        fig.add_shape(type="line", x0=0, x1=v, y0=n, y1=n, line=dict(color=c, width=3), layer="below")
    fig.add_scatter(x=list(values), y=list(names), mode="markers+text", showlegend=False,
                    marker=dict(color=list(colors), size=15, line=dict(color="#0C0724", width=2)),
                    text=[fmt.format(v) for v in values], textposition="middle right",
                    textfont=dict(color="#EDE9FF", size=12), cliponaxis=False,
                    hovertemplate="%{y}: %{text}<extra></extra>")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title=title, height=max(380, 30 * len(values) + 130),
                      xaxis_range=[0, float(max(values)) * 1.2])
    return fig


DONUT_COLORS = ["#FF7A2F", "#8F84FF", "#3DDCAA", "#FFB020", "#FF4F9A", "#5AA9FF", "#C26BE8", "#FFE08A", "#7C75B8"]


def donut(names, values, title, centre="total", fmt="${:,.0f}", height=430):
    """Ring chart with the total in the middle. Long tails fold into 'Other'."""
    pairs = sorted(zip(map(str, names), map(float, values)), key=lambda t: -t[1])
    if len(pairs) > 8:
        pairs = pairs[:8] + [("Other", sum(v for _, v in pairs[8:]))]
    labels, vals = zip(*pairs)
    fig = go.Figure(go.Pie(labels=list(labels), values=list(vals), hole=0.62, sort=False,
                           marker=dict(colors=DONUT_COLORS[:len(vals)], line=dict(color="#150E45", width=3)),
                           textinfo="percent", textfont=dict(color="#0C0724", size=13),
                           hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b style='font-size:26px'>{fmt.format(sum(vals))}</b><br>{centre}",
                       showarrow=False, font=dict(color="#EDE9FF", size=14))
    fig.update_layout(title=title, height=height, legend=dict(orientation="v", y=0.5, x=1.0, font=dict(size=13)),
                      margin=dict(l=20, r=20, t=60, b=20))
    return fig


def show(fig, caption):
    """Chart followed by a one-line plain-English description."""
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<div class='cap'>{html.escape(caption)}</div>", unsafe_allow_html=True)


# ================================================================ UI helpers
def money(x, d=0):
    return f"${x:,.{d}f}"


def hm(x, d=0):
    """Money for raw HTML blocks ($ as an entity so markdown never reads it as math)."""
    return f"&#36;{x:,.{d}f}"


def page_header(title, sub):
    bar = "".join(f"<i style='background:{c}'></i>" for c in RAMP)
    st.markdown(f"<div class='ph'><div class='pbar'>{bar}</div><h1>{html.escape(title)}</h1>"
                f"<p>{html.escape(sub)}</p></div>", unsafe_allow_html=True)


def section(title, sub=""):
    p = f"<p>{html.escape(sub)}</p>" if sub else ""
    st.markdown(f"<div class='sec'><h3>{html.escape(title)}</h3>{p}</div>", unsafe_allow_html=True)


def kpis(items):
    """items: (label, value_html, note, tone) with tone in {'', 'good', 'bad'}."""
    cells = "".join(
        f"<div class='kpi {t}'><div class='kl'>{html.escape(l)}</div>"
        f"<div class='kv {t}'>{v}</div><div class='kn'>{html.escape(n)}</div></div>"
        for l, v, n, t in items)
    st.markdown(f"<div class='kpis'>{cells}</div>", unsafe_allow_html=True)


def cost_stripes(daily):
    """One stripe per day (weekly means when the history is long), cool to hot."""
    d = daily.sort_values("date")
    if len(d) > 240:
        d = d.set_index("date")["total_cost"].resample("W").mean().reset_index()
    v = d["total_cost"].to_numpy(dtype=float)
    lo, hi = np.percentile(v, [4, 96])
    norm = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    cells = "".join(
        f"<i style='background:{ramp_color(n)}' title='{pd.Timestamp(dt):%d %b %Y}: {hm(c, 2)}'></i>"
        for n, dt, c in zip(norm, d["date"], v))
    first, last = pd.Timestamp(d["date"].iloc[0]), pd.Timestamp(d["date"].iloc[-1])
    axis = (f"<div class='stripe-axis'><span>{first:%d %b %Y}</span>"
            f"<span>cheaper day<span class='lgbar' style='background:{RAMP_CSS}'></span>costlier day</span>"
            f"<span>{last:%d %b %Y}</span></div>")
    return (f"<div class='stripes-wrap'><div class='stripes bloom' aria-hidden='true'>{cells}</div>"
            f"<div class='stripes'>{cells}</div></div>{axis}")


def opportunity_rows(df, n=6):
    rows = []
    for _, r in df.head(n).iterrows():
        col = STATUS_COLORS.get(r["Status"], ACCENT)
        move = (f"{r['Instance Type']} to {r['Recommended Type']}"
                if r["Recommended Type"] != r["Instance Type"] else str(r["Instance Type"]))
        rows.append(
            f"<div class='opp'><span class='tick' style='background:{col};color:{col}'></span>"
            f"<div><div class='oh'><b>{html.escape(str(r['Resource ID']))}</b>"
            f"<span class='os' style='color:{col}'>{html.escape(str(r['Status']))}</span></div>"
            f"<div class='ometa'>{html.escape(str(r['Environment']))}, {html.escape(move)}</div>"
            f"<div class='oa'>{html.escape(str(r['Recommended Action']))}</div></div>"
            f"<div class='ov'>{hm(r['Est. Monthly Savings ($)'])}<small>saved a month</small></div></div>")
    st.markdown(f"<div class='opps'>{''.join(rows)}</div>", unsafe_allow_html=True)


def statement(total_monthly, total_saving, cat):
    """Ledger whose displayed (rounded) numbers always add up exactly."""
    spend = int(round(total_monthly))
    waste = int(round(total_saving))
    vals = [int(round(v)) for v in cat["Est. Monthly Savings ($)"]]
    if vals:
        vals[0] += waste - sum(vals)          # absorb rounding in the largest line
    lines = [f"<div class='ln'><span>Estimated monthly spend</span><i></i><b>{hm(spend)}</b></div>"]
    for (_, row), v in zip(cat.iterrows(), vals):
        c = STATUS_COLORS.get(row["Status"], ACCENT)
        lines.append(f"<div class='ln cat'><span><em class='sq' style='background:{c}'></em>"
                     f"{html.escape(str(row['Status']))}</span><i></i><b>-{hm(v)}</b></div>")
    lines.append(f"<div class='ln hot'><span>Recoverable waste</span><i></i><b>-{hm(waste)}</b></div>")
    lines.append(f"<div class='ln total'><span>Spend after optimisation</span><i></i>"
                 f"<b>{hm(max(spend - waste, 0))}</b></div>")
    lines.append(f"<div class='foot'>Applying every recommendation saves about <b>{hm(waste * 12)}</b> a year.</div>")
    st.markdown(f"<div class='stmt'><h3>Monthly statement</h3>{''.join(lines)}</div>", unsafe_allow_html=True)


INSIGHT_TEXT = {
    "Off-hours waste": "dev, test or staging resources run around the clock. An office-hours schedule saves {s} a month.",
    "Underutilized": "resources use under 20% of their CPU. Moving to a smaller size saves {s} a month.",
    "Idle": "resources are almost completely idle. Stopping them saves {s} a month.",
    "Commitment candidate": "steady production servers qualify for a savings plan worth {s} a month.",
    "Overloaded": "resources run above 85% CPU. They need more capacity to avoid slowdowns.",
}


def insight_cards(recs, cat):
    cells = []
    for status in list(cat["Status"].head(3)):
        n = int((recs["Status"] == status).sum())
        s = float(cat.loc[cat["Status"] == status, "Est. Monthly Savings ($)"].iloc[0])
        cells.append((status, n, INSIGHT_TEXT.get(status, "resources need attention.").format(s=hm(s))))
    n_over = int((recs["Status"] == "Overloaded").sum())
    if n_over and "Overloaded" not in [c[0] for c in cells]:
        cells.append(("Overloaded", n_over, INSIGHT_TEXT["Overloaded"]))
    html_cells = "".join(
        f"<div class='ins'><div class='in' style='color:{STATUS_COLORS.get(k, ACCENT)}'>{n}</div>"
        f"<div class='it'>{txt}</div></div>" for k, n, txt in cells)
    st.markdown(f"<div class='insights'>{html_cells}</div>", unsafe_allow_html=True)


# ================================================================ data helpers
@st.cache_data(show_spinner=False)
def load_default():
    return load_and_clean_data(C.DATA_PATH, return_report=True)


@st.cache_data(show_spinner=False)
def load_uploaded(raw: bytes):
    return clean_data(pd.read_csv(io.BytesIO(raw)))


@st.cache_resource(show_spinner=False)
def load_model(mtime: float):
    return joblib.load(C.MODEL_PATH)


@st.cache_data(show_spinner=False)
def cached_recs(df: pd.DataFrame, lookback: int):
    return analyze_resource_efficiency(df, lookback or None)


@st.cache_data(show_spinner=False)
def cached_anomalies(df: pd.DataFrame, z: float):
    return detect_anomalies(df, z_threshold=z)


# ===================================================================== pages
def page_overview():
    top5 = float(actionable.head(5)["Est. Monthly Savings ($)"].sum())
    pct = 100 * total_saving / max(total_monthly, 1)
    if total_saving > 0:
        headline = f"{hm(total_saving)} a month<br>is going to waste"
        body = (f"That is <b>{pct:.0f}%</b> of your {hm(total_monthly)} monthly cloud bill, spent on idle, "
                f"oversized or always-on resources. The five biggest fixes alone recover <b>{hm(top5)}</b> a month.")
    else:
        headline = "Your cloud spend<br>looks efficient"
        body = "No idle, oversized or always-on resources were found in this period."
    if score >= 75:
        sc_color = "#3DDCAA"
    elif score >= 50:
        sc_color = "#FFB020"
    else:
        sc_color = "#FF6A3D"
    st.markdown(
        f"<div class='hero'><div class='hero-top'><div><h1>{headline}</h1><p>{body}</p></div>"
        f"<div class='hero-score'><div class='sc' style='color:{sc_color}'>{score:.0f}</div>"
        f"<div class='sl'>efficiency score<br>out of 100</div></div></div>"
        f"{cost_stripes(daily)}</div>", unsafe_allow_html=True)

    cat = savings_by_category(recs)
    if not cat.empty:
        insight_cards(recs, cat)

    l, r = st.columns([3, 2], gap="large")
    with l:
        d = daily.assign(ma7=daily["total_cost"].rolling(7, min_periods=1).mean())
        peak = d.loc[d["total_cost"].idxmax()]
        fig = go.Figure()
        fig.add_scatter(x=d["date"], y=d["total_cost"], mode="lines", name="Daily spend",
                        line=dict(color="rgba(255,224,138,0.42)", width=1.3),
                        hovertemplate="%{x|%d %b}: $%{y:,.0f}<extra>Daily spend</extra>")
        fig.add_scatter(x=d["date"], y=d["ma7"], mode="lines", name="7-day average",
                        line=dict(color="#FFB020", width=4, shape="spline", smoothing=0.7),
                        fill="tozeroy", fillcolor="rgba(255,106,61,0.16)",
                        hovertemplate="%{x|%d %b}: $%{y:,.0f}<extra>7-day average</extra>")
        fig.add_scatter(x=[peak["date"]], y=[peak["total_cost"]], mode="markers+text", showlegend=False,
                        marker=dict(color="#FFE08A", size=11, line=dict(color="#0C0724", width=2)),
                        text=[f"Peak ${peak['total_cost']:,.0f} on {pd.Timestamp(peak['date']):%d %b}"],
                        textposition="top left", textfont=dict(color="#FFE08A", size=13),
                        hovertemplate="%{x|%d %b}: $%{y:,.0f}<extra>Peak day</extra>")
        lo = float(d["ma7"].min()) * 0.8
        fig.update_layout(title="Daily spend (thin) and 7-day average (thick)", yaxis_title="USD per day",
                          yaxis_range=[lo, float(peak["total_cost"]) * 1.12], height=440, showlegend=False)
        show(fig, "The thin line is each day's total cloud bill. The thick line smooths it over 7 days so the real trend is easy to see.")
    with r:
        statement(total_monthly, total_saving, cat)

    section("Biggest opportunities", "Start here. Each line says what to change and what it saves.")
    if actionable.empty:
        st.success("Nothing to optimise right now.")
    else:
        opportunity_rows(actionable, 6)


def page_cost_analysis():
    page_header("Cost analysis", "Where the money goes, and how hard each resource works for it.")
    t1, t2, t3, t4 = st.tabs(["Spend breakdown", "Utilisation vs cost", "Heatmap", "Resource drill-down"])
    with t1:
        by = st.selectbox("Group by", ["instance_type", "environment", "region", "resource_id"])
        g = fdf.groupby(by)["cost_usd"].sum().sort_values(ascending=False).reset_index()
        fig = donut(g[by], g["cost_usd"], f"Share of total cost by {by.replace('_', ' ')}", centre="total spend")
        show(fig, "Each slice is one group's share of your total cloud bill, so the biggest slices show where the money is concentrated.")
        trend = fdf.groupby(["date", "environment"], as_index=False)["cost_usd"].sum()
        fig = px.area(trend, x="date", y="cost_usd", color="environment",
                      title="Daily cost by environment, stacked").update_layout(yaxis_title="USD per day")
        fig.update_traces(line=dict(width=1.2), opacity=0.92)
        show(fig, "Daily cost stacked by environment: the full height is the day's total and each colour is one environment's share.")
        fig = px.histogram(fdf, x="cost_usd", color="instance_type", nbins=50, title="How daily cost is distributed")
        fig.update_traces(marker_line_width=0, opacity=0.95)
        fig.update_layout(bargap=0.05)
        show(fig, 'How many resource-days fall at each cost level. A long tail on the right means a few unusually expensive days.')
    with t2:
        s = fdf.sample(min(len(fdf), 3000), random_state=1)
        fig = px.scatter(s, x="cpu_utilization", y="cost_usd", color="instance_type",
                         hover_data=["resource_id", "environment"], opacity=.75,
                         title="CPU utilisation vs daily cost")
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        show(fig, 'Each dot is one resource on one day. Dots with high cost but low CPU are paying for capacity they do not use.')
        fig = px.scatter(recs, x="Avg CPU (%)", y="Avg Memory (%)", size="Monthly Cost ($)",
                         color="Status", color_discrete_map=STATUS_COLORS, hover_name="Resource ID",
                         title="Fleet map: low CPU and memory means waste (bubble size = monthly cost)")
        fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1.5, color="#0C0724")))
        show(fig, 'Each bubble is one resource. The bottom-left corner (low CPU and memory) is where waste hides, and bigger bubbles cost more.')
    with t3:
        wk = fdf.assign(week=fdf["date"].dt.to_period("W").astype(str)).pivot_table(
            index="resource_id", columns="week", values="cpu_utilization", aggfunc="mean")
        fig = px.imshow(wk, aspect="auto", color_continuous_scale=[HOT, "#7B3FC4", "#3DDCAA"],
                        title="Weekly average CPU % per resource (orange means idle)")
        fig.update_traces(xgap=2, ygap=2)
        show(fig, 'Weekly average CPU for every resource. Orange rows barely work, and teal rows are well used.')
    with t4:
        rid = st.selectbox("Resource", sorted(fdf["resource_id"].unique()))
        one = fdf[fdf["resource_id"] == rid].sort_values("date")
        fig = go.Figure()
        fig.add_bar(x=one["date"], y=one["cost_usd"], name="Cost ($)", yaxis="y2",
                    marker_color="rgba(165,157,219,.30)")
        fig.add_scatter(x=one["date"], y=one["cpu_utilization"], name="CPU %", line=dict(color=ACCENT, width=3))
        fig.add_scatter(x=one["date"], y=one["memory_utilization"], name="Memory %", line=dict(color=GOOD, width=3))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="USD", showgrid=False),
                          yaxis_title="Utilisation %", title=f"{rid}: utilisation and cost", height=420)
        show(fig, "Lines show the chosen resource's CPU and memory use. The faint bars show what it cost each day.")
        row = recs[recs["Resource ID"] == rid]
        if not row.empty:
            st.info(f"{row.iloc[0]['Status']}: {row.iloc[0]['Recommended Action']}")


def page_forecast():
    page_header("Spend forecast", "Where the bill is heading, and how far the recommendations pull it down.")
    if len(daily) < 14:
        st.warning("At least 14 days of data are needed to forecast."); st.stop()
    h = st.slider("Forecast horizon (days)", 7, 90, 30)
    fc = forecast_total_cost(daily, h)
    proj = monthly_projection(fc, min(30, h))
    fut = fc[fc["kind"] == "forecast"]
    tpd = trend_per_day(daily)
    trend_txt = f"{'+' if tpd > 0 else '-'}&#36;{abs(tpd):.2f}/day"
    kpis([
        ("Expected next 30 days", hm(proj["expected"]), "if nothing changes", ""),
        ("90% range", f"{hm(proj['low'])} to {hm(proj['high'])}", "forecast uncertainty", ""),
        ("Trend", trend_txt, "daily spend is rising" if tpd > 0 else "daily spend is falling",
         "bad" if tpd > 0 else "good"),
        ("With recommendations", hm(max(proj["expected"] - total_saving, 0)), f"saves about {hm(total_saving)}", "good"),
    ])
    fig = go.Figure()
    fig.add_scatter(x=fc["date"], y=fc["actual"], name="Actual", line=dict(color="#FFFFFF", width=2.6))
    fig.add_scatter(x=fut["date"], y=fut["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fut["date"], y=fut["lower"], fill="tonexty", line=dict(width=0),
                    fillcolor="rgba(255,106,61,.20)", name="90% range", hoverinfo="skip")
    fig.add_scatter(x=fut["date"], y=fut["forecast"], name="Forecast", line=dict(dash="dash", color=HOT, width=3.4))
    fig.add_scatter(x=fut["date"], y=(fut["forecast"] - total_saving / 30).clip(lower=0),
                    name="With recommendations applied", line=dict(color=GOOD, width=3.4))
    fig.update_layout(title="Daily fleet spend: history and forecast", yaxis_title="USD per day", height=470)
    show(fig, 'White is actual spend, dashed orange is the forecast with its likely range, and green is spend if you apply every recommendation.')
    st.caption("Method: linear trend plus day-of-week pattern, with a band from the residual spread.")


def page_cost_predictor():
    page_header("Cost predictor", "Describe a server and the machine learning model estimates its daily cost.")
    if not C.MODEL_PATH.exists():
        st.warning("The model is not trained yet.")
        if st.button("Train model now"):
            from src.train_model import train_and_save_model
            with st.spinner("Training…"):
                train_and_save_model(verbose=False)
            st.rerun()
        st.stop()
    bundle = load_model(C.MODEL_PATH.stat().st_mtime)
    model = bundle["model"]
    meta = json.loads(C.METRICS_PATH.read_text()) if C.METRICS_PATH.exists() else {}
    c1, c2 = st.columns(2)
    itype = c1.selectbox("Instance type", C.INSTANCE_TYPES, index=2)
    env = c1.selectbox("Environment", C.ENVIRONMENTS)
    region = c1.selectbox("Region", C.REGIONS)
    storage = c1.number_input("Storage (GB)", 10, 2000, 100, step=10)
    cpu = c2.slider("CPU utilisation (%)", 0.0, 100.0, 45.0)
    mem = c2.slider("Memory utilisation (%)", 0.0, 100.0, 50.0)
    hours = c2.slider("Running hours per day", 0.0, 24.0, 24.0, 0.5)
    net = c2.number_input("Network egress (GB per day)", 0.0, 500.0, 10.0)
    weekend = c2.checkbox("Weekend day")
    X = pd.DataFrame([{
        "instance_type_code": C.TYPE_TO_CODE[itype], "cpu_utilization": cpu,
        "memory_utilization": mem, "running_hours": hours, "storage_gb": storage,
        "network_gb": net, "region_code": C.REGION_TO_CODE[region],
        "env_code": C.ENV_TO_CODE[env], "is_weekend": int(weekend),
        "day_of_week": 5 if weekend else 2}])[bundle["features"]]
    pred = float(model.predict(X)[0])
    iv = meta.get("interval_90", {"lo": 0, "hi": 0})
    lo, hi = max(pred + iv["lo"], 0), pred + iv["hi"]
    formula = (C.HOURLY_RATE[itype] * C.REGION_MULTIPLIER[region] * hours
               + storage * C.STORAGE_PRICE_GB_MONTH / 30 + net * C.NETWORK_PRICE_GB)
    kpis([
        ("Predicted daily cost", hm(pred, 2), f"model: {bundle['name']}", ""),
        ("90% interval", f"{hm(lo, 2)} to {hm(hi, 2)}", "typical prediction error range", ""),
        ("Projected monthly cost", hm(pred * 30, 2), "30 days at this profile", ""),
        ("Price-list check", hm(formula, 2), "rate × hours + storage + network", ""),
    ])
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=bundle["features"]).sort_values()
        imp = imp.sort_values(ascending=False)
        mx = float(imp.max()) or 1.0
        fig = lollipop(imp.index.astype(str), imp.values, [ramp_color(v / mx, CHART_RAMP) for v in imp.values],
                       "What drives the model's predictions", fmt="{:.1%}")
        fig.update_layout(xaxis_title="Importance")
        show(fig, 'Longer stems mean that input moves the predicted cost more.')


def page_anomalies():
    page_header("Cost anomalies", "Days when a resource cost far more than its own normal.")
    z = st.slider("Sensitivity (lower finds more)", 2.0, 8.0, 3.5, 0.5)
    an = cached_anomalies(fdf, z)
    kpis([
        ("Anomalies found", f"{len(an)}", "at this sensitivity", "bad" if len(an) else "good"),
        ("Excess spend", hm(an["excess_cost"].sum() if len(an) else 0, 2), "above each resource's baseline", ""),
        ("Critical", f"{int((an['severity'] == 'Critical').sum()) if len(an) else 0}", "most extreme spikes", ""),
    ])
    if an.empty:
        st.success("No anomalies at this sensitivity.")
    else:
        base = daily_totals(fdf)
        flagged = an.groupby("date", as_index=False)["excess_cost"].sum()
        fig = go.Figure()
        fig.add_scatter(x=base["date"], y=base["total_cost"], name="Daily spend", line=dict(color="#FFB020", width=2.4))
        fig.add_scatter(x=flagged["date"], y=base.set_index("date").loc[flagged["date"], "total_cost"],
                        mode="markers", marker=dict(color=HOT, size=11, line=dict(color="#FFE08A", width=2)),
                        name="Anomaly day")
        fig.update_layout(title="Fleet spend with anomaly days marked", yaxis_title="USD per day", height=380)
        show(fig, 'The line is daily fleet spend, and the dots mark days that cost unusually more than normal.')
        st.dataframe(an.round({c: 3 for c in ["cost_usd", "baseline_cost", "excess_cost", "robust_z"]}),
                     use_container_width=True, hide_index=True)
        st.download_button("Download anomalies (CSV)", to_csv_bytes(an), "anomalies.csv", "text/csv")
    st.caption("Method: per-resource robust z-score (median and MAD) combined with Isolation Forest.")


def page_recommendations():
    page_header("Recommendations", "Every resource with the rule that fired and the money it saves.")
    kpis([
        ("Resources to act on", f"{len(actionable)}", f"of {len(recs)} analysed", ""),
        ("Saving per month", hm(total_saving), f"{100 * total_saving / max(total_monthly, 1):.0f}% of spend", "good"),
        ("Saving per year", hm(total_saving * 12), "if all are applied", "good"),
        ("High priority", f"{int((recs['Priority'] == 'High').sum())}", "do these first", "bad"),
    ])
    f1, f2, f3 = st.columns(3)
    stat = f1.multiselect("Status", sorted(recs["Status"].unique()), default=sorted(recs["Status"].unique()))
    pri = f2.multiselect("Priority", ["High", "Medium", "Low", "None"], default=["High", "Medium", "Low"])
    only = f3.checkbox("Only show resources with savings", value=True)
    view = recs[recs["Status"].isin(stat) & recs["Priority"].isin(pri)]
    if only:
        view = view[view["Est. Monthly Savings ($)"] > 0]
    if view.empty:
        st.info("No resources match these filters.")
    else:
        mx = max(float(view["Est. Monthly Savings ($)"].max()), 1.0)
        cfg = {
            "Est. Monthly Savings ($)": st.column_config.ProgressColumn(
                "Saving per month", format="$%.0f", min_value=0.0, max_value=mx),
            "Monthly Cost ($)": st.column_config.NumberColumn("Monthly cost", format="$%.0f"),
            "Savings %": st.column_config.NumberColumn("Saving share", format="%.0f%%"),
        }
        st.dataframe(view, use_container_width=True, hide_index=True, column_config=cfg)
        fig = px.bar(view.head(15), x="Resource ID", y="Est. Monthly Savings ($)", color="Status",
                     color_discrete_map=STATUS_COLORS, title="Largest savings opportunities")
        fig.update_traces(marker_line_width=0)
        fig.update_layout(yaxis_title="USD per month", xaxis_title=None, bargap=0.35, legend_title=None)
        show(fig, 'The 15 biggest monthly savings, one column per resource, coloured by what is wrong with it.')
    section("Export")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download CSV", to_csv_bytes(view), "recommendations.csv", "text/csv")
    xl = to_excel_bytes({"Recommendations": view, "All resources": recs})
    if xl:
        d2.download_button("Download Excel", xl, "recommendations.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    proj = monthly_projection(forecast_total_cost(daily, 30)) if len(daily) >= 14 else None
    d3.download_button("Download HTML report", html_report(recs, proj), "cost_report.html", "text/html")
    with st.expander("How the rules work"):
        st.markdown(f"""
- **Idle**: average CPU below {C.THRESHOLDS['idle_cpu']}% and memory below {C.THRESHOLDS['idle_mem']}%. Stop or terminate.
- **Underutilised**: CPU below {C.THRESHOLDS['underused_cpu']}% and memory below {C.THRESHOLDS['underused_mem']}%. Right-size to the cheapest type that fits peak (p95) demand plus 25% headroom.
- **Off-hours**: dev, test or staging running more than {C.THRESHOLDS['offhours_running_hours']} hours a day. Switch to an office-hours schedule.
- **Commitment**: steady 24×7 production load. A savings plan cuts about {C.THRESHOLDS['commitment_discount']:.0%}.
- **Overloaded**: CPU above {C.THRESHOLDS['overloaded_cpu']}%. Upsize to avoid performance problems.

Savings are never double counted. All thresholds live in `src/config.py`.""")


def page_savings_simulator():
    page_header("Savings simulator", "Pick the recommendations you would accept and see the effect on the bill.")
    if actionable.empty:
        st.success("Nothing to optimise."); st.stop()
    mode = st.radio("Accept", ["All", "High priority only", "Custom"], horizontal=True)
    if mode == "All":
        chosen = actionable["Resource ID"].tolist()
    elif mode == "High priority only":
        chosen = actionable[actionable["Priority"] == "High"]["Resource ID"].tolist()
    else:
        chosen = st.multiselect("Resources", actionable["Resource ID"].tolist(),
                                default=actionable["Resource ID"].head(5).tolist())
    w = what_if(recs, chosen)
    inr = st.number_input("USD to INR rate", 50.0, 150.0, 84.0)
    kpis([
        ("Current per month", hm(w["current_monthly"]), "as things are today", ""),
        ("After optimisation", hm(w["new_monthly"]), f"{len(chosen)} changes applied", ""),
        ("Saved per month", hm(w["savings"]), f"{w['pct']:.1f}% lower", "good"),
        ("Saved per year", hm(w["annual_savings"]), f"about ₹{w['annual_savings'] * inr:,.0f}", "good"),
    ])
    cur, new = float(w["current_monthly"]), float(w["new_monthly"])
    fig = go.Figure(go.Waterfall(
        x=["Today", "Saved by your changes", "After optimisation"],
        measure=["absolute", "relative", "total"], y=[cur, -(cur - new), 0],
        text=[f"${cur:,.0f}", f"-${cur - new:,.0f}", f"${new:,.0f}"], textposition="outside",
        textfont=dict(color="#EDE9FF", size=14),
        increasing=dict(marker=dict(color=HOT)), decreasing=dict(marker=dict(color=GOOD)),
        totals=dict(marker=dict(color=ACCENT)), connector=dict(line=dict(color="rgba(190,182,240,.6)", width=2))))
    fig.update_layout(title="Monthly spend: today to after optimisation", yaxis_title="USD per month",
                      height=430, showlegend=False, yaxis_range=[0, cur * 1.15])
    show(fig, "Starts from today's monthly bill, subtracts the savings from the changes you accepted, and ends at the new bill.")


def page_model_performance():
    page_header("Model performance", "How well each model predicts cost it has never seen.")
    if not C.METRICS_PATH.exists():
        st.warning("Train the model first: run `python src/train_model.py`."); st.stop()
    m = json.loads(C.METRICS_PATH.read_text())
    best = m["results"][m["best_model"]]
    kpis([
        ("Selected model", m["best_model"], "lowest test RMSE", ""),
        ("MAE", hm(best["MAE"], 3), "average dollar error", ""),
        ("RMSE", hm(best["RMSE"], 3), "penalises large misses", ""),
        ("R²", f"{best['R2']:.3f}", "share of variation explained", ""),
    ])
    st.caption(f"Trained on {m['n_train']} rows up to {m['train_end']}; tested on {m['n_test']} later rows "
               f"from {m['test_start']}. The split is chronological, so the model never sees the future.")
    res = pd.DataFrame(m["results"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(res.round(4), use_container_width=True, hide_index=True)
    fig = px.bar(res.melt(id_vars="Model", value_vars=["MAE", "RMSE"]), x="Model", y="value",
                 color="variable", barmode="group", title="Error by model (lower is better)",
                 color_discrete_sequence=[GOOD, HOT])
    fig.update_traces(marker_line_width=0)
    fig.update_layout(yaxis_title="USD", xaxis_title=None, legend_title=None, bargap=0.35, bargroupgap=0.08)
    show(fig, 'Average prediction error of each model in dollars. Shorter columns mean more accurate predictions.')
    if m.get("feature_importance"):
        imp = pd.Series(m["feature_importance"]).sort_values()
        fig = donut(imp.index, imp.values, "Feature importance: share of the model's decisions",
                    centre="features compared", fmt="{:.0%}")
        show(fig, "How much each input contributes to the model's decisions. Bigger slices matter more.")
    st.caption("RMSE is higher than MAE because the data contains rare cost spikes.")
    if st.button("Retrain on the current data file"):
        from src.train_model import train_and_save_model
        with st.spinner("Training…"):
            train_and_save_model(verbose=False)
        st.cache_resource.clear(); st.rerun()


# ============================================================== navigation
LOGO, MARK = C.ROOT / "assets" / "logo.svg", C.ROOT / "assets" / "logo_mark.svg"
try:
    st.logo(str(LOGO), size="large", icon_image=str(MARK))
except Exception:
    pass

pg = st.navigation({
    "Monitor": [
        st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
        st.Page(page_cost_analysis, title="Cost analysis", icon=":material/bar_chart:", url_path="cost-analysis"),
        st.Page(page_forecast, title="Forecast", icon=":material/query_stats:", url_path="forecast"),
    ],
    "Optimise": [
        st.Page(page_recommendations, title="Recommendations", icon=":material/lightbulb:", url_path="recommendations"),
        st.Page(page_savings_simulator, title="Savings simulator", icon=":material/savings:", url_path="simulator"),
        st.Page(page_anomalies, title="Anomalies", icon=":material/crisis_alert:", url_path="anomalies"),
    ],
    "Model": [
        st.Page(page_cost_predictor, title="Cost predictor", icon=":material/model_training:", url_path="predictor"),
        st.Page(page_model_performance, title="Model performance", icon=":material/monitoring:", url_path="performance"),
    ],
})

# ================================================================== sidebar
st.sidebar.markdown("<div class='side-h'>Fleet health</div>", unsafe_allow_html=True)
health_slot = st.sidebar.container()

st.sidebar.markdown("<div class='side-h'>Data</div>", unsafe_allow_html=True)
source = st.sidebar.radio("Data source", ["Sample data", "My CSV"], horizontal=True,
                          label_visibility="collapsed")

report = {}
if source == "My CSV":
    up = st.sidebar.file_uploader("CSV file", type="csv")
    st.sidebar.caption("Required columns: " + ", ".join(C.REQUIRED_COLUMNS))
    if up is None:
        st.info("Upload a CSV in the sidebar, or switch back to the sample data.")
        st.stop()
    try:
        df, report = load_uploaded(up.getvalue())
    except DataValidationError as e:
        st.error(str(e)); st.stop()
    except Exception as e:
        st.error(f"Could not read the file: {e}"); st.stop()
else:
    if not C.DATA_PATH.exists():
        st.warning("No dataset found yet.")
        if st.button("Generate synthetic dataset"):
            from generate_data import generate_synthetic_data
            generate_synthetic_data(); st.cache_data.clear(); st.rerun()
        st.stop()
    df, report = load_default()

if df.empty:
    st.error("The dataset is empty after cleaning."); st.stop()

with st.sidebar.expander("Filters", expanded=False):
    envs = st.multiselect("Environment", sorted(df["environment"].unique()),
                          default=sorted(df["environment"].unique()))
    regs = st.multiselect("Region", sorted(df["region"].unique()),
                          default=sorted(df["region"].unique()))
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    drange = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
fdf = df[df["environment"].isin(envs) & df["region"].isin(regs)]
if isinstance(drange, tuple) and len(drange) == 2:
    fdf = fdf[(fdf["date"].dt.date >= drange[0]) & (fdf["date"].dt.date <= drange[1])]
if fdf.empty:
    st.warning("No data matches the selected filters."); st.stop()

lookback = st.sidebar.select_slider("Optimizer look-back", [0, 14, 30, 60, 90], value=0,
                                    format_func=lambda x: "All data" if x == 0 else f"{x} days")
with st.sidebar.expander("Data quality report"):
    st.json({k: v for k, v in report.items()})

recs = cached_recs(fdf, lookback)
daily = daily_totals(fdf)
total_monthly = float(recs["Monthly Cost ($)"].sum())
total_saving = float(recs["Est. Monthly Savings ($)"].sum())
score = efficiency_score(recs)
actionable = recs[recs["Est. Monthly Savings ($)"] > 0]

with health_slot:
    st.markdown(
        f"<div class='health'><div class='hl'>Efficiency score</div>"
        f"<div class='hrow'><span class='hscore'>{score:.0f}</span><span class='hof'>out of 100</span></div>"
        f"<div class='gauge'><i style='left:calc({max(0, min(score, 100)):.0f}% - 2px)'></i></div>"
        f"<div class='hnote'><b>{hm(total_saving)}</b> a month could be saved across "
        f"<b>{len(actionable)}</b> resources.</div></div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='side-note'>Academic demonstration on synthetic data. "
                    "Advisory only: it never changes live cloud resources.</div>", unsafe_allow_html=True)

pg.run()
