"""Export helpers: CSV, Excel and a self-contained HTML executive report."""
from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd

from .optimizer import efficiency_score


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets: dict) -> bytes | None:
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            for name, frame in sheets.items():
                frame.to_excel(xw, sheet_name=name[:31], index=False)
        return buf.getvalue()
    except ImportError:
        return None


def html_report(recs: pd.DataFrame, projection: dict | None = None, title: str = "Cloud Cost Optimization Report") -> str:
    total = recs["Monthly Cost ($)"].sum()
    sav = recs["Est. Monthly Savings ($)"].sum()
    score = efficiency_score(recs)
    top = recs[recs["Est. Monthly Savings ($)"] > 0].head(15)
    rows = "".join(
        f"<tr><td>{html.escape(str(r['Resource ID']))}</td><td>{html.escape(str(r['Status']))}</td>"
        f"<td>{html.escape(str(r['Priority']))}</td><td>{html.escape(str(r['Recommended Action']))}</td>"
        f"<td style='text-align:right'>${r['Est. Monthly Savings ($)']:,.2f}</td></tr>"
        for _, r in top.iterrows())
    proj = ""
    if projection:
        proj = (f"<p>Projected next-30-day spend: <b>${projection['expected']:,.0f}</b> "
                f"(90% range ${projection['low']:,.0f} - ${projection['high']:,.0f}).</p>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@700;800&family=Instrument+Sans:wght@400;600&display=swap');
*{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
body{{font-family:'Instrument Sans',Segoe UI,Arial,sans-serif;margin:0;padding:44px 48px;color:#E6E1FB;
background:radial-gradient(800px 380px at 100% 0,rgba(255,90,54,.22),transparent 65%),#0C0724;min-height:100vh}}
h1,h2{{font-family:'Bricolage Grotesque','Arial Black',sans-serif;color:#F4F0FF;letter-spacing:-.02em}}
h1{{font-size:40px;margin:0 0 6px;border-bottom:6px solid transparent;border-image:linear-gradient(90deg,#2A1A9E,#6B1FA8,#C2276B,#FF5A36,#FFA62B,#FFE08A) 1;padding-bottom:14px}}
p{{color:#A59DDB}} b{{color:#F4F0FF}}
table{{border-collapse:collapse;width:100%;background:#150E45;border-radius:10px;overflow:hidden}}
th,td{{border-bottom:1px solid #271D6B;padding:10px 12px;font-size:13px}}
th{{background:#1E1560;text-align:left;color:#FFB27A}}
.k{{display:inline-block;margin:10px 14px 10px 0;padding:14px 20px;background:#150E45;border:1px solid #33288A;border-top:3px solid #FF7A2F;border-radius:10px;color:#A59DDB}}
.k b{{font-family:'Bricolage Grotesque',sans-serif;font-size:22px}}
</style></head><body><h1>{html.escape(title)}</h1>
<p>Generated {datetime.now():%Y-%m-%d %H:%M}. Academic demonstration on synthetic/uploaded telemetry.</p>
<div class="k">Monthly spend<br><b>${total:,.0f}</b></div>
<div class="k">Potential savings<br><b>${sav:,.0f}/mo (${sav*12:,.0f}/yr)</b></div>
<div class="k">Efficiency score<br><b>{score}/100</b></div>{proj}
<h2>Top recommendations</h2><table><tr><th>Resource</th><th>Status</th><th>Priority</th>
<th>Action</th><th>Savings/mo</th></tr>{rows}</table></body></html>"""
