"""
research/report_generator_html.py — HTML dashboard report generator.

Produces a self-contained HTML file that matches the backtest console
output style but as an interactive visual dashboard.

Called automatically at the end of run_backtest.py after the existing
console metrics print.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional


def generate_html_report(
    metrics:      dict,
    trades:       list,
    equity_curve: list[float],
    symbol:       str,
    args,
    output_dir:   str = "research/reports",
) -> str:
    """
    Generate a self-contained HTML backtest dashboard.

    Args:
        metrics:      Dict from compute_metrics() — same object used for CSV.
        trades:       List of SimulatedTrade objects.
        equity_curve: List of balance floats per bar (from BacktestResult).
        symbol:       Instrument name (e.g. "EURUSD").
        args:         CLI args namespace (bars, balance, spread, etc.).
        output_dir:   Folder to write the HTML file into.

    Returns:
        Absolute path to the saved HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{symbol}_backtest_{ts}_dashboard.html"
    filepath  = os.path.join(output_dir, filename)

    # ── Equity curve — downsample to ~100 points for chart performance ────────
    eq   = equity_curve
    step = max(1, len(eq) // 100)
    eq_sampled  = eq[::step]
    if eq[-1] not in eq_sampled:
        eq_sampled.append(eq[-1])
    labels      = [str(i * step) for i in range(len(eq_sampled))]

    # ── Validation results ────────────────────────────────────────────────────
    thresholds = {
        "min_trades":    (">=", 30),
        "win_rate":      (">=", 45.0),
        "profit_factor": (">=", 1.20),
        "max_drawdown":  ("<=", 5.0),
    }
    val_rows = []
    val_failed = []
    total_trades = int(metrics.get("total_trades", 0))
    win_rate_val = float(metrics.get("win_rate", 0)) * 100
    pf_val       = float(metrics.get("profit_factor", 0))
    dd_val       = float(metrics.get("max_drawdown", 0)) * 100

    checks = {
        "min_trades":    (total_trades,  ">=", 30,   f"requires >= 30 · got {total_trades}"),
        "win_rate":      (win_rate_val,  ">=", 45.0, f"requires >= 45% · got {win_rate_val:.1f}%"),
        "profit_factor": (pf_val,        ">=", 1.20, f"requires >= 1.20 · got {pf_val:.2f}"),
        "max_drawdown":  (dd_val,        "<=", 5.0,  f"requires <= 5% · got {dd_val:.2f}%"),
    }
    for name, (val, op, threshold, desc) in checks.items():
        if op == ">=":
            passed = val >= threshold
        else:
            passed = val <= threshold
        if not passed:
            val_failed.append(name)
        pill_class = "pill-ok" if passed else "pill-fail"
        pill_text  = "OK" if passed else "FAIL"
        val_rows.append(
            f'<div class="val-row">'
            f'<div><div class="val-name">{name}</div>'
            f'<div class="val-sub">{desc}</div></div>'
            f'<span class="pill {pill_class}">{pill_text}</span></div>'
        )

    overall_pass   = len(val_failed) == 0
    overall_class  = "val-pass" if overall_pass else "val-fail"
    overall_text   = "PASS" if overall_pass else "FAIL"
    failed_count   = len(val_failed)
    passed_count   = 4 - failed_count

    # ── Metric card helper ────────────────────────────────────────────────────
    net_profit  = float(metrics.get("net_profit", 0))
    net_pct     = (net_profit / float(getattr(args, "balance", 10000))) * 100
    profit_color = "danger" if net_profit < 0 else "success"

    # ── Run info ──────────────────────────────────────────────────────────────
    start_date = getattr(args, "start_date", None)
    end_date   = getattr(args, "end_date", None)
    bars_val   = getattr(args, "bars", 50000)
    if start_date and end_date:
        range_str = f"{start_date} → {end_date}"
    else:
        range_str = f"{bars_val:,} bars"

    timeframe   = "M5"
    run_time    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wins        = int(metrics.get("wins",   0))
    losses      = int(metrics.get("losses", 0))

    eq_json     = json.dumps([round(v, 2) for v in eq_sampled])
    labels_json = json.dumps(labels)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alpha Quant — {symbol} Backtest {ts}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  :root{{
    --bg:#0A0F1C;--bg2:#111827;--bg3:#1F2937;
    --txt:#F3F4F6;--txt2:#9CA3AF;--txt3:#6B7280;
    --brd:#374151;--brd2:#4B5563;
    --red:#F43F5E;--red-bg:rgba(244,63,94,0.1);--red-brd:rgba(244,63,94,0.3);
    --grn:#10B981;--grn-bg:rgba(16,185,129,0.1);--grn-brd:rgba(16,185,129,0.3);
    --info:#38BDF8;--info-bg:rgba(56,189,248,0.1);
    --accent:#6366F1;
    --mono:'JetBrains Mono',monospace;
    --sans:'Inter',-apple-system,sans-serif;
    --rad:8px;--rad-lg:16px;
  }}
  body{{background:var(--bg);color:var(--txt);font-family:var(--sans);font-size:14px;line-height:1.5;padding:32px 24px;background-image:radial-gradient(ellipse at top, #1E1B4B 0%, var(--bg) 50%);background-attachment:fixed;}}
  .page{{max-width:1100px;margin:0 auto}}
  .sep{{border:none;border-top:1px solid var(--brd);margin:16px 0}}
  .panel{{background:var(--bg2);border:1px solid var(--brd);border-radius:var(--rad-lg);padding:24px;margin-bottom:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1),0 2px 4px -2px rgba(0,0,0,0.1);position:relative;overflow:hidden}}
  .panel::after{{content:"";position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);}}
  .panel-dark{{background:var(--bg3);border:1px solid var(--brd)}}
  .hdr-title{{font-size:24px;font-weight:700;color:var(--txt);letter-spacing:-0.02em}}
  .hdr-sub{{font-size:13px;color:var(--info);margin-top:6px;font-family:var(--mono)}}
  .cfg-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:16px}}
  .cfg-item{{background:var(--bg3);border-radius:var(--rad);padding:12px;border:1px solid var(--brd);position:relative;overflow:hidden}}
  .cfg-label{{font-size:10px;color:var(--txt2);text-transform:uppercase;letter-spacing:0.1em;font-weight:600}}
  .cfg-val{{font-size:14px;font-weight:600;color:var(--txt);margin-top:6px;font-family:var(--mono)}}
  .metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:16px}}
  .mcard{{border-radius:var(--rad-lg);padding:20px;border:1px solid;position:relative;background:var(--bg3);display:flex;flex-direction:column;justify-content:center}}
  .mcard::before{{content:"";position:absolute;top:0;left:0;bottom:0;width:4px;border-top-left-radius:var(--rad-lg);border-bottom-left-radius:var(--rad-lg)}}
  .mcard .lbl{{font-size:12px;color:var(--txt2);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;font-weight:600}}
  .mcard .val{{font-size:32px;font-weight:700;letter-spacing:-0.03em;font-family:var(--mono)}}
  .mcard .sub{{font-size:12px;margin-top:8px;font-family:var(--mono);opacity:0.8}}
  .mcard-danger{{border-color:var(--red-brd);color:var(--red)}}
  .mcard-danger::before{{background:var(--red)}}
  .mcard-success{{border-color:var(--grn-brd);color:var(--grn)}}
  .mcard-success::before{{background:var(--grn)}}
  .stats-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
  .block-title{{font-size:12px;font-weight:700;color:var(--txt);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px;display:flex;align-items:center}}
  .block-title::before{{content:"";display:inline-block;width:8px;height:8px;background:var(--accent);border-radius:50%;margin-right:8px;box-shadow:0 0 8px var(--accent)}}
  .stat-row{{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--brd)}}
  .stat-row:last-child{{border-bottom:none}}
  .sk{{font-size:13px;color:var(--txt2)}}
  .sv{{font-size:13px;font-weight:600;font-family:var(--mono)}}
  .sv-good{{color:var(--grn);text-shadow:0 0 10px rgba(16,185,129,0.3)}}
  .sv-bad{{color:var(--red);text-shadow:0 0 10px rgba(244,63,94,0.3)}}
  .val-header{{display:flex;align-items:center;gap:12px;margin-bottom:16px}}
  .val-badge{{font-size:12px;font-weight:700;padding:4px 12px;border-radius:6px;letter-spacing:0.05em;text-transform:uppercase}}
  .val-fail{{background:var(--red-bg);color:var(--red);border:1px solid var(--red-brd);box-shadow:0 0 15px rgba(244,63,94,0.2)}}
  .val-pass{{background:var(--grn-bg);color:var(--grn);border:1px solid var(--grn-brd);box-shadow:0 0 15px rgba(16,185,129,0.2)}}
  .val-row{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:var(--bg3);border:1px solid var(--brd);border-radius:var(--rad);margin-bottom:8px}}
  .val-name{{font-size:14px;color:var(--txt);font-weight:600}}
  .val-sub{{font-size:12px;color:var(--txt3);margin-top:4px;font-family:var(--mono)}}
  .pill{{font-size:11px;font-weight:700;padding:4px 10px;border-radius:6px;letter-spacing:0.05em}}
  .pill-ok{{background:var(--grn-bg);color:var(--grn);border:1px solid var(--grn-brd)}}
  .pill-fail{{background:var(--red-bg);color:var(--red);border:1px solid var(--red-brd)}}
  .report-path{{font-family:var(--mono);font-size:12px;color:var(--accent);word-break:break-all;margin-top:6px;background:var(--bg);padding:8px 12px;border-radius:var(--rad);border:1px solid var(--brd)}}
  .footer{{text-align:center;font-size:12px;color:var(--txt3);margin-top:32px;font-family:var(--mono)}}
  @media(max-width:900px){{
    .cfg-grid{{grid-template-columns:repeat(3,1fr)}}
  }}
  @media(max-width:600px){{
    .metric-grid,.cfg-grid{{grid-template-columns:repeat(2,1fr)}}
    .stats-grid{{grid-template-columns:1fr}}
  }}
</style>
</head>
<body>
<div class="page">

<div class="panel">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
    <div>
      <div class="hdr-title">HMM Bot — Quantitative Research Framework</div>
      <div class="hdr-sub">Backtest completed &middot; {range_str} &middot; {symbol} &middot; {timeframe}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:11px;color:var(--txt3)">run at</div>
      <div style="font-size:12px;font-weight:600">{run_time}</div>
    </div>
  </div>
</div>

<div class="cfg-grid">
  <div class="cfg-item"><div class="cfg-label">Mode</div><div class="cfg-val">backtest</div></div>
  <div class="cfg-item"><div class="cfg-label">Bars (test)</div><div class="cfg-val">{total_trades} trades on {bars_val:,} bars</div></div>
  <div class="cfg-item"><div class="cfg-label">Starting balance</div><div class="cfg-val">${getattr(args,'balance',10000):,.2f}</div></div>
  <div class="cfg-item"><div class="cfg-label">HMM</div><div class="cfg-val">trained</div></div>
  <div class="cfg-item"><div class="cfg-label">Spread / slippage</div><div class="cfg-val">{getattr(args,'spread',1.5)}p / {getattr(args,'slippage',0.5)}p</div></div>
  <div class="cfg-item"><div class="cfg-label">Commission</div><div class="cfg-val">${getattr(args,'commission',6.0):.2f} / lot</div></div>
</div>

<div class="metric-grid">
  <div class="mcard mcard-{profit_color}">
    <div class="lbl">Net profit</div>
    <div class="val">${net_profit:+,.2f}</div>
    <div class="sub">{net_pct:+.2f}% return</div>
  </div>
  <div class="mcard mcard-{'success' if win_rate_val >= 45 else 'danger'}">
    <div class="lbl">Win rate</div>
    <div class="val">{win_rate_val:.1f}%</div>
    <div class="sub">{wins}W / {losses}L / {total_trades} total</div>
  </div>
  <div class="mcard mcard-{'success' if pf_val >= 1.2 else 'danger'}">
    <div class="lbl">Profit factor</div>
    <div class="val">{pf_val:.2f}</div>
    <div class="sub">needs &gt; 1.20</div>
  </div>
  <div class="mcard mcard-{'success' if dd_val <= 5 else 'danger'}">
    <div class="lbl">Max drawdown</div>
    <div class="val">{dd_val:.2f}%</div>
    <div class="sub">limit: 5%</div>
  </div>
</div>

<div class="panel">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <div class="block-title" style="margin:0">Equity curve</div>
    <div style="display:flex;gap:16px;font-size:11px;color:var(--txt3)">
      <span style="display:flex;align-items:center;gap:4px">
        <span style="width:14px;height:2px;background:#E24B4A;display:inline-block"></span>Balance
      </span>
      <span style="display:flex;align-items:center;gap:4px">
        <span style="width:14px;height:2px;background:#888780;display:inline-block;border-top:2px dashed #888780"></span>Baseline
      </span>
    </div>
  </div>
  <div style="position:relative;width:100%;height:220px">
    <canvas id="eqChart"></canvas>
  </div>
</div>

<div class="stats-grid">
  <div class="panel">
    <div class="block-title">PnL breakdown</div>
    <div class="stat-row"><span class="sk">Gross profit</span><span class="sv sv-good">+${float(metrics.get('gross_profit',0)):,.2f}</span></div>
    <div class="stat-row"><span class="sk">Gross loss</span><span class="sv sv-bad">-${abs(float(metrics.get('gross_loss',0))):,.2f}</span></div>
    <div class="stat-row"><span class="sk">Avg win</span><span class="sv sv-good">+${float(metrics.get('avg_win',0)):,.2f}</span></div>
    <div class="stat-row"><span class="sk">Avg loss</span><span class="sv sv-bad">-${abs(float(metrics.get('avg_loss',0))):,.2f}</span></div>
    <div class="stat-row"><span class="sk">Expectancy</span><span class="sv sv-bad">${float(metrics.get('expectancy',0)):+,.2f} / trade</span></div>
  </div>
  <div class="panel">
    <div class="block-title">Risk metrics</div>
    <div class="stat-row"><span class="sk">Sharpe ratio</span><span class="sv sv-bad">{float(metrics.get('sharpe_ratio',0)):.3f}</span></div>
    <div class="stat-row"><span class="sk">Sortino ratio</span><span class="sv sv-bad">{float(metrics.get('sortino_ratio',0)):.3f}</span></div>
    <div class="stat-row"><span class="sk">Max drawdown</span><span class="sv sv-bad">{dd_val:.2f}%</span></div>
    <div class="stat-row"><span class="sk">Total trades</span><span class="sv">{total_trades}</span></div>
    <div class="stat-row"><span class="sk">Wins / Losses</span><span class="sv">{wins} / {losses}</span></div>
  </div>
</div>

</div>

<div class="footer">HMM Adaptive Trading Bot &middot; {symbol} &middot; Generated {run_time}</div>

</div>

<script>
const eq = {eq_json};
const labels = {labels_json};
const baseline = eq.map(() => {getattr(args,'balance',10000)});
const gridCol = 'rgba(255,255,255,0.05)';
const tickCol = '#6B7280';
new Chart(document.getElementById('eqChart'), {{
  type:'line',
  data:{{
    labels,
    datasets:[
      {{label:'Balance',data:eq,borderColor:'#06B6D4',borderWidth:2,pointRadius:0,pointHoverRadius:4,tension:0.3,fill:true,backgroundColor:'rgba(6,182,212,0.1)'}},
      {{label:'Baseline',data:baseline,borderColor:'#4B5563',borderWidth:1,borderDash:[4,4],pointRadius:0,tension:0}}
    ]
  }},
  options:{{
    responsive:true,maintainAspectRatio:false,
    interaction:{{mode:'index',intersect:false}},
    plugins:{{
      legend:{{display:false}},
      tooltip:{{
        backgroundColor:'#1F2937',
        borderColor:'#374151',
        borderWidth:1,
        titleColor:'#F3F4F6',
        bodyColor:'#9CA3AF',
        padding:12,
        titleFont:{{family:'JetBrains Mono',size:13}},
        bodyFont:{{family:'JetBrains Mono',size:13}},
        callbacks:{{label:ctx=>ctx.dataset.label==='Balance'?' $'+ctx.parsed.y.toFixed(2):null}}
      }}
    }},
    scales:{{
      x:{{grid:{{color:gridCol,drawTicks:false}},ticks:{{color:tickCol,font:{{family:'JetBrains Mono',size:10}},maxTicksLimit:10,maxRotation:0}},border:{{display:false}}}},
      y:{{grid:{{color:gridCol}},ticks:{{color:tickCol,font:{{family:'JetBrains Mono',size:11}},callback:v=>'$'+v.toLocaleString()}},border:{{display:false}}}}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.abspath(filepath)