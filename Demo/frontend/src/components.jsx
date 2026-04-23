/* global React */
const { useState, useEffect, useMemo } = React;

// ---------------- status helpers ----------------
function StatusPill({ level, size }) {
  const lv = (level || "").toLowerCase();
  const cls = lv === "low" ? "pill-low" : lv === "medium" ? "pill-medium" : lv === "high" ? "pill-high" : "pill-ghost";
  return (
    <span className={`pill ${cls} ${size === "lg" ? "pill-lg" : ""}`}>
      <span className="pdot" />
      {level || "—"}
    </span>
  );
}

function MetricCard({ title, value, unit, sub, subStatus }) {
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="metric">
        <div className="metric-value mono">
          {value}
          {unit && <span className="unit">{unit}</span>}
        </div>
        {sub && (
          <div className="metric-sub mono">
            {subStatus && <StatusPill level={subStatus} />}
            {subStatus && "  "}
            {sub}
          </div>
        )}
      </div>
    </div>
  );
}

function ForecastTile({ day, value, status, confidence }) {
  return (
    <div className="forecast-tile">
      <div className="day mono">{day}</div>
      <div className="val mono">
        {value.toFixed(2)}
        <span className="unit">ACPS</span>
      </div>
      <div>
        <StatusPill level={status} />
      </div>
      <div className="conf mono">± {confidence.toFixed(2)} · 95% CI</div>
    </div>
  );
}

function Segmented({ options, value, onChange }) {
  return (
    <div className="segmented">
      {options.map((o) => {
        const v = typeof o === "object" ? o.value : o;
        const lbl = typeof o === "object" ? o.label : o;
        return (
          <button key={v} className={value === v ? "on" : ""} onClick={() => onChange(v)}>
            {lbl}
          </button>
        );
      })}
    </div>
  );
}

function Slider({ min, max, step, value, label, formatValue, onChange, scale }) {
  return (
    <div style={{ marginTop: 14 }}>
      <div className="slider-row">
        <div className="slider-label">{label}</div>
        <div className="slider-val mono">{formatValue ? formatValue(value) : value}</div>
      </div>
      <input
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="slider-scale">
        {(scale || [min, max]).map((s, i) => (
          <span key={i}>{s}</span>
        ))}
      </div>
    </div>
  );
}

function Toggle({ label, value, onChange }) {
  return (
    <div className="toggle" onClick={() => onChange(!value)} style={{ cursor: "pointer" }}>
      <div className="toggle-label">{label}</div>
      <div className={`switch ${value ? "on" : ""}`} />
    </div>
  );
}

function FeatureBar({ feature, importance, max }) {
  const pct = Math.max(4, Math.round((importance / max) * 100));
  return (
    <div className="featurebar">
      <div className="fname">{feature}</div>
      <div className="ftrack">
        <div className="ffill" style={{ width: `calc(${pct}% + 2px)` }} />
      </div>
      <div className="fval">{importance.toFixed(3)}</div>
    </div>
  );
}

function HealthBadge({ online }) {
  return (
    <span className={`health ${online ? "" : "offline"}`}>
      <span className="dot" />
      {online ? "BACKEND ONLINE" : "BACKEND OFFLINE"}
    </span>
  );
}

// ---------------- chart placeholder ----------------
function ChartPlaceholder({ height, note, kind }) {
  // kind: 'line' | 'bar'
  const yTicks = ["90", "80", "70", "60", "50"];
  const xTicks = ["DEC 12", "DEC 14", "DEC 16", "DEC 18", "DEC 20", "DEC 22"];
  // rough threshold at y=78 → position ~ (90-78)/(90-50) = 0.30 from top
  const thresholdTop = `calc(12px + (100% - 40px) * 0.30)`;
  // target line for date 2025-12-19: index 7 of 12 context days → x ≈ 58%
  const targetLeft = `calc(48px + (100% - 60px) * 0.636)`;
  return (
    <div className="chart-box" style={{ height: height || 380 }}>
      <div className="chart-placeholder-tag">{note || "LINE CHART · Recharts <LineChart>"}</div>
      <div className="chart-legend">
        <span className="legend-item"><span className="legend-swatch" /> ACTUAL</span>
        <span className="legend-item"><span className="legend-swatch dashed" /> PREDICTED</span>
      </div>
      {kind !== "bar" && (
        <>
          <div className="chart-yaxis mono">
            {yTicks.map((t) => <span key={t}>{t}</span>)}
          </div>
          <div className="chart-xaxis mono">
            {xTicks.map((t) => <span key={t}>{t}</span>)}
          </div>
          <div className="chart-threshold" style={{ top: thresholdTop }}>
            <span className="chart-threshold-label">HIGH THRESHOLD · 78 ACPS</span>
          </div>
          <div className="chart-target-line" style={{ left: targetLeft }}>
            <span className="chart-target-label">TARGET · DEC 19</span>
          </div>
          {/* faux sparkline via SVG to hint at shape without being real data */}
          <svg
            viewBox="0 0 800 380"
            preserveAspectRatio="none"
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          >
            {/* actual: solid */}
            <polyline
              fill="none"
              stroke="var(--text-primary)"
              strokeWidth="2"
              points="60,240 120,260 180,290 240,220 300,190 360,180 420,170 480,160 540,190 600,220 660,250 720,260"
            />
            {/* predicted: dashed */}
            <polyline
              fill="none"
              stroke="var(--accent)"
              strokeWidth="2"
              strokeDasharray="6 5"
              points="60,234 120,256 180,282 240,226 300,196 360,184 420,166 480,150 540,184 600,214 660,246 720,258"
            />
            {/* dots */}
            {[[60,240],[120,260],[180,290],[240,220],[300,190],[360,180],[420,170],[480,160],[540,190],[600,220],[660,250],[720,260]].map(([x,y],i)=>(
              <circle key={i} cx={x} cy={y} r="3" fill="var(--bg-primary)" stroke="var(--text-primary)" strokeWidth="1.5" />
            ))}
          </svg>
        </>
      )}
      {kind === "bar" && (
        <div style={{ position: "absolute", inset: 24, display: "flex", flexDirection: "column", gap: 10, justifyContent: "center" }}>
          {/* bar chart handled separately via FeatureBar list */}
        </div>
      )}
    </div>
  );
}

// ---------------- confusion matrix ----------------
function ConfusionMatrix({ labels, matrix }) {
  const max = Math.max(...matrix.flat());
  const cellBg = (r, c, v) => {
    const t = v / max;
    if (r === c) {
      // diagonal: ok-tinted
      return `color-mix(in oklab, var(--ok) ${Math.round(t * 40)}%, var(--bg-surface))`;
    }
    if (v === 0) return "var(--bg-surface)";
    return `color-mix(in oklab, var(--crit) ${Math.round(t * 25 + 6)}%, var(--bg-surface))`;
  };
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "80px repeat(3, 1fr)", gap: 1, background: "var(--border)", border: "1px solid var(--border)" }}>
        <div className="cm-cell cm-head" style={{ background: "var(--bg-surface-2)" }}>PRED ▸</div>
        {labels.map((l) => (
          <div key={l} className="cm-cell cm-head" style={{ background: "var(--bg-surface-2)" }}>{l.toUpperCase()}</div>
        ))}
        {matrix.map((row, r) => (
          <React.Fragment key={r}>
            <div className="cm-cell cm-head" style={{ background: "var(--bg-surface-2)" }}>▾ {labels[r].toUpperCase()}</div>
            {row.map((v, c) => (
              <div key={c} className="cm-cell" style={{ background: cellBg(r, c, v), minHeight: 78 }}>
                <span className="cm-val mono">{v}</span>
              </div>
            ))}
          </React.Fragment>
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10, fontFamily: "'JetBrains Mono', monospace", fontSize: 10.5, color: "var(--text-muted)", letterSpacing: "0.12em", textTransform: "uppercase" }}>
        <span>Rows: actual · Cols: predicted</span>
        <span>n = {matrix.flat().reduce((a, b) => a + b, 0)}</span>
      </div>
    </div>
  );
}

// ---------------- baseline table ----------------
function BaselineTable({ rows }) {
  return (
    <table className="btable">
      <thead>
        <tr>
          <th>Model</th>
          <th className="num">MAE</th>
          <th className="num">RMSE</th>
          <th className="num">R²</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className={i === 0 ? "ours" : ""}>
            <td>{r.model}{i === 0 && <span style={{ color: "var(--accent)", marginLeft: 10, fontSize: 10 }}>◆ SELECTED</span>}</td>
            <td className="num">{r.mae.toFixed(3)}</td>
            <td className="num">{r.rmse.toFixed(3)}</td>
            <td className="num">{r.r2.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

Object.assign(window, {
  StatusPill, MetricCard, ForecastTile, Segmented, Slider, Toggle,
  FeatureBar, HealthBadge, ChartPlaceholder, ConfusionMatrix, BaselineTable,
});
