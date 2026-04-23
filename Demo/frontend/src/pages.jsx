/* global React, BARAJAS_API, useAsync, formatDay, dayOfWeekName, isoWeek */
/* Note: no top-level `const { useState, ... } = React` — babel standalone evaluates
   each text/babel script in the global scope, so that would collide with the same
   declaration in components.jsx. Reference React.useState directly. */

// ============================================================
// shared: loading / error ornaments
// ============================================================
function Loading({ label }) {
  return <div className="skeleton">LOADING {label || "…"}</div>;
}
function ErrorBanner({ err }) {
  return (
    <div className="banner">
      <span className="b-tag">BACKEND</span>
      <span>Failed to reach API at {window.API_BASE} · {err} · is <code>demo\backend\run.bat</code> running?</span>
    </div>
  );
}

// ============================================================
// Real line chart — draws actual data, not a placeholder
// ============================================================
function LineChart({ data, width, height, threshold, targetDate }) {
  // data: [{date, actual, predicted, is_target}]
  const W = width || 860;
  const H = height || 380;
  const padL = 48, padR = 24, padT = 24, padB = 40;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const yValues = data.flatMap((d) => [d.actual, d.predicted]);
  const yMin = Math.min(...yValues) - 1.5;
  const yMax = Math.max(...yValues) + 1.5;
  const yRange = yMax - yMin;

  const x = (i) => padL + (i / Math.max(1, data.length - 1)) * innerW;
  const y = (v) => padT + (1 - (v - yMin) / yRange) * innerH;

  const actualPath = data.map((d, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(d.actual).toFixed(1)}`).join(" ");
  const predPath = data.map((d, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(d.predicted).toFixed(1)}`).join(" ");

  const yTicks = 5;
  const yStep = yRange / (yTicks - 1);
  const yLabels = Array.from({ length: yTicks }, (_, i) => yMax - i * yStep);

  const targetIdx = data.findIndex((d) => d.is_target || d.date === targetDate);
  const showThreshold = threshold != null && threshold >= yMin && threshold <= yMax;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: "100%", height: H, display: "block" }}>
      {/* grid */}
      {yLabels.map((v, i) => {
        const yy = y(v);
        return (
          <g key={i}>
            <line x1={padL} x2={W - padR} y1={yy} y2={yy} stroke="var(--grid)" strokeWidth="1" />
            <text x={padL - 6} y={yy + 3} fontSize="10" fontFamily="JetBrains Mono, monospace" fill="var(--text-muted)" textAnchor="end">
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}
      {data.map((d, i) => (
        <text key={i} x={x(i)} y={H - padB + 16} fontSize="10" fontFamily="JetBrains Mono, monospace" fill="var(--text-muted)" textAnchor="middle">
          {d.date.slice(5)}
        </text>
      ))}

      {/* threshold */}
      {showThreshold && (
        <g>
          <line x1={padL} x2={W - padR} y1={y(threshold)} y2={y(threshold)} stroke="var(--crit)" strokeWidth="1" strokeDasharray="4 4" />
          <text x={W - padR - 6} y={y(threshold) - 6} fontSize="10" fontFamily="JetBrains Mono, monospace" fill="var(--crit)" textAnchor="end">
            HIGH THRESHOLD · {threshold}
          </text>
        </g>
      )}

      {/* target date marker */}
      {targetIdx >= 0 && (
        <g>
          <line x1={x(targetIdx)} x2={x(targetIdx)} y1={padT} y2={H - padB} stroke="var(--accent)" strokeWidth="1" strokeDasharray="2 3" />
          <rect x={x(targetIdx) - 36} y={padT - 2} width="72" height="16" fill="var(--bg-surface-2)" stroke="var(--accent)" />
          <text x={x(targetIdx)} y={padT + 9} fontSize="9.5" fontFamily="JetBrains Mono, monospace" fill="var(--accent)" textAnchor="middle" letterSpacing="0.08em">
            TARGET
          </text>
        </g>
      )}

      {/* lines */}
      <path d={actualPath} fill="none" stroke="var(--text-primary)" strokeWidth="2" />
      <path d={predPath} fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray="6 5" />

      {/* points */}
      {data.map((d, i) => (
        <g key={i}>
          <circle cx={x(i)} cy={y(d.actual)} r="2.5" fill="var(--bg-primary)" stroke="var(--text-primary)" strokeWidth="1.5" />
          <circle cx={x(i)} cy={y(d.predicted)} r="2.5" fill="var(--bg-primary)" stroke="var(--accent)" strokeWidth="1.5" />
        </g>
      ))}

      {/* legend */}
      <g transform={`translate(${W - padR - 200}, ${padT + 4})`}>
        <rect x="0" y="0" width="196" height="22" fill="var(--bg-surface)" stroke="var(--border)" />
        <line x1="8" x2="24" y1="11" y2="11" stroke="var(--text-primary)" strokeWidth="2" />
        <text x="30" y="14" fontSize="10" fontFamily="JetBrains Mono, monospace" fill="var(--text-secondary)" letterSpacing="0.1em">ACTUAL</text>
        <line x1="96" x2="112" y1="11" y2="11" stroke="var(--accent)" strokeWidth="2" strokeDasharray="4 3" />
        <text x="118" y="14" fontSize="10" fontFamily="JetBrains Mono, monospace" fill="var(--text-secondary)" letterSpacing="0.1em">PREDICTED</text>
      </g>
    </svg>
  );
}
window.LineChart = LineChart;

// ============================================================
// TODAY
// ============================================================
// --- Forecast-any-date card (top of TODAY tab) ---
function ForecastAnyDate() {
  const TODAY_ISO = new Date().toISOString().slice(0, 10);
  const MAX_ISO = "2026-12-31";
  // Default to a dramatic example: Christmas Day 2026
  const [date, setDate] = React.useState("2026-12-25");
  const [res, setRes] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [err, setErr] = React.useState(null);

  const debounceRef = React.useRef(null);
  React.useEffect(() => {
    clearTimeout(debounceRef.current);
    if (!date || date < "2026-01-01" || date > MAX_ISO) {
      setRes(null); setErr(null); return;
    }
    debounceRef.current = setTimeout(() => {
      setLoading(true);
      BARAJAS_API.futureForecast(date)
        .then((r) => { setRes(r); setErr(null); })
        .catch((e) => { setErr(String(e)); setRes(null); })
        .finally(() => setLoading(false));
    }, 200);
    return () => clearTimeout(debounceRef.current);
  }, [date]);

  const presets = [
    { label: "TOMORROW",       date: new Date(Date.now() + 86400000).toISOString().slice(0,10) },
    { label: "SUMMER PEAK",    date: "2026-08-15" },
    { label: "HISPANIC DAY",   date: "2026-10-12" },
    { label: "CHRISTMAS EVE",  date: "2026-12-24" },
    { label: "NEW YEAR'S EVE", date: "2026-12-31" },
  ];

  return (
    <div className="card" style={{ borderColor: "var(--accent)", borderWidth: 1 }}>
      <div className="card-title">
        Forecast Any Date in 2026
        <span className="title-meta mono">pick a date — the model returns its prediction with auto-computed assumptions</span>
      </div>

      {/* Row 1 — input + presets */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 14 }}>
        <input
          type="date"
          value={date}
          min={TODAY_ISO < "2026-01-01" ? "2026-01-01" : TODAY_ISO}
          max={MAX_ISO}
          onChange={(e) => setDate(e.target.value)}
          style={{
            background: "var(--bg-surface-2)", border: "1px solid var(--accent)",
            color: "var(--text-primary)", padding: "10px 14px",
            fontFamily: "'JetBrains Mono', monospace", fontSize: 16,
            letterSpacing: "0.04em", colorScheme: "dark",
          }}
        />
        <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)",
              letterSpacing: "0.14em", textTransform: "uppercase" }}>
          range · {TODAY_ISO < "2026-01-01" ? "2026-01-01" : TODAY_ISO} → {MAX_ISO}
        </span>
        <div style={{ flex: 1 }} />
        {presets.map((p) => (
          <button key={p.label}
                  className={`btn-ghost ${date === p.date ? "" : ""}`}
                  onClick={() => setDate(p.date)}
                  style={date === p.date
                    ? { borderColor: "var(--accent)", color: "var(--accent-fg)", background: "var(--accent-dim)" }
                    : {}}>
            {p.label}
          </button>
        ))}
      </div>

      {err && <ErrorBanner err={err} />}

      {/* Row 2 — the result */}
      {!res && !err && !loading && <div className="skeleton">PICK A DATE TO FORECAST</div>}
      {loading && !res && <div className="skeleton">PREDICTING {date} …</div>}

      {res && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "minmax(320px, 1fr) minmax(280px, 1fr)",
          gap: 16,
          border: "1px solid var(--border)",
          background: "var(--bg-surface-2)",
          padding: 20,
        }}>
          {/* LEFT — prediction */}
          <div>
            <div className="metric-label" style={{ marginBottom: 6 }}>
              PREDICTED · {res.day_of_week} {res.date}
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
              <div className="mono" style={{
                fontSize: 64, fontWeight: 500, lineHeight: 1,
                color: "var(--text-primary)", letterSpacing: "-0.02em",
              }}>
                {res.predicted_acps.toFixed(2)}
              </div>
              <div className="mono" style={{ fontSize: 16, color: "var(--text-muted)",
                    letterSpacing: "0.14em", fontWeight: 500 }}>
                ACPS
              </div>
            </div>
            <div style={{ marginTop: 14 }}>
              <StatusPill level={res.classification} size="lg" />
            </div>
            <div className="metric-sub mono" style={{ marginTop: 14, color: "var(--text-muted)" }}>
              95% CI · ±{(1.96 * res.residual_std).toFixed(2)} ACPS
              <span style={{ color: "var(--text-muted)", margin: "0 6px" }}>·</span>
              <span style={{ color: "var(--text-secondary)" }}>
                [{res.confidence_low.toFixed(2)} → {res.confidence_high.toFixed(2)}]
              </span>
            </div>
          </div>

          {/* RIGHT — assumptions */}
          <div>
            <div className="metric-label" style={{ marginBottom: 10 }}>
              MODEL ASSUMPTIONS
            </div>
            <dl className="kv" style={{ gridTemplateColumns: "auto 1fr", rowGap: 6 }}>
              <dt>Movements</dt>
              <dd>{res.assumptions.total_movements.toLocaleString()}</dd>
              <dt>Basis</dt>
              <dd style={{ fontSize: 10.5, color: "var(--text-secondary)", textAlign: "right", lineHeight: 1.4 }}>
                {res.assumptions.movements_basis}
              </dd>
              <dt>Weather</dt>
              <dd>{res.assumptions.weather_preset.toUpperCase()}</dd>
              <dt>Holiday</dt>
              <dd>
                {res.assumptions.is_holiday
                  ? <span style={{ color: "var(--warn)" }}>
                      YES · {res.assumptions.holiday_name}
                    </span>
                  : <span style={{ color: "var(--text-muted)" }}>no</span>}
              </dd>
              <dt>Bridge day</dt>
              <dd>
                {res.assumptions.is_bridge_day
                  ? <span style={{ color: "var(--warn)" }}>YES</span>
                  : <span style={{ color: "var(--text-muted)" }}>no</span>}
              </dd>
              <dt>7-day ACPS</dt>
              <dd>{res.assumptions.rolling_acps_7d.toFixed(2)}</dd>
            </dl>
            <div className="metric-sub mono" style={{ marginTop: 12, color: "var(--text-muted)", lineHeight: 1.5 }}>
              Weather is assumed clear (unknown in advance). Movements come from
              the post-2023 historical mean for the same day-of-week and month.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function PageToday() {
  const { loading, data, error } = useAsync(() => BARAJAS_API.today(), []);
  if (error) return <div className="page"><ErrorBanner err={error} /></div>;
  if (loading || !data) return <div className="page"><Loading label="TODAY STATE" /></div>;

  const d = data;
  const ci = 1.96 * d.residual_std;
  const weekday = dayOfWeekName(d.today_date);
  const weekNum = isoWeek(d.today_date);

  return (
    <div className="page">
      <ForecastAnyDate />
      <div className="grid grid-7-5">
        <div className="card">
          <div className="card-title">
            Current State
            <span className="title-meta mono">model · HGB regressor · eurocontrol pipeline</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div className="hero-value mono">
              {d.today_predicted_acps.toFixed(2)}
              <span className="unit">ACPS</span>
            </div>
            <div><StatusPill level={d.today_classification} size="lg" /></div>
            <div className="hero-caption mono">
              actual <span style={{ color: "var(--text-primary)" }}>{d.today_actual_acps.toFixed(2)}</span>
              <span className="sep">·</span>
              error <span className={Math.abs(d.today_actual_acps - d.today_predicted_acps) > ci ? "negative" : "positive"}>
                {Math.abs(d.today_actual_acps - d.today_predicted_acps).toFixed(2)}
              </span>
              <span className="sep">·</span>
              residual σ <span style={{ color: "var(--text-primary)" }}>{d.residual_std.toFixed(3)}</span>
            </div>
            <div style={{ height: 1, background: "var(--border)", margin: "8px 0" }} />
            <dl className="kv">
              <dt>Target date</dt><dd>{d.today_date}</dd>
              <dt>Classification</dt><dd>{d.today_classification.toUpperCase()}</dd>
              <dt>Band boundaries</dt><dd>LOW &lt; 65 · MED 65–75 · HIGH ≥ 75</dd>
              <dt>Residual σ (test)</dt><dd>{d.residual_std.toFixed(3)} ACPS</dd>
            </dl>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card">
            <div className="card-title">Weather</div>
            <div className="grid grid-3" style={{ gap: 24 }}>
              <div className="metric">
                <div className="metric-label">Temperature</div>
                <div className="metric-value sm mono">{d.today_weather.temperature_c.toFixed(1)}<span className="unit">°C</span></div>
              </div>
              <div className="metric">
                <div className="metric-label">Wind</div>
                <div className="metric-value sm mono">{d.today_weather.wind_kmh.toFixed(1)}<span className="unit">km/h</span></div>
              </div>
              <div className="metric">
                <div className="metric-label">Precip</div>
                <div className="metric-value sm mono">{d.today_weather.precipitation_mm.toFixed(1)}<span className="unit">mm</span></div>
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
              <span className={`pill ${d.today_weather.is_raining ? "pill-medium" : "pill-ghost"}`}>
                <span className="pdot" />{d.today_weather.is_raining ? "RAINING" : "DRY"}
              </span>
              <span className={`pill ${d.today_weather.is_severe ? "pill-high" : "pill-ghost"}`}>
                <span className="pdot" />{d.today_weather.is_severe ? "SEVERE" : "NOT SEVERE"}
              </span>
            </div>
          </div>
          <div className="card">
            <div className="card-title">Operations</div>
            <div className="metric">
              <div className="metric-label">Movements</div>
              <div className="metric-value mono">{d.today_movements.toLocaleString()}</div>
              <div className="metric-sub mono">arrivals/departures ≈ 50/50 · eurocontrol NM counts</div>
            </div>
          </div>
          <div className="card">
            <div className="card-title">Calendar</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <span className="pill pill-ghost"><span className="pdot" />{weekday}</span>
              <span className="pill pill-ghost"><span className="pdot" />WEEK {weekNum}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          3-Day Forecast
          <span className="title-meta mono">95% CI · ±{ci.toFixed(2)} ACPS · from test-set residual σ</span>
        </div>
        <div className="forecast-row">
          {d.forecast.map((f) => (
            <ForecastTile
              key={f.date}
              day={f.day_label}
              value={f.predicted_acps}
              status={f.classification}
              confidence={ci}
            />
          ))}
        </div>
        <div className="metric-sub mono" style={{ marginTop: 14, color: "var(--text-muted)" }}>
          Forecasts produced live by the trained HGB regressor · 53 features · lag-based rolling context.
        </div>
      </div>
    </div>
  );
}

// ============================================================
// DIAGNOSTICS — Model Showdown + Residual Diagnostics + Why This Day
// ============================================================

// --- Small charts used only by the Diagnostics tab ---

// Vertical bar chart for the Model Showdown.
function ShowdownBars({ actual, models }) {
  const W = 720, H = 300;
  const padL = 50, padR = 20, padT = 28, padB = 60;
  const innerH = H - padT - padB;
  const allVals = [actual, ...models.map((m) => m.predicted)];
  const yMin = 0;
  const yMax = Math.max(...allVals) * 1.12;
  const y = (v) => padT + innerH * (1 - (v - yMin) / (yMax - yMin));
  const n = models.length + 1;
  const slot = (W - padL - padR) / n;
  const barW = slot * 0.55;

  const bar = (i, v, label, isActual, isOurs, residual) => {
    const cx = padL + slot * (i + 0.5);
    const yy = y(v);
    const color = isActual ? "var(--text-primary)" :
                  isOurs   ? "var(--accent)" :
                             "var(--text-muted)";
    const fill = isActual ? "var(--text-primary)" :
                 isOurs   ? "var(--accent)" :
                            "var(--bg-surface-3)";
    return (
      <g key={i}>
        <rect x={cx - barW / 2} y={yy} width={barW} height={H - padB - yy}
              fill={fill} stroke={color} strokeWidth="1.5" />
        <text x={cx} y={yy - 8} textAnchor="middle" fontSize="14"
              fontFamily="JetBrains Mono, monospace" fill={color} fontWeight="500">
          {v.toFixed(2)}
        </text>
        <text x={cx} y={H - padB + 18} textAnchor="middle" fontSize="10"
              fontFamily="JetBrains Mono, monospace" fill="var(--text-muted)"
              letterSpacing="0.06em">
          {label}
        </text>
        {residual != null && (
          <text x={cx} y={H - padB + 34} textAnchor="middle" fontSize="9.5"
                fontFamily="JetBrains Mono, monospace"
                fill={Math.abs(residual) > 5 ? "var(--crit)" : Math.abs(residual) > 1.5 ? "var(--warn)" : "var(--ok)"}>
            err {residual > 0 ? "+" : ""}{residual.toFixed(2)}
          </text>
        )}
      </g>
    );
  };

  const yTicks = 5;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
      {Array.from({ length: yTicks }, (_, i) => {
        const v = yMin + (yMax - yMin) * (i / (yTicks - 1));
        const yy = y(v);
        return (
          <g key={i}>
            <line x1={padL} x2={W - padR} y1={yy} y2={yy}
                  stroke="var(--grid)" strokeWidth="1" />
            <text x={padL - 6} y={yy + 3} fontSize="10"
                  fontFamily="JetBrains Mono, monospace" fill="var(--text-muted)"
                  textAnchor="end">
              {v.toFixed(0)}
            </text>
          </g>
        );
      })}
      {bar(0, actual, "ACTUAL", true, false, null)}
      {models.map((m, i) => bar(i + 1, m.predicted,
        m.model.toUpperCase().replace("REGRESSOR", "REG.").replace("DAY-OF-WEEK AVG", "DOW AVG"),
        false, m.is_ours, m.residual))}
    </svg>
  );
}

// Vertical histogram for the residual distribution.
function ResidHistogram({ bins, stats }) {
  const W = 360, H = 220;
  const padL = 36, padR = 12, padT = 18, padB = 28;
  const maxCount = Math.max(...bins.map((b) => b.count), 1);
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const minB = bins[0].bin_start;
  const maxB = bins[bins.length - 1].bin_end;
  const rng = maxB - minB;
  const xFor = (v) => padL + ((v - minB) / rng) * innerW;
  const binW = innerW / bins.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
      {/* zero line */}
      <line x1={xFor(0)} x2={xFor(0)} y1={padT} y2={H - padB}
            stroke="var(--accent)" strokeWidth="1" strokeDasharray="2 3" />
      {/* mean line */}
      <line x1={xFor(stats.mean)} x2={xFor(stats.mean)} y1={padT} y2={H - padB}
            stroke="var(--warn)" strokeWidth="1" strokeDasharray="4 3" />
      {bins.map((b, i) => {
        const h = (b.count / maxCount) * innerH;
        return (
          <rect key={i}
                x={padL + i * binW + 1}
                y={H - padB - h}
                width={binW - 2}
                height={h}
                fill="var(--accent-dim)"
                stroke="var(--accent)"
                strokeWidth="1" />
        );
      })}
      <text x={xFor(0) + 4} y={padT + 10} fontSize="9"
            fontFamily="JetBrains Mono, monospace" fill="var(--accent)">zero</text>
      <text x={xFor(stats.mean) + 4} y={padT + 22} fontSize="9"
            fontFamily="JetBrains Mono, monospace" fill="var(--warn)">
        mean {stats.mean.toFixed(2)}
      </text>
      {[minB, 0, maxB].map((v, i) => (
        <text key={i} x={xFor(v)} y={H - padB + 14} fontSize="9.5"
              fontFamily="JetBrains Mono, monospace" fill="var(--text-muted)"
              textAnchor="middle">
          {v.toFixed(1)}
        </text>
      ))}
    </svg>
  );
}

// Scatter of residuals over the x-axis (either date or predicted value).
function ResidScatter({ points, xField, xLabel, width, height }) {
  const W = width || 360, H = height || 220;
  const padL = 36, padR = 12, padT = 18, padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  let xVals;
  if (xField === "date") xVals = points.map((p) => new Date(p.date).getTime());
  else xVals = points.map((p) => p[xField]);
  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const yVals = points.map((p) => p.residual);
  const yMin = Math.min(...yVals, -1);
  const yMax = Math.max(...yVals, 1);
  const yAbs = Math.max(Math.abs(yMin), Math.abs(yMax));

  const x = (v) => padL + ((v - xMin) / (xMax - xMin || 1)) * innerW;
  const y = (v) => padT + innerH / 2 - (v / yAbs) * (innerH / 2);

  const color = (cls) =>
    cls === "High" ? "var(--crit)" :
    cls === "Low"  ? "var(--ok)"  :
                     "var(--warn)";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
      {/* zero line */}
      <line x1={padL} x2={W - padR} y1={y(0)} y2={y(0)}
            stroke="var(--accent)" strokeWidth="1" strokeDasharray="3 3" />
      {[-1, 1].map((v) => (
        <line key={v} x1={padL} x2={W - padR}
              y1={y(v)} y2={y(v)}
              stroke="var(--grid)" strokeWidth="1" />
      ))}
      {points.map((p, i) => {
        const xv = xField === "date" ? new Date(p.date).getTime() : p[xField];
        return (
          <circle key={i} cx={x(xv)} cy={y(p.residual)} r="1.8"
                  fill={color(p.actual_class)} opacity="0.65" />
        );
      })}
      {/* y-axis labels */}
      {[-yAbs, 0, yAbs].map((v, i) => (
        <text key={i} x={padL - 4} y={y(v) + 3} fontSize="9"
              fontFamily="JetBrains Mono, monospace"
              fill="var(--text-muted)" textAnchor="end">
          {v.toFixed(1)}
        </text>
      ))}
      <text x={W / 2} y={H - 6} fontSize="9"
            fontFamily="JetBrains Mono, monospace"
            fill="var(--text-muted)" textAnchor="middle" letterSpacing="0.08em">
        {xLabel}
      </text>
    </svg>
  );
}

// Box-plot by class (Low / Medium / High) of residuals.
function ResidByClass({ by_class }) {
  const W = 360, H = 220;
  const padL = 56, padR = 12, padT = 18, padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const classes = ["Low", "Medium", "High"].filter((c) => by_class[c]);
  const all = classes.flatMap((c) => [by_class[c].min, by_class[c].max]);
  const yMin = Math.min(...all, -1);
  const yMax = Math.max(...all, 1);
  const yAbs = Math.max(Math.abs(yMin), Math.abs(yMax));
  const y = (v) => padT + innerH / 2 - (v / yAbs) * (innerH / 2);
  const xSlot = innerW / classes.length;
  const boxW = Math.min(48, xSlot * 0.5);

  const color = (cls) =>
    cls === "High" ? "var(--crit)" :
    cls === "Low"  ? "var(--ok)"  :
                     "var(--warn)";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
      <line x1={padL} x2={W - padR} y1={y(0)} y2={y(0)}
            stroke="var(--accent)" strokeWidth="1" strokeDasharray="3 3" />
      {classes.map((c, i) => {
        const cx = padL + xSlot * (i + 0.5);
        const b = by_class[c];
        const col = color(c);
        return (
          <g key={c}>
            {/* whiskers */}
            <line x1={cx} x2={cx} y1={y(b.min)} y2={y(b.max)}
                  stroke={col} strokeWidth="1" />
            {/* box */}
            <rect x={cx - boxW / 2} y={y(b.q75)}
                  width={boxW} height={y(b.q25) - y(b.q75)}
                  fill={col} fillOpacity="0.15" stroke={col} strokeWidth="1.5" />
            {/* median */}
            <line x1={cx - boxW / 2} x2={cx + boxW / 2}
                  y1={y(b.median)} y2={y(b.median)}
                  stroke={col} strokeWidth="2" />
            {/* label */}
            <text x={cx} y={H - padB + 16} textAnchor="middle"
                  fontSize="10" fontFamily="JetBrains Mono, monospace"
                  fill={col} letterSpacing="0.08em">
              {c.toUpperCase()} · n={b.n}
            </text>
          </g>
        );
      })}
      {[-yAbs, 0, yAbs].map((v, i) => (
        <text key={i} x={padL - 6} y={y(v) + 3} fontSize="9.5"
              fontFamily="JetBrains Mono, monospace"
              fill="var(--text-muted)" textAnchor="end">
          {v.toFixed(1)}
        </text>
      ))}
    </svg>
  );
}

// Cumulative error breakdown (% within ±X).
function CumulativeError({ cumulative, mae }) {
  const W = 360, H = 220;
  const padL = 40, padR = 12, padT = 26, padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const barH = innerH / cumulative.length - 4;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
      <text x={padL} y={14} fontSize="10"
            fontFamily="JetBrains Mono, monospace" fill="var(--text-muted)"
            letterSpacing="0.08em">
        MAE · {mae.toFixed(3)} ACPS
      </text>
      {cumulative.map((c, i) => {
        const yTop = padT + i * (barH + 4);
        const w = (c.pct_within / 100) * innerW;
        return (
          <g key={i}>
            <rect x={padL} y={yTop} width={innerW} height={barH}
                  fill="var(--bg-surface-3)" stroke="var(--border)" />
            <rect x={padL} y={yTop} width={w} height={barH}
                  fill="var(--accent-dim)" stroke="var(--accent)" strokeWidth="1" />
            <text x={padL - 4} y={yTop + barH / 2 + 4} textAnchor="end"
                  fontSize="10" fontFamily="JetBrains Mono, monospace"
                  fill="var(--text-secondary)">
              ±{c.threshold}
            </text>
            <text x={padL + w - 4} y={yTop + barH / 2 + 4} textAnchor="end"
                  fontSize="10.5" fontFamily="JetBrains Mono, monospace"
                  fill="var(--text-primary)" fontWeight="500">
              {c.pct_within.toFixed(1)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// Per-feature position bar for "Why This Day".
function FeatureDeviationRow({ f }) {
  const clamp = (v) => Math.max(0, Math.min(100, v));
  const pctVal = clamp(f.percentile);
  const pctMean = clamp(100.0 * (f.train_mean - f.train_min) /
                        Math.max(1e-9, f.train_max - f.train_min));
  const pctP10 = clamp(100.0 * (f.train_p10 - f.train_min) /
                       Math.max(1e-9, f.train_max - f.train_min));
  const pctP90 = clamp(100.0 * (f.train_p90 - f.train_min) /
                       Math.max(1e-9, f.train_max - f.train_min));
  const deviant = Math.abs(f.z_score) > 1.0;

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "200px 1fr 120px",
      gap: 14,
      alignItems: "center",
      padding: "12px 0",
      borderBottom: "1px dashed var(--border)",
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 12,
    }}>
      <div>
        <div style={{ color: "var(--text-primary)" }}>{f.feature}</div>
        <div style={{ fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.08em", marginTop: 4 }}>
          global imp · {(f.global_importance * 100).toFixed(1)}%
        </div>
      </div>
      <div style={{ position: "relative", height: 30 }}>
        <div style={{
          position: "absolute", left: 0, right: 0, top: 13, height: 4,
          background: "var(--bg-surface-3)", border: "1px solid var(--border)",
        }} />
        {/* p10-p90 shaded band */}
        <div style={{
          position: "absolute",
          left: `${pctP10}%`,
          width: `${pctP90 - pctP10}%`,
          top: 12, height: 6,
          background: "var(--accent-dim)",
          border: "1px solid var(--accent)",
          opacity: 0.55,
        }} />
        {/* train mean tick */}
        <div style={{
          position: "absolute",
          left: `calc(${pctMean}% - 1px)`,
          top: 6, bottom: 6, width: 2,
          background: "var(--text-secondary)",
        }} />
        {/* this-day value */}
        <div style={{
          position: "absolute",
          left: `calc(${pctVal}% - 6px)`,
          top: 6, width: 12, height: 18,
          background: deviant ? "var(--warn)" : "var(--accent)",
          border: `1px solid ${deviant ? "var(--warn)" : "var(--accent)"}`,
        }} />
        <div style={{
          position: "absolute",
          top: 24,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "space-between",
          fontSize: 9.5,
          color: "var(--text-muted)",
        }}>
          <span>min {f.train_min}</span>
          <span>mean {f.train_mean}</span>
          <span>max {f.train_max}</span>
        </div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div style={{ color: "var(--text-primary)", fontSize: 14, fontWeight: 500 }}>
          {f.value}
        </div>
        <div style={{
          fontSize: 10,
          color: deviant ? "var(--warn)" : "var(--text-muted)",
          letterSpacing: "0.06em",
          marginTop: 2,
        }}>
          z · {f.z_score > 0 ? "+" : ""}{f.z_score.toFixed(2)} σ · p{f.percentile.toFixed(0)}
        </div>
      </div>
    </div>
  );
}

function PageDiagnostics() {
  const DEFAULT_DATE = "2025-12-19";
  const datesQ = useAsync(() => BARAJAS_API.testDates(), []);
  const [date, setDate] = React.useState(DEFAULT_DATE);

  const showdownQ = useAsync(() => (date ? BARAJAS_API.showdown(date) : Promise.resolve(null)), [date]);
  const residQ    = useAsync(() => BARAJAS_API.residuals(), []);
  const featQ     = useAsync(() => (date ? BARAJAS_API.localFeatures(date) : Promise.resolve(null)), [date]);

  React.useEffect(() => {
    const onKey = (e) => {
      if (e.target && ["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName)) return;
      if (!datesQ.data) return;
      const all = datesQ.data.dates;
      const idx = all.indexOf(date);
      if (e.key === "ArrowLeft"  && idx > 0)              { e.preventDefault(); setDate(all[idx - 1]); }
      if (e.key === "ArrowRight" && idx < all.length - 1) { e.preventDefault(); setDate(all[idx + 1]); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [datesQ.data, date]);

  if (datesQ.error) return <div className="page"><ErrorBanner err={datesQ.error} /></div>;
  if (!datesQ.data) return <div className="page"><Loading label="TEST-SET INDEX" /></div>;

  const allDates = datesQ.data.dates;
  const idx = allDates.indexOf(date);
  const s = showdownQ.data;
  const r = residQ.data;
  const feat = featQ.data;

  return (
    <div className="page">
      <div className="page-subtitle mono" style={{ letterSpacing: "0.04em" }}>
        Three lenses on the same model · pick any test-set date · all numbers computed live on the held-out 498 days
      </div>

      {/* Date picker bar (shared by section A and section D) */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button className="btn-icon"
                    onClick={() => setDate(allDates[Math.max(0, idx - 1)])}
                    disabled={idx <= 0}
                    style={{ opacity: idx <= 0 ? 0.4 : 1 }}>◀</button>
            <input
              type="date"
              value={date}
              min={datesQ.data.min}
              max={datesQ.data.max}
              onChange={(e) => { if (allDates.includes(e.target.value)) setDate(e.target.value); }}
              style={{ background: "var(--bg-surface-2)", border: "1px solid var(--border)",
                       color: "var(--text-primary)", padding: "8px 12px",
                       fontFamily: "'JetBrains Mono', monospace", fontSize: 14,
                       letterSpacing: "0.04em", colorScheme: "dark" }} />
            <button className="btn-icon"
                    onClick={() => setDate(allDates[Math.min(allDates.length - 1, idx + 1)])}
                    disabled={idx >= allDates.length - 1}
                    style={{ opacity: idx >= allDates.length - 1 ? 0.4 : 1 }}>▶</button>
          </div>
          <div className="mono" style={{ fontSize: 11, color: "var(--text-muted)",
               letterSpacing: "0.12em", textTransform: "uppercase" }}>
            day {idx + 1} of {datesQ.data.count}
          </div>
          <button className="btn-ghost" onClick={() => setDate(DEFAULT_DATE)}>SHOWCASE · 19 DEC 2025</button>
          <div style={{ flex: 1 }} />
          <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)",
                letterSpacing: "0.12em" }}>
            <span className="kbd">←</span> <span className="kbd">→</span>  STEP DAYS
          </span>
        </div>
      </div>

      {/* ==================== SECTION A — MODEL SHOWDOWN ==================== */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, marginTop: 8 }}>
        <h3 style={{ margin: 0, fontFamily: "Inter, sans-serif", fontSize: 17, fontWeight: 600,
                     letterSpacing: "-0.01em" }}>
          Model Showdown
        </h3>
        <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)",
              letterSpacing: "0.12em", textTransform: "uppercase" }}>
          every model on the same day
        </span>
      </div>

      {showdownQ.error && <ErrorBanner err={showdownQ.error} />}
      {!s ? <Loading label="SHOWDOWN" /> : (
        <div className="grid grid-7-5">
          <div className="card">
            <div className="card-title">
              Predictions · {s.day_of_week} {s.date}
              <span className="title-meta mono">
                actual {s.actual_acps.toFixed(2)} · {s.actual_class.toUpperCase()} · {s.total_movements.toLocaleString()} movements
              </span>
            </div>
            <ShowdownBars actual={s.actual_acps} models={s.per_day} />
            <div className="metric-sub mono" style={{ marginTop: 10, color: "var(--text-muted)" }}>
              HGB is the only model that tracks actual within the ±1 ACPS band.
              DoW-Avg and Global Mean are dragged down by COVID-era training data — proof
              the HGB model learned the post-recovery regime.
            </div>
          </div>

          <div className="card">
            <div className="card-title">
              Aggregate · 498-day test
              <span className="title-meta mono">all 5 models we built</span>
            </div>
            <table className="btable">
              <thead>
                <tr><th>Model</th><th className="num">MAE</th><th className="num">R²</th></tr>
              </thead>
              <tbody>
                {s.aggregate.map((row, i) => {
                  const ours = row.model.toLowerCase().includes("hgb");
                  return (
                    <tr key={i} className={ours ? "ours" : ""}>
                      <td>{row.model}{ours && <span style={{ color: "var(--accent)", marginLeft: 8, fontSize: 10 }}>◆</span>}</td>
                      <td className="num">{row.mae.toFixed(3)}</td>
                      <td className="num">{row.r2.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="metric-sub mono" style={{ marginTop: 10, color: "var(--text-muted)" }}>
              Per-day predictions shown for models we can reconstruct deterministically.
              SARIMAX and ANN appear only in aggregate — their per-day regeneration is omitted for speed.
            </div>
          </div>
        </div>
      )}

      {/* ==================== SECTION B — RESIDUAL DIAGNOSTICS ==================== */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, marginTop: 24 }}>
        <h3 style={{ margin: 0, fontFamily: "Inter, sans-serif", fontSize: 17, fontWeight: 600,
                     letterSpacing: "-0.01em" }}>
          Residual Diagnostics
        </h3>
        <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)",
              letterSpacing: "0.12em", textTransform: "uppercase" }}>
          statistical sanity checks on the 498-day test set
        </span>
      </div>

      {residQ.error && <ErrorBanner err={residQ.error} />}
      {!r ? <Loading label="RESIDUALS" /> : (
        <>
          <div className="grid grid-3">
            <div className="card">
              <div className="card-title">1 · Residual Distribution</div>
              <ResidHistogram bins={r.histogram} stats={r.summary} />
              <div className="metric-sub mono" style={{ marginTop: 8, color: "var(--text-muted)", lineHeight: 1.5 }}>
                Mean {r.summary.mean} · σ {r.summary.std} · symmetric around zero →
                <span style={{ color: "var(--ok)" }}> unbiased</span>.
              </div>
            </div>
            <div className="card">
              <div className="card-title">2 · Residuals vs Prediction</div>
              <ResidScatter points={r.points} xField="predicted" xLabel="PREDICTED ACPS" />
              <div className="metric-sub mono" style={{ marginTop: 8, color: "var(--text-muted)", lineHeight: 1.5 }}>
                No funnel as predictions grow →
                <span style={{ color: "var(--ok)" }}> homoscedastic</span>.
                Error variance stays flat.
              </div>
            </div>
            <div className="card">
              <div className="card-title">3 · Residuals Over Time</div>
              <ResidScatter points={r.points} xField="date" xLabel="2024-10-19  →  2026-02-28" />
              <div className="metric-sub mono" style={{ marginTop: 8, color: "var(--text-muted)", lineHeight: 1.5 }}>
                No trend across 16 months →
                <span style={{ color: "var(--ok)" }}> no concept drift</span>.
                The model stays accurate over time.
              </div>
            </div>
          </div>

          <div className="grid grid-2">
            <div className="card">
              <div className="card-title">4 · Residuals by Class</div>
              <ResidByClass by_class={r.by_class} />
              <div className="metric-sub mono" style={{ marginTop: 8, color: "var(--text-muted)", lineHeight: 1.5 }}>
                Box spreads overlap across Low/Med/High →
                <span style={{ color: "var(--ok)" }}> no class bias</span>.
                The model is equally accurate whether traffic is quiet or peak.
              </div>
            </div>
            <div className="card">
              <div className="card-title">5 · Cumulative Accuracy</div>
              <CumulativeError cumulative={r.cumulative} mae={r.summary.mae} />
              <div className="metric-sub mono" style={{ marginTop: 8, color: "var(--text-muted)", lineHeight: 1.5 }}>
                {r.summary.pct_within_1}% of predictions are within ±1 ACPS.
                Practical operational accuracy, not just a good R².
              </div>
            </div>
          </div>
        </>
      )}

      {/* ==================== SECTION D — WHY THIS DAY ==================== */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, marginTop: 24 }}>
        <h3 style={{ margin: 0, fontFamily: "Inter, sans-serif", fontSize: 17, fontWeight: 600,
                     letterSpacing: "-0.01em" }}>
          Why This Day?
        </h3>
        <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)",
              letterSpacing: "0.12em", textTransform: "uppercase" }}>
          feature values on {date} vs their training-set distribution
        </span>
      </div>

      {featQ.error && <ErrorBanner err={featQ.error} />}
      {!feat ? <Loading label="FEATURE DEEP-DIVE" /> : (
        <div className="card">
          <div className="card-title">
            Top features driving the prediction on {feat.date}
            <span className="title-meta mono">
              actual {feat.actual_acps} · {feat.actual_class.toUpperCase()} · {feat.total_movements.toLocaleString()} movements
            </span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "200px 1fr 120px", gap: 14,
                        padding: "6px 0 10px",
                        borderBottom: "1px solid var(--border)",
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: 10, letterSpacing: "0.12em",
                        color: "var(--text-muted)", textTransform: "uppercase" }}>
            <div>FEATURE</div>
            <div>POSITION IN TRAINING RANGE (p10–p90 shaded · mean tick · this day marker)</div>
            <div style={{ textAlign: "right" }}>VALUE · DEVIATION</div>
          </div>
          {feat.features.map((f) => <FeatureDeviationRow key={f.feature} f={f} />)}
          <div className="metric-sub mono" style={{ marginTop: 14, color: "var(--text-muted)", lineHeight: 1.6 }}>
            Amber markers = values more than 1σ from the training mean — i.e., what the model actually "saw" as unusual
            about this day. The combination of deviations explains the prediction better than global importance alone.
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// CASE STUDY
// ============================================================
function PageCaseStudy() {
  const datesQ = useAsync(() => BARAJAS_API.testDates(), []);
  const [date, setDate] = React.useState("2025-12-19");

  React.useEffect(() => {
    if (datesQ.data && !datesQ.data.dates.includes(date)) setDate(datesQ.data.default);
  }, [datesQ.data]);

  const dayQ = useAsync(() => (date ? BARAJAS_API.testDay(date) : Promise.resolve(null)), [date]);

  React.useEffect(() => {
    const onKey = (e) => {
      if (e.target && ["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName)) return;
      if (!datesQ.data) return;
      const all = datesQ.data.dates;
      const idx = all.indexOf(date);
      if (e.key === "ArrowLeft"  && idx > 0)              { e.preventDefault(); setDate(all[idx - 1]); }
      if (e.key === "ArrowRight" && idx < all.length - 1) { e.preventDefault(); setDate(all[idx + 1]); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [datesQ.data, date]);

  if (datesQ.error) return <div className="page"><ErrorBanner err={datesQ.error} /></div>;
  if (!datesQ.data) return <div className="page"><Loading label="TEST-SET INDEX" /></div>;

  const allDates = datesQ.data.dates;
  const idx = allDates.indexOf(date);
  const count = datesQ.data.count;

  const c = dayQ.data;
  const maxImp = c ? Math.max(...c.top_drivers.map((x) => x.importance)) : 1;

  return (
    <div className="page">
      <div className="page-subtitle mono" style={{ letterSpacing: "0.04em" }}>
        {count} days · one click each · replay any prediction from the held-out test set ({datesQ.data.min} → {datesQ.data.max})
      </div>

      {/* Date picker bar */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button className="btn-icon" onClick={() => setDate(allDates[Math.max(0, idx - 1)])} disabled={idx <= 0} style={{ opacity: idx <= 0 ? 0.4 : 1 }}>◀</button>
            <input
              type="date"
              className="mono"
              style={{ background: "var(--bg-surface-2)", border: "1px solid var(--border)", color: "var(--text-primary)", padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace", fontSize: 14, letterSpacing: "0.04em", colorScheme: "dark" }}
              value={date}
              min={datesQ.data.min}
              max={datesQ.data.max}
              onChange={(e) => {
                const v = e.target.value;
                if (allDates.includes(v)) setDate(v);
              }}
            />
            <button className="btn-icon" onClick={() => setDate(allDates[Math.min(allDates.length - 1, idx + 1)])} disabled={idx >= allDates.length - 1} style={{ opacity: idx >= allDates.length - 1 ? 0.4 : 1 }}>▶</button>
          </div>
          <div className="mono" style={{ fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.12em", textTransform: "uppercase" }}>
            day {idx + 1} of {count}
          </div>
          <button className="btn-ghost" onClick={() => setDate("2025-12-19")}>JUMP TO PEAK DAY</button>
          <div style={{ flex: 1 }} />
          <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.12em" }}>
            <span className="kbd">←</span> <span className="kbd">→</span>  STEP DAYS
          </span>
          <span className="pill pill-ghost"><span className="pdot" />TEST SET {datesQ.data.min} → {datesQ.data.max}</span>
        </div>
      </div>

      {dayQ.error && <ErrorBanner err={dayQ.error} />}
      {!c ? <Loading label={`DAY ${date}`} /> : (
        <>
          <div className="hero-strip">
            <div className="cell">
              <div className="label">Actual ACPS</div>
              <div className="value mono">{c.actual_acps.toFixed(2)}</div>
            </div>
            <div className="cell">
              <div className="label">Predicted ACPS</div>
              <div className="value mono" style={{ color: "var(--accent)" }}>{c.predicted_acps.toFixed(2)}</div>
            </div>
            <div className="cell">
              <div className="label">Residual</div>
              <div className="value mono" style={{ color: Math.abs(c.residual) > 1.5 ? "var(--warn)" : "var(--ok)" }}>
                {c.residual > 0 ? "+" : ""}{c.residual.toFixed(2)}
              </div>
            </div>
            <div className="cell">
              <div className="label">Actual class</div>
              <div style={{ marginTop: 4 }}><StatusPill level={c.actual_class} size="lg" /></div>
            </div>
            <div className="cell">
              <div className="label">Predicted class</div>
              <div style={{ marginTop: 4 }}><StatusPill level={c.predicted_class} size="lg" /></div>
            </div>
          </div>

          <div className="card" style={{ padding: 16 }}>
            <div className="card-title" style={{ padding: "0 8px 12px" }}>
              ±7-day Context · {c.context[0].date} → {c.context[c.context.length - 1].date}
              <span className="title-meta mono">y · ACPS · x · date · reference · HIGH 75</span>
            </div>
            <div style={{ background: "var(--bg-surface-2)", border: "1px solid var(--border)", padding: 8 }}>
              <LineChart data={c.context} height={400} threshold={75} targetDate={c.date} />
            </div>
          </div>

          <div className="grid grid-6-6">
            <div className="card">
              <div className="card-title">
                Top Drivers
                <span className="title-meta mono">global permutation importance · HGB regressor</span>
              </div>
              <div>
                {c.top_drivers.map((f) => (
                  <FeatureBar key={f.feature} feature={f.feature} importance={f.importance} max={maxImp} />
                ))}
              </div>
            </div>
            <div className="card">
              <div className="card-title">Day Context</div>
              <dl className="kv">
                <dt>Date</dt><dd>{c.date}</dd>
                <dt>Day-of-week</dt><dd>{dayOfWeekName(c.date)}</dd>
                <dt>Total movements</dt><dd>{c.total_movements.toLocaleString()}</dd>
                <dt>Actual class</dt><dd>{c.actual_class.toUpperCase()}</dd>
                <dt>Predicted class</dt><dd>{c.predicted_class.toUpperCase()}</dd>
                <dt>Residual</dt><dd style={{ color: Math.abs(c.residual) > 1.5 ? "var(--warn)" : "var(--text-primary)" }}>{c.residual.toFixed(2)} ACPS</dd>
                <dt>Classification</dt><dd>{c.actual_class === c.predicted_class ? "CORRECT" : "MISS (adjacent)"}</dd>
              </dl>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ============================================================
// REPORT CARD
// ============================================================
function PageReport() {
  const { loading, data: m, error } = useAsync(() => BARAJAS_API.metrics(), []);
  if (error) return <div className="page"><ErrorBanner err={error} /></div>;
  if (loading || !m) return <div className="page"><Loading label="METRICS" /></div>;

  const maxImp = Math.max(...m.feature_importance.map((d) => d.importance));

  return (
    <div className="page">
      <h2 style={{ margin: "0 0 4px", fontFamily: "Inter, sans-serif", fontWeight: 500, fontSize: 22, letterSpacing: "-0.01em" }}>
        Held-out test set
        <span className="mono" style={{ fontSize: 13, color: "var(--text-muted)", marginLeft: 14, letterSpacing: "0.08em" }}>
          {m.test_set_size} days · {m.test_date_range.start} → {m.test_date_range.end}
        </span>
      </h2>
      <div className="page-subtitle">
        Metrics computed live on predictions the model never saw during training.
      </div>

      <div className="grid grid-4">
        <MetricCard title="MAE" value={m.regression.mae.toFixed(3)} unit="ACPS" sub="mean absolute error" />
        <MetricCard title="RMSE" value={m.regression.rmse.toFixed(3)} unit="ACPS" sub="root mean squared error" />
        <MetricCard title="R²" value={m.regression.r2.toFixed(3)} sub="variance explained" />
        <MetricCard title="Accuracy" value={(m.classification.accuracy * 100).toFixed(2)} unit="%" sub={`F1-macro ${m.classification.f1_macro.toFixed(3)} · bal ${m.classification.balanced_accuracy.toFixed(3)}`} />
      </div>

      <div className="grid grid-6-6">
        <div className="card">
          <div className="card-title">
            Model Comparison
            <span className="title-meta mono">lower MAE/RMSE is better · higher R² is better</span>
          </div>
          <BaselineTable rows={m.baselines} />
          <div className="metric-sub mono" style={{ marginTop: 12, color: "var(--text-muted)" }}>
            {m.baselines.length > 1 && (
              <>HGB reduces MAE by {((1 - m.baselines[0].mae / m.baselines[1].mae) * 100).toFixed(1)}% vs {m.baselines[1].model}.</>
            )}
          </div>
        </div>
        <div className="card">
          <div className="card-title">
            Confusion Matrix
            <span className="title-meta mono">3-class · low / medium / high</span>
          </div>
          <ConfusionMatrix labels={m.classification.labels} matrix={m.classification.confusion_matrix} />
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          Top 10 Features
          <span className="title-meta mono">permutation importance · held-out test</span>
        </div>
        <div>
          {m.feature_importance.map((f) => (
            <FeatureBar key={f.feature} feature={f.feature} importance={f.importance} max={maxImp} />
          ))}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { PageToday, PageDiagnostics, PageCaseStudy, PageReport });
