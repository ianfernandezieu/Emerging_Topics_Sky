/* global React, ReactDOM, BARAJAS_API, useAsync */

const TABS = [
  { id: "today",       label: "TODAY",       key: "1" },
  { id: "diagnostics", label: "DIAGNOSTICS", key: "2" },
  { id: "case",        label: "CASE STUDY",  key: "3" },
  { id: "report",      label: "REPORT CARD", key: "4" },
];

function AppShell() {
  const [tab, setTab] = React.useState(() => localStorage.getItem("barajas_tab") || "today");
  const health = window.useAsync(() => BARAJAS_API.health(), []);
  const online = !health.error && !!health.data;

  React.useEffect(() => { localStorage.setItem("barajas_tab", tab); }, [tab]);
  React.useEffect(() => { document.documentElement.setAttribute("data-theme", "navy"); }, []);

  React.useEffect(() => {
    const onKey = (e) => {
      if (e.target && ["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName)) return;
      const hit = TABS.find((t) => t.key === e.key);
      if (hit) setTab(hit.id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const Page =
    tab === "today"       ? window.PageToday :
    tab === "diagnostics" ? window.PageDiagnostics :
    tab === "case"        ? window.PageCaseStudy :
                            window.PageReport;

  const h = health.data;
  const today = new Date().toISOString().slice(0, 10);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="glyph">◈</span>
          BARAJAS CONGESTION CONSOLE
        </div>
        <div className="live-stamp" style={{ marginLeft: -8 }}>
          <span className="live-dot" style={{ background: online ? "var(--ok)" : "var(--crit)" }} />
          LIVE <span style={{ color: "var(--text-muted)" }}>·</span> {today}
        </div>
        <nav className="tabs" role="tablist" aria-label="Primary">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              <span className="keyhint">{t.key}</span>{t.label}
            </button>
          ))}
        </nav>
      </header>

      <div style={{
        borderBottom: "1px solid var(--border)",
        background: "var(--bg-surface)",
        padding: "8px 24px",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10.5,
        letterSpacing: "0.1em",
        color: "var(--text-muted)",
        display: "flex",
        gap: 24,
        textTransform: "uppercase",
        justifyContent: "space-between",
      }}>
        <span>REG · {h?.regressor_loaded ? <>LOADED  <span style={{ color: "var(--ok)" }}>●</span></> : <>— <span style={{ color: "var(--crit)" }}>●</span></>}</span>
        <span>CLF · {h?.classifier_loaded ? <>LOADED  <span style={{ color: "var(--ok)" }}>●</span></> : <>— <span style={{ color: "var(--crit)" }}>●</span></>}</span>
        <span>ROWS · <span style={{ color: "var(--text-secondary)" }}>{h ? h.rows.toLocaleString() : "—"}</span></span>
        <span>FEATURES · <span style={{ color: "var(--text-secondary)" }}>{h?.features ?? "—"}</span></span>
        <span>RESIDUAL σ · <span style={{ color: "var(--text-secondary)" }}>{h ? h.residual_std.toFixed(3) : "—"}</span></span>
        <span>API · <span style={{ color: online ? "var(--ok)" : "var(--crit)" }}>{online ? "ONLINE" : "OFFLINE"}</span> · 127.0.0.1:8000</span>
      </div>

      <main>
        {health.error ? (
          <div className="page">
            <div className="banner">
              <span className="b-tag">BACKEND</span>
              <span>Cannot reach API at {window.API_BASE}. Start <code>demo\backend\run.bat</code> first, then refresh.</span>
            </div>
          </div>
        ) : <Page />}
      </main>

      <footer style={{
        borderTop: "1px solid var(--border)",
        padding: "12px 24px",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
        letterSpacing: "0.12em",
        color: "var(--text-muted)",
        display: "flex",
        justifyContent: "space-between",
        textTransform: "uppercase",
        marginTop: 16,
      }}>
        <span>HGB REGRESSOR · EUROCONTROL PIPELINE · MADRID-BARAJAS (LEMD)</span>
        <span>PRESS <span className="kbd">1</span>–<span className="kbd">4</span> TO SWITCH TABS · ←→ STEPS DAYS IN DIAGNOSTICS / CASE STUDY</span>
      </footer>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<AppShell />);
