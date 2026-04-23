/* Backend client. Every value in the UI comes from here.
   Backend: FastAPI on 127.0.0.1:8000 — see demo/backend/main.py */

window.API_BASE = "http://127.0.0.1:8000";

async function apiGet(path) {
  const res = await fetch(window.API_BASE + path);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(window.API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

window.BARAJAS_API = {
  health:        () => apiGet("/api/health"),
  today:         () => apiGet("/api/today"),
  predict:       (body) => apiPost("/api/predict", body),
  testDates:     () => apiGet("/api/test-set/dates"),
  testDay:       (date) => apiGet("/api/test-set/" + date),
  metrics:       () => apiGet("/api/metrics"),
  showdown:      (date) => apiGet("/api/model-showdown/" + date),
  residuals:     () => apiGet("/api/residuals"),
  localFeatures: (date) => apiGet("/api/local-features/" + date),
  futureForecast: (date) => apiGet("/api/future-forecast/" + date),
};

window.useAsync = function(asyncFn, deps) {
  const [state, setState] = React.useState({ loading: true, data: null, error: null });
  React.useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    asyncFn()
      .then((data) => { if (alive) setState({ loading: false, data, error: null }); })
      .catch((error) => { if (alive) setState({ loading: false, data: null, error: String(error) }); });
    return () => { alive = false; };
  }, deps || []);
  return state;
};

window.formatDay = function(iso) {
  const d = new Date(iso + "T00:00:00Z");
  const dow = ["SUN","MON","TUE","WED","THU","FRI","SAT"][d.getUTCDay()];
  const mon = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"][d.getUTCMonth()];
  return `${dow} ${mon} ${String(d.getUTCDate()).padStart(2,"0")}`;
};

window.dayOfWeekName = function(iso) {
  const d = new Date(iso + "T00:00:00Z");
  return ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"][d.getUTCDay()];
};

window.isoWeek = function(iso) {
  const d = new Date(iso + "T00:00:00Z");
  const t = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  const dayNum = t.getUTCDay() || 7;
  t.setUTCDate(t.getUTCDate() + 4 - dayNum);
  const y0 = new Date(Date.UTC(t.getUTCFullYear(), 0, 1));
  return Math.ceil(((t - y0) / 86400000 + 1) / 7);
};
