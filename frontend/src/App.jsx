import { useState } from "react";
import "./App.css";

const API_BASE = "/api";

function MetricCard({ label, value, sub, type = "" }) {
  return (
    <div className={`metric-card ${type}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

function StatusBadge({ status }) {
  const normalized = String(status || "").toUpperCase();

  return (
    <span
      className={`status-badge ${
        normalized === "ANOMALOUS" ? "danger" : "normal"
      }`}
    >
      <span className="status-dot" />
      {normalized || "UNKNOWN"}
    </span>
  );
}

function BarChart({ results }) {
  if (!results?.length) return null;

  const fixedValues = results.map((r) =>
    Number(r.fixed_measurements ?? r.fixed_reads ?? 0)
  );

  const falconValues = results.map((r) =>
    Number(r.falcon_measurements ?? r.measurements ?? r.reads ?? 0)
  );

  const max = Math.max(...fixedValues, ...falconValues, 1);

  return (
    <div className="bar-chart">
      {results.map((r, index) => {
        const fixed = Number(
          r.fixed_measurements ?? r.fixed_reads ?? 0
        );

        const falcon = Number(
          r.falcon_measurements ?? r.measurements ?? r.reads ?? 0
        );

        const id =
          r.component_id ||
          r.id ||
          `C${String(index + 1).padStart(2, "0")}`;

        return (
          <div className="bar-group" key={id}>
            <div className="bars">
              <div
                className="bar fixed"
                style={{
                  height: `${(fixed / max) * 150}px`,
                }}
                title={`Fixed: ${fixed}`}
              >
                <span>{fixed}</span>
              </div>

              <div
                className="bar falcon"
                style={{
                  height: `${(falcon / max) * 150}px`,
                }}
                title={`FALCON: ${falcon}`}
              >
                <span>{falcon}</span>
              </div>
            </div>

            <div className="bar-label">{id}</div>
          </div>
        );
      })}
    </div>
  );
}

function ScoreBar({ score, threshold }) {
  const s = Number(score ?? 0);
  const t = Number(threshold ?? 1);

  const max = Math.max(s, t) * 1.15 || 1;

  const scorePercent = Math.min((s / max) * 100, 100);
  const thresholdPercent = Math.min((t / max) * 100, 100);

  return (
    <div className="score-wrapper">
      <div className="score-track">
        <div
          className="score-fill"
          style={{
            width: `${scorePercent}%`,
          }}
        />

        <div
          className="threshold-marker"
          style={{
            left: `${thresholdPercent}%`,
          }}
        />
      </div>

      <div className="score-legend">
        <span>
          Score: <strong>{s.toFixed(2)}</strong>
        </span>

        <span>
          Threshold: <strong>{t.toFixed(2)}</strong>
        </span>
      </div>
    </div>
  );
}

function App() {
  const [normalComponents, setNormalComponents] = useState(4);
  const [anomalousComponents, setAnomalousComponents] = useState(1);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ==========================================================
  // CSV STATE
  // ==========================================================

  const [csvFile, setCsvFile] = useState(null);
  const [csvUploading, setCsvUploading] = useState(false);
  const [csvResult, setCsvResult] = useState(null);
  const [csvError, setCsvError] = useState("");

  // ==========================================================
  // SIMULATION ANALYSIS
  // ==========================================================

  async function runAnalysis() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_BASE}/simulation/run`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            normal_components: Number(normalComponents),
            anomalous_components: Number(anomalousComponents),
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Backend returned HTTP ${response.status}`
        );
      }

      const data = await response.json();

      console.log("FALCON API RESPONSE:", data);

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to FALCON backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  // ==========================================================
  // CSV FILE SELECTION
  // ==========================================================

  function handleCsvChange(event) {
    const selectedFile = event.target.files?.[0];

    setCsvError("");
    setCsvResult(null);

    if (!selectedFile) {
      setCsvFile(null);
      return;
    }

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setCsvError("Please select a CSV file.");
      setCsvFile(null);
      return;
    }

    setCsvFile(selectedFile);
  }

  // ==========================================================
  // CSV UPLOAD
  // ==========================================================

  async function uploadCsv() {
    if (!csvFile) {
      setCsvError("Please select a CSV file first.");
      return;
    }

    setCsvUploading(true);
    setCsvError("");
    setCsvResult(null);

    try {
      const formData = new FormData();

      formData.append("file", csvFile);

      const response = await fetch(
        `${API_BASE}/upload/csv`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "CSV upload failed."
        );
      }

      console.log("CSV UPLOAD RESPONSE:", data);

      setCsvResult(data);
    } catch (err) {
      console.error(err);

      setCsvError(
        err.message || "Unable to upload CSV dataset."
      );
    } finally {
      setCsvUploading(false);
    }
  }

  // ==========================================================
  // NORMALIZE BACKEND RESPONSE
  // ==========================================================

  const components =
    result?.components ||
    result?.results ||
    result?.component_results ||
    [];

  const reductionRaw = Number(
    result?.measurement_reduction ??
      result?.simulation_reduction ??
      result?.measurement_reduction_ratio ??
      0
  );

  const agreementRaw = Number(
    result?.decision_agreement ??
      result?.decision_agreement_ratio ??
      0
  );

  const reduction =
    reductionRaw >= 0 && reductionRaw <= 1
      ? reductionRaw * 100
      : reductionRaw;

  const agreement =
    agreementRaw >= 0 && agreementRaw <= 1
      ? agreementRaw * 100
      : agreementRaw;

  const calculatedFalconMeasurements =
    components.reduce(
      (sum, component) =>
        sum +
        Number(
          component.falcon_measurements ??
            component.measurements ??
            component.reads ??
            0
        ),
      0
    );

  const calculatedFixedMeasurements =
    components.reduce(
      (sum, component) =>
        sum +
        Number(
          component.fixed_measurements ??
            component.fixed_reads ??
            0
        ),
      0
    );

  const totalMeasurements = Number(
    result?.falcon_measurements ??
      result?.falcon_total_measurements ??
      result?.total_measurements ??
      result?.measurements ??
      calculatedFalconMeasurements
  );

  const fixedMeasurements = Number(
    result?.fixed_measurements ??
      result?.fixed_total_measurements ??
      result?.fixed_total ??
      calculatedFixedMeasurements
  );

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app">

      {/* ================================================== */}
      {/* TOP BAR */}
      {/* ================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            F
          </div>

          <div>

            <div className="brand-name">
              FALCON
            </div>

            <div className="brand-subtitle">
              Fault Analysis Logic for Component
              Optimization & Burn-in
            </div>

          </div>

        </div>

        <div className="system-status">
          <span className="live-dot" />
          SYSTEM ONLINE
        </div>

      </header>

      {/* ================================================== */}
      {/* MAIN */}
      {/* ================================================== */}

      <main className="main">

        {/* ================================================= */}
        {/* HERO */}
        {/* ================================================= */}

        <section className="hero">

          <div>

            <div className="eyebrow">
              AI-DRIVEN COMPONENT SCREENING
            </div>

            <h1>
              Intelligent Burn-in
              <br />
              <span>
                Screening Platform
              </span>
            </h1>

            <p>
              FALCON continuously models component
              trajectories and collects additional
              measurements only when available
              evidence could still change the
              screening decision.
            </p>

          </div>

          <div className="hero-status">

            <div className="status-icon">
              ◉
            </div>

            <div>

              <div className="hero-status-title">
                AI ENGINE
              </div>

              <div className="hero-status-value">
                READY
              </div>

            </div>

          </div>

        </section>

        {/* ================================================= */}
        {/* CSV DATASET */}
        {/* ================================================= */}

        <section className="control-panel">

          <div className="section-heading">

            <div>

              <div className="section-kicker">
                DATASET
              </div>

              <h2>
                Upload Measurement Data
              </h2>

            </div>

            <div className="pipeline-indicator">
              <span>UPLOAD</span>
              <i>→</i>
              <span>VALIDATE</span>
              <i>→</i>
              <span>ANALYZE</span>
            </div>

          </div>

          <div className="controls">

            <label
              className="control csv-file-control"
            >
              <span>
                CSV Dataset
              </span>

              <input
                type="file"
                accept=".csv,text/csv"
                onChange={handleCsvChange}
              />
            </label>

            <div className="csv-file-name">
              {csvFile
                ? csvFile.name
                : "No dataset selected"}
            </div>

            <button
              className="run-button"
              onClick={uploadCsv}
              disabled={
                !csvFile ||
                csvUploading
              }
            >
              {csvUploading ? (
                <>
                  <span className="spinner" />
                  VALIDATING...
                </>
              ) : (
                <>
                  UPLOAD CSV
                  <span>↑</span>
                </>
              )}
            </button>

          </div>

          {csvError && (
            <div className="error-box">
              {csvError}
            </div>
          )}

          {csvResult && (
            <div className="csv-result">

              <div className="csv-result-header">
                <span>✓</span>
                DATASET VALIDATED
              </div>

              <div className="csv-summary">

                <div>
                  <span>FILE</span>
                  <strong>
                    {csvResult.filename}
                  </strong>
                </div>

                <div>
                  <span>ROWS</span>
                  <strong>
                    {csvResult.summary.rows}
                  </strong>
                </div>

                <div>
                  <span>COMPONENTS</span>
                  <strong>
                    {csvResult.summary.components}
                  </strong>
                </div>

                <div>
                  <span>VALUE RANGE</span>
                  <strong>
                    {csvResult.summary.value_min}
                    {" → "}
                    {csvResult.summary.value_max}
                  </strong>
                </div>

              </div>

            </div>
          )}

        </section>

        {/* ================================================= */}
        {/* SIMULATION CONFIGURATION */}
        {/* ================================================= */}

        <section className="control-panel">

          <div className="section-heading">

            <div>

              <div className="section-kicker">
                EXPERIMENT
              </div>

              <h2>
                Simulation Configuration
              </h2>

            </div>

            <div className="pipeline-indicator">
              <span>OBSERVE</span>
              <i>→</i>
              <span>MODEL</span>
              <i>→</i>
              <span>DECIDE</span>
            </div>

          </div>

          <div className="controls">

            <label className="control">

              <span>
                Normal Components
              </span>

              <input
                type="number"
                min="1"
                max="50"
                value={normalComponents}
                onChange={(e) =>
                  setNormalComponents(
                    e.target.value
                  )
                }
              />

            </label>

            <label className="control">

              <span>
                Anomalous Components
              </span>

              <input
                type="number"
                min="0"
                max="20"
                value={anomalousComponents}
                onChange={(e) =>
                  setAnomalousComponents(
                    e.target.value
                  )
                }
              />

            </label>

            <button
              className="run-button"
              onClick={runAnalysis}
              disabled={loading}
            >

              {loading ? (
                <>
                  <span className="spinner" />
                  ANALYZING...
                </>
              ) : (
                <>
                  RUN FALCON ANALYSIS
                  <span>→</span>
                </>
              )}

            </button>

          </div>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

        </section>

        {/* ================================================= */}
        {/* EMPTY STATE */}
        {/* ================================================= */}

        {!result && !loading && (
          <section className="empty-state">

            <div className="empty-icon">
              ⌁
            </div>

            <h2>
              Ready for Analysis
            </h2>

            <p>
              Configure the experiment above
              and run FALCON to generate
              AI-driven screening results.
            </p>

          </section>
        )}

        {/* ================================================= */}
        {/* LOADING */}
        {/* ================================================= */}

        {loading && (
          <section className="analysis-loading">

            <div className="loading-ring" />

            <h2>
              FALCON is analyzing
              trajectories...
            </h2>

            <p>
              Gaussian Process modeling
              → Decision Instability
              → Adaptive Scheduling
              → Anomaly Classification
            </p>

          </section>
        )}

        {/* ================================================= */}
        {/* RESULTS */}
        {/* ================================================= */}

        {result && !loading && (
          <>

            {/* METRICS */}

            <section className="metrics-grid">

              <MetricCard
                label="MEASUREMENT REDUCTION"
                value={`${reduction.toFixed(2)}%`}
                sub="Simulation result"
                type="highlight"
              />

              <MetricCard
                label="DECISION AGREEMENT"
                value={`${agreement.toFixed(2)}%`}
                sub="FALCON vs fixed-rate"
              />

              <MetricCard
                label="FALCON MEASUREMENTS"
                value={totalMeasurements}
                sub={`Fixed baseline: ${fixedMeasurements}`}
              />

              <MetricCard
                label="COMPONENTS ANALYZED"
                value={components.length}
                sub={`${normalComponents} normal + ${anomalousComponents} anomalous`}
              />

            </section>

            {/* DASHBOARD */}

            <section className="dashboard-grid">

              {/* CHART */}

              <div className="panel chart-panel">

                <div className="panel-header">

                  <div>

                    <div className="section-kicker">
                      ADAPTIVE SAMPLING
                    </div>

                    <h2>
                      Measurement Efficiency
                    </h2>

                  </div>

                  <div className="chart-legend">

                    <span>
                      <i className="legend-fixed" />
                      Fixed-rate
                    </span>

                    <span>
                      <i className="legend-falcon" />
                      FALCON
                    </span>

                  </div>

                </div>

                <BarChart
                  results={components}
                />

                <div className="chart-note">
                  FALCON dynamically reduces
                  measurements while maintaining
                  decision agreement with the
                  fixed-rate reference.
                </div>

              </div>

              {/* PIPELINE */}

              <div className="panel pipeline-panel">

                <div className="panel-header">

                  <div>

                    <div className="section-kicker">
                      DECISION PIPELINE
                    </div>

                    <h2>
                      FALCON Processing
                    </h2>

                  </div>

                </div>

                <div className="pipeline">

                  {[
                    [
                      "01",
                      "Observe",
                      "Initial trajectory",
                    ],

                    [
                      "02",
                      "Model",
                      "Gaussian Process",
                    ],

                    [
                      "03",
                      "Predict",
                      "Posterior trajectories",
                    ],

                    [
                      "04",
                      "Check",
                      "Decision instability",
                    ],

                    [
                      "05",
                      "Schedule",
                      "Adaptive measurement",
                    ],

                    [
                      "06",
                      "Classify",
                      "Normal / Anomalous",
                    ],
                  ].map(
                    ([
                      number,
                      title,
                      description,
                    ]) => (
                      <div
                        className="pipeline-step"
                        key={number}
                      >

                        <div className="step-number">
                          {number}
                        </div>

                        <div>

                          <strong>
                            {title}
                          </strong>

                          <span>
                            {description}
                          </span>

                        </div>

                      </div>
                    )
                  )}

                </div>

              </div>

            </section>

            {/* COMPONENT RESULTS */}

            <section className="panel results-panel">

              <div className="panel-header">

                <div>

                  <div className="section-kicker">
                    SCREENING OUTPUT
                  </div>

                  <h2>
                    Component Analysis
                  </h2>

                </div>

                <div className="threshold-display">
                  Anomaly threshold

                  <strong>
                    {Number(
                      result?.threshold ?? 0
                    ).toFixed(4)}
                  </strong>

                </div>

              </div>

              <div className="table-wrapper">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Component
                      </th>

                      <th>
                        Ground Truth
                      </th>

                      <th>
                        Fixed Decision
                      </th>

                      <th>
                        FALCON Decision
                      </th>

                      <th>
                        Anomaly Score
                      </th>

                      <th>
                        Measurements
                      </th>

                      <th>
                        Evidence
                      </th>

                    </tr>

                  </thead>

                  <tbody>

                    {components.map(
                      (component, index) => {

                        const id =
                          component.component_id ||
                          component.id ||
                          `C${String(
                            index + 1
                          ).padStart(2, "0")}`;

                        const truth =
                          component.truth ||
                          component.ground_truth ||
                          "—";

                        const fixed =
                          component.fixed_decision ||
                          component.fixed ||
                          "—";

                        const falcon =
                          component.falcon_decision ||
                          component.decision ||
                          "—";

                        const score =
                          Number(
                            component.anomaly_score ??
                              component.score ??
                              0
                          );

                        const measurements =
                          Number(
                            component.falcon_measurements ??
                              component.measurements ??
                              component.reads ??
                              0
                          );

                        const totalPoints =
                          Number(
                            component.total_points ??
                              101
                          );

                        const sampledPercent =
                          Math.round(
                            (measurements /
                              Math.max(
                                totalPoints,
                                1
                              )) *
                              100
                          );

                        return (
                          <tr key={id}>

                            <td>
                              <span className="component-id">
                                {id}
                              </span>
                            </td>

                            <td>
                              <StatusBadge
                                status={truth}
                              />
                            </td>

                            <td>
                              <StatusBadge
                                status={fixed}
                              />
                            </td>

                            <td>
                              <StatusBadge
                                status={falcon}
                              />
                            </td>

                            <td className="score-cell">

                              <ScoreBar
                                score={score}
                                threshold={
                                  result?.threshold
                                }
                              />

                            </td>

                            <td>

                              <strong>
                                {measurements}
                              </strong>

                              <span className="muted">
                                {" "}
                                / {totalPoints}
                              </span>

                            </td>

                            <td>

                              <div className="evidence">
                                {sampledPercent}%
                                {" "}
                                sampled
                              </div>

                            </td>

                          </tr>
                        );
                      }
                    )}

                  </tbody>

                </table>

              </div>

            </section>

            {/* TECHNICAL FOOTER */}

            <section className="technical-footer">

              <div>

                <strong>
                  FALCON AI/ML ENGINE
                </strong>

                <span>
                  Gaussian Process trajectory
                  modeling · Decision Instability
                  · Mahalanobis anomaly detection
                </span>

              </div>

              <div>

                <strong>
                  MODE
                </strong>

                <span>
                  Software Simulation / Safe Proxy
                </span>

              </div>

            </section>

          </>
        )}

      </main>

      {/* FOOTER */}

      <footer>

        <span>
          FALCON · AnomalyX
        </span>

        <span>
          AI-Driven Component Screening
        </span>

      </footer>

    </div>
  );
}

export default App;