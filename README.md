# FALCON — SIH26170 Ultimate Demo Build

FALCON is a local engineering dashboard for component burn-in/screening. This build is designed to run **without hardware** while keeping hardware integration ready.

## What is active

- Premium animated React/Vite dashboard
- FALCON logo integrated in login, sidebar and reports
- Admin + Engineer login with password authentication
- Node backend with authenticated endpoints
- Server-Sent Events telemetry stream
- Hardware telemetry ingestion endpoint
- Hardware-free Demo Mode with Healthy / Stress / Fault scenarios
- Transparent Arrhenius calculation
- Demo evidence/risk engine (explicitly labelled synthetic)
- Gemini server-side AI Copilot when `GEMINI_API_KEY` is configured
- Consolidated print-ready engineering report with logo + charts + provenance
- 24-component review catalog
- Command palette (`⌘K` / `Ctrl+K`)
- Dark/light theme
- Responsive UI and reduced-motion support

## Start on macOS

```bash
cd ~/Desktop
unzip -o FALCON_SIH26170_ULTIMATE_TOP_NOTCH.zip -d falcon_ultimate
cd falcon_ultimate
cp .env.example .env
npm install
./start_falcon.command
```

Open: http://localhost:5173

## Demo login

Engineer:

- Username: `engineer`
- Password: `ENGINEER@2026`

Admin:

- Username: `admin`
- Password: `FALCON@2026`

Change these values in `.env` before using the project outside the demo environment.

## Gemini AI

Edit `.env`:

```env
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
```

Restart `./start_falcon.command` after changing `.env`.

The AI Copilot is instructed to stay evidence-bound and not invent measurements, accuracy, specifications or validation results.

## No hardware? Use Demo Mode

1. Login.
2. Open **Live Burn-In**.
3. Choose `healthy`, `stress`, or `fault`.
4. Press **Start simulation**.
5. Watch SSE telemetry, charts, Arrhenius, risk evidence and reports update live.

The UI labels this data as **DEMO / SIMULATION**. It is not represented as a real hardware measurement.

## Hardware integration

The authenticated hardware bridge can POST:

```http
POST http://localhost:8787/api/telemetry
Authorization: Bearer <session-token>
Content-Type: application/json
```

```json
{
  "componentId": "DUT-01",
  "voltage": 12.1,
  "current": 0.42,
  "temperature": 71.4,
  "timestamp": "2026-10-01T17:30:00Z"
}
```

The dashboard receives accepted measurements through `/api/stream`.

## Backend endpoints

- `GET /api/health`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/stream`
- `GET /api/telemetry/latest`
- `POST /api/telemetry`
- `GET /api/demo/status`
- `POST /api/demo/start`
- `POST /api/demo/stop`
- `POST /api/llm`
- `POST /api/report`

## Engineering boundary

The demo evidence engine is intentionally not presented as a trained production ML model. Calibrated healthy populations, validated model artifacts and real hardware data are required before making production screening claims.

## ULTIMATE TOP-NOTCH BUILD NOTES

- UI uses a cinematic neon visual system with animated telemetry, hover states, orbit effects, gradient motion and reduced-motion support.
- Sidebar/report/login FALCON marks use a transparent artwork variant to prevent clipping and white-box artifacts.
- Gemini model default is `gemini-3.8-flash`; configure `GEMINI_API_KEY` in `.env` and restart the backend.
- Demo mode is the hardware-free path: it uses the same authenticated backend, SSE stream, deterministic physics layer, dashboard, AI copilot and report pipeline, but telemetry is explicitly labelled `DEMO_SIMULATION`.
- Hardware telemetry can later POST to `/api/telemetry` without changing the frontend architecture.

### Gemini setup

```bash
cp .env.example .env
# edit .env and add GEMINI_API_KEY=YOUR_KEY
# GEMINI_MODEL=gemini-3.8-flash
./start_falcon.command
```

If an existing `.env` still contains an older model, update it with:

```bash
sed -i '' 's/^GEMINI_MODEL=.*/GEMINI_MODEL=gemini-3.8-flash/' .env
```


## Gemini AI setup (current)

FALCON uses Google Gemini 3.8 Flash through the Gemini Interactions API. The browser never receives the API key; the local Node backend calls Google server-side.

1. Copy `.env.example` to `.env`.
2. Set `GEMINI_API_KEY=YOUR_KEY`.
3. Keep `GEMINI_MODEL=gemini-3.8-flash`.
4. Restart `./start_falcon.command`.
5. Open the AI Copilot and wait for the connection check to show ONLINE.

If an older `GEMINI_MODEL=gemini-2.5-flash` is present, the backend automatically normalizes that legacy value to `gemini-3.8-flash`.
