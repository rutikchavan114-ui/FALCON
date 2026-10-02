#!/bin/bash
set -e
cd "$(dirname "$0")"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  FALCON / SIH26170 — ULTIMATE ENGINEERING BUILD         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
[ -f .env ] || cp .env.example .env

cleanup(){
  if [ -n "$BACKEND_PID" ]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

echo "→ Starting backend: http://localhost:8787"
node --env-file-if-exists=.env server.mjs &
BACKEND_PID=$!
sleep 0.7

echo "→ Starting dashboard: http://localhost:5173"
echo "→ Login: Engineer / ENGINEER@2026"
echo "→ Admin:    admin / FALCON@2026"
echo ""
npm run dev -- --host 127.0.0.1
