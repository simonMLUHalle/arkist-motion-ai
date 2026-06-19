#!/bin/zsh

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$ROOT_DIR/docs"
PORT=8000
URL="http://localhost:${PORT}/therapy_feedback_showcase/index.html"

cd "$DOCS_DIR"

echo "ARKIST showcase server starting in $DOCS_DIR"
echo "Open: $URL"
echo ""
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Existing server will be reused."
  SERVER_PID=""
else
  echo "Press Ctrl+C in this window to stop the server."
  python3 -m http.server "$PORT" &
  SERVER_PID=$!
  sleep 1
fi

if command -v open >/dev/null 2>&1; then
  open -a Safari "$URL" || true
fi

if [ -n "$SERVER_PID" ]; then
  wait "$SERVER_PID"
fi
