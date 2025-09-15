#!/usr/bin/env bash
set -euo pipefail

URL=${1:?health check URL required}
TIMEOUT=${2:-30}

echo "Health check $URL with timeout ${TIMEOUT}s"
end=$((SECONDS+TIMEOUT))
while [ $SECONDS -lt $end ]; do
  if curl -fsS "$URL" >/dev/null; then
    echo "Healthy"
    exit 0
  fi
  sleep 2
done
echo "Unhealthy after ${TIMEOUT}s"
exit 1
