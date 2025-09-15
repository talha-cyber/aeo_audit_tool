#!/usr/bin/env bash
set -euo pipefail

# Runs Alembic migrations inside a temporary container.
IMAGE=${1:?image required}

echo "Running DB migrations with $IMAGE"
docker run --rm --network host -e APP_ENV=production "$IMAGE" alembic upgrade head
echo "Migrations complete"
