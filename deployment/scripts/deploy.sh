#!/usr/bin/env bash
set -euo pipefail

# Usage: IMAGE_TAG=<image> ./deployment/scripts/deploy.sh [staging|production] [blue-green|rolling]
STAGE=${1:-staging}
STRATEGY=${2:-blue-green}

IMAGE_TAG=${IMAGE_TAG:-}
if [[ -z "${IMAGE_TAG}" ]]; then
  echo "IMAGE_TAG not set" >&2
  exit 1
fi

echo "Deploying $IMAGE_TAG to $STAGE with strategy $STRATEGY"

# Default to docker-compose deployment
COMPOSE_FILE="docker-compose.prod.yml"

if [[ "$STRATEGY" == "blue-green" ]]; then
  # Run a side-by-side stack (suffix -green), health check, then swap
  SUFFIX=$RANDOM
  export AEO_STACK="aeo_${STAGE}_${SUFFIX}"
  echo "Starting temporary stack $AEO_STACK for health check"
  docker compose -f "$COMPOSE_FILE" pull
  docker compose -f "$COMPOSE_FILE" up -d --no-deps --quiet-pull web

  # Override image for web and worker
  docker stop aeo-web-new 2>/dev/null || true
  docker rm aeo-web-new 2>/dev/null || true
  docker run -d --name aeo-web-new -p 8001:8000 -e APP_ENV=$STAGE "$IMAGE_TAG"

  echo "Health checking new instance"
  ./deployment/scripts/health-check.sh http://localhost:8001/api/v1/health 30 || {
    echo "Health check failed; aborting rollout" >&2
    docker rm -f aeo-web-new || true
    exit 1
  }

  echo "Promoting new version"
  docker rm -f aeo-web-old 2>/dev/null || true
  docker rename aeo-web aeo-web-old 2>/dev/null || true
  docker rename aeo-web-new aeo-web
  docker stop aeo-web-old 2>/dev/null || true
  docker rm aeo-web-old 2>/dev/null || true

else
  echo "Rolling deployment via compose"
  docker compose -f "$COMPOSE_FILE" pull
  IMAGE_OVERRIDE=$IMAGE_TAG docker compose -f "$COMPOSE_FILE" up -d --no-deps web worker
fi

echo "Deployment complete"
