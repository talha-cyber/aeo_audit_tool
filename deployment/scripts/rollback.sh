#!/usr/bin/env bash
set -euo pipefail

# Usage: IMAGE_TAG=<image> ./deployment/scripts/rollback.sh
IMAGE_TAG=${IMAGE_TAG:-}
if [[ -z "$IMAGE_TAG" ]]; then
  echo "IMAGE_TAG not set" >&2
  exit 1
fi

echo "Rolling back to $IMAGE_TAG"
docker stop aeo-web || true
docker rm aeo-web || true
docker run -d --name aeo-web -p 8000:8000 -e APP_ENV=production "$IMAGE_TAG"
echo "Rollback complete"
