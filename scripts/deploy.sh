#!/bin/bash

# Production deployment script for AEO Competitive Intelligence Tool
set -e

echo "🚀 Starting production deployment..."

# Configuration
ENV_FILE=".env"
COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    log_error "Please don't run as root for security reasons"
    exit 1
fi

# Check if required files exist
if [ ! -f "$ENV_FILE" ]; then
    log_error "Environment file $ENV_FILE not found"
    log_info "Please create $ENV_FILE from .env.example and configure your values"
    exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
    log_error "Docker compose file $COMPOSE_FILE not found"
    exit 1
fi

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed"
    exit 1
fi

# Create backup directory
log_info "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup database if services are running
if docker-compose -f "$COMPOSE_FILE" ps db | grep -q "Up"; then
    log_info "Creating database backup..."
    docker-compose -f "$COMPOSE_FILE" exec -T db pg_dump -U postgres aeo_audit | gzip > "$BACKUP_DIR/database_backup.sql.gz"
    log_success "Database backup created"
fi

# Backup reports directory
if [ -d "reports" ] && [ "$(ls -A reports)" ]; then
    log_info "Backing up reports directory..."
    tar -czf "$BACKUP_DIR/reports_backup.tar.gz" reports/
    log_success "Reports backup created"
fi

# Pull latest images
log_info "Pulling latest Docker images..."
docker-compose -f "$COMPOSE_FILE" pull

# Build application images
log_info "Building application images..."
docker-compose -f "$COMPOSE_FILE" build --no-cache

# Stop existing services
log_info "Stopping existing services..."
docker-compose -f "$COMPOSE_FILE" down

# Start database and Redis first
log_info "Starting database and Redis..."
docker-compose -f "$COMPOSE_FILE" up -d db redis

# Wait for database to be ready
log_info "Waiting for database to be ready..."
timeout=60
while ! docker-compose -f "$COMPOSE_FILE" exec -T db pg_isready -U postgres -d aeo_audit >/dev/null 2>&1; do
    if [ $timeout -le 0 ]; then
        log_error "Database failed to start within expected time"
        exit 1
    fi
    sleep 2
    timeout=$((timeout-2))
done

# Run database migrations
log_info "Running database migrations..."
docker-compose -f "$COMPOSE_FILE" run --rm web alembic upgrade head

# Start all services
log_info "Starting all services..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be healthy
log_info "Waiting for services to be healthy..."
sleep 30

# Health check
log_info "Performing health checks..."
if curl -f http://localhost:8000/api/v1/health >/dev/null 2>&1; then
    log_success "Web service is healthy"
else
    log_error "Web service health check failed"
    docker-compose -f "$COMPOSE_FILE" logs web
    exit 1
fi

# Check Celery worker
if docker-compose -f "$COMPOSE_FILE" exec -T worker celery -A celery_worker inspect ping >/dev/null 2>&1; then
    log_success "Celery worker is healthy"
else
    log_warning "Celery worker health check failed"
    docker-compose -f "$COMPOSE_FILE" logs worker
fi

# Cleanup old Docker images
log_info "Cleaning up old Docker images..."
docker image prune -f

# Show service status
log_info "Service status:"
docker-compose -f "$COMPOSE_FILE" ps

# Show URLs
echo ""
log_success "🎉 Deployment completed successfully!"
echo ""
echo "📊 Service URLs:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/api/v1/health"
echo "  - Grafana: http://localhost:3000 (admin/\$GRAFANA_ADMIN_PASSWORD)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Flower (Celery): http://localhost:5555"
echo ""
echo "📁 Backup created in: $BACKUP_DIR"
echo ""
echo "🔧 To monitor logs:"
echo "  docker-compose -f $COMPOSE_FILE logs -f"
echo ""
echo "🛑 To stop services:"
echo "  docker-compose -f $COMPOSE_FILE down"
echo ""

log_success "Deployment script completed!"