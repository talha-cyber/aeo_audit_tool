# AEO Audit Tool - Setup Guide

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root with the following configuration:

```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/aeo_audit

# Redis Configuration
REDIS_URL=redis://localhost:6379

# AI Platform API Keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
GOOGLE_AI_API_KEY=your_google_ai_key_here

# Application Configuration
SECRET_KEY=your_secret_key_here
DEBUG=True
LOG_LEVEL=INFO

# Rate Limiting (requests per minute)
OPENAI_RATE_LIMIT=50
ANTHROPIC_RATE_LIMIT=100
PERPLEXITY_RATE_LIMIT=20
GOOGLE_AI_RATE_LIMIT=60
```

### 2. Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Docker Setup (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Testing

```bash
# Run tests
pytest

# Test the health endpoint directly
curl http://localhost:8000/health
```

### 5. API Documentation

Once running, visit:
- API Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## Services

- **Web API**: FastAPI application on port 8000
- **Worker**: Celery worker for background tasks
- **Database**: PostgreSQL on port 5432
- **Redis**: Redis cache/broker on port 6379

## Project Structure

```
AEO_Audit_tool/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application with /health endpoint
├── tests/
│   ├── __init__.py
│   └── test_health.py       # Health endpoint tests
├── celery_worker.py         # Celery configuration with test task
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Multi-service setup
├── requirements.txt         # Python dependencies
└── README_setup.md         # This file
```

## Next Steps

This is the basic skeleton. To implement the full AEO audit functionality, refer to the ARCHITECTURE.md file for detailed implementation phases including:

- Database models and migrations
- AI platform integrations
- Brand detection engine
- Question generation
- Report generation
- API endpoints
