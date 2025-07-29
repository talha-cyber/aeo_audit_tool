# AEO Competitive Intelligence Tool

## Overview
This platform simulates real user questions across multiple AI platforms (ChatGPT, Claude, Perplexity, Google AI) to monitor brand visibility, extract sentiment, and track competitor presence. Built for white-label use by SEO/AEO agencies.

## Core Features
- Query simulation across AI engines
- Brand & competitor detection with sentiment scoring
- PDF reports (Executive Summary, Platform Analysis, Recommendations)
- Background task processing via Celery
- REST API with audit config, run, and status endpoints
- Dockerized with Postgres & Redis

## Tech Stack
- **Backend**: Python 3.11+, FastAPI
- **Queue**: Celery + Redis
- **DB**: PostgreSQL + SQLAlchemy (with pgvector)
- **LLM APIs**: OpenAI, Anthropic, Perplexity, Google AI
- **NLP**: spaCy, Transformers, fastText
- **Reporting**: ReportLab, Matplotlib
- **Deployment**: Docker + docker-compose

## Folder Structure
aeo-audit-tool/
├── app/
│ ├── api/
│ ├── config/
│ ├── models/
│ ├── services/
│ ├── tasks/
│ └── utils/
├── docker/
├── tests/
├── scripts/
├── alembic/
├── frontend/ (optional)



## Getting Started (Dev)
```bash
git clone https://github.com/your-org/aeo-audit-tool
cd aeo-audit-tool
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
docker-compose up --build


| Task            | Command                                  |
| --------------- | ---------------------------------------- |
| Run API         | `uvicorn app.main:app --reload`          |
| Start worker    | `celery -A app.tasks.audit_tasks worker` |
| Start scheduler | `celery -A app.tasks.audit_tasks beat`   |
| Run tests       | `pytest`                                 |


Environment Variables

Set your .env file using the provided template:

DATABASE_URL=postgresql://postgres:password@localhost:5432/aeo_audit
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
PERPLEXITY_API_KEY=...
GOOGLE_AI_API_KEY=...


Deployment

All services run in Docker (API, Redis, Postgres, Celery)
PDF reports are saved to reports/ and can be mounted as volume
Notes

This project is async-first and modular
All logic is structured for testability and CI/CD
Cursor AI tools reference ARCHITECTURE.md for module responsibilities
