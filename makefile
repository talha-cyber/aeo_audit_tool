# Makefile

.PHONY: healthcheck

healthcheck:
	@echo "🔧 Running AEO Audit Tool Health Check...\n"

	@echo "1️⃣  Checking Alembic migration..." && \
	docker-compose exec web alembic upgrade head || exit 1

	@echo "\n2️⃣  Checking environment config via Pydantic..." && \
	docker-compose exec web python -c 'from app.core.config import settings; print(settings.model_dump())' || exit 1

	@echo "\n3️⃣  Running Ruff (lint)..." && \
	docker-compose exec web ruff check . || exit 1

	@echo "\n5️⃣  Running Pytest with coverage..." && \
	docker-compose exec web pytest --cov=app --cov-report=term-missing || exit 1

	@echo "\n✅ All health checks passed!"
