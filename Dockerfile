FROM python:3.13.7-slim

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

# Install build essentials, then clean cache to reduce image surface area
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies early for build caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLP assets (commented out for now)
# RUN python -m spacy download en_core_web_sm

# Copy application source
COPY . .

# Prepare runtime directories before switching users
RUN mkdir -p reports logs

# Create and switch to non-root user for runtime safety
RUN groupadd -r aeo && useradd -r -g aeo --home /app aeo \
    && chown -R aeo:aeo /app

USER aeo

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
