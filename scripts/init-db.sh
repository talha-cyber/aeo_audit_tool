#!/bin/bash

# Database initialization script for production
set -e

# Create extensions if they don't exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create pgvector extension if available (for future vector similarity features)
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Create text search extensions for better search functionality
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    CREATE EXTENSION IF NOT EXISTS btree_gin;
    CREATE EXTENSION IF NOT EXISTS unaccent;
    
    -- Set up proper encoding
    UPDATE pg_database SET datcollate='C.UTF-8', datctype='C.UTF-8' WHERE datname='$POSTGRES_DB';
    
    -- Create performance indexes
    \c $POSTGRES_DB;
    
    -- These will be created by Alembic migrations, but this ensures basic setup
    -- Performance tuning
    ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
    ALTER SYSTEM SET max_connections = 200;
    ALTER SYSTEM SET shared_buffers = '256MB';
    ALTER SYSTEM SET effective_cache_size = '1GB';
    ALTER SYSTEM SET maintenance_work_mem = '64MB';
    ALTER SYSTEM SET checkpoint_completion_target = 0.9;
    ALTER SYSTEM SET wal_buffers = '16MB';
    ALTER SYSTEM SET default_statistics_target = 100;
    ALTER SYSTEM SET random_page_cost = 1.1;
    ALTER SYSTEM SET effective_io_concurrency = 200;
    
    SELECT pg_reload_conf();
EOSQL