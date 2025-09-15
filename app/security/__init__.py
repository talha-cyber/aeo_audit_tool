"""
Security package: auth, encryption, audit logging, validation, monitoring.

Modules are async-first where applicable, use structlog for logging, and prefer
Redis for shared state (sessions, rate limiting). Keep interfaces small and
composable so they can be swapped by environment.
"""
