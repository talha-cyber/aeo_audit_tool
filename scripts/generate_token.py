#!/usr/bin/env python3
"""
Generates a JWT for a given subject.
"""
import sys
from app.security.auth.jwt_handler import get_jwt_handler

def generate_token(subject: str):
    """
    Generates a JWT for the given subject.
    """
    jwt_handler = get_jwt_handler()
    token = jwt_handler.create_token(subject=subject)
    print(token)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_token.py <subject>")
        sys.exit(1)
    generate_token(sys.argv[1])
