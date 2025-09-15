from __future__ import annotations

import os
from typing import Optional


class SecretsManager:
    """Simple secrets manager facade.

    For now reads from environment. Extend to cloud secret stores later.
    """

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(name, default)
