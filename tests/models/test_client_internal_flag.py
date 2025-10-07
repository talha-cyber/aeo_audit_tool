from __future__ import annotations

import pytest

from app.models.audit import Client


def test_client_is_internal_defaults_false() -> None:
    """Clients should expose an is_internal flag defaulting to False."""

    client = Client(id="client-default", name="Default Client")

    # Accessing the attribute should yield False without explicit assignment.
    assert client.is_internal is False


@pytest.mark.parametrize("value", [True, False])
def test_client_is_internal_assignment(value: bool) -> None:
    """The is_internal flag should be assignable at construction time."""

    client = Client(id="client-explicit", name="Explicit Client", is_internal=value)
    assert client.is_internal is value
