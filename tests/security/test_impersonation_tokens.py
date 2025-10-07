from __future__ import annotations

from app.security.auth.jwt_handler import get_jwt_handler


def test_mint_act_as_token_embeds_target_client_context() -> None:
    handler = get_jwt_handler()

    token = handler.mint_act_as_token(
        subject="admin@example.com",
        actor_id="admin@example.com",
        target_client_id="client-internal",
        ttl_seconds=900,
    )

    claims = handler.verify_token(token)
    assert "act_as" in claims
    assert claims["act_as"]["client_id"] == "client-internal"
    assert claims["act_as"]["actor"] == "admin@example.com"

    lifetime = int(claims["exp"]) - int(claims["iat"])
    assert 1 <= lifetime <= 900

    context = handler.parse_act_as_claims(claims)
    assert context is not None
    assert context["client_id"] == "client-internal"


def test_mint_act_as_token_supports_custom_ttl() -> None:
    handler = get_jwt_handler()

    token = handler.mint_act_as_token(
        subject="admin@example.com",
        actor_id="admin@example.com",
        target_client_id="client-short-ttl",
        ttl_seconds=120,
    )
    claims = handler.verify_token(token)
    lifetime = int(claims["exp"]) - int(claims["iat"])
    assert lifetime <= 120
