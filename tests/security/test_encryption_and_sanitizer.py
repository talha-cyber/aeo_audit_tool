from app.security.encryption.field_encryption import FieldEncryptor
from app.security.validation.input_sanitizer import sanitize_text


def test_field_encryption_roundtrip():
    enc = FieldEncryptor()
    secret = "my$ecretValue123"
    token = enc.encrypt(secret)
    assert token != secret
    plain = enc.decrypt(token)
    assert plain == secret


def test_input_sanitizer():
    raw = "  <script>alert('x')</script>\x00Hello  "
    cleaned = sanitize_text(raw)
    # '<' should be escaped, control char removed, trimmed
    assert cleaned.startswith("&lt;script&gt;alert")
    assert "\x00" not in cleaned
    assert cleaned.endswith("Hello")
