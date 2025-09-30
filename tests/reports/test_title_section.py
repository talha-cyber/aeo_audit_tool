import base64

from reportlab.platypus import Image

from app.reports.v2.sections import title
from app.reports.v2.theme import Theme


def _write_logo(tmp_path):
    logo_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    logo_path = tmp_path / "logo.png"
    logo_path.write_bytes(logo_bytes)
    return logo_path


def test_title_section_includes_logo(tmp_path):
    logo_path = _write_logo(tmp_path)
    theme = Theme(name="test", logo_path=str(logo_path))
    data = {"client_name": "Acme"}

    flowables = title.build(theme, data)

    assert any(isinstance(flowable, Image) for flowable in flowables)
