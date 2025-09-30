from pathlib import Path
import shutil

import pytest
import yaml

from app.services.question_engine_v2.persona_extractor import PersonaExtractor
from app.services.question_engine_v2.schemas import (
    PersonaMode,
    PersonaSelection,
)


@pytest.fixture()
def extractor() -> PersonaExtractor:
    return PersonaExtractor()


@pytest.fixture()
def custom_catalog_dir(tmp_path: Path) -> Path:
    source_dir = (
        Path(__file__).resolve().parents[3] / "app" / "services" / "question_engine_v2" / "catalogs"
    )
    filenames = [
        "b2c_roles.yaml",
        "b2c_drivers.yaml",
        "b2c_contexts.yaml",
        "b2c_voices.yaml",
    ]

    for name in filenames:
        shutil.copy(source_dir / name, tmp_path / name)

    roles_path = tmp_path / "b2c_roles.yaml"
    roles_data = yaml.safe_load(roles_path.read_text())
    roles_data["skeptic"].pop("default_context_stage", None)
    roles_path.write_text(yaml.safe_dump(roles_data, sort_keys=False))

    return tmp_path


@pytest.fixture()
def custom_extractor(custom_catalog_dir: Path) -> PersonaExtractor:
    return PersonaExtractor(base_path=custom_catalog_dir)


def test_load_catalog_b2c(extractor: PersonaExtractor) -> None:
    catalog = extractor.load_catalog(PersonaMode.B2C)

    assert catalog.mode == PersonaMode.B2C
    assert "advocate" in catalog.roles
    assert "deal_hunter" in catalog.drivers
    assert "value_shopper" in catalog.voices


def test_resolve_personas_with_presets_and_custom(extractor: PersonaExtractor) -> None:
    personas = extractor.resolve_personas(
        mode=PersonaMode.B2B,
        voices=["cost_cutter_cfo"],
        selections=[
            PersonaSelection(role="vp_sales", driver="scalability", contexts=["evaluation"])
        ],
    )

    assert len(personas) == 2
    preset = next(p for p in personas if p.voice == "cost_cutter_cfo")
    assert preset.emotional_anchor == "fear_of_waste"
    custom = next(p for p in personas if p.voice is None)
    assert custom.role == "vp_sales"
    assert custom.contexts == ["evaluation"]


def test_resolve_personas_ignores_unknown_voice(extractor: PersonaExtractor) -> None:
    personas = extractor.resolve_personas(
        mode=PersonaMode.B2C,
        voices=["missing_voice", "value_shopper"],
    )

    assert len(personas) == 1
    assert personas[0].voice == "value_shopper"


def test_resolve_personas_falls_back_to_default_context(extractor: PersonaExtractor) -> None:
    personas = extractor.resolve_personas(
        mode=PersonaMode.B2C,
        selections=[
            PersonaSelection(
                role="advocate",
                driver="values_aligned",
                contexts=["unknown_stage"],
            )
        ],
    )

    assert len(personas) == 1
    assert personas[0].contexts == ["loyalty"]


def test_persona_extractor_capabilities(
    custom_extractor: PersonaExtractor, custom_catalog_dir: Path
) -> None:
    personas = custom_extractor.resolve_personas(
        mode=PersonaMode.B2C,
        voices=["value_shopper", "value_shopper"],
        selections=[
            PersonaSelection(
                role="skeptic",
                driver="deal_hunter",
                contexts=["unknown_context"],
            )
        ],
    )

    assert len(personas) == 2
    preset = next(p for p in personas if p.voice == "value_shopper")
    custom = next(p for p in personas if p.voice is None)

    assert preset.emotional_anchor == "fear_of_overpaying"
    assert custom.emotional_anchor == "fear_of_overpaying"
    assert custom.contexts == ["awareness"]

    catalog_payload = custom_extractor.export_catalog(PersonaMode.B2C)
    assert catalog_payload["roles"]["skeptic"]["label"] == "Skeptical Buyer"
    assert catalog_payload["roles"]["skeptic"].get("default_context_stage") is None
    assert catalog_payload["voices"]["value_shopper"]["contexts"] == [
        "evaluation",
        "purchase",
    ]

    catalog_initial = custom_extractor.load_catalog(PersonaMode.B2C)
    assert catalog_initial.drivers["deal_hunter"].emotional_anchor == "fear_of_overpaying"

    drivers_path = custom_catalog_dir / "b2c_drivers.yaml"
    drivers_data = yaml.safe_load(drivers_path.read_text())
    drivers_data["deal_hunter"]["emotional_anchor"] = "optimism"
    drivers_path.write_text(yaml.safe_dump(drivers_data, sort_keys=False))

    catalog_cached = custom_extractor.load_catalog(PersonaMode.B2C)
    assert catalog_cached.drivers["deal_hunter"].emotional_anchor == "fear_of_overpaying"

    custom_extractor.clear_cache()
    catalog_updated = custom_extractor.load_catalog(PersonaMode.B2C)
    assert catalog_updated.drivers["deal_hunter"].emotional_anchor == "optimism"
