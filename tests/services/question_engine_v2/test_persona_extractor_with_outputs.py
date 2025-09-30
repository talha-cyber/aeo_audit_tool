"""
Enhanced version of persona extractor tests that show actual outputs.
Copy this to see outputs during testing.
"""

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


def print_personas(personas, title: str):
    """Helper to print persona details."""
    print(f"\n{title}")
    print(f"{'='*50}")
    print(f"Found {len(personas)} personas:")
    
    for i, persona in enumerate(personas):
        print(f"\n--- Persona {i + 1} ---")
        print(f"Mode: {persona.mode.value}")
        print(f"Role: {persona.role}")
        print(f"Driver: {persona.driver}")
        print(f"Contexts: {persona.contexts}")
        print(f"Emotional Anchor: {persona.emotional_anchor}")
        print(f"Voice: {persona.voice or 'Custom (no preset)'}")


def test_load_catalog_b2c_with_output(extractor: PersonaExtractor) -> None:
    """Test catalog loading and show what's available."""
    catalog = extractor.load_catalog(PersonaMode.B2C)
    
    print(f"\n{'='*50}")
    print(f"B2C CATALOG LOADED")
    print(f"{'='*50}")
    print(f"Mode: {catalog.mode}")
    print(f"Available roles: {list(catalog.roles.keys())}")
    print(f"Available drivers: {list(catalog.drivers.keys())}")
    print(f"Available contexts: {list(catalog.contexts.keys())}")
    print(f"Available voice presets: {list(catalog.voices.keys())}")
    
    # Show one example of each
    print(f"\nExample Role ('advocate'):")
    advocate = catalog.roles['advocate']
    print(f"  Label: {advocate.label}")
    print(f"  Description: {advocate.description}")
    print(f"  Default Context: {advocate.default_context_stage}")
    
    print(f"\nExample Driver ('values_aligned'):")
    driver = catalog.drivers['values_aligned']
    print(f"  Label: {driver.label}")
    print(f"  Emotional Anchor: {driver.emotional_anchor}")
    print(f"  Weight: {driver.weight}")
    
    print(f"\nExample Voice Preset ('loyalist'):")
    voice = catalog.voices['loyalist']
    print(f"  Role: {voice.role}")
    print(f"  Driver: {voice.driver}")
    print(f"  Contexts: {voice.contexts}")

    # Original assertions
    assert catalog.mode == PersonaMode.B2C
    assert "advocate" in catalog.roles
    assert "deal_hunter" in catalog.drivers
    assert "value_shopper" in catalog.voices


def test_resolve_personas_with_presets_and_custom_with_output(extractor: PersonaExtractor) -> None:
    """Test mixed persona resolution and show the results."""
    personas = extractor.resolve_personas(
        mode=PersonaMode.B2B,
        voices=["cost_cutter_cfo"],
        selections=[
            PersonaSelection(role="vp_sales", driver="scalability", contexts=["evaluation"])
        ],
    )
    
    print_personas(personas, "MIXED PERSONAS TEST (B2B)")
    
    print(f"\nDETAILED ANALYSIS:")
    preset = next(p for p in personas if p.voice == "cost_cutter_cfo")
    custom = next(p for p in personas if p.voice is None)
    
    print(f"Preset persona details:")
    print(f"  - Uses voice: {preset.voice}")
    print(f"  - Emotional anchor: {preset.emotional_anchor}")
    print(f"  - Role: {preset.role}")
    
    print(f"Custom persona details:")
    print(f"  - Custom combination of role + driver + contexts")
    print(f"  - Role: {custom.role}")
    print(f"  - Driver: {custom.driver}")
    print(f"  - Contexts: {custom.contexts}")

    # Original assertions
    assert len(personas) == 2
    assert preset.emotional_anchor == "fear_of_waste"
    assert custom.role == "vp_sales"
    assert custom.contexts == ["evaluation"]


def test_resolve_personas_ignores_unknown_voice_with_output(extractor: PersonaExtractor) -> None:
    """Test error handling and show graceful degradation."""
    print(f"\n{'='*50}")
    print(f"ERROR HANDLING TEST")
    print(f"{'='*50}")
    print(f"Requesting: ['missing_voice', 'value_shopper']")
    print(f"Expected: Should ignore 'missing_voice' and return only 'value_shopper'")
    
    personas = extractor.resolve_personas(
        mode=PersonaMode.B2C,
        voices=["missing_voice", "value_shopper"],
    )

    print_personas(personas, "RESULT")
    print(f"\nAs expected: Unknown voice was ignored, valid voice was processed.")

    # Original assertions
    assert len(personas) == 1
    assert personas[0].voice == "value_shopper"


def test_resolve_personas_falls_back_to_default_context_with_output(extractor: PersonaExtractor) -> None:
    """Test fallback behavior and show how defaults work."""
    print(f"\n{'='*50}")
    print(f"FALLBACK BEHAVIOR TEST")
    print(f"{'='*50}")
    print(f"Requesting role 'advocate' with unknown context 'unknown_stage'")
    print(f"Expected: Should fall back to role's default context")
    
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

    print_personas(personas, "FALLBACK RESULT")
    
    print(f"\nAnalysis:")
    print(f"- Requested context: ['unknown_stage']")
    print(f"- Actual context: {personas[0].contexts}")
    print(f"- Fallback worked: Used role's default context instead")

    # Original assertions
    assert len(personas) == 1
    assert personas[0].contexts == ["loyalty"]

