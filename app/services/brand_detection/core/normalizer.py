import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Set

from ..market_adapters.base import MarketAdapterFactory

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result of brand name normalization"""

    original: str
    normalized_forms: Set[str]
    primary_form: str
    confidence: float
    market_specific_forms: Dict[str, Set[str]]
    metadata: Dict[str, any]


class BrandNormalizer:
    """Advanced brand name normalization engine"""

    def __init__(self):
        self.common_abbreviations = {
            "corporation": ["corp", "co"],
            "company": ["co", "comp"],
            "incorporated": ["inc", "incorp"],
            "limited": ["ltd", "lim"],
            "gesellschaft": ["ges"],
            "aktiengesellschaft": ["ag"],
            "gmbh": ["gesellschaft mit beschränkter haftung"],
            "international": ["intl", "int'l"],
            "technologies": ["tech", "technologies"],
            "software": ["sw", "soft"],
            "systems": ["sys", "syst"],
        }

        # Unicode normalization patterns
        self.unicode_patterns = [
            # Smart quotes
            (r'["""]', '"'),
            (r"['']", "'"),
            # Dashes
            (r"[‒–—―]", "-"),
            # Spaces
            (r"[\u00A0\u2000-\u200B\u2028\u2029\u202F\u205F\u3000]", " "),
        ]

        # Common brand name patterns
        self.brand_patterns = [
            # Camel case splitting
            (r"([a-z])([A-Z])", r"\1 \2"),
            # Number-letter boundaries
            (r"(\d)([A-Za-z])", r"\1 \2"),
            (r"([A-Za-z])(\d)", r"\1 \2"),
        ]

    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters in brand name"""
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Apply pattern replacements
        for pattern, replacement in self.unicode_patterns:
            text = re.sub(pattern, replacement, text)

        return text.strip()

    def normalize_basic(self, brand: str) -> Set[str]:
        """Basic normalization - case, spacing, punctuation"""
        if not brand:
            return set()

        variations = set()

        # Original
        variations.add(brand)

        # Unicode normalized
        normalized = self.normalize_unicode(brand)
        variations.add(normalized)

        # Case variations
        variations.add(brand.lower())
        variations.add(brand.upper())
        variations.add(brand.title())

        # Remove punctuation variations
        no_punct = re.sub(r"[^\w\s]", "", brand)
        if no_punct != brand:
            variations.add(no_punct)
            variations.add(no_punct.lower())

        # Normalize whitespace
        normalized_space = re.sub(r"\s+", " ", brand).strip()
        variations.add(normalized_space)

        # Remove extra characters
        clean = re.sub(r"[^\w\s-]", "", brand)
        variations.add(clean)

        return {v for v in variations if v and v.strip()}

    def normalize_comprehensive(
        self, brand: str, market_code: str = "DE"
    ) -> NormalizationResult:
        """Comprehensive brand normalization"""

        if not brand or not brand.strip():
            return NormalizationResult(
                original="",
                normalized_forms=set(),
                primary_form="",
                confidence=0.0,
                market_specific_forms={},
                metadata={"error": "Empty brand name"},
            )

        all_variations = set()
        market_specific = {}

        try:
            # Get market adapter
            adapter = MarketAdapterFactory.create_adapter(market_code)

            # Basic normalization
            basic_forms = self.normalize_basic(brand)
            all_variations.update(basic_forms)

            # Market-specific normalization
            market_forms = adapter.normalize_brand_name(brand)
            all_variations.update(market_forms)
            market_specific[market_code] = market_forms

            # Clean up variations
            clean_variations = {
                v.strip()
                for v in all_variations
                if v and v.strip() and len(v.strip()) >= 2
            }

            # Determine primary form (most common or original)
            primary_form = self._determine_primary_form(brand, clean_variations)

            # Calculate confidence based on consistency
            confidence = self._calculate_normalization_confidence(
                brand, clean_variations
            )

            metadata = {
                "total_variations": len(clean_variations),
                "market_code": market_code,
                "has_market_specific": len(market_forms) > 0,
                "normalization_methods": ["basic", "market_specific"],
            }

            return NormalizationResult(
                original=brand,
                normalized_forms=clean_variations,
                primary_form=primary_form,
                confidence=confidence,
                market_specific_forms=market_specific,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Normalization failed for brand '{brand}': {e}")
            return NormalizationResult(
                original=brand,
                normalized_forms={brand},
                primary_form=brand,
                confidence=0.5,
                market_specific_forms={},
                metadata={"error": str(e)},
            )

    def _determine_primary_form(self, original: str, variations: Set[str]) -> str:
        """Determine the primary normalized form"""
        if not variations:
            return original

        # Preference order:
        # 1. Original if it exists in variations
        if original in variations:
            return original

        # 2. Title case version
        title_versions = [v for v in variations if v.istitle()]
        if title_versions:
            return min(title_versions, key=len)  # Shortest title case

        # 3. First uppercase version
        upper_versions = [v for v in variations if v[0].isupper()]
        if upper_versions:
            return min(upper_versions, key=len)

        # 4. Shortest variation
        return min(variations, key=len)

    def _calculate_normalization_confidence(
        self, original: str, variations: Set[str]
    ) -> float:
        """Calculate confidence in normalization quality"""
        if not variations:
            return 0.0

        base_confidence = 0.5

        # More variations generally mean better coverage
        variation_boost = min(0.3, len(variations) * 0.02)
        base_confidence += variation_boost

        # Prefer if original is preserved
        if original in variations:
            base_confidence += 0.1

        # Check for reasonable variation distribution
        length_variance = len(set(len(v) for v in variations))
        if length_variance > 1:  # Good variation in lengths
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def get_normalization_stats(self) -> Dict[str, any]:
        """Get normalization performance statistics"""
        return {
            "abbreviation_mappings": len(self.common_abbreviations),
            "unicode_patterns": len(self.unicode_patterns),
            "brand_patterns": len(self.brand_patterns),
        }
