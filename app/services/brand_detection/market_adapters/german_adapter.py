import re
from typing import Dict, List, Set

from ..models.brand_mention import DetectionMethod
from .base import BaseMarketAdapter, MarketAdapterFactory


class GermanMarketAdapter(BaseMarketAdapter):
    """German market-specific brand detection logic"""

    def __init__(self):
        super().__init__("DE", "de")

        # German-specific character mappings
        self.umlaut_mappings = {
            "ä": ["ae", "a"],
            "ö": ["oe", "o"],
            "ü": ["ue", "u"],
            "Ä": ["Ae", "AE", "A"],
            "Ö": ["Oe", "OE", "O"],
            "Ü": ["Ue", "UE", "U"],
            "ß": ["ss", "s"],
        }

        # German compound word patterns
        self.compound_patterns = [
            r"([A-ZÄÖÜ][a-zäöüß]+)([A-ZÄÖÜ][a-zäöüß]+)",  # CamelCase compounds
            r"([a-zäöüß]+)-([a-zäöüß]+)",  # Hyphenated compounds
        ]

        # German business context indicators
        self.business_context_indicators = [
            "unternehmen",
            "firma",
            "gesellschaft",
            "konzern",
            "gruppe",
            "software",
            "lösung",
            "anbieter",
            "hersteller",
            "dienstleister",
        ]

    def _get_company_suffixes(self) -> List[str]:
        """German company legal forms"""
        return [
            "GmbH",
            "AG",
            "KG",
            "OHG",
            "mbH",
            "eV",
            "UG",
            "GmbH & Co. KG",
            "SE",
            "KGaA",
            "eG",
            # Variations
            "gmbh",
            "ag",
            "kg",
            "ohg",
            "mbh",
            "ev",
            "ug",
        ]

    def _get_business_keywords(self) -> List[str]:
        """German business context keywords"""
        return [
            "unternehmen",
            "firma",
            "gesellschaft",
            "konzern",
            "gruppe",
            "anbieter",
            "hersteller",
            "dienstleister",
            "entwickler",
            "software",
            "lösung",
            "system",
            "plattform",
            "tool",
            "service",
            "dienst",
            "produkt",
            "marke",
            "brand",
        ]

    def _get_industry_patterns(self) -> Dict[str, List[str]]:
        """German industry-specific patterns"""
        return {
            "software": [
                "softwareunternehmen",
                "softwareanbieter",
                "softwarehersteller",
                "it-unternehmen",
                "technologieunternehmen",
            ],
            "automotive": [
                "automobilhersteller",
                "autobauer",
                "fahrzeughersteller",
                "autozulieferer",
                "automobilkonzern",
            ],
            "finance": [
                "bank",
                "sparkasse",
                "versicherung",
                "finanzdienstleister",
                "kreditinstitut",
                "geldinstitut",
            ],
            "retail": [
                "einzelhändler",
                "handelskette",
                "kaufhaus",
                "supermarkt",
                "onlineshop",
                "e-commerce",
            ],
        }

    def normalize_brand_name(self, brand: str) -> Set[str]:
        """Generate German-specific brand variations"""
        variations = {brand, brand.lower(), brand.upper()}

        # Handle umlauts
        for original, replacements in self.umlaut_mappings.items():
            if original in brand:
                for replacement in replacements:
                    variations.add(brand.replace(original, replacement))
                    variations.add(brand.replace(original, replacement).lower())

        # Handle company suffixes
        brand_without_suffix = self._remove_company_suffix(brand)
        if brand_without_suffix != brand:
            variations.add(brand_without_suffix)
            variations.add(brand_without_suffix.lower())

            # Add variations with different suffixes
            for suffix in self.company_suffixes[:5]:  # Top 5 most common
                variations.add(f"{brand_without_suffix} {suffix}")

        # Handle compound words
        compound_parts = self._split_compound_word(brand)
        if len(compound_parts) > 1:
            # Add individual parts
            variations.update(compound_parts)
            # Add different combinations
            variations.add(" ".join(compound_parts))
            variations.add("-".join(compound_parts))

        # Handle acronyms (German companies often use them)
        if len(brand.split()) > 1:
            acronym = "".join(word[0].upper() for word in brand.split() if word)
            variations.add(acronym)
            variations.add(acronym.lower())

        # Clean up variations
        return {v.strip() for v in variations if v.strip() and len(v.strip()) >= 2}

    def _remove_company_suffix(self, brand: str) -> str:
        """Remove German company suffixes"""
        for suffix in self.company_suffixes:
            patterns = [
                f" {suffix}$",
                f" {suffix.lower()}$",
                f"-{suffix}$",
                f"-{suffix.lower()}$",
            ]
            for pattern in patterns:
                brand = re.sub(pattern, "", brand, flags=re.IGNORECASE)
        return brand.strip()

    def _split_compound_word(self, word: str) -> List[str]:
        """Split German compound words"""
        parts = [word]

        for pattern in self.compound_patterns:
            matches = re.finditer(pattern, word)
            for match in matches:
                parts.extend(match.groups())

        # Remove duplicates and original word if split occurred
        unique_parts = list(set(parts))
        if len(unique_parts) > 1 and word in unique_parts:
            unique_parts.remove(word)

        return [part for part in unique_parts if len(part) >= 2]

    def calculate_market_confidence(
        self, original_text: str, brand: str, context: str
    ) -> float:
        """Calculate German market-specific confidence"""
        base_confidence = 0.5

        # Boost for German business context
        german_context_score = self._calculate_german_context_score(context)
        base_confidence += german_context_score * 0.3

        # Boost for proper German capitalization
        if self._has_proper_german_capitalization(brand):
            base_confidence += 0.1

        # Boost for company suffix presence
        if self._has_german_company_suffix(brand):
            base_confidence += 0.15

        # Boost for compound word structure
        if self._is_german_compound_structure(brand):
            base_confidence += 0.1

        # Penalty for non-German characteristics
        if self._has_non_german_patterns(brand):
            base_confidence -= 0.2

        return min(1.0, max(0.0, base_confidence))

    def _calculate_german_context_score(self, context: str) -> float:
        """Calculate how German the context appears"""
        if not context:
            return 0.0

        context_lower = context.lower()
        german_indicator_count = 0

        # Check for German business keywords
        for keyword in self.business_context_indicators:
            if keyword in context_lower:
                german_indicator_count += 1

        # Check for German articles and prepositions
        german_words = [
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "und",
            "oder",
            "mit",
            "für",
            "von",
        ]
        for word in german_words:
            if f" {word} " in context_lower:
                german_indicator_count += 0.5

        # Normalize score
        return min(1.0, german_indicator_count / 5)

    def _has_proper_german_capitalization(self, brand: str) -> bool:
        """Check if brand follows German capitalization rules"""
        if not brand:
            return False

        # German nouns are capitalized
        words = brand.split()
        if len(words) == 1:
            return words[0][0].isupper() if words[0] else False

        # For multi-word brands, major words should be capitalized
        capitalized_count = sum(1 for word in words if word and word[0].isupper())
        return capitalized_count / len(words) >= 0.5

    def _has_german_company_suffix(self, brand: str) -> bool:
        """Check if brand has German company suffix"""
        brand_lower = brand.lower()
        return any(suffix.lower() in brand_lower for suffix in self.company_suffixes)

    def _is_german_compound_structure(self, brand: str) -> bool:
        """Check if brand has German compound word structure"""
        return any(re.search(pattern, brand) for pattern in self.compound_patterns)

    def _has_non_german_patterns(self, brand: str) -> bool:
        """Check for patterns that are unlikely in German brands"""
        non_german_patterns = [
            r"\b(Inc|Corp|LLC|Ltd)\b",  # English suffixes
            r"^[a-z]+$",  # All lowercase (unusual for German)
            r"[xqy]{2,}",  # Letter combinations rare in German
        ]

        return any(
            re.search(pattern, brand, re.IGNORECASE) for pattern in non_german_patterns
        )

    def extract_business_context(self, text: str) -> Dict[str, any]:
        """Extract German business context"""
        context = {
            "industry_indicators": [],
            "business_relationships": [],
            "company_actions": [],
            "market_position": [],
            "german_specific": {
                "legal_forms": [],
                "compound_words": [],
                "industry_terms": [],
            },
        }

        text_lower = text.lower()

        # Extract industry indicators
        for industry, terms in self.industry_patterns.items():
            found_terms = [term for term in terms if term in text_lower]
            if found_terms:
                context["industry_indicators"].append(
                    {"industry": industry, "terms": found_terms}
                )

        # Extract legal forms
        legal_forms = [
            suffix for suffix in self.company_suffixes if suffix.lower() in text_lower
        ]
        context["german_specific"]["legal_forms"] = legal_forms

        # Extract compound words
        compound_matches = []
        for pattern in self.compound_patterns:
            matches = re.finditer(pattern, text)
            compound_matches.extend([match.group() for match in matches])
        context["german_specific"]["compound_words"] = list(set(compound_matches))

        return context

    def _market_specific_preprocessing(self, text: str) -> str:
        """German-specific text preprocessing"""
        if not text:
            return ""

        # Normalize quotation marks (German uses different quotes)
        text = re.sub(r'[„""]', '"', text)
        text = re.sub(r"[‚" "]", "'", text)

        # Handle German-specific punctuation
        text = re.sub(
            r"(?<=[a-zäöüß])\.(?=[A-ZÄÖÜ])", ". ", text
        )  # Add space after period

        # Normalize umlauts for consistent processing
        # (Keep originals but also create normalized versions)
        return text

    def _market_specific_validation(
        self, brand: str, context: str, detection_method: DetectionMethod
    ) -> bool:
        """
        German-specific validation kept permissive to maximize recall; rely on fuzzy/semantic thresholds to control precision.
        """
        return True


# Register the adapter
MarketAdapterFactory.register_adapter("DE", GermanMarketAdapter)
