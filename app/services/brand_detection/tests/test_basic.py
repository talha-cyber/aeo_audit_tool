from unittest.mock import patch

import pytest

from ..core.detector import BrandDetectionEngine, DetectionConfig
from ..market_adapters.german_adapter import GermanMarketAdapter
from ..models.brand_mention import BrandMention, DetectionMethod


class TestBrandDetectionBasic:
    """Basic tests for brand detection engine"""

    @pytest.fixture
    def detection_config(self):
        """Test configuration"""
        return DetectionConfig(
            confidence_threshold=0.7,
            similarity_threshold=0.8,
            market_code="DE",
            language_code="de",
            enable_caching=False,  # Disable for testing
        )

    @pytest.fixture
    def sample_german_text(self):
        """Sample German text for testing"""
        return """
        Salesforce ist eine führende CRM-Software, die von vielen Unternehmen verwendet wird.
        HubSpot bietet auch gute Marketing-Automation-Funktionen.
        Microsoft Dynamics ist eine weitere Alternative.
        """

    @pytest.fixture
    def target_brands(self):
        """Target brands for testing"""
        return ["Salesforce", "HubSpot", "Microsoft Dynamics"]

    def test_detection_config_creation(self, detection_config):
        """Test detection configuration creation"""
        assert detection_config.confidence_threshold == 0.7
        assert detection_config.market_code == "DE"
        assert detection_config.language_code == "de"

    def test_german_adapter_creation(self):
        """Test German market adapter creation"""
        adapter = GermanMarketAdapter()
        assert adapter.market_code == "DE"
        assert adapter.language_code == "de"
        assert "GmbH" in adapter.company_suffixes
        assert "unternehmen" in adapter.business_keywords

    def test_brand_normalization(self):
        """Test brand name normalization"""
        adapter = GermanMarketAdapter()
        variations = adapter.normalize_brand_name("SAP AG")

        assert "SAP AG" in variations
        assert "sap ag" in variations
        assert "SAP" in variations
        assert len(variations) > 3

    def test_german_preprocessing(self):
        """Test German text preprocessing"""
        adapter = GermanMarketAdapter()
        text = "Salesforce  ist   eine    gute  Software."
        processed = adapter.preprocess_text(text)

        # Should normalize whitespace
        assert "  " not in processed
        assert processed.strip() == processed

    def test_brand_mention_creation(self):
        """Test brand mention object creation"""
        mention = BrandMention(
            brand="Salesforce",
            original_text="Salesforce",
            confidence=0.9,
            detection_method=DetectionMethod.FUZZY,
            language="de",
        )

        assert mention.brand == "Salesforce"
        assert mention.confidence == 0.9
        assert mention.detection_method == DetectionMethod.FUZZY
        assert mention.mention_count == 0  # No contexts added yet

    def test_relevance_score_calculation(self):
        """Test relevance score calculation"""
        mention = BrandMention(
            brand="Salesforce",
            original_text="Salesforce",
            confidence=0.8,
            detection_method=DetectionMethod.FUZZY,
            language="de",
        )

        # Base relevance should be close to confidence
        relevance = mention.calculate_relevance_score()
        assert 0.7 <= relevance <= 1.0

    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        with patch("openai.AsyncOpenAI"):
            engine = BrandDetectionEngine("fake_api_key")

            # Empty text
            result = await engine.detect_brands("", ["Salesforce"])
            assert result.total_brands_found == 0

            # Empty brands list
            result = await engine.detect_brands("Some text", [])
            assert result.total_brands_found == 0

    def test_cache_key_generation(self):
        """Test cache key generation"""
        with patch("openai.AsyncOpenAI"):
            engine = BrandDetectionEngine("fake_api_key")
            config = DetectionConfig()

            key1 = engine._generate_cache_key("test text", ["brand1"], config)
            key2 = engine._generate_cache_key("test text", ["brand1"], config)
            key3 = engine._generate_cache_key("different text", ["brand1"], config)

            # Same inputs should generate same key
            assert key1 == key2
            # Different inputs should generate different keys
            assert key1 != key3

    def test_brand_candidate_extraction(self):
        """Test brand candidate extraction"""
        with patch("openai.AsyncOpenAI"):
            engine = BrandDetectionEngine("fake_api_key")
            adapter = GermanMarketAdapter()

            text = "Salesforce und Microsoft sind beide große Softwareunternehmen."
            candidates = engine._extract_brand_candidates(text, adapter)

            # Should find capitalized words
            candidate_texts = [c["text"] for c in candidates]
            assert "Salesforce" in candidate_texts
            assert "Microsoft" in candidate_texts

    def test_sentence_extraction(self):
        """Test sentence extraction around mentions"""
        with patch("openai.AsyncOpenAI"):
            engine = BrandDetectionEngine("fake_api_key")

            text = "This is first sentence. Salesforce is mentioned here. This is third sentence."
            start = text.find("Salesforce")
            end = start + len("Salesforce")

            sentence = engine._extract_sentence(text, start, end)
            assert "Salesforce is mentioned here" in sentence

    def test_statistics_collection(self):
        """Test statistics collection"""
        with patch("openai.AsyncOpenAI"):
            engine = BrandDetectionEngine("fake_api_key")

            stats = engine.get_detection_statistics()

            assert "detection_stats" in stats
            assert "success_rate_percent" in stats
            assert isinstance(stats["detection_stats"]["total_detections"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
