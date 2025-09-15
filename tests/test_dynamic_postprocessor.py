from app.services.providers.dynamic_provider import PostProcessor


def test_postprocessor_classification_en():
    raw = "\n".join(
        [
            "BrandA vs BrandB: which is better?",  # comparison
            "What is the price of BrandA?",  # pricing
            "Does BrandA integrate with Salesforce?",  # integrations
            "Is BrandA SOC 2 compliant?",  # security_compliance
            "How long to implement BrandA?",  # implementation_migration
            "BrandA ROI compared to BrandB?",  # roi_tco
            "BrandA SLA and uptime?",  # support_reliability
            "Top features of BrandA?",  # features
            "BrandB reviews and ratings?",  # reviews
            "Is BrandA available in the EU?",  # geography
        ]
    )
    pp = PostProcessor()
    qs = pp.process(raw, 50, language="en")
    subcats = {q.metadata.get("sub_category") for q in qs}
    assert {
        "comparison",
        "pricing",
        "integrations",
        "security_compliance",
        "implementation_migration",
        "roi_tco",
        "support_reliability",
        "features",
        "reviews",
        "geography",
    }.issubset(subcats)


def test_postprocessor_classification_de():
    raw = "\n".join(
        [
            "BrandA vs. BrandB: Was ist besser?",  # comparison
            "Wie viel kostet BrandA?",  # pricing
            "Integriert sich BrandA mit Salesforce?",  # integrations
            "Erfüllt BrandA SOC 2?",  # security_compliance
            "Wie lange dauert die Implementierung von BrandA?",  # implementation
            "ROI von BrandA im Vergleich zu BrandB?",  # roi_tco
            "BrandA SLA und Verfügbarkeit?",  # support
            "Top-Funktionen von BrandA?",  # features
            "BrandB Bewertungen und Rezensionen?",  # reviews
            "Ist BrandA in der EU verfügbar?",  # geography
        ]
    )
    pp = PostProcessor()
    qs = pp.process(raw, 50, language="de")
    subcats = {q.metadata.get("sub_category") for q in qs}
    assert {
        "comparison",
        "pricing",
        "integrations",
        "security_compliance",
        "implementation_migration",
        "roi_tco",
        "support_reliability",
        "features",
        "reviews",
        "geography",
    }.issubset(subcats)
