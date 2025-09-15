from typing import Dict, List


class IndustryKnowledge:
    """Minimal static knowledge to enrich seeds and templates.

    Intentionally small and easily extendable without external calls.
    """

    _integrations: Dict[str, List[str]] = {
        # key by product_type for pragmatic matching
        "CRM": [
            "Salesforce",
            "HubSpot",
            "Zapier",
            "Slack",
            "Outlook",
            "Google Workspace",
            "QuickBooks",
        ],
        "SaaS": ["Okta", "Azure AD", "Snowflake", "Slack", "Jira"],
    }

    _compliance: Dict[str, List[str]] = {
        "SaaS": ["SOC 2", "ISO 27001", "GDPR"],
        "CRM": ["SOC 2", "ISO 27001", "GDPR"],
        "Healthcare": ["HIPAA", "SOC 2", "GDPR"],
        "Finance": ["PCI DSS", "SOC 2", "GDPR"],
    }

    @classmethod
    def integrations(cls, product_type: str) -> List[str]:
        return cls._integrations.get(product_type, [])

    @classmethod
    def compliance(cls, industry: str) -> List[str]:
        return cls._compliance.get(industry, cls._compliance.get("SaaS", []))
