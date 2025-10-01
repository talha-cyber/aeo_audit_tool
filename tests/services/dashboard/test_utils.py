import math

from app.services.dashboard.utils import cron_to_label, safe_ratio


def test_cron_to_label_presets():
    assert cron_to_label("0 9 1 * *") == "Monthly on day 1 at 09:00"
    assert cron_to_label("0 6 * * 1") == "Weekly on Monday at 06:00"


def test_cron_to_label_custom():
    assert cron_to_label("*") == "*"


def test_safe_ratio_handles_zero():
    assert safe_ratio(1, 0) is None


def test_safe_ratio_regular():
    assert math.isclose(safe_ratio(2, 4), 0.5)
