"""
Tests for trigger system functionality.

Tests all trigger types, factory patterns, and scheduling logic
for the comprehensive scheduling system.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.services.scheduling.triggers.base import (
    BaseTrigger,
    TriggerValidationError,
)
from app.services.scheduling.triggers.cron_trigger import CronTrigger
from app.services.scheduling.triggers.date_trigger import DateTrigger
from app.services.scheduling.triggers.dependency_trigger import (
    DependencyTrigger,
    DependencyType,
)
from app.services.scheduling.triggers.factory import TriggerFactory, get_trigger_factory
from app.services.scheduling.triggers.interval_trigger import IntervalTrigger


class TestBaseTrigger:
    """Test suite for BaseTrigger abstract class functionality"""

    def test_base_trigger_initialization(self):
        """Test base trigger initialization"""
        config = {"trigger_type": "test", "timezone": "UTC", "misfire_grace_time": 300}

        # Create a concrete implementation for testing
        class TestTrigger(BaseTrigger):
            def validate_config(self):
                pass

            async def get_next_run_time(self, previous_run_time):
                return datetime.now(timezone.utc)

        trigger = TestTrigger(config)

        assert trigger.config == config
        assert trigger.timezone.zone == "UTC"
        assert trigger.misfire_grace_time == 300

    def test_get_config_value_with_default(self):
        """Test getting config values with defaults"""

        class TestTrigger(BaseTrigger):
            def validate_config(self):
                pass

            async def get_next_run_time(self, previous_run_time):
                return None

        trigger = TestTrigger({"existing_key": "value"})

        assert trigger.get_config_value("existing_key") == "value"
        assert trigger.get_config_value("missing_key", default="default") == "default"

    def test_get_config_value_required_missing(self):
        """Test getting required config value that's missing"""

        class TestTrigger(BaseTrigger):
            def validate_config(self):
                pass

            async def get_next_run_time(self, previous_run_time):
                return None

        trigger = TestTrigger({})

        with pytest.raises(TriggerValidationError, match="Required configuration"):
            trigger.get_config_value("missing_required", required=True)

    def test_validate_datetime_config_valid(self):
        """Test datetime config validation with valid input"""

        class TestTrigger(BaseTrigger):
            def validate_config(self):
                pass

            async def get_next_run_time(self, previous_run_time):
                return None

        config = {
            "iso_datetime": "2024-12-25T09:00:00Z",
            "datetime_obj": datetime.now(timezone.utc),
        }

        trigger = TestTrigger(config)

        result1 = trigger.validate_datetime_config("iso_datetime")
        assert isinstance(result1, datetime)
        assert result1.tzinfo is not None

        result2 = trigger.validate_datetime_config("datetime_obj")
        assert isinstance(result2, datetime)

    def test_validate_datetime_config_invalid(self):
        """Test datetime config validation with invalid input"""

        class TestTrigger(BaseTrigger):
            def validate_config(self):
                pass

            async def get_next_run_time(self, previous_run_time):
                return None

        trigger = TestTrigger({"invalid_datetime": "not-a-date"})

        with pytest.raises(TriggerValidationError):
            trigger.validate_datetime_config("invalid_datetime")


class TestCronTrigger:
    """Test suite for CronTrigger"""

    def test_cron_trigger_initialization_valid(self):
        """Test cron trigger with valid expression"""
        config = {"trigger_type": "cron", "expression": "0 9 * * *"}  # Daily at 9 AM

        trigger = CronTrigger(config)

        assert trigger.cron_expression == "0 9 * * *"
        assert trigger.croniter_obj is not None

    def test_cron_trigger_invalid_expression(self):
        """Test cron trigger with invalid expression"""
        config = {"trigger_type": "cron", "expression": "invalid cron"}

        with pytest.raises(TriggerValidationError, match="Invalid cron expression"):
            CronTrigger(config)

    def test_cron_trigger_missing_expression(self):
        """Test cron trigger without expression"""
        config = {"trigger_type": "cron"}

        with pytest.raises(TriggerValidationError, match="Required configuration"):
            CronTrigger(config)

    @pytest.mark.asyncio
    async def test_cron_get_next_run_time_first_run(self):
        """Test cron next run time calculation for first run"""
        config = {"trigger_type": "cron", "expression": "0 9 * * *"}  # Daily at 9 AM

        trigger = CronTrigger(config)

        # Mock current time to ensure predictable result
        with patch("app.services.scheduling.triggers.cron_trigger.datetime") as mock_dt:
            mock_now = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)  # 8 AM
            mock_dt.now.return_value = mock_now

            next_run = await trigger.get_next_run_time(None)

            assert next_run is not None
            assert next_run.hour == 9  # Should be 9 AM today
            assert next_run.minute == 0

    @pytest.mark.asyncio
    async def test_cron_get_next_run_time_subsequent_run(self):
        """Test cron next run time calculation for subsequent runs"""
        config = {"trigger_type": "cron", "expression": "0 9 * * *"}  # Daily at 9 AM

        trigger = CronTrigger(config)

        # Previous run was yesterday at 9 AM
        previous_run = datetime(2024, 1, 14, 9, 0, tzinfo=timezone.utc)

        with patch("app.services.scheduling.triggers.cron_trigger.datetime") as mock_dt:
            mock_now = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)  # 10 AM today
            mock_dt.now.return_value = mock_now

            next_run = await trigger.get_next_run_time(previous_run)

            assert next_run is not None
            # Should be tomorrow at 9 AM since today's run was missed
            assert next_run.day == 16
            assert next_run.hour == 9

    def test_cron_get_trigger_info(self):
        """Test cron trigger info generation"""
        config = {"trigger_type": "cron", "expression": "0 9 * * *"}  # Daily at 9 AM

        trigger = CronTrigger(config)
        info = trigger.get_trigger_info()

        assert info["type"] == "cron"
        assert info["expression"] == "0 9 * * *"
        assert "description" in info
        assert "next_runs" in info

    def test_cron_should_skip_run_within_grace(self):
        """Test cron should skip run logic within grace time"""
        config = {
            "trigger_type": "cron",
            "expression": "0 9 * * *",
            "misfire_grace_time": 300,  # 5 minutes
        }

        trigger = CronTrigger(config)

        scheduled_time = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
        current_time = datetime(
            2024, 1, 15, 9, 3, tzinfo=timezone.utc
        )  # 3 minutes late

        should_skip = trigger.should_skip_run(scheduled_time, current_time)
        assert should_skip is False

    def test_cron_should_skip_run_beyond_grace(self):
        """Test cron should skip run logic beyond grace time"""
        config = {
            "trigger_type": "cron",
            "expression": "0 9 * * *",
            "misfire_grace_time": 300,  # 5 minutes
        }

        trigger = CronTrigger(config)

        scheduled_time = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
        current_time = datetime(
            2024, 1, 15, 9, 10, tzinfo=timezone.utc
        )  # 10 minutes late

        should_skip = trigger.should_skip_run(scheduled_time, current_time)
        assert should_skip is True


class TestIntervalTrigger:
    """Test suite for IntervalTrigger"""

    def test_interval_trigger_initialization_valid(self):
        """Test interval trigger with valid configuration"""
        config = {"trigger_type": "interval", "minutes": 30, "seconds": 15}

        trigger = IntervalTrigger(config)

        assert trigger.minutes == 30
        assert trigger.seconds == 15
        assert trigger.total_seconds == 30 * 60 + 15  # 1815 seconds

    def test_interval_trigger_no_time_units(self):
        """Test interval trigger without any time units"""
        config = {"trigger_type": "interval"}

        with pytest.raises(TriggerValidationError, match="At least one time unit"):
            IntervalTrigger(config)

    def test_interval_trigger_negative_values(self):
        """Test interval trigger with negative time values"""
        config = {"trigger_type": "interval", "minutes": -5}

        with pytest.raises(TriggerValidationError, match="must be non-negative"):
            IntervalTrigger(config)

    def test_interval_trigger_minimum_interval(self):
        """Test interval trigger with interval below minimum"""
        config = {
            "trigger_type": "interval",
            "seconds": 0.5,  # Less than 1 second
            "min_interval": 1,
        }

        with pytest.raises(TriggerValidationError, match="must be at least"):
            IntervalTrigger(config)

    @pytest.mark.asyncio
    async def test_interval_get_next_run_time_first_run(self):
        """Test interval next run time for first execution"""
        config = {"trigger_type": "interval", "minutes": 30}

        trigger = IntervalTrigger(config)

        with patch(
            "app.services.scheduling.triggers.interval_trigger.datetime"
        ) as mock_dt:
            mock_now = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now

            next_run = await trigger.get_next_run_time(None)

            assert next_run is not None
            assert next_run >= mock_now  # Should be now or in the future

    @pytest.mark.asyncio
    async def test_interval_get_next_run_time_subsequent(self):
        """Test interval next run time for subsequent executions"""
        config = {"trigger_type": "interval", "minutes": 30}

        trigger = IntervalTrigger(config)

        previous_run = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
        expected_next = previous_run + timedelta(minutes=30)

        next_run = await trigger.get_next_run_time(previous_run)

        assert next_run == expected_next

    @pytest.mark.asyncio
    async def test_interval_with_end_date_passed(self):
        """Test interval trigger with end date in the past"""
        config = {
            "trigger_type": "interval",
            "minutes": 30,
            "end_date": "2024-01-01T00:00:00Z",  # Past date
        }

        trigger = IntervalTrigger(config)

        with patch(
            "app.services.scheduling.triggers.interval_trigger.datetime"
        ) as mock_dt:
            mock_now = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now

            next_run = await trigger.get_next_run_time(None)

            assert next_run is None

    def test_interval_get_trigger_info(self):
        """Test interval trigger info generation"""
        config = {"trigger_type": "interval", "hours": 2, "minutes": 30}

        trigger = IntervalTrigger(config)
        info = trigger.get_trigger_info()

        assert info["type"] == "interval"
        assert info["interval_seconds"] == 2.5 * 3600  # 2.5 hours
        assert "Every 2 hours and 30 minutes" in info["description"]

    def test_interval_get_description_short(self):
        """Test interval description for short intervals"""
        config = {"trigger_type": "interval", "seconds": 45}

        trigger = IntervalTrigger(config)
        description = trigger.get_interval_description()

        assert description == "45s"


class TestDateTrigger:
    """Test suite for DateTrigger"""

    def test_date_trigger_initialization_valid(self):
        """Test date trigger with valid run date"""
        future_date = datetime.now(timezone.utc) + timedelta(hours=1)
        config = {"trigger_type": "date", "run_date": future_date.isoformat()}

        trigger = DateTrigger(config)

        assert trigger.run_date == future_date
        assert trigger.has_executed is False

    def test_date_trigger_missing_run_date(self):
        """Test date trigger without run date"""
        config = {"trigger_type": "date"}

        with pytest.raises(TriggerValidationError, match="Required configuration"):
            DateTrigger(config)

    def test_date_trigger_past_date_warning(self):
        """Test date trigger with past date (should log warning but not fail)"""
        past_date = datetime.now(timezone.utc) - timedelta(hours=1)
        config = {"trigger_type": "date", "run_date": past_date.isoformat()}

        with patch(
            "app.services.scheduling.triggers.date_trigger.logger"
        ) as mock_logger:
            trigger = DateTrigger(config)

            assert trigger.run_date == past_date
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_date_get_next_run_time_not_executed(self):
        """Test date trigger next run time when not yet executed"""
        future_date = datetime.now(timezone.utc) + timedelta(hours=1)
        config = {"trigger_type": "date", "run_date": future_date.isoformat()}

        trigger = DateTrigger(config)

        next_run = await trigger.get_next_run_time(None)

        assert next_run == future_date

    @pytest.mark.asyncio
    async def test_date_get_next_run_time_already_executed(self):
        """Test date trigger next run time when already executed"""
        future_date = datetime.now(timezone.utc) + timedelta(hours=1)
        config = {"trigger_type": "date", "run_date": future_date.isoformat()}

        trigger = DateTrigger(config)

        # Simulate previous execution
        previous_run = datetime.now(timezone.utc) - timedelta(minutes=30)

        next_run = await trigger.get_next_run_time(previous_run)

        assert next_run is None

    @pytest.mark.asyncio
    async def test_date_get_next_run_time_past_within_grace(self):
        """Test date trigger with run date slightly in the past"""
        config = {
            "trigger_type": "date",
            "run_date": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
            "past_date_grace_seconds": 300,  # 5 minutes grace
        }

        trigger = DateTrigger(config)

        with patch("app.services.scheduling.triggers.date_trigger.datetime") as mock_dt:
            mock_now = datetime.now(timezone.utc)
            mock_dt.now.return_value = mock_now

            next_run = await trigger.get_next_run_time(None)

            assert next_run == mock_now  # Should schedule for immediate execution

    @pytest.mark.asyncio
    async def test_date_get_next_run_time_past_beyond_grace(self):
        """Test date trigger with run date far in the past"""
        config = {
            "trigger_type": "date",
            "run_date": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "past_date_grace_seconds": 300,  # 5 minutes grace
        }

        trigger = DateTrigger(config)

        next_run = await trigger.get_next_run_time(None)

        assert next_run is None  # Should skip execution

    def test_date_get_trigger_info_scheduled(self):
        """Test date trigger info when scheduled"""
        future_date = datetime.now(timezone.utc) + timedelta(hours=2)
        config = {"trigger_type": "date", "run_date": future_date.isoformat()}

        trigger = DateTrigger(config)
        info = trigger.get_trigger_info()

        assert info["type"] == "date"
        assert info["status"] == "scheduled"
        assert info["time_until"] is not None
        assert "in 2 hour" in info["description"] or "in 1 hour" in info["description"]

    def test_date_get_trigger_info_completed(self):
        """Test date trigger info when completed"""
        past_date = datetime.now(timezone.utc) - timedelta(hours=1)
        config = {
            "trigger_type": "date",
            "run_date": past_date.isoformat(),
            "has_executed": True,
        }

        trigger = DateTrigger(config)
        info = trigger.get_trigger_info()

        assert info["type"] == "date"
        assert info["status"] == "completed"
        assert info["has_executed"] is True

    def test_date_mark_executed(self):
        """Test marking date trigger as executed"""
        future_date = datetime.now(timezone.utc) + timedelta(hours=1)
        config = {"trigger_type": "date", "run_date": future_date.isoformat()}

        trigger = DateTrigger(config)
        assert trigger.has_executed is False

        trigger.mark_executed()

        assert trigger.has_executed is True
        assert trigger.config["has_executed"] is True


class TestDependencyTrigger:
    """Test suite for DependencyTrigger"""

    def test_dependency_trigger_initialization_valid(self):
        """Test dependency trigger with valid configuration"""
        config = {
            "trigger_type": "dependency",
            "depends_on": ["job-1", "job-2"],
            "dependency_type": "success",
        }

        trigger = DependencyTrigger(config)

        assert trigger.depends_on_jobs == ["job-1", "job-2"]
        assert trigger.dependency_type == DependencyType.SUCCESS
        assert trigger.require_all is True

    def test_dependency_trigger_missing_depends_on(self):
        """Test dependency trigger without depends_on"""
        config = {"trigger_type": "dependency"}

        with pytest.raises(TriggerValidationError, match="Required configuration"):
            DependencyTrigger(config)

    def test_dependency_trigger_empty_depends_on(self):
        """Test dependency trigger with empty depends_on list"""
        config = {"trigger_type": "dependency", "depends_on": []}

        with pytest.raises(TriggerValidationError, match="cannot be empty"):
            DependencyTrigger(config)

    def test_dependency_trigger_invalid_dependency_type(self):
        """Test dependency trigger with invalid dependency type"""
        config = {
            "trigger_type": "dependency",
            "depends_on": ["job-1"],
            "dependency_type": "invalid_type",
        }

        with pytest.raises(
            TriggerValidationError, match="dependency_type must be one of"
        ):
            DependencyTrigger(config)

    @pytest.mark.asyncio
    async def test_dependency_get_next_run_time_already_executed(self):
        """Test dependency trigger when already executed"""
        config = {"trigger_type": "dependency", "depends_on": ["job-1"]}

        trigger = DependencyTrigger(config)

        # Simulate previous execution
        previous_run = datetime.now(timezone.utc) - timedelta(hours=1)

        next_run = await trigger.get_next_run_time(previous_run)

        assert next_run is None

    @pytest.mark.asyncio
    async def test_dependency_get_next_run_time_dependencies_not_satisfied(self):
        """Test dependency trigger when dependencies not satisfied"""
        config = {"trigger_type": "dependency", "depends_on": ["job-1"]}

        trigger = DependencyTrigger(config)

        # Mock dependency checking to return False
        with patch.object(trigger, "_check_dependencies", return_value=False):
            with patch(
                "app.services.scheduling.triggers.dependency_trigger.datetime"
            ) as mock_dt:
                mock_now = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
                mock_dt.now.return_value = mock_now

                next_run = await trigger.get_next_run_time(None)

                # Should return future time for next check
                assert next_run is not None
                assert next_run > mock_now

    @pytest.mark.asyncio
    async def test_dependency_get_next_run_time_dependencies_satisfied(self):
        """Test dependency trigger when dependencies satisfied"""
        config = {
            "trigger_type": "dependency",
            "depends_on": ["job-1"],
            "delay_seconds": 60,
        }

        trigger = DependencyTrigger(config)

        completion_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        # Mock dependency checking to return True
        with patch.object(trigger, "_check_dependencies", return_value=True):
            with patch.object(
                trigger,
                "_get_earliest_dependency_completion_time",
                return_value=completion_time,
            ):
                next_run = await trigger.get_next_run_time(None)

                # Should return completion time + delay
                expected_time = completion_time + timedelta(seconds=60)
                assert next_run == expected_time

    @pytest.mark.asyncio
    async def test_dependency_check_dependencies_all_required(self):
        """Test dependency checking with require_all=True"""
        config = {
            "trigger_type": "dependency",
            "depends_on": ["job-1", "job-2"],
            "require_all": True,
        }

        trigger = DependencyTrigger(config)

        # Mock that only one dependency is satisfied
        trigger._dependency_status = {"job-1": True, "job-2": False}

        result = await trigger._check_dependencies()

        assert result is False  # Not all dependencies satisfied

    @pytest.mark.asyncio
    async def test_dependency_check_dependencies_any_required(self):
        """Test dependency checking with require_all=False"""
        config = {
            "trigger_type": "dependency",
            "depends_on": ["job-1", "job-2"],
            "require_all": False,
        }

        trigger = DependencyTrigger(config)

        # Mock that only one dependency is satisfied
        trigger._dependency_status = {"job-1": True, "job-2": False}

        result = await trigger._check_dependencies()

        assert result is True  # At least one dependency satisfied

    @pytest.mark.asyncio
    async def test_dependency_completion_time_selection(self):
        """Ensure earliest completion time is respected."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        config = {
            "trigger_type": "dependency",
            "depends_on": ["job-1", "job-2"],
            "dependency_type": "success",
        }

        trigger = DependencyTrigger(config)
        trigger.update_dependency_status("job-1", True, completed_at=base_time)
        trigger.update_dependency_status(
            "job-2", True, completed_at=base_time + timedelta(minutes=10)
        )

        earliest = await trigger._get_earliest_dependency_completion_time()

        assert earliest == trigger.normalize_datetime(base_time)

    @pytest.mark.asyncio
    async def test_dependency_completion_time_none_when_unsatisfied(self):
        """Earliest completion returns None when no dependencies satisfied."""
        config = {
            "trigger_type": "dependency",
            "depends_on": ["job-1", "job-2"],
        }

        trigger = DependencyTrigger(config)
        trigger.update_dependency_status("job-1", False)
        trigger.update_dependency_status("job-2", None)

        assert await trigger._get_earliest_dependency_completion_time() is None

    def test_dependency_get_trigger_info(self):
        """Test dependency trigger info generation"""
        config = {
            "trigger_type": "dependency",
            "depends_on": ["job-1", "job-2"],
            "dependency_type": "success",
            "delay_seconds": 300,
        }

        trigger = DependencyTrigger(config)
        info = trigger.get_trigger_info()

        assert info["type"] == "dependency"
        assert info["depends_on"] == ["job-1", "job-2"]
        assert info["dependency_type"] == "success"
        assert "after 5 minutes delay" in info["description"]

    def test_dependency_add_dependency(self):
        """Test adding new dependency"""
        config = {"trigger_type": "dependency", "depends_on": ["job-1"]}

        trigger = DependencyTrigger(config)

        trigger.add_dependency("job-2")

        assert "job-2" in trigger.depends_on_jobs
        assert trigger.config["depends_on"] == ["job-1", "job-2"]

    def test_dependency_remove_dependency(self):
        """Test removing dependency"""
        config = {"trigger_type": "dependency", "depends_on": ["job-1", "job-2"]}

        trigger = DependencyTrigger(config)

        trigger.remove_dependency("job-1")

        assert "job-1" not in trigger.depends_on_jobs
        assert trigger.config["depends_on"] == ["job-2"]


class TestTriggerFactory:
    """Test suite for TriggerFactory"""

    def test_trigger_factory_initialization(self):
        """Test trigger factory initialization"""
        factory = TriggerFactory()

        assert len(factory.TRIGGER_REGISTRY) > 0
        assert factory._custom_triggers == {}

    def test_create_trigger_cron(self):
        """Test creating cron trigger via factory"""
        factory = TriggerFactory()

        config = {"trigger_type": "cron", "expression": "0 9 * * *"}

        trigger = factory.create_trigger(config)

        assert isinstance(trigger, CronTrigger)
        assert trigger.cron_expression == "0 9 * * *"

    def test_create_trigger_interval(self):
        """Test creating interval trigger via factory"""
        factory = TriggerFactory()

        config = {"trigger_type": "interval", "minutes": 30}

        trigger = factory.create_trigger(config)

        assert isinstance(trigger, IntervalTrigger)
        assert trigger.minutes == 30

    def test_create_trigger_date(self):
        """Test creating date trigger via factory"""
        factory = TriggerFactory()

        future_date = datetime.now(timezone.utc) + timedelta(hours=1)
        config = {"trigger_type": "date", "run_date": future_date.isoformat()}

        trigger = factory.create_trigger(config)

        assert isinstance(trigger, DateTrigger)
        assert trigger.run_date == future_date

    def test_create_trigger_dependency(self):
        """Test creating dependency trigger via factory"""
        factory = TriggerFactory()

        config = {"trigger_type": "dependency", "depends_on": ["job-1"]}

        trigger = factory.create_trigger(config)

        assert isinstance(trigger, DependencyTrigger)
        assert trigger.depends_on_jobs == ["job-1"]

    def test_create_trigger_invalid_type(self):
        """Test creating trigger with invalid type"""
        factory = TriggerFactory()

        config = {"trigger_type": "invalid_type"}

        with pytest.raises(TriggerValidationError, match="Invalid trigger type"):
            factory.create_trigger(config)

    def test_create_trigger_manual_type(self):
        """Test creating manual trigger (should fail)"""
        factory = TriggerFactory()

        config = {"trigger_type": "manual"}

        with pytest.raises(
            TriggerValidationError, match="Manual triggers do not use trigger instances"
        ):
            factory.create_trigger(config)

    def test_create_trigger_missing_type(self):
        """Test creating trigger without trigger_type"""
        factory = TriggerFactory()

        config = {}

        with pytest.raises(TriggerValidationError, match="Missing 'trigger_type'"):
            factory.create_trigger(config)

    def test_register_custom_trigger(self):
        """Test registering custom trigger type"""
        factory = TriggerFactory()

        class CustomTrigger(BaseTrigger):
            def validate_config(self):
                pass

            async def get_next_run_time(self, previous_run_time):
                return None

        factory.register_custom_trigger("custom", CustomTrigger)

        assert "custom" in factory._custom_triggers
        assert factory._custom_triggers["custom"] == CustomTrigger

    def test_register_custom_trigger_invalid_class(self):
        """Test registering invalid custom trigger class"""
        factory = TriggerFactory()

        class InvalidTrigger:
            pass

        with pytest.raises(TriggerValidationError, match="must extend BaseTrigger"):
            factory.register_custom_trigger("invalid", InvalidTrigger)

    def test_unregister_custom_trigger(self):
        """Test unregistering custom trigger type"""
        factory = TriggerFactory()

        class CustomTrigger(BaseTrigger):
            def validate_config(self):
                pass

            async def get_next_run_time(self, previous_run_time):
                return None

        factory.register_custom_trigger("custom", CustomTrigger)
        assert "custom" in factory._custom_triggers

        result = factory.unregister_custom_trigger("custom")

        assert result is True
        assert "custom" not in factory._custom_triggers

    def test_unregister_nonexistent_trigger(self):
        """Test unregistering non-existent trigger type"""
        factory = TriggerFactory()

        result = factory.unregister_custom_trigger("nonexistent")

        assert result is False

    def test_get_supported_trigger_types(self):
        """Test getting supported trigger types"""
        factory = TriggerFactory()

        supported = factory.get_supported_trigger_types()

        assert "CRON" in supported
        assert "INTERVAL" in supported
        assert "DATE" in supported
        assert "DEPENDENCY" in supported

        for trigger_type, info in supported.items():
            assert "type" in info
            assert "class" in info
            assert "description" in info

    def test_validate_trigger_config_valid(self):
        """Test trigger config validation with valid config"""
        factory = TriggerFactory()

        config = {"trigger_type": "cron", "expression": "0 9 * * *"}

        result = factory.validate_trigger_config(config)

        assert result["valid"] is True
        assert result["trigger_type"] == "cron"
        assert "info" in result

    def test_validate_trigger_config_invalid(self):
        """Test trigger config validation with invalid config"""
        factory = TriggerFactory()

        config = {"trigger_type": "cron", "expression": "invalid cron"}

        result = factory.validate_trigger_config(config)

        assert result["valid"] is False
        assert "error" in result

    def test_create_trigger_from_job_config(self):
        """Test creating trigger from job config"""
        factory = TriggerFactory()

        job_config = {
            "name": "test_job",
            "trigger_config": {"trigger_type": "interval", "minutes": 15},
        }

        trigger = factory.create_trigger_from_job_config(job_config)

        assert isinstance(trigger, IntervalTrigger)
        assert trigger.minutes == 15

    def test_create_trigger_from_job_config_missing(self):
        """Test creating trigger from job config without trigger_config"""
        factory = TriggerFactory()

        job_config = {"name": "test_job"}

        with pytest.raises(TriggerValidationError, match="missing 'trigger_config'"):
            factory.create_trigger_from_job_config(job_config)

    def test_get_trigger_factory_singleton(self):
        """Test get_trigger_factory returns same instance"""
        factory1 = get_trigger_factory()
        factory2 = get_trigger_factory()

        assert factory1 is factory2
