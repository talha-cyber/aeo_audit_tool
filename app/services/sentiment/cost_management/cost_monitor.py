"""
Cost Monitoring and Resource Management System.

Tracks usage, estimates costs, and provides recommendations for cost-effective
sentiment analysis operations.
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import psutil

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UsageMetrics:
    """Usage metrics for a specific operation"""
    operation_type: str
    timestamp: float
    duration: float
    memory_used_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: float
    input_tokens: int
    output_tokens: int
    model_name: str
    provider_name: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class CostEstimate:
    """Cost estimate for operations"""
    compute_cost: float
    memory_cost: float
    storage_cost: float
    total_cost: float
    cost_per_request: float
    currency: str = "USD"


class CostCalculator:
    """Calculate costs for different types of operations"""

    def __init__(self):
        # Cost rates (rough estimates for local operations)
        self.rates = {
            "cpu_hour": 0.02,      # $0.02 per hour of CPU usage
            "gpu_hour": 0.10,      # $0.10 per hour of GPU usage
            "memory_gb_hour": 0.001, # $0.001 per GB-hour of memory
            "storage_gb_month": 0.02, # $0.02 per GB-month of storage
            "inference_request": 0.0001, # $0.0001 per inference request
            "training_hour": 0.05,  # $0.05 per hour of training
        }

    def calculate_inference_cost(
        self,
        duration_seconds: float,
        memory_mb: float,
        gpu_memory_mb: float,
        model_size: str = "small"
    ) -> CostEstimate:
        """Calculate cost for inference operations"""

        # Base compute cost
        if gpu_memory_mb > 0:
            compute_cost = (duration_seconds / 3600) * self.rates["gpu_hour"]
        else:
            compute_cost = (duration_seconds / 3600) * self.rates["cpu_hour"]

        # Memory cost
        memory_gb = memory_mb / 1024
        memory_cost = (duration_seconds / 3600) * memory_gb * self.rates["memory_gb_hour"]

        # Model size multiplier
        size_multipliers = {
            "ultra_small": 0.5,
            "small": 1.0,
            "medium": 2.0,
            "large": 4.0
        }
        multiplier = size_multipliers.get(model_size, 1.0)

        total_cost = (compute_cost + memory_cost) * multiplier

        return CostEstimate(
            compute_cost=compute_cost,
            memory_cost=memory_cost,
            storage_cost=0.0,
            total_cost=total_cost,
            cost_per_request=total_cost
        )

    def calculate_training_cost(
        self,
        duration_hours: float,
        memory_gb: float,
        num_samples: int,
        model_size: str = "small"
    ) -> CostEstimate:
        """Calculate cost for training operations"""

        # Training is more expensive than inference
        base_cost = duration_hours * self.rates["training_hour"]

        # Memory cost
        memory_cost = duration_hours * memory_gb * self.rates["memory_gb_hour"] * 2  # Training uses more memory

        # Sample complexity cost
        sample_cost = num_samples * 0.00001  # $0.00001 per sample

        # Model size multiplier
        size_multipliers = {
            "ultra_small": 0.3,
            "small": 1.0,
            "medium": 3.0,
            "large": 8.0
        }
        multiplier = size_multipliers.get(model_size, 1.0)

        total_cost = (base_cost + memory_cost + sample_cost) * multiplier

        return CostEstimate(
            compute_cost=base_cost,
            memory_cost=memory_cost,
            storage_cost=sample_cost,
            total_cost=total_cost,
            cost_per_request=total_cost / max(1, num_samples)
        )

    def calculate_storage_cost(
        self,
        storage_gb: float,
        duration_days: float = 30
    ) -> CostEstimate:
        """Calculate storage costs"""

        monthly_cost = storage_gb * self.rates["storage_gb_month"]
        daily_cost = monthly_cost / 30
        total_cost = daily_cost * duration_days

        return CostEstimate(
            compute_cost=0.0,
            memory_cost=0.0,
            storage_cost=total_cost,
            total_cost=total_cost,
            cost_per_request=0.0
        )


class ResourceTracker:
    """Track resource usage in real-time"""

    def __init__(self):
        self.tracking_active = False
        self.current_usage = {
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "gpu_memory_mb": 0.0,
            "disk_usage_gb": 0.0
        }
        self._tracking_thread = None

    def start_tracking(self):
        """Start resource tracking"""
        if self.tracking_active:
            return

        self.tracking_active = True
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True
        )
        self._tracking_thread.start()
        logger.info("Started resource tracking")

    def stop_tracking(self):
        """Stop resource tracking"""
        self.tracking_active = False
        if self._tracking_thread:
            self._tracking_thread.join(timeout=1.0)
        logger.info("Stopped resource tracking")

    def _tracking_loop(self):
        """Resource tracking loop"""
        while self.tracking_active:
            try:
                # CPU usage
                self.current_usage["cpu_percent"] = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()
                process = psutil.Process()
                self.current_usage["memory_mb"] = process.memory_info().rss / (1024 * 1024)

                # GPU memory (if available)
                gpu_memory = 0.0
                try:
                    import torch
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            gpu_memory += torch.cuda.memory_allocated(i) / (1024 * 1024)
                except ImportError:
                    pass

                self.current_usage["gpu_memory_mb"] = gpu_memory

                # Disk usage
                disk = psutil.disk_usage('/')
                self.current_usage["disk_usage_gb"] = disk.used / (1024 * 1024 * 1024)

                time.sleep(5)  # Update every 5 seconds

            except Exception as e:
                logger.warning(f"Resource tracking error: {e}")
                time.sleep(10)

    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return self.current_usage.copy()


class CostMonitor:
    """
    Comprehensive cost monitoring system for sentiment analysis operations.

    Features:
    - Real-time usage tracking
    - Cost estimation and budgeting
    - Usage analytics and reporting
    - Cost optimization recommendations
    - Budget alerts and limits
    """

    def __init__(
        self,
        storage_dir: str = "cost_monitoring",
        budget_limit: float = 10.0,  # $10 monthly budget
        alert_threshold: float = 0.8  # Alert at 80% of budget
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        self.budget_limit = budget_limit
        self.alert_threshold = alert_threshold

        # Components
        self.calculator = CostCalculator()
        self.resource_tracker = ResourceTracker()

        # Usage tracking
        self.usage_history: deque = deque(maxlen=10000)  # Keep last 10k operations
        self.daily_costs = defaultdict(float)
        self.monthly_costs = defaultdict(float)

        # Real-time metrics
        self.current_session_cost = 0.0
        self.operations_count = 0

        # Alerts
        self.alerts_sent = set()

        # Load historical data
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical cost data"""
        try:
            history_file = self.storage_dir / "usage_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)

                # Restore usage history
                for item in data.get("usage_history", []):
                    self.usage_history.append(UsageMetrics(**item))

                # Restore cost data
                self.daily_costs.update(data.get("daily_costs", {}))
                self.monthly_costs.update(data.get("monthly_costs", {}))

                logger.info("Loaded historical cost data")

        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")

    def _save_historical_data(self):
        """Save historical cost data"""
        try:
            history_file = self.storage_dir / "usage_history.json"

            data = {
                "usage_history": [asdict(metric) for metric in list(self.usage_history)],
                "daily_costs": dict(self.daily_costs),
                "monthly_costs": dict(self.monthly_costs),
                "last_updated": time.time()
            }

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save historical data: {e}")

    def start_monitoring(self):
        """Start cost monitoring"""
        self.resource_tracker.start_tracking()
        logger.info("Cost monitoring started")

    def stop_monitoring(self):
        """Stop cost monitoring"""
        self.resource_tracker.stop_tracking()
        self._save_historical_data()
        logger.info("Cost monitoring stopped")

    def record_operation(
        self,
        operation_type: str,
        duration: float,
        model_name: str,
        provider_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Record a completed operation"""

        # Get current resource usage
        usage = self.resource_tracker.get_current_usage()

        # Create usage metric
        metric = UsageMetrics(
            operation_type=operation_type,
            timestamp=time.time(),
            duration=duration,
            memory_used_mb=usage["memory_mb"],
            cpu_usage_percent=usage["cpu_percent"],
            gpu_memory_mb=usage["gpu_memory_mb"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=model_name,
            provider_name=provider_name,
            success=success,
            error_message=error_message
        )

        # Add to history
        self.usage_history.append(metric)
        self.operations_count += 1

        # Calculate cost
        if operation_type == "inference":
            cost_estimate = self.calculator.calculate_inference_cost(
                duration,
                usage["memory_mb"],
                usage["gpu_memory_mb"],
                self._extract_model_size(model_name)
            )
        elif operation_type == "training":
            cost_estimate = self.calculator.calculate_training_cost(
                duration / 3600,  # Convert to hours
                usage["memory_mb"] / 1024,  # Convert to GB
                input_tokens,  # Use as sample count
                self._extract_model_size(model_name)
            )
        else:
            # Default cost calculation
            cost_estimate = self.calculator.calculate_inference_cost(
                duration,
                usage["memory_mb"],
                usage["gpu_memory_mb"]
            )

        # Update cost tracking
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")

        self.daily_costs[today] += cost_estimate.total_cost
        self.monthly_costs[month] += cost_estimate.total_cost
        self.current_session_cost += cost_estimate.total_cost

        # Check for budget alerts
        self._check_budget_alerts()

        logger.debug(f"Recorded operation: {operation_type}, cost: ${cost_estimate.total_cost:.6f}")

    def _extract_model_size(self, model_name: str) -> str:
        """Extract model size from model name"""
        model_name_lower = model_name.lower()

        if "ultra" in model_name_lower or "tiny" in model_name_lower:
            return "ultra_small"
        elif "small" in model_name_lower or "base" in model_name_lower:
            return "small"
        elif "medium" in model_name_lower:
            return "medium"
        elif "large" in model_name_lower:
            return "large"
        else:
            return "small"  # Default

    def _check_budget_alerts(self):
        """Check if budget alerts should be sent"""
        month = datetime.now().strftime("%Y-%m")
        current_monthly_cost = self.monthly_costs[month]

        alert_threshold_cost = self.budget_limit * self.alert_threshold

        if current_monthly_cost >= alert_threshold_cost:
            alert_key = f"{month}_threshold"
            if alert_key not in self.alerts_sent:
                self._send_budget_alert(
                    f"Budget alert: ${current_monthly_cost:.2f} / ${self.budget_limit:.2f} "
                    f"({current_monthly_cost/self.budget_limit*100:.1f}%)"
                )
                self.alerts_sent.add(alert_key)

        if current_monthly_cost >= self.budget_limit:
            alert_key = f"{month}_limit"
            if alert_key not in self.alerts_sent:
                self._send_budget_alert(
                    f"Budget limit exceeded: ${current_monthly_cost:.2f} / ${self.budget_limit:.2f}"
                )
                self.alerts_sent.add(alert_key)

    def _send_budget_alert(self, message: str):
        """Send budget alert (log for now, could extend to email/slack)"""
        logger.warning(f"BUDGET ALERT: {message}")

    def get_cost_summary(self, period: str = "month") -> Dict:
        """Get cost summary for specified period"""

        if period == "day":
            today = datetime.now().strftime("%Y-%m-%d")
            total_cost = self.daily_costs.get(today, 0.0)
            period_key = today
        elif period == "month":
            month = datetime.now().strftime("%Y-%m")
            total_cost = self.monthly_costs.get(month, 0.0)
            period_key = month
        else:
            # Session
            total_cost = self.current_session_cost
            period_key = "current_session"

        # Calculate statistics from recent operations
        recent_operations = [
            m for m in self.usage_history
            if time.time() - m.timestamp < (86400 if period == "day" else 86400 * 30)
        ]

        if recent_operations:
            avg_duration = sum(op.duration for op in recent_operations) / len(recent_operations)
            success_rate = sum(1 for op in recent_operations if op.success) / len(recent_operations)
            total_operations = len(recent_operations)
        else:
            avg_duration = 0.0
            success_rate = 1.0
            total_operations = 0

        return {
            "period": period,
            "period_key": period_key,
            "total_cost": total_cost,
            "budget_limit": self.budget_limit,
            "budget_used_percent": (total_cost / self.budget_limit * 100) if self.budget_limit > 0 else 0,
            "total_operations": total_operations,
            "avg_cost_per_operation": total_cost / max(1, total_operations),
            "avg_duration": avg_duration,
            "success_rate": success_rate,
            "estimated_monthly_cost": total_cost * 30 if period == "day" else total_cost
        }

    def get_usage_analytics(self) -> Dict:
        """Get detailed usage analytics"""

        if not self.usage_history:
            return {"error": "No usage data available"}

        # Group by operation type
        by_operation = defaultdict(list)
        by_provider = defaultdict(list)
        by_model = defaultdict(list)

        for metric in self.usage_history:
            by_operation[metric.operation_type].append(metric)
            by_provider[metric.provider_name].append(metric)
            by_model[metric.model_name].append(metric)

        # Calculate analytics
        analytics = {
            "total_operations": len(self.usage_history),
            "time_range": {
                "start": min(m.timestamp for m in self.usage_history),
                "end": max(m.timestamp for m in self.usage_history)
            },
            "by_operation_type": {},
            "by_provider": {},
            "by_model": {},
            "performance_metrics": self._calculate_performance_metrics()
        }

        # Operation type analytics
        for op_type, metrics in by_operation.items():
            analytics["by_operation_type"][op_type] = {
                "count": len(metrics),
                "avg_duration": sum(m.duration for m in metrics) / len(metrics),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
                "total_cost": sum(
                    self._estimate_operation_cost(m) for m in metrics
                )
            }

        # Provider analytics
        for provider, metrics in by_provider.items():
            analytics["by_provider"][provider] = {
                "count": len(metrics),
                "avg_duration": sum(m.duration for m in metrics) / len(metrics),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
                "avg_memory_mb": sum(m.memory_used_mb for m in metrics) / len(metrics)
            }

        # Model analytics
        for model, metrics in by_model.items():
            analytics["by_model"][model] = {
                "count": len(metrics),
                "avg_duration": sum(m.duration for m in metrics) / len(metrics),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
                "avg_memory_mb": sum(m.memory_used_mb for m in metrics) / len(metrics)
            }

        return analytics

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.usage_history:
            return {}

        recent_metrics = list(self.usage_history)[-100:]  # Last 100 operations

        return {
            "avg_response_time": sum(m.duration for m in recent_metrics) / len(recent_metrics),
            "p95_response_time": sorted([m.duration for m in recent_metrics])[int(len(recent_metrics) * 0.95)],
            "error_rate": 1 - (sum(1 for m in recent_metrics if m.success) / len(recent_metrics)),
            "avg_memory_usage": sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics),
            "throughput_ops_per_minute": len(recent_metrics) / ((time.time() - recent_metrics[0].timestamp) / 60)
            if len(recent_metrics) > 1 else 0
        }

    def _estimate_operation_cost(self, metric: UsageMetrics) -> float:
        """Estimate cost for a single operation"""
        if metric.operation_type == "inference":
            cost_estimate = self.calculator.calculate_inference_cost(
                metric.duration,
                metric.memory_used_mb,
                metric.gpu_memory_mb,
                self._extract_model_size(metric.model_name)
            )
        else:
            cost_estimate = self.calculator.calculate_training_cost(
                metric.duration / 3600,
                metric.memory_used_mb / 1024,
                metric.input_tokens,
                self._extract_model_size(metric.model_name)
            )

        return cost_estimate.total_cost

    def get_optimization_recommendations(self) -> List[str]:
        """Get cost optimization recommendations"""
        recommendations = []

        if not self.usage_history:
            return ["No usage data available for recommendations"]

        # Analyze recent usage
        recent_metrics = list(self.usage_history)[-100:]
        analytics = self.get_usage_analytics()

        # High memory usage recommendation
        avg_memory = analytics["performance_metrics"].get("avg_memory_usage", 0)
        if avg_memory > 2000:  # 2GB
            recommendations.append(
                f"High memory usage detected ({avg_memory:.0f}MB avg). "
                "Consider using model quantization (4-bit or 8-bit) to reduce memory footprint."
            )

        # Slow response time recommendation
        avg_response_time = analytics["performance_metrics"].get("avg_response_time", 0)
        if avg_response_time > 2.0:  # 2 seconds
            recommendations.append(
                f"Slow response time detected ({avg_response_time:.2f}s avg). "
                "Consider using smaller models or enabling model compilation."
            )

        # Error rate recommendation
        error_rate = analytics["performance_metrics"].get("error_rate", 0)
        if error_rate > 0.05:  # 5%
            recommendations.append(
                f"High error rate detected ({error_rate*100:.1f}%). "
                "Check model configurations and input validation."
            )

        # Budget recommendation
        month_summary = self.get_cost_summary("month")
        if month_summary["budget_used_percent"] > 50:
            recommendations.append(
                f"Budget usage is at {month_summary['budget_used_percent']:.1f}%. "
                "Consider optimizing frequently used models or increasing cache hit rates."
            )

        # Provider efficiency recommendation
        if "by_provider" in analytics:
            provider_efficiency = {}
            for provider, stats in analytics["by_provider"].items():
                efficiency = stats["success_rate"] / (stats["avg_duration"] + 0.1)
                provider_efficiency[provider] = efficiency

            if len(provider_efficiency) > 1:
                best_provider = max(provider_efficiency.keys(), key=lambda x: provider_efficiency[x])
                recommendations.append(
                    f"Most efficient provider: {best_provider}. "
                    "Consider using it for more operations."
                )

        if not recommendations:
            recommendations.append("No specific optimizations needed. System is performing well!")

        return recommendations

    def export_report(self, format: str = "json") -> str:
        """Export comprehensive cost report"""
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "daily": self.get_cost_summary("day"),
                "monthly": self.get_cost_summary("month"),
                "session": self.get_cost_summary("session")
            },
            "analytics": self.get_usage_analytics(),
            "recommendations": self.get_optimization_recommendations(),
            "budget_info": {
                "limit": self.budget_limit,
                "alert_threshold": self.alert_threshold,
                "alerts_sent": list(self.alerts_sent)
            }
        }

        if format == "json":
            report_file = self.storage_dir / f"cost_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            return str(report_file)

        # Could add other formats (CSV, PDF) here
        return json.dumps(report_data, indent=2)

    async def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        self._save_historical_data()
        logger.info("Cost monitor cleaned up")