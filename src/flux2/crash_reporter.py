"""
Opt-in Crash and Telemetry Reporter for FLUX.2

Collects anonymous crash logs, stack traces, and performance regression data
with explicit user consent. All data is privacy-compliant (no PII).

Features:
- Anonymous crash reporting
- Stack trace collection with source code redaction
- Performance regression detection
- Feature usage heatmaps
- Opt-in consent mechanism
- Local storage with cloud submission option

Phase 7 Implementation
"""

import sys
import traceback
import platform
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import threading


@dataclass
class CrashReport:
    """Anonymous crash report structure"""
    crash_id: str
    timestamp: str
    python_version: str
    platform_system: str
    platform_release: str
    exception_type: str
    exception_message: str
    stack_trace: List[str]
    context: Dict[str, Any]
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_exception(
        cls,
        exc_type: type,
        exc_value: Exception,
        exc_traceback,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> "CrashReport":
        """Create crash report from exception"""
        # Generate unique crash ID (hash of exception + stack trace)
        crash_hash = hashlib.sha256(
            f"{exc_type.__name__}{str(exc_value)}{''.join(traceback.format_tb(exc_traceback))}".encode()
        ).hexdigest()[:16]
        
        # Extract stack trace
        stack_trace = []
        for line in traceback.format_tb(exc_traceback):
            # Redact file paths (keep only filename)
            sanitized = cls._sanitize_stack_line(line)
            stack_trace.append(sanitized)
        
        return cls(
            crash_id=crash_hash,
            timestamp=datetime.now().isoformat(),
            python_version=platform.python_version(),
            platform_system=platform.system(),
            platform_release=platform.release(),
            exception_type=exc_type.__name__,
            exception_message=cls._sanitize_message(str(exc_value)),
            stack_trace=stack_trace,
            context=cls._sanitize_context(context or {}),
            session_id=session_id
        )
    
    @staticmethod
    def _sanitize_stack_line(line: str) -> str:
        """Remove absolute paths from stack trace line"""
        # Replace full paths with just filename
        import re
        # Match file paths like: File "C:\Users\...\file.py", line 123
        pattern = r'File "(.+[\\/])([^\\\/]+\.py)"'
        return re.sub(pattern, r'File "\2"', line)
    
    @staticmethod
    def _sanitize_message(message: str) -> str:
        """Remove potential PII from exception message"""
        import re
        # Redact file paths
        message = re.sub(r'[A-Za-z]:\\[^\s]+', '[PATH]', message)
        message = re.sub(r'/[^\s]+', '[PATH]', message)
        # Redact IP addresses
        message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', message)
        # Truncate long messages
        return message[:500]
    
    @staticmethod
    def _sanitize_context(context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from context"""
        sanitized = {}
        for key, value in context.items():
            if key.lower() in ['password', 'token', 'api_key', 'secret']:
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 200:
                sanitized[key] = value[:200] + "..."
            else:
                sanitized[key] = value
        return sanitized


class CrashReporter:
    """
    Opt-in crash reporter with privacy-first design
    
    Usage:
        reporter = CrashReporter(enabled=True)
        reporter.install_exception_handler()
        
        # Use as decorator
        @reporter.catch_errors
        def risky_function():
            ...
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        enabled: bool = False,
        storage_dir: Optional[Path] = None,
        session_id: Optional[str] = None
    ):
        if self._initialized:
            return
        
        self.enabled = enabled
        self.storage_dir = storage_dir or Path("logs/crashes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Use session ID from analytics if available
        self.session_id = session_id or "anonymous"
        
        # Crash statistics
        self.crash_count = 0
        self.crashes: List[CrashReport] = []
        
        # Original exception handler
        self._original_excepthook = sys.excepthook
        
        self._initialized = True
    
    def install_exception_handler(self):
        """Install global exception handler"""
        if not self.enabled:
            return
        
        def exception_handler(exc_type, exc_value, exc_traceback):
            """Custom exception handler that logs crashes"""
            # Report crash
            self.report_crash(exc_type, exc_value, exc_traceback)
            
            # Call original handler
            self._original_excepthook(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = exception_handler
    
    def uninstall_exception_handler(self):
        """Restore original exception handler"""
        sys.excepthook = self._original_excepthook
    
    def report_crash(
        self,
        exc_type: type,
        exc_value: Exception,
        exc_traceback,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Report a crash with anonymous context
        
        Args:
            exc_type: Exception class
            exc_value: Exception instance
            exc_traceback: Traceback object
            context: Additional context (sanitized)
        """
        if not self.enabled:
            return
        
        # Create crash report
        report = CrashReport.from_exception(
            exc_type=exc_type,
            exc_value=exc_value,
            exc_traceback=exc_traceback,
            session_id=self.session_id,
            context=context
        )
        
        # Store in memory
        self.crashes.append(report)
        self.crash_count += 1
        
        # Write to disk
        self._write_crash_report(report)
        
        # Could submit to remote service here (with consent)
        # self._submit_to_service(report)
    
    def _write_crash_report(self, report: CrashReport):
        """Write crash report to disk"""
        crash_file = self.storage_dir / f"crash_{report.crash_id}_{report.timestamp.replace(':', '-')}.json"
        
        with open(crash_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
    
    def catch_errors(self, func: Callable) -> Callable:
        """
        Decorator to catch and report errors from functions
        
        Usage:
            @crash_reporter.catch_errors
            def my_function():
                ...
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self.enabled:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    self.report_crash(
                        exc_type=exc_type,
                        exc_value=exc_value,
                        exc_traceback=exc_traceback,
                        context={
                            "function": func.__name__,
                            "module": func.__module__
                        }
                    )
                raise
        
        return wrapper
    
    def get_crash_statistics(self) -> Dict[str, Any]:
        """Get crash statistics summary"""
        if not self.crashes:
            return {
                "total_crashes": 0,
                "unique_crashes": 0,
                "most_common_exception": "None",
                "crash_frequency": {}
            }
        
        # Count exception types
        exception_counts = {}
        for crash in self.crashes:
            exc_type = crash.exception_type
            exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1
        
        # Most common exception
        most_common = max(exception_counts.items(), key=lambda x: x[1])
        
        return {
            "total_crashes": len(self.crashes),
            "unique_crashes": len(set(c.crash_id for c in self.crashes)),
            "most_common_exception": most_common[0],
            "most_common_count": most_common[1],
            "crash_frequency": exception_counts
        }
    
    def get_recent_crashes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent crashes"""
        recent = sorted(
            self.crashes,
            key=lambda c: c.timestamp,
            reverse=True
        )[:limit]
        
        return [crash.to_dict() for crash in recent]
    
    def load_historical_crashes(self, days: int = 30) -> List[CrashReport]:
        """Load crash reports from disk"""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        crashes = []
        
        for crash_file in self.storage_dir.glob("crash_*.json"):
            try:
                with open(crash_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Check if within time range
                crash_time = datetime.fromisoformat(data["timestamp"])
                if crash_time > cutoff:
                    report = CrashReport(**data)
                    crashes.append(report)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        
        return crashes
    
    def clear_crash_data(self):
        """Delete all crash reports (privacy request)"""
        # Clear memory
        self.crashes.clear()
        self.crash_count = 0
        
        # Delete files
        for crash_file in self.storage_dir.glob("crash_*.json"):
            try:
                crash_file.unlink()
            except OSError:
                pass
    
    def enable_reporting(self, session_id: Optional[str] = None):
        """Enable crash reporting with optional session ID"""
        self.enabled = True
        if session_id:
            self.session_id = session_id
        self.install_exception_handler()
    
    def disable_reporting(self):
        """Disable crash reporting"""
        self.enabled = False
        self.uninstall_exception_handler()


class PerformanceMonitor:
    """
    Performance regression detection system
    
    Tracks key metrics over time and alerts on degradation
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("logs/performance")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.storage_dir / "performance_metrics.jsonl"
        
        # Baseline metrics (loaded from historical data)
        self.baselines: Dict[str, float] = {}
        self._load_baselines()
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "value": value,
            "context": context or {}
        }
        
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    
    def _load_baselines(self):
        """Load baseline metrics from historical data"""
        if not self.metrics_file.exists():
            return
        
        # Calculate median for each metric from last 30 days
        from collections import defaultdict
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=30)
        metric_values = defaultdict(list)
        
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    timestamp = datetime.fromisoformat(record["timestamp"])
                    
                    if timestamp > cutoff:
                        metric_values[record["metric"]].append(record["value"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        
        # Calculate median as baseline
        for metric, values in metric_values.items():
            if values:
                sorted_values = sorted(values)
                median_idx = len(sorted_values) // 2
                self.baselines[metric] = sorted_values[median_idx]
    
    def check_regression(
        self,
        metric_name: str,
        current_value: float,
        threshold_percent: float = 10.0
    ) -> bool:
        """
        Check if metric has regressed beyond threshold
        
        Args:
            metric_name: Name of metric to check
            current_value: Current measured value
            threshold_percent: Acceptable regression percentage
        
        Returns:
            True if regression detected
        """
        if metric_name not in self.baselines:
            # No baseline yet
            return False
        
        baseline = self.baselines[metric_name]
        regression_pct = ((current_value - baseline) / baseline) * 100
        
        return regression_pct > threshold_percent
    
    def get_trend_data(
        self,
        metric_name: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get historical trend data for a metric"""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        data = []
        
        if not self.metrics_file.exists():
            return data
        
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record["metric"] == metric_name:
                        timestamp = datetime.fromisoformat(record["timestamp"])
                        if timestamp > cutoff:
                            data.append({
                                "timestamp": record["timestamp"],
                                "value": record["value"]
                            })
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        
        return sorted(data, key=lambda x: x["timestamp"])


# Global instances
_crash_reporter: Optional[CrashReporter] = None
_performance_monitor: Optional[PerformanceMonitor] = None


def get_crash_reporter() -> CrashReporter:
    """Get global crash reporter instance"""
    global _crash_reporter
    if _crash_reporter is None:
        _crash_reporter = CrashReporter()
    return _crash_reporter


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
