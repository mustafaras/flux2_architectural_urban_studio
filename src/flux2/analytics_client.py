"""
Privacy-First Analytics Client for FLUX.2 Professional Image Generator

Tracks user behavior and feature usage WITHOUT collecting personally identifiable
information (PII). All analytics are session-based with anonymous IDs.

Features:
- Anonymous session tracking (UUID-based, no user identification)
- PII sanitization for all metadata
- Opt-in/opt-out with global disable flag
- Local storage with configurable retention
- Batch submission to minimize overhead
- Event categorization and filtering
- Privacy-compliant reporting

Phase 7 Implementation
"""

import uuid
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import defaultdict, deque
import re


class EventType(Enum):
    """Event categories for analytics tracking"""
    # Generation events
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_FAILED = "generation_failed"
    GENERATION_CANCELLED = "generation_cancelled"
    
    # Model events
    MODEL_LOADED = "model_loaded"
    MODEL_SWITCHED = "model_switched"
    MODEL_CACHE_HIT = "model_cache_hit"
    MODEL_CACHE_MISS = "model_cache_miss"
    
    # UI events
    TAB_SWITCHED = "tab_switched"
    SETTINGS_CHANGED = "settings_changed"
    PRESET_APPLIED = "preset_applied"
    EXPORT_EXECUTED = "export_executed"
    
    # Queue events
    QUEUE_ITEM_ADDED = "queue_item_added"
    QUEUE_ITEM_REMOVED = "queue_item_removed"
    QUEUE_ITEM_REORDERED = "queue_item_reordered"
    QUEUE_PROCESSED = "queue_processed"
    
    # Safety events
    SAFETY_CHECK_PASSED = "safety_check_passed"
    SAFETY_CHECK_FAILED = "safety_check_failed"
    SAFETY_RULE_ADDED = "safety_rule_added"
    SAFETY_APPEAL_SUBMITTED = "safety_appeal_submitted"
    
    # Performance events
    PERFORMANCE_MEASURED = "performance_measured"
    MEMORY_WARNING = "memory_warning"
    GPU_THROTTLING = "gpu_throttling"
    
    # Error events
    ERROR_OCCURRED = "error_occurred"
    ERROR_RECOVERED = "error_recovered"
    
    # Session events
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"

    # Architecture workflow events
    CREATE_PROJECT = "create_project"
    COMPOSE_PROMPT = "compose_prompt"
    GENERATE_OPTION = "generate_option"
    SHORTLIST_OPTION = "shortlist_option"
    EXPORT_BOARD = "export_board"


class PrivacyLevel(Enum):
    """Privacy levels for analytics collection"""
    DISABLED = 0  # No analytics collected
    MINIMAL = 1   # Only critical metrics (errors, crashes)
    STANDARD = 2  # Feature usage + performance (default)
    FULL = 3      # Detailed telemetry (still no PII)


class AnalyticsEvent:
    """Individual analytics event with sanitized metadata"""
    
    def __init__(
        self,
        event_type: EventType,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[uuid.UUID] = None
    ):
        self.event_id = uuid.uuid4()
        self.event_type = event_type
        self.timestamp = datetime.now()
        self.session_id = session_id or uuid.uuid4()
        self.metadata = self._sanitize_metadata(metadata or {})
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove any PII from metadata"""
        sanitized = {}
        
        # PII patterns to remove
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'(?i)(password|token|api_key|secret)',  # Credentials
        ]
        
        for key, value in metadata.items():
            if isinstance(value, str):
                # Check for PII patterns
                contains_pii = any(re.search(pattern, value) for pattern in pii_patterns)
                if contains_pii:
                    sanitized[key] = "[REDACTED]"
                # Truncate long strings (possible prompts with PII)
                elif len(value) > 200:
                    sanitized[key] = value[:200] + "..."
                else:
                    sanitized[key] = value
            elif isinstance(value, (int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_metadata(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_metadata({"item": item})["item"] 
                    if isinstance(item, dict) else item 
                    for item in value[:100]  # Limit array size
                ]
            else:
                sanitized[key] = str(value)[:200]
        
        return sanitized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for storage"""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": str(self.session_id),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalyticsEvent":
        """Reconstruct event from dictionary"""
        event = cls.__new__(cls)
        event.event_id = uuid.UUID(data["event_id"])
        event.event_type = EventType(data["event_type"])
        event.timestamp = datetime.fromisoformat(data["timestamp"])
        event.session_id = uuid.UUID(data["session_id"])
        event.metadata = data["metadata"]
        return event


class AnalyticsClient:
    """
    Privacy-first analytics client for tracking usage without PII
    
    Usage:
        client = AnalyticsClient()
        client.log_event(EventType.GENERATION_STARTED, {"model": "klein-4b"})
        client.batch_submit()
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global analytics client"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
        max_events_in_memory: int = 1000,
        retention_days: int = 90
    ):
        if self._initialized:
            return
        
        self.session_id = uuid.uuid4()
        self.privacy_level = privacy_level
        self.enabled = privacy_level != PrivacyLevel.DISABLED
        self.max_events_in_memory = max_events_in_memory
        self.retention_days = retention_days
        
        # Storage
        self.storage_dir = storage_dir or Path("logs/analytics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory event buffer
        self.events: deque[AnalyticsEvent] = deque(maxlen=max_events_in_memory)
        
        # Session start
        self._session_start_time = datetime.now()
        self.log_event(EventType.SESSION_STARTED, {
            "privacy_level": privacy_level.name,
            "session_start": self._session_start_time.isoformat()
        })
        
        # Statistics cache
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_time: Optional[datetime] = None
        self._cache_ttl = 60  # seconds
        
        self._initialized = True
    
    def log_event(
        self,
        event_type: EventType,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an analytics event with privacy sanitization
        
        Args:
            event_type: Type of event from EventType enum
            metadata: Additional event data (will be sanitized)
        """
        if not self.enabled:
            return
        
        # Filter events based on privacy level
        if self.privacy_level == PrivacyLevel.MINIMAL:
            # Only log errors and crashes
            if event_type not in [
                EventType.ERROR_OCCURRED,
                EventType.GENERATION_FAILED,
                EventType.SESSION_STARTED,
                EventType.SESSION_ENDED
            ]:
                return
        
        event = AnalyticsEvent(
            event_type=event_type,
            metadata=metadata,
            session_id=self.session_id
        )
        
        self.events.append(event)
        
        # Auto-flush if buffer nearly full
        if len(self.events) >= self.max_events_in_memory * 0.9:
            self.flush_to_disk()
    
    def flush_to_disk(self):
        """Write in-memory events to disk"""
        if not self.events:
            return
        
        # Create daily log file
        log_file = self.storage_dir / f"analytics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, "a", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict()) + "\n")
        
        self.events.clear()
    
    def batch_submit(self):
        """
        Submit analytics batch (currently stores locally)
        
        Future: Could send to analytics service with user consent
        """
        self.flush_to_disk()
        self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """Remove analytics files older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.storage_dir.glob("analytics_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("analytics_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    log_file.unlink()
            except (ValueError, OSError):
                pass
    
    def get_statistics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get aggregated statistics from all events
        
        Args:
            force_refresh: Bypass cache and recalculate
        
        Returns:
            Dictionary with statistics
        """
        # Check cache
        if not force_refresh and self._stats_cache_time:
            age = (datetime.now() - self._stats_cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._stats_cache
        
        # Load all events
        all_events = self._load_all_events()
        
        # Calculate statistics
        stats = {
            "total_events": len(all_events),
            "session_count": len(set(e.session_id for e in all_events)),
            "event_counts": defaultdict(int),
            "this_week": self._get_weekly_stats(all_events),
            "performance": self._get_performance_stats(all_events),
            "errors": self._get_error_stats(all_events),
            "popular_models": self._get_popular_models(all_events),
            "feature_usage": self._get_feature_usage(all_events)
        }
        
        for event in all_events:
            stats["event_counts"][event.event_type.value] += 1
        
        # Cache results
        self._stats_cache = stats
        self._stats_cache_time = datetime.now()
        
        return stats
    
    def _load_all_events(self, days: int = 30) -> List[AnalyticsEvent]:
        """Load events from disk (last N days)"""
        events = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Load from files
        for log_file in sorted(self.storage_dir.glob("analytics_*.jsonl")):
            try:
                date_str = log_file.stem.replace("analytics_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    continue
                
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            events.append(AnalyticsEvent.from_dict(data))
                        except (json.JSONDecodeError, KeyError):
                            pass
            except (ValueError, OSError):
                pass
        
        # Add in-memory events
        events.extend(list(self.events))
        
        return events
    
    def _get_weekly_stats(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Calculate statistics for the last 7 days"""
        week_ago = datetime.now() - timedelta(days=7)
        week_events = [e for e in events if e.timestamp > week_ago]
        
        generation_events = [
            e for e in week_events 
            if e.event_type == EventType.GENERATION_COMPLETED
        ]
        
        # Calculate average generation time
        total_time = 0
        for event in generation_events:
            if "duration_ms" in event.metadata:
                total_time += event.metadata["duration_ms"]
        
        avg_time = (total_time / len(generation_events) / 1000) if generation_events else 0
        
        # Most used model
        model_counts = defaultdict(int)
        for event in generation_events:
            if "model" in event.metadata:
                model_counts[event.metadata["model"]] += 1
        
        most_used_model = max(model_counts.items(), key=lambda x: x[1]) if model_counts else ("N/A", 0)
        
        # Favorite preset
        preset_counts = defaultdict(int)
        for event in week_events:
            if event.event_type == EventType.PRESET_APPLIED:
                if "preset_name" in event.metadata:
                    preset_counts[event.metadata["preset_name"]] += 1
        
        favorite_preset = max(preset_counts.items(), key=lambda x: x[1]) if preset_counts else ("N/A", 0)
        
        return {
            "total_generations": len(generation_events),
            "avg_generation_time_seconds": round(avg_time, 1),
            "most_used_model": most_used_model[0],
            "most_used_model_percentage": round(most_used_model[1] / len(generation_events) * 100, 1) if generation_events else 0,
            "favorite_preset": favorite_preset[0],
            "favorite_preset_percentage": round(favorite_preset[1] / len(week_events) * 100, 1) if week_events else 0
        }
    
    def _get_performance_stats(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Calculate performance-related statistics"""
        generation_events = [
            e for e in events 
            if e.event_type == EventType.GENERATION_COMPLETED
        ]
        
        failed_events = [
            e for e in events 
            if e.event_type == EventType.GENERATION_FAILED
        ]
        
        total_attempts = len(generation_events) + len(failed_events)
        success_rate = (len(generation_events) / total_attempts * 100) if total_attempts > 0 else 0
        
        # Queue wait times
        queue_events = [
            e for e in events 
            if e.event_type == EventType.QUEUE_PROCESSED
        ]
        
        total_wait = sum(
            e.metadata.get("wait_time_ms", 0) 
            for e in queue_events
        )
        avg_wait = (total_wait / len(queue_events) / 1000) if queue_events else 0
        
        return {
            "success_rate_percentage": round(success_rate, 1),
            "total_generations": len(generation_events),
            "total_failures": len(failed_events),
            "avg_queue_wait_seconds": round(avg_wait, 1)
        }
    
    def _get_error_stats(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_events = [
            e for e in events 
            if e.event_type == EventType.ERROR_OCCURRED
        ]
        
        error_types = defaultdict(int)
        for event in error_events:
            error_type = event.metadata.get("error_type", "Unknown")
            error_types[error_type] += 1
        
        return {
            "total_errors": len(error_events),
            "error_breakdown": dict(error_types),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else "None"
        }
    
    def _get_popular_models(self, events: List[AnalyticsEvent]) -> List[Dict[str, Any]]:
        """Get most popular models"""
        model_usage = defaultdict(lambda: {"count": 0, "total_time_ms": 0})
        
        for event in events:
            if event.event_type == EventType.GENERATION_COMPLETED:
                model = event.metadata.get("model", "unknown")
                model_usage[model]["count"] += 1
                model_usage[model]["total_time_ms"] += event.metadata.get("duration_ms", 0)
        
        # Sort by usage count
        sorted_models = sorted(
            model_usage.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        return [
            {
                "model": model,
                "usage_count": stats["count"],
                "avg_time_seconds": round(stats["total_time_ms"] / stats["count"] / 1000, 1)
            }
            for model, stats in sorted_models[:10]
        ]
    
    def _get_feature_usage(self, events: List[AnalyticsEvent]) -> Dict[str, int]:
        """Track which features are most used"""
        feature_counts = defaultdict(int)
        
        for event in events:
            if event.event_type == EventType.TAB_SWITCHED:
                tab = event.metadata.get("tab_name", "unknown")
                feature_counts[f"tab_{tab}"] += 1
            elif event.event_type == EventType.EXPORT_EXECUTED:
                format_type = event.metadata.get("format", "unknown")
                feature_counts[f"export_{format_type}"] += 1
            elif event.event_type == EventType.SAFETY_CHECK_FAILED:
                feature_counts["safety_blocks"] += 1
            elif event.event_type == EventType.CREATE_PROJECT:
                feature_counts["create_project"] += 1
            elif event.event_type == EventType.COMPOSE_PROMPT:
                feature_counts["compose_prompt"] += 1
            elif event.event_type == EventType.GENERATE_OPTION:
                feature_counts["generate_option"] += 1
            elif event.event_type == EventType.SHORTLIST_OPTION:
                feature_counts["shortlist_option"] += 1
            elif event.event_type == EventType.EXPORT_BOARD:
                feature_counts["export_board"] += 1
        
        return dict(feature_counts)
    
    def export_to_csv(self, output_path: Path, days: int = 30) -> Path:
        """
        Export analytics data to CSV
        
        Args:
            output_path: Where to save CSV file
            days: Number of days to include
        
        Returns:
            Path to created CSV file
        """
        import csv
        
        events = self._load_all_events(days=days)
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Event ID", "Event Type", "Timestamp", 
                "Session ID", "Metadata"
            ])
            
            for event in events:
                writer.writerow([
                    str(event.event_id),
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    str(event.session_id),
                    json.dumps(event.metadata)
                ])
        
        return output_path
    
    def clear_all_data(self):
        """Delete all analytics data (user privacy request)"""
        # Clear in-memory
        self.events.clear()
        self._stats_cache.clear()
        
        # Delete files
        for log_file in self.storage_dir.glob("analytics_*.jsonl"):
            try:
                log_file.unlink()
            except OSError:
                pass
    
    def set_privacy_level(self, level: PrivacyLevel):
        """Change privacy level (requires app restart to take full effect)"""
        self.privacy_level = level
        self.enabled = level != PrivacyLevel.DISABLED
        
        if level == PrivacyLevel.DISABLED:
            self.clear_all_data()
    
    def shutdown(self):
        """Close session and flush remaining events"""
        self.log_event(EventType.SESSION_ENDED, {
            "session_duration_seconds": (datetime.now() - self._session_start_time).total_seconds()
        })
        self.flush_to_disk()


# Global analytics client instance
_analytics_client: Optional[AnalyticsClient] = None


def get_analytics() -> AnalyticsClient:
    """Get global analytics client instance"""
    global _analytics_client
    if _analytics_client is None:
        _analytics_client = AnalyticsClient()
    return _analytics_client


def log_analytics_event(event_type: EventType, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to log analytics event"""
    get_analytics().log_event(event_type, metadata)
