"""
Advanced Safety & Content Filtering Pipeline

Multi-stage safety checks with configurable rules and explainability.
Supports prompt analysis, image detection, and custom rule definitions.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety filtering strictness levels"""
    STRICT = "strict"  # Blocks ambiguous content
    MODERATE = "moderate"  # Current default
    PERMISSIVE = "permissive"  # Flags only clear violations
    CUSTOM = "custom"  # User-defined rules


class ViolationType(Enum):
    """Types of safety violations"""
    NSFW_EXPLICIT = "nsfw_explicit"
    NSFW_SUGGESTIVE = "nsfw_suggestive"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    ILLEGAL_CONTENT = "illegal_content"
    PERSONAL_INFO = "personal_info"
    SPAM = "spam"
    MISLEADING = "misleading"
    COPYRIGHT = "copyright"
    CUSTOM_RULE = "custom_rule"


@dataclass
class SafetyResult:
    """Result from safety check"""
    is_safe: bool
    confidence: float  # 0.0 to 1.0
    violations: List['Violation'] = field(default_factory=list)
    flagged_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    explanation: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "is_safe": self.is_safe,
            "confidence": self.confidence,
            "violations": [v.to_dict() for v in self.violations],
            "flagged_regions": self.flagged_regions,
            "explanation": self.explanation,
            "timestamp": self.timestamp
        }


@dataclass
class Violation:
    """Specific safety violation"""
    type: ViolationType
    severity: float  # 0.0 to 1.0
    description: str
    matched_pattern: Optional[str] = None
    suggested_alternative: Optional[str] = None
    region: Optional[Tuple[int, int, int, int]] = None  # Image region if applicable
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "type": self.type.value,
            "severity": self.severity,
            "description": self.description,
            "matched_pattern": self.matched_pattern,
            "suggested_alternative": self.suggested_alternative,
            "region": self.region
        }


@dataclass
class SafetyRule:
    """Custom safety rule definition"""
    name: str
    pattern: str  # Regex pattern
    violation_type: ViolationType
    severity: float
    enabled: bool = True
    case_sensitive: bool = False
    description: str = ""
    suggested_fix: str = ""


class SafetyPipeline:
    """
    Multi-stage safety pipeline with configurable rules.
    
    Features:
    - Prompt-based filtering (keyword + semantic)
    - Image-based NSFW detection
    - Custom rule support (YAML)
    - Violation tracking and reporting
    - Configurable safety levels
    """
    
    def __init__(self, 
                 safety_level: SafetyLevel = SafetyLevel.MODERATE,
                 config_path: Optional[Path] = None):
        self.safety_level = safety_level
        # Make config path configurable via environment variable
        if config_path is None:
            import os
            config_env = os.environ.get("FLUX2_SAFETY_CONFIG_PATH")
            config_path = Path(config_env) if config_env else Path("src/flux2/safety_config.yaml")
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        # Built-in keyword lists (moderate level)
        self.nsfw_keywords = self._load_nsfw_keywords()
        self.violence_keywords = self._load_violence_keywords()
        self.hate_keywords = self._load_hate_keywords()
        self.semantic_patterns = self._load_semantic_patterns()
        self.whitelist_patterns: List[Dict[str, str]] = []
        self.blacklist_patterns: List[Dict[str, Any]] = []
        
        # Custom rules
        self.custom_rules: List[SafetyRule] = []

        # Defaults (overridden by config)
        self.skin_tone_threshold = 0.40
        self.contrast_threshold = 0.85
        self.max_history = 1000
        
        # Statistics
        self.violation_history: List[Dict] = []
        self.check_count = 0
        self.violation_count = 0
        
        # NSFW threshold based on safety level
        self.nsfw_threshold = self._get_threshold_for_level(safety_level)

        # Load/override from config
        self.reload_config()
        
        logger.info(f"SafetyPipeline initialized with level: {safety_level.value}")
    
    def _get_threshold_for_level(self, level: SafetyLevel) -> float:
        """Get NSFW detection threshold based on safety level"""
        thresholds = {
            SafetyLevel.STRICT: 0.70,  # More sensitive
            SafetyLevel.MODERATE: 0.85,
            SafetyLevel.PERMISSIVE: 0.95,  # Less sensitive
            SafetyLevel.CUSTOM: 0.85  # Default, can be overridden
        }
        return thresholds.get(level, 0.85)
    
    def _load_nsfw_keywords(self) -> List[str]:
        """Load NSFW keyword list (moderate severity)"""
        # Basic NSFW terms - production would load from external file
        return [
            r'\bnude\b', r'\bnaked\b', r'\bnsfw\b', r'\bexplicit\b',
            r'\bporn\b', r'\bxxx\b', r'\berotic\b', r'\blewd\b',
            r'\bsexy\b', r'\badult\b.*\bcontent\b'
        ]
    
    def _load_violence_keywords(self) -> List[str]:
        """Load violence keyword list"""
        return [
            r'\bgore\b', r'\bviolence\b', r'\bblood\b', r'\bmurder\b',
            r'\bkilling\b', r'\btorture\b', r'\bweapon\b.*\battack\b'
        ]
    
    def _load_hate_keywords(self) -> List[str]:
        """Load hate speech keyword list"""
        return [
            r'\bhate\b.*\bspeech\b', r'\bracist\b', r'\bdiscrimination\b',
            r'\bslur\b', r'\bhomophobic\b', r'\bxenophobic\b'
        ]

    def _load_semantic_patterns(self) -> Dict[ViolationType, List[str]]:
        """Load phrase-level semantic risk patterns."""
        return {
            ViolationType.NSFW_SUGGESTIVE: [
                "adult roleplay",
                "sexual fantasy",
                "seductive pose",
                "provocative scene",
            ],
            ViolationType.VIOLENCE: [
                "graphic injury",
                "brutal execution",
                "detailed dismemberment",
                "violent assault",
            ],
            ViolationType.ILLEGAL_CONTENT: [
                "how to make illegal drugs",
                "counterfeit documents",
                "evade law enforcement",
                "weapon trafficking",
            ],
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config file if available."""
        if not self.config_path.exists():
            return {}

        try:
            import yaml
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                return loaded if isinstance(loaded, dict) else {}
        except ImportError:
            logger.warning("PyYAML not installed, using in-code safety defaults")
            return {}
        except Exception as e:
            logger.error(f"Failed to load safety config: {e}")
            return {}

    def _apply_config(self, config: Dict[str, Any]):
        """Apply loaded configuration to pipeline settings."""
        self.config = config

        configured_level = config.get("safety_level")
        if configured_level in [level.value for level in SafetyLevel]:
            self.safety_level = SafetyLevel(configured_level)

        presets = config.get("presets", {}) if isinstance(config.get("presets", {}), dict) else {}
        level_preset = presets.get(self.safety_level.value, {}) if isinstance(presets, dict) else {}

        configured_threshold = config.get("nsfw_threshold")
        if isinstance(level_preset, dict) and "nsfw_threshold" in level_preset:
            configured_threshold = level_preset["nsfw_threshold"]

        self.nsfw_threshold = (
            float(configured_threshold)
            if isinstance(configured_threshold, (int, float))
            else self._get_threshold_for_level(self.safety_level)
        )

        image_analysis = config.get("image_analysis", {})
        if isinstance(image_analysis, dict):
            self.skin_tone_threshold = float(image_analysis.get("skin_tone_threshold", self.skin_tone_threshold))
            self.contrast_threshold = float(image_analysis.get("contrast_threshold", self.contrast_threshold))

        violation_response = config.get("violation_response", {})
        if isinstance(violation_response, dict):
            self.max_history = int(violation_response.get("max_history", self.max_history))

        self.whitelist_patterns = config.get("whitelist", []) if isinstance(config.get("whitelist", []), list) else []
        self.blacklist_patterns = config.get("blacklist", []) if isinstance(config.get("blacklist", []), list) else []

        self._load_custom_rules(config=config)
    
    def _load_custom_rules(self, config: Optional[Dict[str, Any]] = None):
        """Load custom safety rules from YAML config"""
        if config is None:
            config = self._load_config()

        self.custom_rules = []

        try:
            rules_data = config.get('custom_rules', [])
            for rule_dict in rules_data:
                rule = SafetyRule(
                    name=rule_dict['name'],
                    pattern=rule_dict['pattern'],
                    violation_type=ViolationType(rule_dict.get('violation_type', 'custom_rule')),
                    severity=rule_dict.get('severity', 0.5),
                    enabled=rule_dict.get('enabled', True),
                    case_sensitive=rule_dict.get('case_sensitive', False),
                    description=rule_dict.get('description', ''),
                    suggested_fix=rule_dict.get('suggested_fix', '')
                )
                self.custom_rules.append(rule)
            
            logger.info(f"Loaded {len(self.custom_rules)} custom safety rules")
            
        except Exception as e:
            logger.error(f"Failed to load custom rules: {e}")

    def reload_config(self) -> bool:
        """Reload safety config from disk without restarting the app."""
        config = self._load_config()
        self._apply_config(config)
        logger.info(
            "Safety configuration reloaded (level=%s, threshold=%.2f, rules=%d)",
            self.safety_level.value,
            self.nsfw_threshold,
            len(self.custom_rules),
        )
        return True

    def _matches_pattern_list(self, text: str, pattern_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return first matching pattern item from a list of pattern dictionaries."""
        for item in pattern_items:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern")
            if not pattern or not isinstance(pattern, str):
                continue
            if re.search(pattern, text, re.IGNORECASE):
                return item
        return None

    def _analyze_semantic_risk(self, text: str) -> List[Violation]:
        """Stage-2 semantic phrase analysis for higher-level unsafe intent."""
        findings: List[Violation] = []

        for violation_type, phrases in self.semantic_patterns.items():
            for phrase in phrases:
                if phrase in text:
                    findings.append(Violation(
                        type=violation_type,
                        severity=0.72,
                        description=f"Semantic risk phrase detected: '{phrase}'",
                        matched_pattern=phrase,
                    ))

        return findings
    
    def check_prompt(self, text: str) -> SafetyResult:
        """
        Check prompt for safety violations.
        
        Multi-stage process:
        1. Keyword matching (fast)
        2. Regex pattern matching (custom rules)
        3. Semantic analysis (optional, future)
        
        Args:
            text: Prompt text to check
            
        Returns:
            SafetyResult with violations if any
        """
        self.check_count += 1
        violations = []
        
        # Stage 0: Global whitelist / blacklist
        text_lower = text.lower()

        blacklisted = self._matches_pattern_list(text_lower, self.blacklist_patterns)
        if blacklisted:
            violation = Violation(
                type=ViolationType.ILLEGAL_CONTENT,
                severity=float(blacklisted.get("severity", 1.0)),
                description=blacklisted.get("description", "Blocked by global blacklist"),
                matched_pattern=blacklisted.get("pattern"),
            )
            result = SafetyResult(
                is_safe=False,
                confidence=max(0.0, 1.0 - violation.severity),
                violations=[violation],
                explanation="Blocked: illegal_content"
            )
            self.violation_count += 1
            self._record_violation(result, prompt=text)
            return result

        whitelisted = self._matches_pattern_list(text_lower, self.whitelist_patterns)
        if whitelisted:
            return SafetyResult(
                is_safe=True,
                confidence=1.0,
                explanation=f"Whitelisted by pattern: {whitelisted.get('pattern', '')}"
            )

        # Stage 1: Keyword matching
        
        # Check NSFW keywords
        for pattern in self.nsfw_keywords:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violations.append(Violation(
                    type=ViolationType.NSFW_EXPLICIT,
                    severity=0.8,
                    description=f"NSFW keyword detected: '{match.group()}'",
                    matched_pattern=pattern,
                    suggested_alternative="Use more appropriate language"
                ))
        
        # Check violence keywords
        for pattern in self.violence_keywords:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violations.append(Violation(
                    type=ViolationType.VIOLENCE,
                    severity=0.7,
                    description=f"Violence keyword detected: '{match.group()}'",
                    matched_pattern=pattern
                ))
        
        # Check hate speech keywords
        for pattern in self.hate_keywords:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violations.append(Violation(
                    type=ViolationType.HATE_SPEECH,
                    severity=0.9,
                    description=f"Inappropriate language detected",
                    matched_pattern=pattern
                ))
        
        # Stage 2: Semantic phrase analysis
        violations.extend(self._analyze_semantic_risk(text_lower))

        # Stage 3: Custom rules
        for rule in self.custom_rules:
            if not rule.enabled:
                continue
            
            flags = 0 if rule.case_sensitive else re.IGNORECASE
            matches = re.finditer(rule.pattern, text, flags)
            
            for match in matches:
                violations.append(Violation(
                    type=rule.violation_type,
                    severity=rule.severity,
                    description=rule.description or f"Matched custom rule: {rule.name}",
                    matched_pattern=rule.pattern,
                    suggested_alternative=rule.suggested_fix
                ))
        
        # Determine if safe based on severity and count
        is_safe = True
        max_severity = 0.0
        
        if violations:
            max_severity = max(v.severity for v in violations)
            
            # Apply safety level thresholds
            if self.safety_level == SafetyLevel.STRICT:
                is_safe = max_severity < 0.5
            elif self.safety_level == SafetyLevel.MODERATE:
                is_safe = max_severity < 0.8
            elif self.safety_level == SafetyLevel.PERMISSIVE:
                is_safe = max_severity < 0.95
        
        explanation = ""
        if not is_safe:
            violation_summary = ", ".join(set(v.type.value for v in violations))
            explanation = f"Blocked: {violation_summary}"
        
        result = SafetyResult(
            is_safe=is_safe,
            confidence=1.0 - max_severity if violations else 1.0,
            violations=violations,
            explanation=explanation
        )
        
        # Track statistics
        if not is_safe:
            self.violation_count += 1
            self._record_violation(result, prompt=text)
        
        return result
    
    def check_image(self, img: Image.Image, prompt: str = "") -> SafetyResult:
        """
        Check generated image for safety violations.
        
        Uses basic heuristics (color analysis, pattern detection).
        Production would integrate actual NSFW detection model.
        
        Args:
            img: PIL Image to check
            prompt: Optional prompt context
            
        Returns:
            SafetyResult with violations if any
        """
        self.check_count += 1
        violations = []
        flagged_regions = []
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Stage 1 (visual analyzer A): skin-tone ratio (simplified NSFW proxy)
        skin_tone_ratio = self._detect_skin_tones(img_array)
        flagged_regions = self._detect_flagged_regions(img_array)
        
        nsfw_score = min(skin_tone_ratio / max(self.skin_tone_threshold, 1e-6), 1.0)
        
        if nsfw_score > self.nsfw_threshold:
            violations.append(Violation(
                type=ViolationType.NSFW_SUGGESTIVE,
                severity=nsfw_score,
                description=f"Image may contain inappropriate content (confidence: {nsfw_score:.2f})",
                region=flagged_regions[0] if flagged_regions else None,
            ))
        
        # Stage 2 (visual analyzer B): high-contrast patterns (violence proxy)
        contrast_score = self._detect_high_contrast(img_array)
        
        if contrast_score > self.contrast_threshold:
            violations.append(Violation(
                type=ViolationType.VIOLENCE,
                severity=contrast_score * 0.6,  # Lower severity for heuristic
                description="Image contains high-contrast patterns"
            ))
        
        is_safe = len(violations) == 0 or all(v.severity < self.nsfw_threshold for v in violations)
        max_severity = max((v.severity for v in violations), default=0.0)
        
        explanation = ""
        if not is_safe:
            explanation = f"Image flagged: NSFW score {nsfw_score:.2f} (threshold: {self.nsfw_threshold:.2f})"
        
        result = SafetyResult(
            is_safe=is_safe,
            confidence=1.0 - max_severity,
            violations=violations,
            flagged_regions=flagged_regions,
            explanation=explanation
        )
        
        if not is_safe:
            self.violation_count += 1
            self._record_violation(
                result,
                image=True,
                prompt=prompt,
                metadata={
                    "regions": flagged_regions,
                    "skin_tone_ratio": float(skin_tone_ratio),
                    "contrast_score": float(contrast_score),
                }
            )
        
        return result

    def _build_skin_mask(self, img_array: np.ndarray) -> np.ndarray:
        """Build boolean skin-tone mask for explainability and region extraction."""
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        return (
            (r > 95) & (g > 40) & (b > 20) &
            ((r > g) & (r > b)) &
            (abs(r - g) > 15)
        )
    
    def _detect_skin_tones(self, img_array: np.ndarray) -> float:
        """
        Detect skin tone pixels as simplified NSFW proxy.
        Real implementation would use trained classifier.
        
        Returns:
            Ratio of skin-tone pixels (0.0 to 1.0)
        """
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return 0.0

        skin_mask = self._build_skin_mask(img_array)
        
        skin_ratio = np.sum(skin_mask) / skin_mask.size
        return float(skin_ratio)

    def _detect_flagged_regions(self, img_array: np.ndarray, grid_size: int = 8) -> List[Tuple[int, int, int, int]]:
        """Estimate suspicious image regions using a coarse skin-mask heatmap grid."""
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return []

        skin_mask = self._build_skin_mask(img_array)
        height, width = skin_mask.shape
        cell_h = max(height // grid_size, 1)
        cell_w = max(width // grid_size, 1)

        regions: List[Tuple[int, int, int, int]] = []
        for row in range(grid_size):
            for col in range(grid_size):
                y0 = row * cell_h
                x0 = col * cell_w
                y1 = height if row == grid_size - 1 else (row + 1) * cell_h
                x1 = width if col == grid_size - 1 else (col + 1) * cell_w

                block = skin_mask[y0:y1, x0:x1]
                if block.size == 0:
                    continue

                block_ratio = float(np.sum(block) / block.size)
                if block_ratio >= self.skin_tone_threshold:
                    regions.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))

        return regions[:20]
    
    def _detect_high_contrast(self, img_array: np.ndarray) -> float:
        """
        Detect high-contrast patterns (violence proxy).
        
        Returns:
            Contrast score (0.0 to 1.0)
        """
        if len(img_array.shape) == 3:
            # Convert to grayscale
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Calculate contrast (std deviation normalized)
        contrast = np.std(gray) / 128.0
        return min(float(contrast), 1.0)
    
    def _record_violation(
        self,
        result: SafetyResult,
        prompt: str = "",
        image: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record violation for statistics and reporting"""
        record = {
            "timestamp": result.timestamp,
            "type": "image" if image else "prompt",
            "prompt": prompt[:100] if prompt else "",  # Truncate for privacy
            "violations": [v.type.value for v in result.violations],
            "max_severity": max((v.severity for v in result.violations), default=0.0),
            "metadata": metadata or {},
        }
        self.violation_history.append(record)
        
        # Keep bounded history
        if len(self.violation_history) > self.max_history:
            self.violation_history = self.violation_history[-self.max_history:]
    
    def get_violations(self, limit: int = 100) -> List[Dict]:
        """
        Get recent violations for dashboard display.
        
        Args:
            limit: Maximum violations to return
            
        Returns:
            List of violation records
        """
        return self.violation_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get safety statistics for dashboard.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_checks": self.check_count,
            "total_violations": self.violation_count,
            "violation_rate": self.violation_count / self.check_count if self.check_count > 0 else 0.0,
            "safety_level": self.safety_level.value,
            "nsfw_threshold": self.nsfw_threshold,
            "skin_tone_threshold": self.skin_tone_threshold,
            "contrast_threshold": self.contrast_threshold,
            "custom_rules_count": len(self.custom_rules),
            "enabled_rules_count": sum(1 for r in self.custom_rules if r.enabled)
        }
        
        # Violation breakdown
        if self.violation_history:
            violation_types = {}
            for record in self.violation_history:
                for vtype in record['violations']:
                    violation_types[vtype] = violation_types.get(vtype, 0) + 1
            
            stats["violation_breakdown"] = violation_types
            stats["most_common_violation"] = max(violation_types.items(), key=lambda x: x[1])[0] if violation_types else None
        else:
            stats["violation_breakdown"] = {}
            stats["most_common_violation"] = None
        
        return stats
    
    def add_custom_rule(self, rule: SafetyRule) -> bool:
        """Add a custom safety rule"""
        try:
            self.custom_rules.append(rule)
            self._save_custom_rules()
            logger.info(f"Added custom rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add custom rule: {e}")
            return False
    
    def remove_custom_rule(self, rule_name: str) -> bool:
        """Remove a custom safety rule by name"""
        initial_count = len(self.custom_rules)
        self.custom_rules = [r for r in self.custom_rules if r.name != rule_name]
        
        if len(self.custom_rules) < initial_count:
            self._save_custom_rules()
            logger.info(f"Removed custom rule: {rule_name}")
            return True
        
        return False
    
    def _save_custom_rules(self):
        """Save custom rules to YAML config"""
        try:
            import yaml

            config_data = self._load_config()
            if not isinstance(config_data, dict):
                config_data = {}

            config_data["safety_level"] = self.safety_level.value
            config_data["nsfw_threshold"] = self.nsfw_threshold
            config_data["custom_rules"] = [
                    {
                        "name": r.name,
                        "pattern": r.pattern,
                        "violation_type": r.violation_type.value,
                        "severity": r.severity,
                        "enabled": r.enabled,
                        "case_sensitive": r.case_sensitive,
                        "description": r.description,
                        "suggested_fix": r.suggested_fix
                    }
                    for r in self.custom_rules
                ]
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info(f"Saved {len(self.custom_rules)} custom rules to {self.config_path}")
            
        except ImportError:
            logger.warning("PyYAML not installed, cannot save custom rules")
        except Exception as e:
            logger.error(f"Failed to save custom rules: {e}")
    
    def set_safety_level(self, level: SafetyLevel):
        """Change safety level and update threshold"""
        self.safety_level = level
        presets = self.config.get("presets", {}) if isinstance(self.config, dict) else {}
        if isinstance(presets, dict) and isinstance(presets.get(level.value), dict):
            self.nsfw_threshold = float(presets[level.value].get("nsfw_threshold", self._get_threshold_for_level(level)))
        else:
            self.nsfw_threshold = self._get_threshold_for_level(level)
        logger.info(f"Safety level changed to: {level.value} (threshold: {self.nsfw_threshold})")
    
    def clear_history(self):
        """Clear violation history"""
        self.violation_history.clear()
        logger.info("Violation history cleared")


# Global safety pipeline instance
_safety_pipeline: Optional[SafetyPipeline] = None


def get_safety_pipeline(safety_level: SafetyLevel = SafetyLevel.MODERATE) -> SafetyPipeline:
    """Get or create global safety pipeline singleton"""
    global _safety_pipeline
    
    if _safety_pipeline is None:
        _safety_pipeline = SafetyPipeline(safety_level=safety_level)
    elif _safety_pipeline.safety_level != safety_level:
        _safety_pipeline.set_safety_level(safety_level)
    
    return _safety_pipeline
