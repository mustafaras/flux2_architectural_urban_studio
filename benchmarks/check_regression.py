"""
Performance regression detection script.

Compares current benchmark results against baseline and alerts if
performance has degraded more than the threshold (default 5%).
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


REGRESSION_THRESHOLD = 5.0  # 5% threshold


def load_baseline() -> Dict[str, Any]:
    """Load baseline metrics from previous run."""
    baseline_file = Path(__file__).parent / "baseline.json"
    release_baseline_file = Path(__file__).parent / "release_quality_baseline.json"
    
    if not baseline_file.exists():
        print("⚠️  No baseline metrics found - cannot detect regressions")
        return {}
    
    payload: Dict[str, Any] = {}
    with open(baseline_file, "r") as f:
        payload.update(json.load(f))

    if release_baseline_file.exists():
        with open(release_baseline_file, "r") as f:
            release_payload = json.load(f)
        if isinstance(release_payload, dict):
            quality = release_payload.get("quality", {})
            if isinstance(quality, dict):
                payload.update({f"quality_{k}": v for k, v in quality.items()})

    return payload


def load_current() -> Dict[str, Any]:
    """Load current benchmark results."""
    results_file = Path(__file__).parent.parent / "reports" / "benchmark.json"
    release_results_file = Path(__file__).parent.parent / "reports" / "release_quality_metrics.json"
    
    if not results_file.exists():
        print("⚠️  No current benchmark results found")
        return {}
    
    payload: Dict[str, Any] = {}
    with open(results_file, "r") as f:
        payload.update(json.load(f))

    if release_results_file.exists():
        with open(release_results_file, "r") as f:
            release_payload = json.load(f)
        if isinstance(release_payload, dict):
            current = release_payload.get("current", {})
            quality = current.get("quality", {}) if isinstance(current, dict) else {}
            if isinstance(quality, dict):
                payload.update({f"quality_{k}": v for k, v in quality.items()})

    return payload


def compare_metrics(baseline: Dict, current: Dict) -> Dict[str, Any]:
    """Compare baseline and current metrics, return regressions."""
    regressions = {
        "detected": False,
        "details": [],
        "threshold": REGRESSION_THRESHOLD,
    }
    
    if not baseline or not current:
        return regressions
    
    # Compare metrics
    for metric_name in baseline:
        if metric_name not in current:
            continue
        
        baseline_val = baseline[metric_name]
        current_val = current[metric_name]
        
        # Skip non-numeric values
        if not isinstance(baseline_val, (int, float)) or not isinstance(current_val, (int, float)):
            continue
        
        # Calculate percentage change
        if baseline_val == 0:
            continue
        
        change_pct = ((current_val - baseline_val) / baseline_val) * 100
        
        # For latency metrics (higher is worse), flag negative changes
        # For throughput metrics (higher is better), flag positive changes
        is_latency = "duration" in metric_name.lower() or "latency" in metric_name.lower()
        
        if is_latency and change_pct > REGRESSION_THRESHOLD:
            regressions["detected"] = True
            regressions["details"].append({
                "metric": metric_name,
                "baseline": baseline_val,
                "current": current_val,
                "change_pct": round(change_pct, 2),
                "status": "🔴 REGRESSION"
            })
        elif not is_latency and change_pct < -REGRESSION_THRESHOLD:
            regressions["detected"] = True
            regressions["details"].append({
                "metric": metric_name,
                "baseline": baseline_val,
                "current": current_val,
                "change_pct": round(change_pct, 2),
                "status": "🔴 REGRESSION"
            })
        else:
            # Show improvement
            if (is_latency and change_pct < 0) or (not is_latency and change_pct > 0):
                status = "🟢 IMPROVEMENT"
            else:
                status = "🟡 STABLE"
            
            regressions["details"].append({
                "metric": metric_name,
                "baseline": baseline_val,
                "current": current_val,
                "change_pct": round(change_pct, 2),
                "status": status
            })
    
    return regressions


def print_report(regressions: Dict[str, Any]) -> None:
    """Print regression report."""
    print("\n" + "=" * 70)
    print("PERFORMANCE REGRESSION ANALYSIS")
    print("=" * 70)
    print(f"Threshold: {regressions['threshold']}%\n")
    
    if not regressions["details"]:
        print("ℹ️  No metrics to compare\n")
        return
    
    print(f"{'Metric':<30} {'Baseline':<15} {'Current':<15} {'Change':<10} {'Status'}")
    print("-" * 70)
    
    for detail in regressions["details"]:
        metric = detail["metric"][:28]
        baseline = f"{detail['baseline']:.4f}"
        current = f"{detail['current']:.4f}"
        change = f"{detail['change_pct']:+.2f}%"
        status = detail["status"]
        
        print(f"{metric:<30} {baseline:<15} {current:<15} {change:<10} {status}")
    
    print("\n" + "=" * 70)
    
    if regressions["detected"]:
        print("⚠️  PERFORMANCE REGRESSIONS DETECTED!")
        print("\nAction items:")
        print("1. Review code changes for performance impact")
        print("2. Consider rollback or optimization")
        print("3. Update baseline if improvements are intentional")
        print("\n")
        return 1
    else:
        print("✅ No significant regressions detected")
        print("\n")
        return 0


def main():
    """Main entry point."""
    baseline = load_baseline()
    current = load_current()
    regressions = compare_metrics(baseline, current)
    
    exit_code = print_report(regressions)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
