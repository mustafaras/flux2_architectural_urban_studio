from __future__ import annotations

import argparse
import json
from pathlib import Path

from flux2.benchmark_harness import (
    build_release_quality_metrics,
    build_release_quality_summary,
    compare_quality_regression,
    load_prompt_suite,
    validate_prompt_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run release-candidate quality harness for architecture/urban prompts.")
    parser.add_argument("--suite", default="benchmarks/release_candidate_prompt_suite.json")
    parser.add_argument("--baseline", default="benchmarks/release_quality_baseline.json")
    parser.add_argument("--output-json", default="reports/release_quality_metrics.json")
    parser.add_argument("--output-md", default="reports/release_quality_summary.md")
    parser.add_argument("--threshold-pct", type=float, default=3.0)
    args = parser.parse_args()

    suite = load_prompt_suite(args.suite)
    ok, issues = validate_prompt_suite(suite, minimum=10)
    if not ok:
        print("Prompt suite validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    current = build_release_quality_metrics(suite)

    baseline_path = Path(args.baseline)
    baseline = None
    comparison = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = compare_quality_regression(baseline, current, max_drop_pct=float(args.threshold_pct))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "current": current,
        "baseline": baseline,
        "comparison": comparison,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        build_release_quality_summary(current=current, baseline=baseline, comparison=comparison),
        encoding="utf-8",
    )

    if comparison and bool(comparison.get("regression_detected", False)):
        print("Release quality regressions detected")
        return 1

    print("Release quality harness completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
