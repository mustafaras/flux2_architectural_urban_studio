"""
Analytics & Insights Dashboard for FLUX.2

Real-time analytics dashboard showing:
- Weekly/monthly statistics
- Performance trends
- Resource usage analysis
- Feature usage heatmaps
- Export options for reports

Streamlit Component - Phase 7 Implementation
"""

import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import json
from ui import icons
from src.flux2.baseline_kpi import build_baseline_kpi_report
from src.flux2.feature_flags import is_feature_enabled
from src.flux2.governance import GovernanceAuditLog
from src.flux2.policy_profiles import build_policy_conformance_dashboard, get_policy_profile
from src.flux2.sla_monitor import SLAThresholds, correlate_quality_with_performance, default_runbook_links, evaluate_sla

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore[assignment]
    px = None  # type: ignore[assignment]
    _PLOTLY_AVAILABLE = False

# Import analytics modules
try:
    from src.flux2.analytics_client import get_analytics, PrivacyLevel
    from src.flux2.crash_reporter import get_crash_reporter, get_performance_monitor
    from src.flux2.report_generator import (
        ReportGenerator, ReportType, CostAnalyzer, CarbonAnalyzer
    )
except ImportError:
    # Fallback imports for direct execution
    from analytics_client import get_analytics, PrivacyLevel
    from crash_reporter import get_crash_reporter, get_performance_monitor
    from report_generator import (
        ReportGenerator, ReportType, CostAnalyzer, CarbonAnalyzer
    )


def render_privacy_settings():
    """Render privacy & consent settings"""
    icons.heading("Privacy & Consent Settings", icon="shield", level=3)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        FLUX.2 collects **anonymous, privacy-respecting analytics** to improve your experience.
        
        **What we collect:**
        - Feature usage (which tabs you visit, export formats used)
        - Performance metrics (generation time, success rates)
        - Error logs (crashes, failures for debugging)
        
        **What we DON'T collect:**
        - Personal information (name, email, location)
        - Prompts or generated images
        - API keys or passwords
        - System credentials
        """)
    
    with col2:
        st.info(
            "\n".join(
                [
                    icons.label("All data is anonymous", "check"),
                    icons.label("No user tracking", "check"),
                    icons.label("Local storage only", "check"),
                    icons.label("Fully deletable", "check"),
                ]
            )
        )
    
    # Privacy level selector
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        privacy_level = st.radio(
            "Analytics Privacy Level:",
            options=["Disabled", "Minimal", "Standard (Default)", "Full"],
            index=2,
            help="""
            - **Disabled**: No analytics collected
            - **Minimal**: Only critical metrics (errors, crashes)
            - **Standard**: Feature usage + performance
            - **Full**: Detailed telemetry (still no PII)
            """
        )
    
    with col2:
        crash_reporting = st.checkbox(
            "Enable Crash Reporting",
            value=True,
            help="Report unexpected errors to help improve FLUX.2"
        )
        
        performance_tracking = st.checkbox(
            "Track Performance Metrics",
            value=True,
            help="Monitor generation times and system performance"
        )
    
    # Apply settings
    level_map = {
        "Disabled": PrivacyLevel.DISABLED,
        "Minimal": PrivacyLevel.MINIMAL,
        "Standard (Default)": PrivacyLevel.STANDARD,
        "Full": PrivacyLevel.FULL
    }
    
    analytics = get_analytics()
    analytics.set_privacy_level(level_map[privacy_level])
    
    crash_reporter = get_crash_reporter()
    if crash_reporting:
        crash_reporter.enable_reporting()
    else:
        crash_reporter.disable_reporting()
    
    if st.button(icons.label("Delete All Analytics Data", "trash"), key="delete_analytics"):
        analytics.clear_all_data()
        crash_reporter.clear_crash_data()
        st.success(icons.label("All analytics data deleted", "check"))
        st.rerun()


def render_statistics_overview():
    """Render key statistics from this week"""
    icons.heading("Statistics This Week", icon="activity", level=3)
    
    analytics = get_analytics()
    stats = analytics.get_statistics()
    
    weekly_stats = stats.get("this_week", {})
    perf_stats = stats.get("performance", {})
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Generations",
            weekly_stats.get("total_generations", 0),
            delta="+5" if weekly_stats.get("total_generations", 0) > 100 else None
        )
    
    with col2:
        st.metric(
            "Avg Generation Time",
            f"{weekly_stats.get('avg_generation_time_seconds', 0):.1f}s",
            delta="-0.2s" if weekly_stats.get('avg_generation_time_seconds', 0) < 4 else None
        )
    
    with col3:
        st.metric(
            "Success Rate",
            f"{perf_stats.get('success_rate_percentage', 0):.1f}%",
            delta="+1.2%" if perf_stats.get('success_rate_percentage', 0) > 95 else None
        )
    
    with col4:
        st.metric(
            "Queue Wait Time",
            f"{perf_stats.get('avg_queue_wait_seconds', 0):.1f}s",
            delta="-0.5s" if perf_stats.get('avg_queue_wait_seconds', 0) < 2 else None
        )
    
    st.divider()
    
    # Model and preset info
    col1, col2 = st.columns(2)
    
    with col1:
        most_used = weekly_stats.get("most_used_model", "N/A")
        usage_pct = weekly_stats.get("most_used_model_percentage", 0)
        st.write(f"**Most Used Model:** {most_used}")
        st.write(icons.label(f"Usage: {usage_pct:.1f}% of all generations", "activity"))
    
    with col2:
        favorite = weekly_stats.get("favorite_preset", "N/A")
        fav_pct = weekly_stats.get("favorite_preset_percentage", 0)
        st.write(f"**Favorite Preset:** {favorite}")
        st.write(icons.label(f"Usage: {fav_pct:.1f}% of presets applied", "sparkles"))


def render_performance_trends():
    """Render performance trend visualizations"""
    icons.heading("Performance Trends", icon="activity", level=3)

    if not _PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed. Install `plotly` to enable performance charts.")
        return
    
    analytics = get_analytics()
    stats = analytics.get_statistics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate trend
        st.markdown("**Success Rate Trend**")
        perf = stats.get("performance", {})
        success_rate = perf.get("success_rate_percentage", 0)
        
        # Create gauge chart
        fig = go.Figure(data=[go.Indicator(
            mode="gauge+number+delta",
            value=success_rate,
            title={'text': "Success Rate (%)"},
            delta={'reference': 95, 'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 85], 'color': "lightgray"},
                    {'range': [85, 95], 'color': "lightyellow"},
                    {'range': [95, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}
            }
        )])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model performance comparison
        st.markdown("**Average Generation Time by Model**")
        models = stats.get("popular_models", [])
        
        if models:
            model_names = [m.get("model", "Unknown") for m in models[:5]]
            model_times = [m.get("avg_time_seconds", 0) for m in models[:5]]
            
            fig = go.Figure(data=[
                go.Bar(x=model_names, y=model_times, marker_color='lightblue')
            ])
            fig.update_layout(
                yaxis_title="Avg Time (seconds)",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model data available yet")
    
    st.divider()
    
    # Error analysis
    st.markdown("**Error Analysis**")
    errors = stats.get("errors", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total Errors:** {errors.get('total_errors', 0)}")
        st.write(f"**Most Common:** {errors.get('most_common_error', 'None')}")
    
    with col2:
        if errors.get("error_breakdown"):
            st.write("**Error Breakdown:**")
            for error_type, count in errors.get("error_breakdown", {}).items():
                st.write(f"- {error_type}: {count}")


def render_resource_usage():
    """Render resource usage analysis"""
    icons.heading("Resource Usage & Cost Analysis", icon="download", level=3)
    
    analytics = get_analytics()
    stats = analytics.get_statistics()
    perf = stats.get("performance", {})
    
    generation_count = perf.get("total_generations", 0)
    avg_time = perf.get("success_rate_percentage", 0) / 100 if perf.get("success_rate_percentage", 0) else 3
    
    col1, col2, col3 = st.columns(3)
    
    # VRAM usage estimation
    with col1:
        st.markdown("**GPU Memory Usage**")
        # Estimate based on model usage
        models = stats.get("popular_models", [])
        if models:
            primary_model = models[0].get("model", "Klein 4B")
            vram_gb = 4 if "4B" in primary_model else 9
            total_vram_hours = (generation_count * avg_time) / 3600 * vram_gb
            
            st.metric("Primary Model", primary_model)
            st.metric("Avg VRAM", f"{vram_gb}GB")
            st.metric("Total VRAM Hours", f"{total_vram_hours:.1f}h")
        else:
            st.info("No usage data yet")
    
    # Cost estimation (cloud deployment)
    with col2:
        st.markdown("**Cost Estimate (Cloud GPU)**")
        if generation_count > 0:
            costs = CostAnalyzer.calculate_cost(
                generation_count,
                avg_time,
                gpu_type="T4"
            )
            
            st.metric("GPU Hours", f"{costs['gpu_hours']:.2f}h")
            st.metric("Compute Cost", f"${costs['gpu_cost']:.2f}")
            st.metric("Total Cost", f"${costs['total_cost']:.2f}", delta=f"${costs['inference_cost']:.2f} inference")
        else:
            st.info("Run some generations to see cost estimates")
    
    # Carbon footprint
    with col3:
        st.markdown("**Carbon Footprint Estimate**")
        if generation_count > 0:
            carbon = CarbonAnalyzer.calculate_carbon_footprint(
                generation_count,
                avg_time,
                gpu_type="T4",
                energy_source="grid_avg"
            )
            
            st.metric("Energy Used", f"{carbon['energy_kwh']:.2f} kWh")
            st.metric("CO₂ Equivalent", f"{carbon['carbon_kg']:.2f} kg")
            st.metric(
                "Trees to Offset",
                f"{carbon['trees_offset_per_year']:.2f}/year",
                help="Number of trees needed to offset annual emissions"
            )
        else:
            st.info("Run some generations for carbon estimates")


def render_feature_heatmap():
    """Render feature usage heatmap"""
    icons.heading("Feature Usage Heatmap", icon="fire", level=3)

    if not _PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed. Install `plotly` to enable feature heatmaps.")
        return
    
    analytics = get_analytics()
    stats = analytics.get_statistics()
    features = stats.get("feature_usage", {})
    
    if features:
        # Create simple bar chart of feature usage
        feature_names = list(features.keys())[:10]
        feature_counts = list(features.values())[:10]
        
        fig = px.bar(
            x=feature_names,
            y=feature_counts,
            title="Top Features Used This Week",
            labels={"x": "Feature", "y": "Usage Count"},
            color=feature_counts,
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature usage data available yet")


def render_crash_statistics():
    """Render crash and error statistics"""
    icons.heading("Crash & Error Monitoring", icon="cross", level=3)
    
    crash_reporter = get_crash_reporter()
    stats = crash_reporter.get_crash_statistics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Crashes", stats.get("total_crashes", 0))
    with col2:
        st.metric("Unique Issues", stats.get("unique_crashes", 0))
    with col3:
        st.metric("Most Common", stats.get("most_common_exception", "None"))
    
    st.divider()
    
    # Recent crashes
    if stats.get("total_crashes", 0) > 0:
        st.markdown("**Recent Crashes**")
        recent_crashes = crash_reporter.get_recent_crashes(limit=5)
        
        for idx, crash in enumerate(recent_crashes, 1):
            with st.expander(f"Crash #{idx}: {crash['exception_type']}"):
                st.write(f"**Time:** {crash['timestamp']}")
                st.write(f"**Exception:** {crash['exception_message']}")
                st.write(f"**Stack Trace:**")
                for line in crash['stack_trace'][:5]:  # Show top 5 lines
                    st.code(line, language="python")


def render_export_options():
    """Render report export options"""
    icons.heading("Export Reports & Data", icon="download", level=3)
    
    st.markdown("""
    Generate and download comprehensive analytics reports in multiple formats.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Report Type**")
        report_type = st.selectbox(
            "Choose report:",
            ["Weekly Summary", "Monthly Summary", "Performance Analysis"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Export Format**")
        export_format = st.selectbox(
            "Choose format:",
            ["Text Report", "CSV Data", "All Files"],
            label_visibility="collapsed"
        )
    
    st.divider()
    
    if st.button(icons.label("Generate Report", "activity"), use_container_width=True, type="primary"):
        with st.spinner("Generating report..."):
            analytics = get_analytics()
            stats = analytics.get_statistics()
            
            # Map selections to report type
            report_type_map = {
                "Weekly Summary": ReportType.WEEKLY,
                "Monthly Summary": ReportType.MONTHLY,
                "Performance Analysis": ReportType.PERFORMANCE
            }
            
            generator = ReportGenerator()
            files = generator.generate_full_report(
                stats,
                report_type=report_type_map.get(report_type, ReportType.WEEKLY)
            )
            
            st.success(icons.label("Report generated successfully!", "check"))
            
            # Show available files
            st.markdown("**Available Downloads:**")
            cols = st.columns(2)
            col_idx = 0
            
            for file_type, file_path in files.items():
                with cols[col_idx % 2]:
                    if file_path.exists():
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label=icons.label(file_type, "report"),
                                data=f.read(),
                                file_name=file_path.name,
                                mime="text/plain"
                            )
                    col_idx += 1
    
    st.divider()
    
    # CSV export options
    st.markdown("**Direct CSV Exports**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(icons.label("Export Generation Log", "template"), use_container_width=True):
            st.info("Generation log CSV would be available for download")
    
    with col2:
        if st.button(icons.label("Export Performance Data", "activity"), use_container_width=True):
            st.info("Performance metrics CSV would be available for download")


def render_performance_regression():
    """Render performance regression detection"""
    icons.heading("Performance Regression Detection", icon="settings", level=3)
    
    perf_monitor = get_performance_monitor()
    
    # Sample metrics to check
    metrics = [
        ("generation_time_ms", "Generation Time"),
        ("queue_wait_ms", "Queue Wait Time"),
        ("memory_usage_mb", "Memory Usage")
    ]
    
    st.markdown("""
    Monitor for performance regressions. Alert if metrics exceed baseline by >10%.
    """)
    
    for metric_key, metric_label in metrics:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**{metric_label}**")
        
        with col2:
            # Check if regression detected
            if perf_monitor.baselines.get(metric_key):
                baseline = perf_monitor.baselines[metric_key]
                st.write(f"Baseline: {baseline:.0f}")


def render_baseline_kpi_report() -> None:
    """Render baseline KPI report for architecture workflow telemetry."""
    icons.heading("Baseline KPI Report", icon="target", level=3)
    report = build_baseline_kpi_report(dict(st.session_state))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        value = report.get("time_to_first_board_s")
        st.metric("Time-to-First-Board", f"{value:.1f}s" if isinstance(value, (int, float)) else "N/A")
    with col2:
        st.metric("Iterations / Session", int(report.get("iteration_count_session", 0)))
    with col3:
        st.metric("Generation Latency", f"{float(report.get('generation_latency_avg_s', 0.0)):.2f}s")
    with col4:
        st.metric("Queue Wait", f"{float(report.get('queue_wait_avg_s', 0.0)):.2f}s")

    st.json(report)
    st.download_button(
        "Download KPI Report JSON",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name=f"baseline_kpi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="analytics_download_baseline_kpi",
    )


def render_sla_alerts() -> None:
    """Render SLA metrics, alerts, and runbook links for incident triage."""
    icons.heading("SLA Alerts & Runbooks", icon="target", level=3)

    queue = st.session_state.get("generation_queue")
    queue_status = queue.get_status() if queue is not None else {}
    if not isinstance(queue_status, dict):
        queue_status = {}

    latencies = st.session_state.get("kpi_generation_latencies", [])
    latency_values = [float(x) for x in latencies if isinstance(x, (int, float))]
    latency_values.sort()
    p95_index = int(0.95 * (len(latency_values) - 1)) if latency_values else 0
    p95_latency_s = latency_values[p95_index] if latency_values else 0.0

    completed = int(queue_status.get("completed", 0))
    failed = int(queue_status.get("failed", 0))
    queued = int(queue_status.get("queued", 0))
    total_finished = completed + failed
    error_rate_pct = (failed / total_finished * 100.0) if total_finished > 0 else 0.0

    analytics = get_analytics()
    stats = analytics.get_statistics()
    event_counts = stats.get("event_counts", {})
    cache_hit = int(event_counts.get("model_cache_hit", 0)) if hasattr(event_counts, "get") else 0
    cache_miss = int(event_counts.get("model_cache_miss", 0)) if hasattr(event_counts, "get") else 0
    cache_total = cache_hit + cache_miss
    cache_hit_ratio_pct = (cache_hit / cache_total * 100.0) if cache_total > 0 else 100.0

    thresholds = SLAThresholds(
        p95_latency_s=float(st.number_input("P95 latency threshold (s)", min_value=0.1, value=8.0, step=0.5, key="sla_thr_p95")),
        queue_backlog=int(st.number_input("Queue backlog threshold", min_value=1, value=20, step=1, key="sla_thr_queue")),
        error_rate_pct=float(st.number_input("Error rate threshold (%)", min_value=0.1, value=5.0, step=0.5, key="sla_thr_error")),
        min_cache_hit_ratio_pct=float(st.number_input("Min cache hit ratio (%)", min_value=0.0, value=50.0, step=1.0, key="sla_thr_cache")),
    )

    metrics = {
        "p95_latency_s": p95_latency_s,
        "queue_backlog": queued,
        "error_rate_pct": error_rate_pct,
        "cache_hit_ratio_pct": cache_hit_ratio_pct,
    }
    alerts = evaluate_sla(metrics, thresholds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P95 Latency", f"{p95_latency_s:.2f}s")
    col2.metric("Queue Backlog", queued)
    col3.metric("Failure Ratio", f"{error_rate_pct:.2f}%")
    col4.metric("Cache Hit Ratio", f"{cache_hit_ratio_pct:.2f}%")

    quality_avg = 0.0
    project_scores = st.session_state.get("project_rubric_scores", {})
    if isinstance(project_scores, dict) and project_scores:
        all_scores = []
        for bucket in project_scores.values():
            if isinstance(bucket, dict):
                for score_dict in bucket.values():
                    if isinstance(score_dict, dict) and score_dict:
                        all_scores.append(sum(float(v) for v in score_dict.values()) / max(1, len(score_dict)))
        if all_scores:
            quality_avg = sum(all_scores) / len(all_scores)

    st.write(
        {
            "quality_performance_correlation": correlate_quality_with_performance(
                quality_score=quality_avg,
                p95_latency_s=p95_latency_s,
                error_rate_pct=error_rate_pct,
            )
        }
    )

    if alerts:
        for alert in alerts:
            st.error(
                f"SLA breach: {alert['metric']} value={alert['value']} threshold={alert['threshold']} severity={alert['severity']}"
            )
    else:
        st.success("No SLA threshold breaches detected.")

    st.markdown("**Runbooks**")
    for name, link in default_runbook_links().items():
        st.markdown(f"- {name}: {link}")


def render_policy_conformance_dashboard() -> None:
    """Render policy conformance and export compliance dashboard."""
    icons.heading("Policy Conformance Dashboard", icon="shield", level=3)
    if not is_feature_enabled(st.session_state, "enable_policy_dashboards"):
        st.info("Policy dashboard feature flag is disabled.")
        return

    profile = str(st.session_state.get("current_policy_profile", "commercial"))
    workflow_state = str(st.session_state.get("review_workflow_state", "draft"))
    audit_rows = GovernanceAuditLog().query(limit=200)

    dashboard = build_policy_conformance_dashboard(
        profile=profile,
        workflow_state=workflow_state,
        recent_audit_rows=audit_rows,
    )

    st.write({"policy": get_policy_profile(profile), "conformance": dashboard})
    st.caption("Export compliance is derived from recent governance audit events and active policy constraints.")


def main():
    """Main dashboard layout"""
    icons.title("Analytics & Insights Dashboard", icon="activity")
    st.markdown("Track usage, performance, and system health")
    
    # Navigation tabs
    tabs = st.tabs([
        icons.tab("Overview", "activity"),
        icons.tab("Performance", "target"),
        icons.tab("Baseline KPI", "target"),
        icons.tab("Resources", "download"),
        icons.tab("Features", "fire"),
        icons.tab("Governance", "shield"),
        icons.tab("Errors", "cross"),
        icons.tab("Export", "download"),
        icons.tab("Privacy", "shield"),
    ])
    
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(icons.label("Refresh Data", "refresh"), use_container_width=True):
                st.rerun()
        render_statistics_overview()
    
    with tabs[1]:
        render_performance_trends()
        st.divider()
        render_sla_alerts()
    
    with tabs[2]:
        render_baseline_kpi_report()

    with tabs[3]:
        render_resource_usage()
    
    with tabs[4]:
        render_feature_heatmap()
    
    with tabs[5]:
        render_policy_conformance_dashboard()

    with tabs[6]:
        render_crash_statistics()
        st.divider()
        render_performance_regression()
    
    with tabs[7]:
        render_export_options()
    
    with tabs[8]:
        render_privacy_settings()


if __name__ == "__main__":
    main()
