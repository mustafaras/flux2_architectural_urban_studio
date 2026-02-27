"""
Safety Dashboard UI for FLUX.2 Professional

Real-time safety metrics, violation tracking, and appeal workflow.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

from flux2.safety_pipeline import get_safety_pipeline, SafetyLevel, SafetyRule, ViolationType
from ui.components_advanced import heatmap_display
from ui import icons


def render():
    """Render the safety dashboard page"""
    icons.page_intro(
        "Safety & Content Filtering Dashboard",
        "Review safety status, violations, custom rules, and appeals in one place.",
        icon="shield",
    )

    if not PLOTLY_AVAILABLE:
        st.warning(
            "Plotly is not installed. Interactive charts are replaced with basic Streamlit charts. "
            "Install with: pip install plotly"
        )
    
    # Get or create safety pipeline
    pipeline = get_safety_pipeline()
    
    # Sidebar: Safety configuration
    with st.sidebar:
        st.header("Safety Configuration")
        
        # Safety level selector
        current_level = pipeline.safety_level
        new_level = st.selectbox(
            "Safety Level",
            options=[level.value for level in SafetyLevel],
            index=[level.value for level in SafetyLevel].index(current_level.value),
            help="Strictness of content filtering"
        )
        
        if new_level != current_level.value:
            pipeline.set_safety_level(SafetyLevel(new_level))
            st.success(f"Safety level updated to: {new_level}")
            st.rerun()
        
        # NSFW threshold
        st.slider(
            "NSFW Threshold",
            min_value=0.5,
            max_value=1.0,
            value=pipeline.nsfw_threshold,
            step=0.05,
            help="Lower = more sensitive detection",
            key="nsfw_threshold_slider",
            disabled=True  # Display only
        )

        st.caption(
            f"Image thresholds — Skin: {pipeline.skin_tone_threshold:.2f}, Contrast: {pipeline.contrast_threshold:.2f}"
        )

        if st.button(icons.label("Reload Safety Config", "recycle"), use_container_width=True):
            pipeline.reload_config()
            st.success("Safety configuration reloaded")
            st.rerun()
        
        # Statistics refresh
        if st.button(icons.label("Refresh Statistics", "refresh"), use_container_width=True):
            st.rerun()
        
        # Clear history
        if st.button(icons.label("Clear Violation History", "trash"), use_container_width=True, type="secondary"):
            pipeline.clear_history()
            st.success("History cleared")
            st.rerun()
    
    # Main content tabs
    tab_overview, tab_violations, tab_rules, tab_appeals = st.tabs([
        icons.tab("Overview", "activity"),
        icons.tab("Violations", "violation"),
        icons.tab("Custom Rules", "template"),
        icons.tab("Appeals", "scales"),
    ])
    
    # Tab 1: Overview
    with tab_overview:
        _render_overview(pipeline)
    
    # Tab 2: Violations
    with tab_violations:
        _render_violations(pipeline)
    
    # Tab 3: Custom Rules
    with tab_rules:
        _render_custom_rules(pipeline)
    
    # Tab 4: Appeals
    with tab_appeals:
        _render_appeals(pipeline)


def _render_overview(pipeline):
    """Render overview statistics"""
    stats = pipeline.get_statistics()
    
    st.header("Safety Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Checks",
            f"{stats['total_checks']:,}",
            help="Total safety checks performed"
        )
    
    with col2:
        st.metric(
            "Total Violations",
            f"{stats['total_violations']:,}",
            delta=f"{stats['violation_rate']*100:.1f}% rate",
            delta_color="inverse",
            help="Total violations detected"
        )
    
    with col3:
        st.metric(
            "Custom Rules",
            f"{stats['enabled_rules_count']}/{stats['custom_rules_count']}",
            help="Active custom safety rules"
        )
    
    with col4:
        safety_score = (1.0 - stats['violation_rate']) * 100
        st.metric(
            "Safety Score",
            f"{safety_score:.1f}%",
            help="Overall content safety rating"
        )
    
    st.divider()
    
    # Violation breakdown chart
    if stats['violation_breakdown']:
        st.subheader("Violation Breakdown")
        
        col_chart, col_table = st.columns([2, 1])
        
        with col_chart:
            # Pie chart
            df_violations = pd.DataFrame([
                {"Type": vtype, "Count": count}
                for vtype, count in stats['violation_breakdown'].items()
            ])

            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    df_violations,
                    values="Count",
                    names="Type",
                    title="Violations by Type",
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                chart_data = df_violations.set_index("Type")
                st.bar_chart(chart_data)
        
        with col_table:
            st.dataframe(
                df_violations.sort_values("Count", ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            if stats['most_common_violation']:
                st.info(f"**Most Common:** {stats['most_common_violation']}")
    else:
        st.info("No violations recorded yet")
    
    st.divider()
    
    # Timeline chart (if violations exist)
    violations = pipeline.get_violations(limit=500)
    if violations:
        st.subheader("Violation Timeline")
        
        # Aggregate by hour
        df_timeline = pd.DataFrame(violations)
        df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])
        df_timeline['hour'] = df_timeline['timestamp'].dt.floor('H')
        
        hourly_counts = df_timeline.groupby('hour').size().reset_index(name='count')

        if PLOTLY_AVAILABLE:
            fig = px.line(
                hourly_counts,
                x='hour',
                y='count',
                title='Violations Over Time',
                labels={'hour': 'Time', 'count': 'Violations'}
            )
            fig.update_traces(line_color='#ff4b4b', fill='tozeroy')
            st.plotly_chart(fig, use_container_width=True)
        else:
            timeline_chart = hourly_counts.set_index('hour')[['count']]
            st.line_chart(timeline_chart)

    st.divider()
    st.subheader("NSFW Region Visualization")
    latest_image = st.session_state.get("current_image")
    region_candidates = []
    if violations:
        latest_violation = violations[0]
        metadata = latest_violation.get("metadata", {}) if isinstance(latest_violation, dict) else {}
        maybe_regions = metadata.get("regions", []) if isinstance(metadata, dict) else []
        if isinstance(maybe_regions, list):
            region_candidates = maybe_regions

    heatmap_display(
        image=latest_image,
        heatmap_regions=region_candidates,
        key_prefix="safety_overview_heatmap",
    )


def _render_violations(pipeline):
    """Render detailed violation log"""
    st.header("Violation Log")
    
    violations = pipeline.get_violations(limit=200)
    
    if not violations:
        st.info("No violations recorded")
        return
    
    # Filters
    col_type, col_severity, col_count = st.columns(3)
    
    with col_type:
        all_types = list(set(
            vtype 
            for v in violations 
            for vtype in v.get('violations', [])
        ))
        selected_types = st.multiselect(
            "Filter by Type",
            options=all_types,
            default=all_types
        )
    
    with col_severity:
        min_severity = st.slider(
            "Minimum Severity",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
    
    with col_count:
        display_count = st.number_input(
            "Display Count",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
    
    # Filter violations
    filtered = [
        v for v in violations
        if v['max_severity'] >= min_severity
        and any(vt in selected_types for vt in v.get('violations', []))
    ][:display_count]
    
    # Display table
    if filtered:
        df = pd.DataFrame(filtered)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['violations'] = df['violations'].apply(lambda x: ', '.join(x))
        df['severity'] = df['max_severity'].apply(lambda x: f"{x:.2f}")
        
        display_df = df[['timestamp', 'type', 'violations', 'severity', 'prompt']]
        display_df.columns = ['Time', 'Source', 'Violations', 'Severity', 'Prompt (Preview)']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Severity": st.column_config.ProgressColumn(
                    "Severity",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0
                )
            }
        )
        
        st.caption(f"Showing {len(filtered)} of {len(violations)} violations")
    else:
        st.info("No violations match the current filters")
    
    # Heatmap: Violation patterns by hour and day
    if len(violations) > 10:
        st.subheader("Violation Heatmap")
        
        df_hm = pd.DataFrame(violations)
        df_hm['timestamp'] = pd.to_datetime(df_hm['timestamp'])
        df_hm['hour'] = df_hm['timestamp'].dt.hour
        df_hm['day'] = df_hm['timestamp'].dt.day_name()
        
        heatmap_data = df_hm.groupby(['day', 'hour']).size().reset_index(name='count')
        heatmap_pivot = heatmap_data.pivot(index='day', columns='hour', values='count').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='Reds',
                hoverongaps=False
            ))

            fig.update_layout(
                title='Violation Patterns by Day and Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Day of Week',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(heatmap_pivot, use_container_width=True)


def _render_custom_rules(pipeline):
    """Render custom rules management"""
    st.header("Custom Safety Rules")
    
    # Display existing rules
    st.subheader("Active Rules")
    
    if pipeline.custom_rules:
        for idx, rule in enumerate(pipeline.custom_rules):
            with st.expander(icons.label(rule.name, "check" if rule.enabled else "cross"), expanded=False):
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.text(f"Pattern: {rule.pattern}")
                    st.text(f"Type: {rule.violation_type.value}")
                    st.text(f"Severity: {rule.severity:.2f}")
                    
                    if rule.description:
                        st.info(rule.description)
                    
                    if rule.suggested_fix:
                        st.success(icons.label(f"Suggestion: {rule.suggested_fix}", "idea"))
                
                with col_actions:
                    if st.button(icons.label("Remove", "trash"), key=f"remove_rule_{idx}"):
                        if pipeline.remove_custom_rule(rule.name):
                            st.success("Rule removed")
                            st.rerun()
                        else:
                            st.error("Failed to remove rule")
    else:
        st.info("No custom rules defined. Add one below.")
    
    st.divider()
    
    # Add new rule form
    st.subheader("Add Custom Rule")
    
    with st.form("add_rule_form"):
        col_name, col_type = st.columns(2)
        
        with col_name:
            rule_name = st.text_input(
                "Rule Name",
                placeholder="e.g., brand_protection",
                help="Unique identifier for this rule"
            )
        
        with col_type:
            rule_type = st.selectbox(
                "Violation Type",
                options=[vt.value for vt in ViolationType]
            )
        
        rule_pattern = st.text_input(
            "Regex Pattern",
            placeholder=r"\b(keyword1|keyword2)\b",
            help="Regular expression pattern to match"
        )
        
        col_severity, col_case = st.columns(2)
        
        with col_severity:
            rule_severity = st.slider(
                "Severity",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Higher = more likely to block"
            )
        
        with col_case:
            case_sensitive = st.checkbox(
                "Case Sensitive",
                value=False
            )
        
        rule_description = st.text_input(
            "Description (optional)",
            placeholder="Explain what this rule detects"
        )
        
        rule_fix = st.text_input(
            "Suggested Fix (optional)",
            placeholder="Suggest alternative phrasing"
        )
        
        submitted = st.form_submit_button(icons.label("Add Rule", "plus"), use_container_width=True)
        
        if submitted:
            if not rule_name or not rule_pattern:
                st.error("Name and pattern are required")
            else:
                # Create new rule
                new_rule = SafetyRule(
                    name=rule_name,
                    pattern=rule_pattern,
                    violation_type=ViolationType(rule_type),
                    severity=rule_severity,
                    case_sensitive=case_sensitive,
                    description=rule_description,
                    suggested_fix=rule_fix
                )
                
                if pipeline.add_custom_rule(new_rule):
                    st.success(f"Rule '{rule_name}' added successfully")
                    st.rerun()
                else:
                    st.error("Failed to add rule")


def _render_appeals(pipeline):
    """Render appeal workflow"""
    st.header("Violation Appeals")
    
    st.info("""
    **Appeal Process:**
    1. Review flagged content below
    2. Provide justification if content is legitimate
    3. Submit appeal for manual review
    4. Approved appeals update safety rules
    """)
    
    # Get recent violations that might need appeals
    violations = pipeline.get_violations(limit=50)
    high_severity = [v for v in violations if v['max_severity'] >= 0.8]
    
    if not high_severity:
        st.success("No high-severity violations to appeal")
        return
    
    st.subheader(f"High-Severity Violations ({len(high_severity)})")
    
    for idx, violation in enumerate(high_severity[:10]):  # Show top 10
        with st.expander(
            icons.label(f"{violation['type'].title()} - Severity {violation['max_severity']:.2f}", "violation"),
            expanded=idx == 0
        ):
            col_info, col_appeal = st.columns([2, 1])
            
            with col_info:
                st.text(f"Time: {violation['timestamp']}")
                st.text(f"Types: {', '.join(violation['violations'])}")
                
                if violation.get('prompt'):
                    st.text_area(
                        "Prompt Preview",
                        value=violation['prompt'],
                        height=80,
                        disabled=True,
                        key=f"prompt_preview_{idx}"
                    )
            
            with col_appeal:
                with st.form(f"appeal_form_{idx}"):
                    justification = st.text_area(
                        "Justification",
                        placeholder="Explain why this should be allowed...",
                        height=100,
                        key=f"justification_{idx}"
                    )
                    
                    appeal_category = st.selectbox(
                        "Category",
                        options=["Educational", "Artistic", "Professional", "Other"],
                        key=f"category_{idx}"
                    )
                    
                    submitted = st.form_submit_button(icons.label("Submit Appeal", "note"))
                    
                    if submitted:
                        if len(justification) < 50:
                            st.error("Justification must be at least 50 characters")
                        else:
                            # In production, this would create an appeal record
                            st.success("Appeal submitted for manual review")
                            st.info(f"Category: {appeal_category}")
    
    st.divider()
    
    # Appeal statistics
    st.subheader("Appeal Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pending Appeals", "0", help="Currently in review")
    
    with col2:
        st.metric("Approved", "0", help="Approved appeals")
    
    with col3:
        st.metric("Rejected", "0", help="Rejected appeals")
    
    st.caption(icons.label("Appeal system is a Phase 6 preview. Full workflow coming in Phase 7.", "warning"))


if __name__ == "__main__":
    render()
