"""
Export, Sharing & Publishing Dashboard.
Multi-format export, social media optimization, cloud storage, and project management.
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from src.flux2.export_manager import (
    get_export_manager, ExportSettings, ExportFormat, SocialPlatform, PLATFORM_SPECS
)
from src.flux2.cloud_storage import (
    get_cloud_storage_manager, GoogleDriveConfig, S3Config, 
    AzureBlobConfig, WebDAVConfig
)
from src.flux2.project_manager import (
    get_project_manager, GenerationSnapshot, CollectionType
)
from src.flux2.annotations import build_annotation_summary
from src.flux2.explainability import build_confidence_labels, build_why_card
from src.flux2.feature_flags import is_feature_enabled
from src.flux2.governance import GovernanceAction, WorkflowState, can_perform_action, get_retention_profile
from src.flux2.governance_artifacts import attach_signed_manifest
from src.flux2.policy_profiles import check_export_policy, check_share_link_policy, get_policy_profile
from ui import icons
from ui.components_advanced import metadata_viewer


def _resolve_export_source_image() -> Optional[Path]:
    """Resolve latest generated image to a concrete file path for export."""
    output_dir = Path(st.session_state.get("output_dir", "outputs"))
    temp_dir = output_dir / "_export_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    current_image = st.session_state.get("current_image")
    if current_image is not None:
        temp_path = temp_dir / "latest_generation.png"
        current_image.save(temp_path, format="PNG")
        return temp_path

    history = st.session_state.get("generation_history", [])
    if history and isinstance(history, list):
        first = history[0] if history else {}
        payload = first.get("image_bytes") if isinstance(first, dict) else None
        if payload:
            temp_path = temp_dir / "latest_history.png"
            temp_path.write_bytes(payload)
            return temp_path

    return None


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".avif": "image/avif",
        ".gif": "image/gif",
        ".pdf": "application/pdf",
        ".json": "application/json",
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".csv": "text/csv",
    }
    return mapping.get(ext, "application/octet-stream")


def render_export_format_selector(key_prefix: str = "format"):
    """Render export format selection interface."""
    icons.heading("Export Format", icon="upload", level=3)
    
    categories = {
        "Images": ["PNG", "JPG", "WebP", "AVIF"],
        "Animations": ["GIF (animated)", "MP4 (video)", "WebP Sequence"],
        "Data": ["JSON (metadata)", "YAML (settings)", "CSV (batch log)"],
        "Print": ["PDF (high-res)", "Poster A4", "Poster A3"]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_category = st.selectbox(
            "Format Category",
            list(categories.keys()),
            help="Choose format category",
            key=f"{key_prefix}_category"
        )
    
    with col2:
        selected_format = st.selectbox(
            "Format",
            categories[selected_category],
            help="Choose specific format",
            key=f"{key_prefix}_format"
        )
    
    return selected_format


def render_export_settings(selected_format: str, key_prefix: str = "settings") -> ExportSettings:
    """Render export settings based on format."""
    icons.heading("Export Settings", icon="settings", level=3)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quality = st.slider(
            "Quality/Compression",
            0, 100, 95,
            help="Higher = better quality but larger file",
            key=f"{key_prefix}_quality"
        )
    
    with col2:
        include_metadata = st.checkbox(
            "Include Metadata",
            value=True,
            help="Embed generation info in file",
            key=f"{key_prefix}_metadata"
        )
    
    with col3:
        watermark = st.checkbox(
            "Add Watermark",
            value=False,
            help="Visible attribution watermark",
            key=f"{key_prefix}_watermark"
        )
    
    # Color profile
    col1, col2 = st.columns(2)
    
    with col1:
        color_profile = st.selectbox(
            "Color Profile",
            ["sRGB", "DisplayP3"],
            help="Color space for image",
            key=f"{key_prefix}_color_profile"
        )
    
    with col2:
        if watermark:
            watermark_placement = st.selectbox(
                "Watermark Placement",
                ["bottom_right", "bottom_left", "top_right", "top_left"],
                help="Where to place watermark",
                key=f"{key_prefix}_watermark_placement"
            )
        else:
            watermark_placement = "bottom_right"
    
    return ExportSettings(
        format=selected_format,
        quality=quality,
        metadata=include_metadata,
        watermark=watermark,
        watermark_placement=watermark_placement,
        color_profile=color_profile
    )


def render_batch_export(key_prefix: str = "batch"):
    """Render batch export interface."""
    icons.heading("Batch Export", icon="zip", level=3)
    
    st.markdown("""
    **Naming Template Variables:**
    - `{prompt}` - First 20 chars of prompt
    - `{date}` - YYYYMMDD format
    - `{model}` - Model name
    - `{seed}` - Generation seed
    - `{index}` - Sequential #001, #002, etc
    """)
    
    naming_template = st.text_input(
        "Naming Template",
        value="flux_{model}_{date}_{index}",
        help="Pattern for batch file names",
        key=f"{key_prefix}_naming_template"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        batch_count = st.number_input(
            "Batch Count",
            min_value=1, max_value=100, value=10,
            help="Number of generations to export",
            key=f"{key_prefix}_count"
        )
    
    with col2:
        zip_output = st.checkbox(
            "Create ZIP",
            value=True,
            help="Bundle all files into ZIP archive",
            key=f"{key_prefix}_zip_output"
        )
    
    with col3:
        namespace = st.text_input(
            "Subfolder",
            value="",
            placeholder="e.g., proj_sunset",
            help="Organize in subdirectory",
            key=f"{key_prefix}_namespace"
        )
    
    return {
        "naming_template": naming_template,
        "batch_count": batch_count,
        "zip_output": zip_output,
        "namespace": namespace
    }


def render_social_media_optimization(key_prefix: str = "social"):
    """Render social media platform optimizer."""
    icons.heading("Social Media Optimization", icon="phone", level=3)
    
    platforms = {
        "Twitter/X": {"max_size": "5 MB", "aspect": "1:1, 16:9", "min_width": "400px"},
        "Instagram": {"max_size": "8 MB", "aspect": "1:1, 4:5", "min_width": "1080x1350px"},
        "Pinterest": {"max_size": "10 MB", "aspect": "2:3, 1:1", "min_width": "1000px"},
        "Discord": {"max_size": "25 MB", "aspect": "16:9, 1:1", "min_width": "512px"}
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_platforms = st.multiselect(
            "Select Platforms",
            list(platforms.keys()),
            default=["Twitter/X"],
            help="Choose platforms to optimize for",
            key=f"{key_prefix}_platforms"
        )
    
    # Show platform specs
    for platform in selected_platforms:
        spec = platforms[platform]
        st.caption(f"**{platform}** - {spec['max_size']}, {spec['aspect']}, min {spec['min_width']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            ["1:1", "16:9", "4:5", "2:3"],
            help="Image aspect ratio",
            key=f"{key_prefix}_aspect_ratio"
        )
    
    with col2:
        add_caption = st.checkbox(
            "Add Caption",
            value=False,
            help="Include text caption",
            key=f"{key_prefix}_add_caption"
        )
    
    caption = ""
    if add_caption:
        caption = st.text_area(
            "Caption Text",
            max_chars=500,
            placeholder="Enter post caption...",
            key=f"{key_prefix}_caption"
        )
    
    return {
        "platforms": selected_platforms,
        "aspect_ratio": aspect_ratio,
        "caption": caption
    }


def render_cloud_storage_integration(key_prefix: str = "cloud"):
    """Render cloud storage configuration."""
    icons.heading("Cloud Storage Integration", icon="cloud", level=3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Auto-Sync Settings**")
        
        auto_sync = st.checkbox("Enable Auto-Sync", value=False, key=f"{key_prefix}_auto_sync")
        
        if auto_sync:
            sync_interval = st.slider(
                "Sync Interval",
                min_value=1, max_value=120, value=5,
                help="Minutes between syncs",
                key=f"{key_prefix}_sync_interval"
            )
        else:
            sync_interval = 5
    
    with col2:
        st.markdown("**Cloud Providers**")
        
        providers = st.multiselect(
            "Select Providers",
            ["Google Drive", "AWS S3", "Azure Blob", "WebDAV"],
            help="Where to sync generations",
            key=f"{key_prefix}_providers"
        )
    
    # Provider-specific settings
    if providers:
        st.markdown("**Provider Configuration**")
        
        for provider in providers:
            with st.expander(f"Configure {provider}"):
                if provider == "Google Drive":
                    st.text_input("API Key", type="password", key=f"{key_prefix}_gdrive_api_key")
                    st.text_input("Folder ID", key=f"{key_prefix}_gdrive_folder_id")
                    st.checkbox("Include Metadata", key=f"{key_prefix}_gdrive_include_metadata")
                
                elif provider == "AWS S3":
                    st.text_input("Access Key", type="password", key=f"{key_prefix}_s3_access_key")
                    st.text_input("Secret Key", type="password", key=f"{key_prefix}_s3_secret_key")
                    st.text_input("Bucket Name", key=f"{key_prefix}_s3_bucket")
                    st.selectbox("Region", ["us-east-1", "eu-west-1", "ap-southeast-1"], key=f"{key_prefix}_s3_region")
                    st.slider("Auto-Delete After (days)", 0, 365, 90, key=f"{key_prefix}_s3_lifecycle_days")
                
                elif provider == "Azure Blob":
                    st.text_input("Account Name", key=f"{key_prefix}_azure_account_name")
                    st.text_input("Account Key", type="password", key=f"{key_prefix}_azure_account_key")
                    st.text_input("Container Name", key=f"{key_prefix}_azure_container_name")
                    st.selectbox("Storage Tier", ["Hot", "Cool", "Archive"], key=f"{key_prefix}_azure_tier")
                
                elif provider == "WebDAV":
                    st.text_input("Server URL", key=f"{key_prefix}_webdav_url")
                    st.text_input("Username", key=f"{key_prefix}_webdav_username")
                    st.text_input("Password", type="password", key=f"{key_prefix}_webdav_password")
                    st.checkbox("Verify SSL", value=True, key=f"{key_prefix}_webdav_verify_ssl")
    
    return {
        "auto_sync": auto_sync,
        "sync_interval": sync_interval,
        "providers": providers
    }


def render_project_management(key_prefix: str = "project"):
    """Render project and collection management."""
    icons.heading("Project Management", icon="folder", level=3)
    
    project_manager = get_project_manager()
    projects = project_manager.list_projects()
    
    # Project selector or create new
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if projects:
            project_names = [p.name for p in projects]
            selected_project = st.selectbox(
                "Current Project",
                project_names,
                help="Select project to export from",
                key=f"{key_prefix}_current_project"
            )
            current_project = next(p for p in projects if p.name == selected_project)
        else:
            current_project = None
            st.info("No projects yet. Create one in the Settings tab.")
    
    with col2:
        if st.button(icons.label("New Project", "plus"), key="export_new_project"):
            st.session_state.show_new_project_dialog = True
    
    if current_project:
        # Show project stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Generations", current_project.total_generations)
        with col2:
            st.metric("Starred", current_project.total_starred)
        with col3:
            st.metric("Collections", len(current_project.collections))
        with col4:
            st.metric("Share Links", len([l for l in current_project.sharing_links.values() 
                                          if l.expires > datetime.now()]))
        
        # Collections management
        st.markdown("**Collections**")
        
        if current_project.collections:
            for collection in current_project.collections.values():
                col_name, col_count, col_action = st.columns([2, 1, 1])
                
                with col_name:
                    st.caption(icons.label(f"{collection.name} ({collection.type.value})", "folder"))
                with col_count:
                    st.caption(f"{len(collection.generation_ids)} items")
                with col_action:
                    if st.button("Export", key=f"export_coll_{collection.id}"):
                        st.session_state.export_collection_id = collection.id
        else:
            st.caption("No collections in this project")


def render_sharing_links(key_prefix: str = "sharing"):
    """Render sharing and publishing options."""
    icons.heading("Sharing & Publishing", icon="link", level=3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Create Share Link**")
        policy_profile = str(st.session_state.get("current_policy_profile", "commercial"))
        policy_ok, policy_reason = check_share_link_policy(profile=policy_profile)
        
        expire_days = st.slider(
            "Link Expiration",
            min_value=1, max_value=365,
            value=30,
            help="Days until link expires",
            key=f"{key_prefix}_expire_days"
        )
        
        max_views = st.number_input(
            "Max Views (0 = unlimited)",
            min_value=0, max_value=10000,
            value=0,
            help="Limit number of views",
            key=f"{key_prefix}_max_views"
        )
        
        col_meta, col_prompts = st.columns(2)
        with col_meta:
            include_metadata = st.checkbox("Include Metadata", value=True, key=f"{key_prefix}_include_metadata")
        with col_prompts:
            include_prompts = st.checkbox("Include Prompts", value=False, key=f"{key_prefix}_include_prompts")
        if not get_policy_profile(policy_profile).get("allow_prompt_exports", True):
            include_prompts = False
            st.caption("Policy enforces prompt export disabled.")
        
        if st.button("Generate Share Link", key=f"{key_prefix}_generate", disabled=not policy_ok):
            project_manager = get_project_manager()
            if project_manager.current_project:
                share_link = project_manager.create_share_link(
                    project_manager.current_project.id,
                    expires_in_days=expire_days,
                    max_views=max_views if max_views > 0 else None,
                    includes_metadata=include_metadata,
                    includes_prompts=include_prompts
                )
                
                if share_link:
                    st.success("Share link created!")
                    st.code(share_link.access_token, language="text")
        if not policy_ok:
            st.warning(f"Policy block: {policy_reason}")
    
    with col2:
        st.markdown("**Active Share Links**")
        
        project_manager = get_project_manager()
        if project_manager.current_project:
            links = project_manager.list_share_links(
                project_manager.current_project.id,
                active_only=True
            )
            
            for link in links:
                col_link, col_actions = st.columns([3, 1])
                with col_link:
                    st.caption(icons.label(f"{link.link_id} • {link.views_count} views", "link"))
                    st.caption(icons.label(f"Expires: {link.expires.strftime('%Y-%m-%d')}", "clock"))
                
                with col_actions:
                    if st.button("Revoke", key=f"revoke_{link.link_id}"):
                        project_manager.revoke_share_link(
                            project_manager.current_project.id,
                            link.link_id
                        )
                        st.rerun()


def render_export_history():
    """Render export history."""
    icons.heading("Export History", icon="activity", level=3)
    
    export_manager = get_export_manager()
    history = export_manager.get_export_history(limit=20)
    
    if history:
        for idx, export in enumerate(reversed(history), 1):
            with st.expander(
                f"Export #{idx} - {export.get('format', 'Unknown')} - "
                f"{export.get('timestamp', '')[:10]}"
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Format", export.get('format', 'N/A'))
                with col2:
                    st.metric("Quality", f"{export.get('quality', 'N/A')}%")
                with col3:
                    st.metric("Size", f"{export.get('file_size_mb', 0)} MB")
                
                if export.get('platform'):
                    st.caption(icons.label(f"Platform: {export['platform']}", "link"))
                
                if export.get('error'):
                    st.error(f"Error: {export['error']}")

                metadata_viewer(
                    export,
                    title="Export Record Details",
                    key_prefix=f"export_history_meta_{idx}",
                )
    else:
        st.info("No exports yet")


def main():
    """Main export dashboard."""
    icons.page_intro(
        "Export, Sharing & Publishing",
        "Export outputs, optimize for sharing, connect cloud destinations, and manage project publishing.",
        icon="download",
    )

    st.markdown("**Multi-format exports, social media optimization, cloud storage sync, and project management.**")
    st.markdown(f"- **{icons.label('Export Formats', 'upload')}**: PNG, JPG, WebP, AVIF, GIF, MP4, PDF, and data formats")
    st.markdown(f"- **{icons.label('Social Media', 'phone')}**: Auto-optimize for Twitter, Instagram, Pinterest, Discord")
    st.markdown(f"- **{icons.label('Cloud Sync', 'cloud')}**: Google Drive, AWS S3, Azure Blob, WebDAV")
    st.markdown(f"- **{icons.label('Projects', 'folder')}**: Version history, collections, starred generations")
    st.markdown(f"- **{icons.label('Sharing', 'link')}**: Expiring share links with view limits")
    
    # Initialize session state
    if 'export_tab' not in st.session_state:
        st.session_state.export_tab = "Single Export"
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        icons.tab("Single Export", "upload"),
        icons.tab("Batch Export", "zip"),
        icons.tab("Social Media", "phone"),
        icons.tab("Cloud Storage", "cloud"),
        icons.tab("Projects & Sharing", "folder"),
    ])
    
    with tab1:
        st.markdown("**Export single generation with customization**")

        role = str(st.session_state.get("current_user_role", "editor"))
        workflow_state = str(st.session_state.get("review_workflow_state", WorkflowState.DRAFT.value))
        retention_profile = str(st.session_state.get("retention_profile", "exploratory"))
        policy_profile = str(st.session_state.get("current_policy_profile", "commercial"))
        can_export = can_perform_action(role, GovernanceAction.EXPORT.value, workflow_state)
        policy_ok, policy_reason = check_export_policy(profile=policy_profile, workflow_state=workflow_state, include_prompts=False)
        can_export = can_export and policy_ok
        if not policy_ok:
            st.warning(f"Policy block: {policy_reason}")
        
        export_manager = get_export_manager()
        export_format = render_export_format_selector(key_prefix="single_format")
        settings = render_export_settings(export_format, key_prefix="single_settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(icons.label("Download", "download"), key="export_download", disabled=not can_export):
                source = _resolve_export_source_image()
                if source is None:
                    st.warning("No generated image found. Generate an image first, then export.")
                else:
                    result = export_manager.export_image(
                        source,
                        settings,
                        generation_metadata=st.session_state.get("generation_metadata"),
                    )
                    st.session_state["export_last_result"] = result
                    if result.get("success"):
                        st.success(f"Export completed: {Path(result['output_path']).name}")
                    else:
                        st.error(f"Export failed: {result.get('error', 'Unknown error')}")
        
        with col2:
            if st.button(icons.label("Upload to Cloud", "cloud"), key="export_cloud"):
                st.info("Upload started...")
        
        with col3:
            if st.button(icons.label("Share on Social", "phone"), key="export_social"):
                st.info("Preparing for social media...")

        last_result = st.session_state.get("export_last_result")
        if isinstance(last_result, dict) and last_result.get("success"):
            output_path = Path(last_result.get("output_path", ""))
            if output_path.exists():
                st.download_button(
                    label=icons.label(f"Download {output_path.name}", "download"),
                    data=output_path.read_bytes(),
                    file_name=output_path.name,
                    mime=_guess_mime(output_path),
                    key="export_last_download_button",
                )
                if last_result.get("warning"):
                    st.caption(icons.label(str(last_result["warning"]), "warning"))

                export_manifest = attach_signed_manifest(
                    {
                        "manifest_version": "export-manifest-v1",
                        "created_at": datetime.now().isoformat(),
                        "output_file": output_path.name,
                        "workflow_state": workflow_state,
                        "retention_profile": retention_profile,
                        "retention_policy": get_retention_profile(retention_profile),
                        "policy_profile": policy_profile,
                        "policy": get_policy_profile(policy_profile),
                        "generation_metadata": st.session_state.get("generation_metadata", {}),
                        "explainability": {
                            "why_card": build_why_card(st.session_state.get("generation_metadata", {}) if isinstance(st.session_state.get("generation_metadata", {}), dict) else {}),
                            "confidence": build_confidence_labels(st.session_state.get("generation_metadata", {}) if isinstance(st.session_state.get("generation_metadata", {}), dict) else {}),
                        },
                        "annotation_summary": build_annotation_summary(
                            st.session_state.get("annotation_threads", []) if isinstance(st.session_state.get("annotation_threads", []), list) else []
                        ),
                    }
                )
                st.download_button(
                    label=icons.label("Download Signed Manifest", "book"),
                    data=json.dumps(export_manifest, indent=2).encode("utf-8"),
                    file_name=f"export_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="export_manifest_download",
                )
    
    with tab2:
        st.markdown("**Export multiple generations with template naming**")
        
        batch_settings = render_batch_export(key_prefix="batch")
        export_format = render_export_format_selector(key_prefix="batch_format")
        settings = render_export_settings(export_format, key_prefix="batch_settings")
        
        if st.button(icons.label("Create Batch Export", "zip"), key="batch_export_btn"):
            st.success(f"Creating batch with {batch_settings['batch_count']} items...")
    
    with tab3:
        st.markdown("**Optimize and share directly to social media**")
        
        social_settings = render_social_media_optimization(key_prefix="social")
        
        if st.button(icons.label("Optimize for Selected Platforms", "phone"), key="social_optimize"):
            if social_settings['platforms']:
                st.success(f"Optimizing for: {', '.join(social_settings['platforms'])}")
            else:
                st.warning("Select at least one platform")
    
    with tab4:
        st.markdown("**Configure automatic cloud synchronization**")
        
        cloud_settings = render_cloud_storage_integration(key_prefix="cloud")
        
        if st.button(icons.label("Connect Providers", "cloud"), key="cloud_connect"):
            if cloud_settings['providers']:
                st.success(f"Connecting to: {', '.join(cloud_settings['providers'])}")
            else:
                st.warning("Select at least one provider")
    
    with tab5:
        render_project_management(key_prefix="project")
        st.divider()
        render_sharing_links(key_prefix="sharing")
    
    st.divider()
    render_export_history()


if __name__ == "__main__":
    main()
