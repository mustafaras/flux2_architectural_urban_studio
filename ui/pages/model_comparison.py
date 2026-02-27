"""
Model Comparison & A/B Testing Page

Enable side-by-side comparison of different models with identical parameters.
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any
import time
import logging

from src.flux2.model_registry import get_model_registry, ModelType, ModelStatus
from src.flux2.streamlit_adapter import get_adapter
from ui.components_advanced import image_comparison, model_selector_advanced
from ui import icons

logger = logging.getLogger(__name__)


def _is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "accelerator" in text)


def _model_name_to_identifier(model_name: str) -> str:
    """Convert display name to model identifier.
    
    Examples:
    - "FLUX.2 Klein 4B" -> "flux.2-klein-4b"
    - "FLUX.2 Klein Base 4B" -> "flux.2-klein-base-4b"
    """
    return model_name.lower().replace(" ", "-").replace(".2 ", ".2-")


def render():
    """Render model comparison page"""
    icons.title("Model Comparison Lab", icon="lab")
    st.markdown("""
    Compare two models side-by-side with identical parameters to evaluate:
    - **Quality**: Visual fidelity and prompt adherence
    - **Speed**: Inference time and throughput
    - **Efficiency**: VRAM usage and power consumption
    """)
    
    registry = get_model_registry()
    adapter = get_adapter()
    
    # Initialize comparison state
    if 'comparison_results' not in st.session_state:
        st.session_state['comparison_results'] = None
    
    # Model selection
    st.subheader("1️⃣ Select Models")
    
    # Get available FLUX models
    available_models = registry.list_available(
        model_type=None,  # Allow all types
        status=ModelStatus.VERIFIED
    )
    
    if len(available_models) < 2:
        st.warning("⚠️ Not enough models available for comparison. Need at least 2 verified models.")
        st.info("💡 Add more models in the Settings page or verify existing models.")
        return
    
    model_lookup = {f"{m.name} ({m.parameter_count})": m for m in available_models}
    model_descriptions = {
        f"{m.name} ({m.parameter_count})": (m.description or "No description available")
        for m in available_models
    }

    # Create model selection columns
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 🅰️ Model A")
        model_a_name = model_selector_advanced(
            model_options=list(model_lookup.keys()),
            descriptions=model_descriptions,
            key_prefix="compare_model_a",
        )
        model_a = model_lookup[model_a_name]
        
        _render_model_info(model_a)
    
    with col_b:
        st.markdown("### 🅱️ Model B")
        model_b_name = model_selector_advanced(
            model_options=list(model_lookup.keys()),
            descriptions=model_descriptions,
            key_prefix="compare_model_b",
        )
        model_b = model_lookup[model_b_name]
        
        _render_model_info(model_b)
    
    # Quick comparison metrics
    st.divider()
    icons.heading("Quick Comparison", icon="activity", level=3)
    _render_quick_comparison(model_a, model_b)
    
    # Generation parameters
    st.divider()
    st.subheader("2️⃣ Configure Test Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_prompt = st.text_area(
            "Test Prompt",
            value="A serene mountain landscape at sunset with golden light",
            height=100,
            help="Identical prompt used for both models"
        )
    
    with col2:
        num_steps = st.slider("Inference Steps", 1, 50, 4, key="comp_steps")
        guidance = st.slider("Guidance", 1.0, 5.0, 3.5, 0.1, key="comp_guidance")
        width = st.selectbox("Width", [512, 768, 1024, 1280], index=2, key="comp_width")
    
    with col3:
        height = st.selectbox("Height", [512, 768, 1024, 1280], index=2, key="comp_height")
        seed = st.number_input("Seed", 0, 999999, 42, help="Fixed seed ensures reproducibility")
        true_cfg = st.slider("True CFG", 1.0, 5.0, 1.0, 0.1, key="comp_true_cfg")
    
    # Run comparison
    st.divider()
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        run_comparison = st.button(icons.label("Run A/B Test", "rocket"), type="primary", use_container_width=True)
    
    with col_btn2:
        if st.session_state['comparison_results']:
            if st.button(icons.label("Clear Results", "trash"), use_container_width=True):
                st.session_state['comparison_results'] = None
                st.rerun()
    
    # Execute comparison
    if run_comparison:
        _execute_comparison(
            model_a=model_a,
            model_b=model_b,
            prompt=test_prompt,
            num_steps=num_steps,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
            true_cfg=true_cfg,
            adapter=adapter
        )
    
    # Display results
    if st.session_state['comparison_results']:
        st.divider()
        st.subheader("3️⃣ Comparison Results")
        _render_results(st.session_state['comparison_results'])


def _render_model_info(model):
    """Render compact model information card"""
    with st.container():
        st.caption(f"**Type:** {model.model_type.value.title()}")
        st.caption(f"**Parameters:** {model.parameter_count}")
        st.caption(f"**VRAM:** {model.vram_requirement_gb:.1f} GB")
        st.caption(f"**Avg Speed:** {model.avg_inference_time_s:.1f}s")
        st.caption(f"**Score:** ⭐ {model.performance_score:.1f}/10")
        
        if model.tags:
            tags_html = " ".join([f"<span style='background-color: #4ECDC4; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; margin-right: 4px;'>{tag}</span>" for tag in model.tags])
            st.markdown(tags_html, unsafe_allow_html=True)


def _render_quick_comparison(model_a, model_b):
    """Render quick comparison metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Speed comparison
    speed_diff = model_b.avg_inference_time_s - model_a.avg_inference_time_s
    speed_pct = (speed_diff / model_a.avg_inference_time_s * 100) if model_a.avg_inference_time_s > 0 else 0
    
    with col1:
        st.metric(
            "Speed Difference",
            f"{abs(speed_diff):.1f}s",
            f"{-speed_pct:.0f}%" if speed_diff < 0 else f"+{speed_pct:.0f}%",
            delta_color="inverse"
        )
    
    # Quality comparison
    quality_diff = model_b.performance_score - model_a.performance_score
    
    with col2:
        st.metric(
            "Quality Difference",
            f"{abs(quality_diff):.1f}",
            f"{quality_diff:+.1f}",
            delta_color="normal"
        )
    
    # VRAM comparison
    vram_diff = model_b.vram_requirement_gb - model_a.vram_requirement_gb
    
    with col3:
        st.metric(
            "VRAM Difference",
            f"{abs(vram_diff):.1f} GB",
            f"{vram_diff:+.1f} GB",
            delta_color="inverse"
        )
    
    # Recommendation
    with col4:
        if speed_pct < -20 and quality_diff >= -0.5:
            st.success(icons.label("Model B is faster", "rocket"))
        elif quality_diff > 1.0 and abs(speed_pct) < 30:
            st.success(icons.label("Model B is better quality", "sparkles"))
        elif vram_diff < -2.0 and quality_diff >= -1.0:
            st.success(icons.label("Model B more efficient", "save"))
        else:
            st.info(icons.label("Models are comparable", "scales"))


def _execute_comparison(model_a, model_b, prompt: str, num_steps: int, 
                       guidance: float, width: int, height: int, 
                       seed: int, true_cfg: float, adapter):
    """Execute side-by-side model comparison"""
    
    results = {
        'model_a': {'metadata': model_a, 'image': None, 'time': 0.0, 'error': None},
        'model_b': {'metadata': model_b, 'image': None, 'time': 0.0, 'error': None},
        'params': {
            'prompt': prompt,
            'num_steps': num_steps,
            'guidance': guidance,
            'width': width,
            'height': height,
            'seed': seed,
            'true_cfg': true_cfg
        }
    }
    
    progress_bar = st.progress(0, text="Initializing comparison...")
    
    # Generate with Model A
    try:
        progress_bar.progress(10, text=f"🅰️ Loading {model_a.name}...")
        logger.info(f"Starting Model A comparison: {model_a.name}")
        
        # Load Model A with correct identifier format
        model_a_id = _model_name_to_identifier(model_a.name)
        logger.info(f"Loading model with identifier: {model_a_id}")
        adapter.load(model_a_id)
        logger.info(f"Model A loaded successfully: {model_a_id}")
        
        progress_bar.progress(20, text=f"🅰️ Generating with {model_a.name}...")
        logger.info(f"Starting generation for Model A with prompt: {prompt[:50]}...")
        
        start_time = time.time()
        logger.info("About to call adapter.generate() for Model A")
        
        image_a = adapter.generate(
            prompt=prompt,
            num_steps=num_steps,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
            true_cfg=true_cfg
        )
        
        logger.info("adapter.generate() returned successfully for Model A")
        generation_time = time.time() - start_time
        logger.info(f"Model A generation completed in {generation_time:.1f}s")
        
        results['model_a']['image'] = image_a
        results['model_a']['time'] = generation_time
        
        progress_bar.progress(50, text="✅ Model A complete")
        
    except Exception as e:
        logger.exception(f"Model A generation failed: {e}")
        results['model_a']['error'] = str(e)
        st.error(f"🅰️ Model A failed: {e}")
    
    # Generate with Model B
    try:
        progress_bar.progress(55, text=f"🅱️ Loading {model_b.name}...")
        logger.info(f"Starting Model B comparison: {model_b.name}")
        
        # Load Model B with correct identifier format
        model_b_id = _model_name_to_identifier(model_b.name)
        logger.info(f"Loading model with identifier: {model_b_id}")
        try:
            adapter.load(model_b_id)
        except Exception as load_exc:
            if not _is_oom_error(load_exc):
                raise
            logger.warning("Model B load OOM; retrying with cpu_offloading=True")
            st.warning("Model B hit GPU memory limit. Retrying with memory-saving mode.")
            adapter.load(
                model_b_id,
                cpu_offloading=True,
                attn_slicing=True,
            )
        logger.info(f"Model B loaded successfully: {model_b_id}")
        
        progress_bar.progress(65, text=f"🅱️ Generating with {model_b.name}...")
        logger.info(f"Starting generation for Model B with prompt: {prompt[:50]}...")
        
        start_time = time.time()
        logger.info("About to call adapter.generate() for Model B")
        
        b_width = width
        b_height = height
        b_steps = num_steps
        image_b = adapter.generate(
            prompt=prompt,
            num_steps=b_steps,
            guidance=guidance,
            width=b_width,
            height=b_height,
            seed=seed,
            true_cfg=true_cfg
        )
        
        logger.info("adapter.generate() returned successfully for Model B")
        generation_time = time.time() - start_time
        logger.info(f"Model B generation completed in {generation_time:.1f}s")
        
        results['model_b']['image'] = image_b
        results['model_b']['time'] = generation_time
        
        progress_bar.progress(100, text="✅ Comparison complete!")
        
    except Exception as e:
        if _is_oom_error(e):
            logger.warning("Model B generation OOM; retrying with reduced settings.")
            try:
                reduced_width = min(width, 768)
                reduced_height = min(height, 768)
                reduced_steps = min(num_steps, 4)
                st.warning(
                    f"Model B ran out of memory again. Retrying at {reduced_width}x{reduced_height}, {reduced_steps} steps."
                )

                start_time = time.time()
                image_b = adapter.generate(
                    prompt=prompt,
                    num_steps=reduced_steps,
                    guidance=guidance,
                    width=reduced_width,
                    height=reduced_height,
                    seed=seed,
                    true_cfg=true_cfg,
                )
                generation_time = time.time() - start_time

                results['model_b']['image'] = image_b
                results['model_b']['time'] = generation_time
                results['model_b']['error'] = None
                progress_bar.progress(100, text="✅ Comparison complete (Model B reduced settings)")
            except Exception as retry_exc:
                logger.exception(f"Model B generation failed after OOM fallback: {retry_exc}")
                results['model_b']['error'] = str(retry_exc)
                st.error(f"🅱️ Model B failed after fallback: {retry_exc}")
        else:
            logger.exception(f"Model B generation failed: {e}")
            results['model_b']['error'] = str(e)
            st.error(f"🅱️ Model B failed: {e}")
    
    # Store results
    st.session_state['comparison_results'] = results
    
    time.sleep(0.5)
    st.rerun()


def _render_results(results: Dict[str, Any]):
    """Render comparison results with images and metrics"""
    
    col_a, col_b = st.columns(2)
    
    # Model A results
    with col_a:
        st.markdown("### 🅰️ Model A Results")
        st.caption(results['model_a']['metadata'].name)
        
        if results['model_a']['error']:
            st.error(f"Generation failed: {results['model_a']['error']}")
        elif results['model_a']['image']:
            st.image(results['model_a']['image'], use_container_width=True)
            st.caption(f"⏱️ Generation time: {results['model_a']['time']:.2f}s")
    
    # Model B results
    with col_b:
        st.markdown("### 🅱️ Model B Results")
        st.caption(results['model_b']['metadata'].name)
        
        if results['model_b']['error']:
            st.error(f"Generation failed: {results['model_b']['error']}")
        elif results['model_b']['image']:
            st.image(results['model_b']['image'], use_container_width=True)
            st.caption(f"⏱️ Generation time: {results['model_b']['time']:.2f}s")
    
    # Performance comparison
    st.divider()
    st.markdown("### 📈 Performance Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    time_a = results['model_a']['time']
    time_b = results['model_b']['time']
    
    with col1:
        st.metric(
            "Model A Time",
            f"{time_a:.2f}s",
            help="Total generation time"
        )
    
    with col2:
        st.metric(
            "Model B Time",
            f"{time_b:.2f}s",
            f"{((time_b - time_a) / time_a * 100):+.0f}%" if time_a > 0 else "N/A",
            delta_color="inverse"
        )
    
    with col3:
        faster_model = "A" if time_a < time_b else "B"
        speedup = max(time_a, time_b) / min(time_a, time_b) if min(time_a, time_b) > 0 else 1.0
        st.metric(
            "Faster Model",
            f"Model {faster_model}",
            f"{speedup:.2f}x faster"
        )
    
    with col4:
        # User preference voting
        st.markdown("**Your Preference**")
        preference = st.radio(
            "Which output do you prefer?",
            ["Model A", "Model B", "Equal"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Test parameters recap
    with st.expander(icons.label("Test Parameters", "template")):
        params = results['params']
        st.json(params)
    
    # Export options
    st.divider()
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if st.button(icons.label("Save Model A", "save"), use_container_width=True):
            if results['model_a']['image']:
                # TODO: Implement save functionality
                st.success("Saved! (Feature in development)")
    
    with col_exp2:
        if st.button(icons.label("Save Model B", "save"), use_container_width=True):
            if results['model_b']['image']:
                # TODO: Implement save functionality
                st.success("Saved! (Feature in development)")
    
    with col_exp3:
        if st.button(icons.label("Save Both", "save"), use_container_width=True):
            # TODO: Implement save functionality
            st.success("Saved! (Feature in development)")

    if results['model_a']['image'] and results['model_b']['image']:
        st.divider()
        image_comparison(
            left_image=results['model_a']['image'],
            right_image=results['model_b']['image'],
            left_label=f"Model A: {results['model_a']['metadata'].name}",
            right_label=f"Model B: {results['model_b']['metadata'].name}",
            key_prefix="model_comparison_advanced",
        )


if __name__ == "__main__":
    render()
