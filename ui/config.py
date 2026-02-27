"""UI configuration: model profiles, color scheme, all English string constants."""

from __future__ import annotations

# ─── Model Profiles ───────────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict] = {
    "flux.2-klein-4b": {
        "display_name": "FLUX.2-Klein 4B",
        "short_label": "Klein 4B",
        "description": (
            "Fastest model. Distilled for 4-step generation with minimal VRAM. "
            "Ideal for rapid iteration and lower-spec hardware."
        ),
        "vram_estimate": "~8 GB",
        "speed_label": "Fast",
        "quality_label": "3/5",
        "recommended": {
            "num_steps": 4,
            "guidance": 1.0,
            "width": 768,
            "height": 768,
            "dtype": "bf16",
        },
        "parameter_ranges": {
            "num_steps": {"min": 1, "max": 20, "step": 1, "fixed": True},
            "guidance": {"min": 0.0, "max": 5.0, "step": 0.1, "fixed": True},
            "width_options": [512, 640, 768, 896, 1024],
            "height_options": [512, 640, 768, 896, 1024],
        },
        "guidance_distilled": True,
        "fixed_params": {"guidance", "num_steps"},
        "default_dtype": "bf16",
    },
    "flux.2-klein-9b": {
        "display_name": "FLUX.2-Klein 9B",
        "short_label": "Klein 9B",
        "description": (
            "Higher-capacity distilled model. Better detail and coherence than 4B "
            "at the cost of additional VRAM."
        ),
        "vram_estimate": "~16 GB",
        "speed_label": "Fast",
        "quality_label": "4/5",
        "recommended": {
            "num_steps": 4,
            "guidance": 1.0,
            "width": 768,
            "height": 768,
            "dtype": "bf16",
        },
        "parameter_ranges": {
            "num_steps": {"min": 1, "max": 20, "step": 1, "fixed": True},
            "guidance": {"min": 0.0, "max": 5.0, "step": 0.1, "fixed": True},
            "width_options": [512, 640, 768, 896, 1024, 1360],
            "height_options": [512, 640, 768, 896, 1024, 1360],
        },
        "guidance_distilled": True,
        "fixed_params": {"guidance", "num_steps"},
        "default_dtype": "bf16",
    },
    "flux.2-klein-base-4b": {
        "display_name": "FLUX.2-Klein Base 4B",
        "short_label": "Klein Base 4B",
        "description": (
            "Non-distilled base model for the 4B architecture. Supports full "
            "guidance range and more customizable generation at slower speed."
        ),
        "vram_estimate": "~8 GB",
        "speed_label": "Slow",
        "quality_label": "4/5",
        "recommended": {
            "num_steps": 50,
            "guidance": 4.0,
            "width": 768,
            "height": 768,
            "dtype": "bf16",
        },
        "parameter_ranges": {
            "num_steps": {"min": 1, "max": 100, "step": 1, "fixed": False},
            "guidance": {"min": 0.0, "max": 10.0, "step": 0.5, "fixed": False},
            "width_options": [512, 640, 768, 896, 1024, 1360],
            "height_options": [512, 640, 768, 896, 1024, 1360],
        },
        "guidance_distilled": False,
        "fixed_params": set(),
        "default_dtype": "bf16",
    },
    "flux.2-klein-base-9b": {
        "display_name": "FLUX.2-Klein Base 9B",
        "short_label": "Klein Base 9B",
        "description": (
            "Non-distilled base model for the 9B architecture. Highest quality "
            "Klein variant; requires most VRAM and longest generation time."
        ),
        "vram_estimate": "~16 GB",
        "speed_label": "Slow",
        "quality_label": "5/5",
        "recommended": {
            "num_steps": 50,
            "guidance": 4.0,
            "width": 768,
            "height": 768,
            "dtype": "bf16",
        },
        "parameter_ranges": {
            "num_steps": {"min": 1, "max": 100, "step": 1, "fixed": False},
            "guidance": {"min": 0.0, "max": 10.0, "step": 0.5, "fixed": False},
            "width_options": [512, 640, 768, 896, 1024, 1360],
            "height_options": [512, 640, 768, 896, 1024, 1360],
        },
        "guidance_distilled": False,
        "fixed_params": set(),
        "default_dtype": "bf16",
    },
    "flux.2-dev": {
        "display_name": "FLUX.2-Dev",
        "short_label": "Dev",
        "description": (
            "Full-size development model. Highest overall quality with Mistral-based "
            "text encoding. Best for fine-grained prompt adherence and complex scenes."
        ),
        "vram_estimate": "~20 GB",
        "speed_label": "Slow",
        "quality_label": "5/5",
        "recommended": {
            "num_steps": 50,
            "guidance": 4.0,
            "width": 1360,
            "height": 768,
            "dtype": "bf16",
        },
        "parameter_ranges": {
            "num_steps": {"min": 1, "max": 100, "step": 1, "fixed": False},
            "guidance": {"min": 0.0, "max": 10.0, "step": 0.5, "fixed": False},
            "width_options": [512, 640, 768, 896, 1024, 1360, 1920],
            "height_options": [512, 640, 768, 896, 1024, 1360, 1080],
        },
        "guidance_distilled": True,
        "fixed_params": set(),
        "default_dtype": "bf16",
    },
}

# Ordered list for the UI selector (most user-friendly first)
MODEL_DISPLAY_ORDER = [
    "flux.2-klein-4b",
    "flux.2-klein-9b",
    "flux.2-klein-base-4b",
    "flux.2-klein-base-9b",
    "flux.2-dev",
]

# ─── Quality Presets (map to step counts per model type) ──────────────────────

QUALITY_PRESETS_DISTILLED = {
    "Draft (2 steps)": {"num_steps": 2},
    "Standard (4 steps)": {"num_steps": 4},
    "High Quality (8 steps)": {"num_steps": 8},
}

QUALITY_PRESETS_BASE = {
    "Draft (10 steps)": {"num_steps": 10, "guidance": 3.5},
    "Standard (25 steps)": {"num_steps": 25, "guidance": 4.0},
    "High Quality (50 steps)": {"num_steps": 50, "guidance": 4.5},
    "Maximum (100 steps)": {"num_steps": 100, "guidance": 5.0},
}

GEN_QUALITY_PRESET_LABEL = "Quality Preset"
GEN_QUALITY_PRESET_HELP = (
    "Quick preset for inference steps (and guidance for non-distilled models). "
    "Pick \"Standard\" for best quality-to-speed balance."
)

# ─── Prompt Presets (aligned to project type and design phase) ────────────────

PROMPT_PRESETS_BY_PROJECT_TYPE: dict[str, dict[str, str]] = {
    "Mixed-use Development": {
        "Concept Massing": "Mixed-use perimeter block, 5-8 floors, active ground floor retail, climate-responsive facade, eye-level street perspective, overcast daylight, competition-board quality",
        "Streetscape Detail": "Mid-rise mixed-use building, permeable block edge, shaded arcade, physically based materials, global illumination, street-level pedestrian view, golden hour lighting",
        "Urban Context": "Mixed-use development cluster, transit-priority corridor, bike-priority network, contextual facade treatment, aerial oblique view, clear daytime, stakeholder-ready rendering",
        "Material & Facade": "Mid-rise facade study, opaque masonry with deep shading fins, glazing ratio 40%, texture roughness 0.45, balanced reflectance, eye-level detail view",
        "Presentation Board": "Mixed-use perimeter massing, courtyard activation, human-scale readability, competition-board quality, global illumination, overcast soft light, wide-angle perspective",
    },
    "Residential": {
        "Concept Massing": "Residential courtyard massing, 4-6 floors, central shared amenity, green buffer zones, aerial perspective, clear daytime, competition-board quality",
        "Unit Detail": "Residential facade detail, timber expression with opaque masonry, glazing ratio 35%, muted reflectance, eye-level view, golden hour, photorealistic materials",
        "Neighborhood Context": "Low-rise residential blocks, bike-priority network, civic plaza activation, tree-lined streets, aerial oblique view, summer season, stakeholder-ready rendering",
        "Common Spaces": "Residential courtyard view, shared green space, permeable edges, natural materials, human-scale readability, overcast soft light, wide perspective",
        "Facade Study": "Residential mid-rise facade, deep balconies, climate-responsive shading, timber and masonry, eye-level detail, morning light, competition-board quality",
    },
    "Commercial": {
        "Office Tower": "High-rise tower-podium, fully glazed tower, active podium retail, transit-priority access, aerial oblique, clear daytime, competition-board quality",
        "Podium Activation": "Commercial podium detail, permeable street edge, retail frontage, shaded arcade, metal panel rhythm, eye-level view, golden hour, stakeholder-ready",
        "Campus Layout": "Commercial campus masterplan, car-free core, civic plaza activation, landscape integration, aerial view, overcast daylight, presentation-ready",
        "Facade Detail": "Commercial high-rise facade, fully glazed curtain wall, metal panel accents, glazing ratio 65%, reflective mood, eye-level detail, noon lighting",
        "Lobby & Entry": "Commercial lobby entrance, fully glazed facade, high ceiling volume, polished materials, balanced reflectance, eye-level perspective, daylight interior",
    },
    "Institutional": {
        "Campus Concept": "Institutional campus layout, low-rise blocks, civic plaza activation, permeable pedestrian network, aerial oblique view, clear daytime, competition-board quality",
        "Courtyard Design": "Institutional courtyard massing, central gathering space, contextual facade, opaque masonry with deep shading, eye-level view, overcast soft light",
        "Entry Sequence": "Institutional entry plaza, car-free core, landscape integration, timber and masonry expression, human-scale readability, golden hour, stakeholder-ready",
        "Building Detail": "Institutional building facade, contextual materials, glazing ratio 30%, deep shading fins, muted reflectance, eye-level detail, morning light",
        "Public Realm": "Institutional public realm, bike-priority network, civic plaza activation, tree canopy, aerial view, summer season, presentation-ready rendering",
    },
    "Public Realm": {
        "Plaza Design": "Public plaza design, car-free core, civic activation nodes, natural paving, integrated seating, aerial oblique, clear daytime, stakeholder-ready",
        "Street Section": "Complete street design, bike-priority network, tree-lined boulevard, permeable edges, transit integration, eye-level view, golden hour, competition-board quality",
        "Waterfront Promenade": "Waterfront public realm, promenade design, permeable edge, natural materials, integrated lighting, eye-level pedestrian view, overcast soft light",
        "Park Integration": "Urban park interface, green buffer zones, bike network, natural materials, human-scale readability, aerial view, summer season, presentation-ready",
        "Night Activation": "Public plaza night scene, integrated lighting, activated edges, material reflectance, eye-level perspective, night lighting, stakeholder-ready rendering",
    },
    "Urban Masterplan": {
        "Masterplan Overview": "Urban masterplan aerial, mixed-use development cluster, transit-priority corridors, green network, permeable block structure, aerial oblique, clear daytime, competition-board quality",
        "District Identity": "Urban district massing, tower-podium typology, mid-rise perimeter, active ground floor, contextual skyline, aerial view, overcast daylight, stakeholder-ready",
        "Mobility Network": "Urban mobility framework, transit-priority corridor, bike-priority network, car-free zones, street hierarchy, aerial oblique, clear daytime, presentation-ready",
        "Public Realm System": "Urban public realm network, civic plaza nodes, green corridors, permeable edges, human-scale activation, aerial view, summer season, stakeholder-ready",
        "Development Phasing": "Urban phasing diagram, mixed-use blocks, incremental density, infrastructure integration, aerial oblique, clear daytime, competition-board quality",
    },
}

EDITOR_PROMPT_PRESET_LABEL = "Prompt Preset"
EDITOR_PROMPT_PRESET_HELP = (
    "Quick-start prompts aligned to your active project type. "
    "Select a preset to populate the prompt field, then customize as needed."
)

# ─── DTYPE Options ────────────────────────────────────────────────────────────

DTYPE_OPTIONS = {
    "bf16": "BF16 — Recommended (best balance of speed and precision)",
    "fp16": "FP16 — May be faster on older GPUs",
    "fp32": "FP32 — Full precision (highest VRAM usage)",
}

# ─── Upsampling Backends ──────────────────────────────────────────────────────

UPSAMPLE_BACKENDS = {
    "none": "Off — Use prompt as-is",
    "local": "Local (Ollama) — Use local Ollama model (no API key required)",
    "openrouter": "OpenRouter API — Use a cloud LLM (requires API key)",
}

UPSAMPLE_BACKEND_HINTS = {
    "none": "No upsampling will be applied.",
    "local": "Latency depends on your local Ollama model size and hardware. Cost: $0.",
    "openrouter": "Latency: Low to medium depending on network/model. Cost: depends on provider pricing.",
}

UPS_OLLAMA_MODEL_LABEL = "Ollama Model"
UPS_OLLAMA_MODEL_HELP = "Select a local Ollama model for prompt expansion."
UPS_OLLAMA_PROFILE_LABEL = "Ollama Profile"

UPS_OLLAMA_PROFILES = {
    "fast": {
        "label": "Fast",
        "default_model": "qwen2.5-coder:1.5b-base",
        "temperature": 0.2,
    },
    "quality": {
        "label": "Quality",
        "default_model": "qwen3:30b",
        "temperature": 0.15,
    },
}

OPENROUTER_MODEL_OPTIONS = [
    "mistralai/pixtral-large-2411",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
]

# ─── Example Prompts ──────────────────────────────────────────────────────────

EXAMPLE_PROMPTS = [
    "Mixed-use perimeter block, 6-8 floors, active ground floor retail, climate-responsive facade, eye-level street perspective, overcast daylight",
    "Tower-podium residential complex with transit plaza, shaded colonnade, native planting, aerial oblique view, golden hour",
    "Adaptive reuse warehouse into innovation hub, retained brick structure, new timber extension, public courtyard, human-scale rendering",
    "Transit-oriented district around metro station, dense mid-rise fabric, protected bike lanes, shared streets, rainy evening atmosphere",
    "Waterfront masterplan with floodable park, stepped embankment, mixed-use frontage, pedestrian priority promenade, early morning light",
    "Courtyard housing cluster in warm-arid climate, deep balconies, perforated shading screens, passive cooling strategy, noon lighting",
    "Civic plaza redesign with integrated bus corridor, tree canopy, seating terraces, inclusive accessibility routes, high-detail urban visualization",
    "University precinct urban scenario: low-carbon material palette, multimodal mobility spine, activated public realm, dusk streetscape view",
]

PROMPT_STYLE_PRESETS = {
    "Photography": "architectural photography, photorealistic, eye-level composition, physically plausible materials, natural lighting",
    "Oil Painting": "architectural oil painting, textured brushstrokes, atmospheric perspective, contextual streetscape composition",
    "Anime / Illustration": "diagrammatic architectural illustration, clean linework, simplified massing, annotated visual hierarchy",
    "Pencil Sketch": "architectural pencil sketch, conceptual massing study, graphite shading, design process aesthetic",
    "3D Render": "architectural 3D render, physically based materials, global illumination, competition-board quality",
    "Watercolor": "urban design watercolor style, soft washes, people and landscape accents, presentation-board mood",
    "Conceptual Art": "architectural concept visualization, massing narrative, public realm focus, contextual composition",
}

# ─── UI Text Strings (all English) ────────────────────────────────────────────

PAGE_TITLE = "FLUX.2 Professional Image Generator"
PAGE_ICON = ""

SIDEBAR_TITLE = "Configuration"
SIDEBAR_ADVANCED_LABEL = "Advanced Mode"
SIDEBAR_ADVANCED_HELP = (
    "Enable Advanced Mode to access full parameter ranges and hardware settings. "
    "Recommended defaults will still be shown as reference."
)

TAB_GENERATOR = "Generate"
TAB_EDITOR = "Image Editor"
TAB_QUEUE = "Queue"
TAB_PROGRESS = "Progress"
TAB_PERFORMANCE = "Performance"
TAB_UPSAMPLER = "Prompt Upsampler"
TAB_SETTINGS = "Settings & Help"

# Generator tab
GEN_PROMPT_LABEL = "Image Description"
GEN_PROMPT_PLACEHOLDER = (
    "Describe the architectural or urban design intent. Include program, massing, "
    "facade/material cues, public realm, view type, and environmental conditions..."
)
GEN_PROMPT_HELP = (
    "Detailed architectural prompts improve consistency. Include: project type, site/context, "
    "massing, facade/material language, mobility/public-realm intent, and lighting/climate conditions."
)
GEN_STYLE_PRESET_LABEL = "Append Style Preset"
GEN_STYLE_PRESET_HELP = "Append a style keyword block to your prompt."
GEN_EXAMPLE_LABEL = "Load Example Prompt"
GEN_MODEL_LABEL = "Model"
GEN_MODEL_HELP = "Select the FLUX.2 model variant to use for generation."
GEN_STEPS_LABEL = "Inference Steps"
GEN_STEPS_HELP = (
    "Number of denoising steps. More steps = higher quality but slower generation. "
    "Distilled (Klein) models are optimized for very low step counts (4 steps recommended)."
)
GEN_GUIDANCE_LABEL = "Guidance Scale"
GEN_GUIDANCE_HELP = (
    "Controls how strongly the model follows your prompt. Higher values = closer to prompt "
    "but potentially less natural. Lower values = more creative freedom."
)
GEN_WIDTH_LABEL = "Width (px)"
GEN_HEIGHT_LABEL = "Height (px)"
GEN_SEED_LABEL = "Seed"
GEN_SEED_HELP = (
    "Fixed seed for reproducible outputs. Use the same seed and parameters to "
    "regenerate an identical image. Set to 0 for a random seed each run."
)
GEN_DTYPE_LABEL = "Data Type"
GEN_DTYPE_HELP = (
    "Numerical precision for model weights. BF16 is recommended for NVIDIA Ampere+ GPUs. "
    "FP16 works on older GPUs. FP32 uses more memory but offers maximum precision."
)
GEN_CPU_OFFLOAD_LABEL = "CPU Offloading"
GEN_CPU_OFFLOAD_HELP = (
    "Move model components to CPU RAM when not in use, allowing generation on GPUs with "
    "limited VRAM. Significantly slower; recommended only when GPU VRAM is insufficient."
)
GEN_ATTN_SLICING_LABEL = "Attention Slicing"
GEN_ATTN_SLICING_HELP = (
    "Process attention in smaller slices to reduce peak VRAM usage. "
    "Slight speed penalty; recommended for GPUs below 12 GB VRAM."
)
GEN_OUTPUT_DIR_LABEL = "Output Directory"
GEN_OUTPUT_DIR_HELP = "Folder where generated images will be saved."
GEN_BUTTON_LABEL = "Generate Image"
GEN_CLEAR_CACHE_LABEL = "Clear GPU Cache"
GEN_CLEAR_CACHE_HELP = "Free unused GPU memory (torch.cuda.empty_cache). Use if encountering OOM errors."
GEN_DOWNLOAD_LABEL = "Download PNG"
GEN_SAVE_LABEL = "Save to Gallery"
GEN_COPY_PROMPT_LABEL = "Copy Prompt"
GEN_SEND_TO_EDITOR_LABEL = "Use as Reference"
GEN_EXPAND_PROMPT_LABEL = "Expand Prompt"
GEN_EXPAND_PROMPT_HELP = "Open the Prompt Upsampler tab to automatically enrich this prompt with AI."
GEN_RESET_PARAMS_LABEL = "↺ Reset to Recommended"
GEN_RESET_PARAMS_HELP = "Restore all parameters to the recommended defaults for the selected model."

# Status messages
STATUS_LOADING_MODEL = "Loading model components — please wait..."
STATUS_GENERATING = "Generating image..."
STATUS_ENCODING_TEXT = "Encoding text prompt..."
STATUS_DENOISING = "Running diffusion denoising..."
STATUS_DECODING = "Decoding latents to image..."
STATUS_DONE = "Generation complete!"
STATUS_CACHE_CLEARED = "GPU cache cleared successfully."

# Errors
ERR_PROMPT_EMPTY = "Please enter a prompt before generating."
ERR_NO_CUDA = (
    "CUDA GPU not detected. Enable **CPU Offloading** in the sidebar to run on CPU "
    "(significantly slower), or connect a CUDA-compatible GPU."
)
ERR_MODEL_LOAD = "Failed to load model: {error}"
ERR_GENERATION = "Generation failed: {error}"
ERR_OOM = (
    "Out of GPU memory. Try: reduce resolution, switch to BF16 dtype, "
    "enable CPU Offloading, or use the Klein 4B model."
)
ERR_SAFETY_PROMPT = (
    "Your prompt was flagged for potential copyright or public persona concerns. "
    "Please revise your prompt and try again."
)
ERR_SAFETY_IMAGE = (
    "The uploaded image was flagged as unsuitable. "
    "Please choose a different reference image."
)
ERR_API_KEY_MISSING = (
    "OpenRouter API key is not configured. "
    "Please add it in the **Settings** tab under *API Keys*."
)

# Warnings
WARN_ADVANCED_PARAMS = (
    "You've modified parameters from their recommended values for this model. "
    "Results may be suboptimal."
)
WARN_STEPS_DISTILLED = (
    "This is a distilled model optimized for {recommended_steps} steps. "
    "Using more steps may not improve quality."
)

# Editor tab
EDITOR_UPLOAD_LABEL = "Upload Reference Image(s)"
EDITOR_UPLOAD_HELP = (
    "Upload one or more reference images to guide generation. "
    "The model will incorporate visual elements from these images."
)
EDITOR_MATCH_SIZE_LABEL = "Match dimensions from image"
EDITOR_MATCH_SIZE_HELP = "Automatically set output width & height to match the selected reference image dimensions."
EDITOR_PROMPT_LABEL = "Edit / Generation Prompt"
EDITOR_PROMPT_HELP = "Describe what you want to create based on the reference image(s)."

# Upsampler tab
UPS_INPUT_LABEL = "Original Prompt"
UPS_INPUT_PLACEHOLDER = "Enter a short or rough prompt to expand and enrich..."
UPS_BACKEND_LABEL = "Upsampling Backend"
UPS_MODEL_LABEL = "OpenRouter Model"
UPS_BUTTON_LABEL = "Expand Prompt"
UPS_APPLY_BUTTON_LABEL = "Apply to Generator"
UPS_RESULT_LABEL = "Expanded Prompt"
UPS_COMPARISON_HEADER = "Before & After"
UPS_REFERENCE_HINT = "Optional reference images for i2i-aware prompt expansion"

# Settings tab
SETTINGS_OUTPUT_SECTION = "Output Settings"
SETTINGS_API_SECTION = "API Keys"
SETTINGS_MODEL_PATHS_SECTION = "Custom Model Paths"
SETTINGS_SAFETY_SECTION = "Content Safety"
SETTINGS_ABOUT_SECTION = "About"
SETTINGS_OPENROUTER_KEY_LABEL = "OpenRouter API Key"
SETTINGS_OPENROUTER_KEY_HELP = "Used for cloud-based prompt upsampling. Get yours at openrouter.ai."
SETTINGS_NSFW_THRESHOLD_LABEL = "NSFW Detection Threshold"
SETTINGS_NSFW_THRESHOLD_HELP = (
    "Confidence threshold above which generated images are flagged as NSFW. "
    "Lower = stricter filtering (0.0 = block everything, 1.0 = block nothing)."
)
SETTINGS_SAFETY_PROMPT_LABEL = "Enable prompt safety checks"
SETTINGS_SAFETY_IMAGE_LABEL = "Enable output image safety checks"

# Queue tab
QUEUE_TAB_TITLE = "Batch Queue"
QUEUE_ENQUEUE_LABEL = "Add to Queue"
QUEUE_START_LABEL = "Start"
QUEUE_PAUSE_LABEL = "Pause"
QUEUE_RESUME_LABEL = "Resume"
QUEUE_CLEAR_LABEL = "Clear"
QUEUE_SCHEDULE_LABEL = "Schedule for (optional)"
QUEUE_PRIORITY_LABEL = "Priority"
QUEUE_PRIORITY_HELP = "Higher values are processed first."

# Help content
HELP_PROMPT_TIPS = """
**Writing Effective Prompts**

- **Be project-specific**: Instead of *"modern building"*, use *"mixed-use mid-rise with active podium, permeable block edge, and shaded arcade"*.
- **Include site context**: Mention climate, topography, adjacency, and mobility constraints.
- **Define design intent**: Add massing language, facade/material cues, and public-realm priorities.
- **Describe viewpoint**: *"aerial oblique"*, *"eye-level streetscape"*, *"courtyard view"*, *"waterfront promenade"*.
- **Specify environment**: *"overcast winter morning"*, *"golden hour"*, *"leaf-off season"*, *"rainy evening"*.
- **Add presentation quality cues**: *"competition-board quality"*, *"human-scale readability"*, *"stakeholder-ready rendering"*.

**Use the Prompt Upsampler** section to automatically enrich short prompts using AI.
"""

HELP_MODEL_COMPARISON = """
| Model | VRAM | Speed | Quality | Best For |
|---|---|---|---|---|
| Klein 4B | ~8 GB | Fast | 3/5 | Rapid iteration, low-VRAM hardware |
| Klein 9B | ~16 GB | Fast | 4/5 | Balanced quality + speed |
| Klein Base 4B | ~8 GB | Slow | 4/5 | Custom guidance on 4B arch. |
| Klein Base 9B | ~16 GB | Slow | 5/5 | Maximum quality on Klein |
| FLUX.2-Dev | ~20 GB | Slow | 5/5 | Professional / top quality |

**Klein distilled models** (4B, 9B) are optimized for exactly 4 inference steps.  
**Base / Dev models** allow full control over steps and guidance.
"""

HELP_UPSAMPLING = """
**What is Prompt Upsampling?**

Prompt upsampling uses a language model to automatically expand a short or rough prompt 
into a detailed, well-structured description that better leverages FLUX.2's capabilities.

**When to use it:**
- Your initial prompt is short or vague
- You want to add consistent technical and stylistic details automatically
- You want to explore richer interpretations of a concept

**Backends:**
- **Local**: Uses the on-device Mistral model (no internet/API key required; slower first run while loading).
- **OpenRouter**: Uses a cloud LLM of your choice via the OpenRouter API (fast, requires API key).

**Tip**: Always review the expanded prompt before generating — you can edit it before applying.
"""

HELP_IMAGE_EDITOR = """
**Image-to-Image Generation**

Upload one or more reference images to guide composition, color, or structure in the output.

**Single reference**: The model will incorporate the visual style and structure of the reference.  
**Multiple references**: Useful for combining elements from different images.

**Tips:**
- Use `Match dimensions from image` to automatically align output resolution to your reference.
- A clear, high-quality reference image produces more consistent results.
- Your text prompt still influences the final output — describe what you want to add or change.
"""

HELP_TROUBLESHOOTING = """
**Common Issues & Solutions**

| Issue | Solution |
|---|---|
| Out of GPU memory (OOM) | Reduce resolution, enable CPU Offloading, switch to BF16, use Klein 4B |
| Slow generation | Use Klein 4B/9B (4 steps), reduce resolution |
| Poor image quality | Increase steps (base/dev models), use Prompt Upsampler, refine your prompt |
| Prompt not followed well | Increase guidance scale, be more specific in your prompt |
| Model not found | Check your Hugging Face login and model access permissions |
| API key error | Verify your OpenRouter API key in Settings |
"""

QUICK_REFERENCE_MD = """
# FLUX.2 UI Quick Reference

## Recommended model picks
- Klein 4B: fastest iteration, lower VRAM
- Klein 9B: better detail with moderate latency
- Klein Base / Dev: higher quality with more steps

## Prompt checklist
- Subject + environment + style + lighting + mood
- Add camera/art keywords for stronger control
- Use Prompt Upsampler for short drafts

## Performance checklist
- OOM: lower resolution, enable CPU offload, use BF16
- Slow: use Klein distilled models, fewer steps

## Safety checklist
- Keep prompt safety checks enabled by default
- Review generated outputs when working with references
""".strip()
