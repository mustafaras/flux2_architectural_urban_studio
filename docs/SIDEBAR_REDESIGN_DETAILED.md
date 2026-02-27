# FLUX.2 Sidebar Redesign — Phase-by-Phase Prompt Pack (Comprehensive Edition)

This document provides a comprehensive set of implementation prompts for redesigning the FLUX.2 sidebar into a more consistent, advanced, and workflow-aware experience.

Each phase is written as a **ready-to-run prompt** for an implementation agent. Prompts are ordered to minimize risk and preserve existing behavior while improving UX quality.

---

## Phase 1 Prompt — Extract Sidebar into a Dedicated Module

### Prompt
You are working in the FLUX.2 codebase. Refactor the sidebar implementation from `ui_flux2_professional.py` into a dedicated module at `ui/sidebar.py` without changing user-visible behavior.

#### Goal
Create a clean sidebar architecture that separates concerns and makes later UX improvements safer and faster.

#### Detailed Context
The current sidebar in `ui_flux2_professional.py` (lines 90–261) spans 170+ lines and contains:
- Workflow toggle (Architecture vs Legacy mode)
- Model selection and sync logic
- Quality preset application logic
- Output size (width/height) controls
- Seed management (random vs manual)
- Queue control buttons (start/pause/resume)
- Queue status display
- History browser with restore capability
- State initialization and reset handlers

This monolithic structure makes it difficult to test, refactor, or add new sidebar features without risk of breaking existing functionality. Extracting to a dedicated module enables safer iteration in subsequent phases.

#### Scope
- Move all sidebar rendering logic from `ui_flux2_professional.py` (lines 90–261) into `ui/sidebar.py`.
- Expose a single public entry point function (recommended name: `render_sidebar()`).
- Preserve **all** existing control widget keys exactly to avoid state breakage.
- Keep current button actions, toggles, status sections, and history behavior unchanged in appearance and function.
- Do not introduce new dependencies or circular imports.

#### Requirements
1. Do not alter business logic during this phase (extract-only refactor).
2. Keep Streamlit keys stable (workflow_mode, selected_model, selected_preset, output_width, output_height, manual_seed, seed_mode, queue_auto_run, queue_paused).
3. Preserve model sync behavior with `state.sync_model_selection_state()`.
4. Keep queue controls and history restore features functionally identical.
5. Keep imports minimal and avoid circular dependencies.
6. Support both navigation modes (Architecture workflow + Legacy tabs) equally.

#### Implementation Notes
- Create helper functions for logical sub-sections (e.g., `_render_workflow_section()`, `_render_model_controls()`, `_render_queue_status()`, `_render_history_browser()`).
- Maintain original order and grouping of sections.
- Document purpose and state dependencies of each extracted section.
- Copy-paste code is acceptable in Phase 1 (avoid premature de-duplication).

#### Deliverables
1. **New module**: `ui/sidebar.py` with `render_sidebar()` function and internal helpers.
2. **Main file update**: `ui_flux2_professional.py` imports and calls `render_sidebar()` instead.
3. **No functional regressions** in sidebar interactions.
4. **No broken imports**.

#### Validation Checklist
- ✅ `ui/sidebar.py` exists and is syntactically valid Python.
- ✅ App launches without import errors.
- ✅ Sidebar renders successfully on first page load (both navigation modes).
- ✅ Workflow toggle switches between Architecture and Legacy without errors.
- ✅ Model dropdown updates `session_state['selected_model']` correctly.
- ✅ Preset selector applies defaults (size, seed) when changed.
- ✅ Manual seed input accepts numeric values and updates state.
- ✅ Random seed toggle works and clears manual seed value.
- ✅ Queue start/pause/resume buttons trigger expected state transitions.
- ✅ History browser populates correctly and restore buttons work.
- ✅ No Streamlit key collision errors.
- ✅ Sidebar state persists correctly across reruns.
- ✅ Both navigation modes work without sidebar breakage.
- ✅ No visual changes to sidebar appearance.

#### Common Pitfalls to Avoid
1. **Circular imports**: Do not import page modules. Use `st.session_state` instead.
2. **Widget key changes**: Keep keys constant; renaming breaks state hydration.
3. **Missing callback wiring**: Ensure all button handlers are preserved exactly.
4. **Model sync logic**: Verify `state.sync_model_selection_state()` is called at the correct point.

#### Non-Goals
- No visual redesign.
- No new features.
- No changes to page layout.
- No reordering of sidebar sections.

---

## Phase 2 Prompt — Normalize Sidebar State and Actions

### Prompt
Improve reliability of sidebar state updates by introducing a thin state/action layer in `ui/state.py` and wiring sidebar controls to these helpers. Reduce scattered direct `st.session_state` writes and establish a single source of truth for state mutations.

#### Goal
Reduce scattered direct `st.session_state` writes and make sidebar behavior deterministic and testable.

#### Detailed Context
Currently, sidebar controls directly manipulate `st.session_state`, leading to:
- Multiple code paths updating the same keys
- Inconsistent update ordering (seed logic vs preset logic race conditions)
- Difficult-to-debug state corruption when features interact
- Duplicate logic between sidebar and page-level controls
- No clear contract for what happens when state X changes

Phase 2 creates a thin action layer: focused, named functions that encapsulate each state mutation.

#### Scope
Add state helper functions to `ui/state.py` for these sidebar actions:

**Model & Defaults**:
- `apply_model_selection(model_name: str)` → Updates selected_model, calls sync, resets incompatible fields.
- `apply_quality_preset(preset_name: str)` → Sets quality_mode, model, width/height, seed.
- `sync_model_recommendations()` → Updates recommendations for selected model.

**Output & Sizing**:
- `set_output_size(width: int, height: int)` → Validates and updates dimensions.
- `apply_aspect_ratio(ratio_key: str)` → Applies common aspect ratios (1:1, 16:9, 9:16, 4:3).

**Seed Management**:
- `set_random_seed_mode()` → Enables random mode, clears manual seed.
- `set_manual_seed(value: int)` → Validates range, updates manual seed, disables random mode.
- `generate_new_random_seed()` → Generates new random seed in valid range.

**Queue & Run Control**:
- `start_queue_auto_run()` → Sets queue_auto_run=True, queue_paused=False.
- `pause_queue()` → Sets queue_paused=True, preserves queue_auto_run.
- `resume_queue()` → Sets queue_paused=False.
- `stop_queue_auto_run()` → Sets queue_auto_run=False, queue_paused=False.

**Session & History**:
- `restore_history_checkpoint(checkpoint_id: str)` → Loads saved state snapshot.
- `clear_generation_session(scope: str)` → Clears transient fields, preserves project context.

#### Requirements
1. Backward-compatible state keys (no renaming or removal).
2. No breaking changes in page components.
3. Keep helper functions focused and testable.
4. Avoid side effects outside intended scope.
5. Document preconditions, postconditions, and side effects.
6. Helpers should not cause unnecessary reruns.

#### Implementation Notes
- Start with most-used actions (model, preset, seed).
- Use consistent naming: `apply_X`, `set_X`, `is_X`, `get_X_state`.
- Add docstrings with usage examples and type hints.
- Log action calls in debug mode for state flow tracking.

#### Deliverables
1. **New/updated helper APIs in `ui/state.py`**: Minimum 10 functions with docstrings and type hints.
2. **Sidebar refactored**: Uses helpers instead of direct state writes for key actions.
3. **Reduced duplicate logic**: Consolidate preset/model logic into shared helpers.

#### Validation Checklist
- ✅ All new helper functions exist with docstrings.
- ✅ Helpers have type hints and error handling.
- ✅ Sidebar calls helpers instead of direct state writes.
- ✅ App launches without errors.
- ✅ Toggling advanced mode is stable across reruns.
- ✅ Model changes update defaults correctly.
- ✅ Preset application is consistent.
- ✅ Seed random/manual toggle is predictable.
- ✅ Queue buttons work as before.
- ✅ History restore preserved.
- ✅ No regressions in Architecture or Legacy modes.

#### Common Pitfalls to Avoid
1. **Over-abstraction**: Focus on reused patterns, not every state write.
2. **Side effect surprises**: Document every state key touched.
3. **Circular logic**: Avoid helpers calling helpers in circular ways.
4. **Backward compatibility breaks**: Don't rename internal keys.
5. **Untested edge cases**: Handle boundary values (0, max_int, negative) and unknown inputs.

#### Non-Goals
- No UX copy changes.
- No visual changes.
- No section reorder.
- No removal of direct state access.

---

## Phase 3 Prompt — Introduce Workflow-Aware Sidebar Sections

### Prompt
Redesign sidebar information architecture to be context-aware and progressive. Show the right controls at the right time based on workflow mode, user progression, and project state. Reduce cognitive load by hiding irrelevant controls and prominently displaying blockers.

#### Goal
Transform sidebar from "all controls visible all the time" to "smart, context-aware progression." Users see: what they need next (clear CTA at top), only relevant workflow controls, live status of dependencies, clear error states and recovery paths.

#### Detailed Context
Users encounter cognitive friction because:
- 15+ controls visible simultaneously regardless of context
- No hierarchy between critical (project setup) and optional (advanced seed)
- No project → sidebar still shows generation controls (confusing)
- Queue status only visible if switching tabs
- New users don't know where to start

Phase 3 restructures sidebar with conditional sections that respect user progression and app state.

#### Scope
Restructure sidebar into 5 sections with visibility rules:

1. **Workflow Mode Section** (always visible):
   - Toggle between "Architecture Workflow" (8-step guided) and "Legacy Tabs".
   - Current mode indicator with brief description.

2. **Project Context Card** (conditionally visible):
   - Project name/ID, type, phase, geography.
   - CTA: "Start New Project" if no active project.
   - Visibility: If `active_project_id` exists, show rich card; else show prompt.

3. **Global Generation Controls** (conditionally visible):
   - Model selection, quality preset, output size, seed management, advanced toggle.
   - Visibility: If `active_project_id` exists, show full; else gray out with tooltip.

4. **Operations Panel** (always visible, adaptive):
   - Live queue status, auto-run toggle, last generation summary.
   - Visibility: Always visible but density varies (compact when empty, expanded when active).

5. **Session Tools** (collapsible):
   - History browser, session reset, export links, debug tools.
   - Visibility: Collapsed by default; expanded if clicked or error detected.

#### Requirements
1. Keep all existing functionality (no deletions, only conditional hiding).
2. Add conditional visibility based on state (project, queue, history, errors).
3. Maintain both navigation modes (Architecture + Legacy).
4. Preserve state continuity across reruns.

#### Implementation Notes
- Use `st.expander()` for collapsible sections.
- Use `st.columns()` for side-by-side status chips.
- Use `st.info()`, `st.warning()`, `st.error()` for feedback.
- Store visibility state: `session_state['sidebar_sections']`.
- Implement visibility helpers: `should_show_X()` for each section.
- Use consistent icons/emojis for status (✓, ⏸, ⚠).

#### Deliverables
1. **Restructured sidebar**: 5 sections with visibility rules.
2. **Context card**: Project metadata and CTA.
3. **Visibility helpers**: Codified rules in helper functions.
4. **Reduced clutter**: Same functionality, better organization.

#### Validation Checklist
- ✅ Fresh start: "Project Context" shows "Start New Project" CTA.
- ✅ After project creation: Card shows project name, type, phase.
- ✅ Generation controls activate when project exists.
- ✅ Queue starts: Operations panel expands with queue count.
- ✅ Queue finishes: Operations panel compacts.
- ✅ Sections collapse/expand based on visibility state.
- ✅ Workflow toggle switches without section loss.
- ✅ History collapsed by default.
- ✅ Error states show warning badges.
- ✅ New users find starting point in <10 seconds.
- ✅ Active project always obvious.
- ✅ Legacy mode works identically.
- ✅ No state regressions.

#### Common Pitfalls to Avoid
1. **Premature hiding**: Use expanders, not if statements (avoids state loss).
2. **State loss on toggle**: Keep widgets in DOM, hide via expander.
3. **Ignored blockers**: Gray out controls, show tooltip explaining why.
4. **Missing feedback**: Update operations panel live without requiring interaction.
5. **Confusing labels**: Use action-oriented, clear terminology.

#### Non-Goals
- No visual overhaul.
- No new features.
- No control removal.

---

## Phase 4 Prompt — Visual and Interaction Polish (UX Consistency)

### Prompt
Upgrade the sidebar's visual quality, scanning hierarchy, and usability while staying consistent with Streamlit theme constraints and architecture/urban design domain language. Focus on clarity, conciseness, and professional presentation.

#### Goal
Make sidebar cohesive, professional, and intellectually engaging. Users should scan and understand controls in <3 seconds per section.

#### Detailed Context (Common Issues)
- Section headings are generic ("Model", "Seed", "Queue") lacking domain tone.
- Microcopy is inconsistent (buttons say "Start Queue" vs "Begin Auto-Run").
- Status scattered (queue count buried in history, generation time missing).
- Expanders default-open for low-value sections, closed for critical ones.
- No visual feedback for disabled states.
- Model names bare (`flux-base`) without quality/performance hints.

#### Scope
**Improve section headings**:
- "Workflow Mode" → "🎯 Design Workflow"
- "Project Context" → "📍 Active Project"
- "Generation Controls" → "✨ Generation Parameters"
- "Operations" → "⚙️ Generation Queue"
- "Session Tools" → "🛠 Session Control"

**Use concise captions and status chips**:
- Replace: `st.write("Selected Model: flux-base")`
- With: `st.caption("Model")` + `st.write("🚀 **FLUX Base** | Fast, Balanced | ~8GB VRAM")`
- Model profiles show: Speed tier, Quality tier, Est. VRAM.

**Standardize action labels**:
- All start buttons: "▶ Start Auto-Run"
- All pause buttons: "⏸ Pause"
- All reset actions: "↻ Clear" or "↻ Reset"
- Success: "✓ [Action] complete" (green, 2s auto-dismiss)
- Error: "✗ [Action] failed: [reason]" (red, stay visible)

**Rebalance expanders: smart defaults**:
- Default-OPEN: "Generation Controls", "Project Context"
- Default-OPEN (if queue active): "Operations"
- Default-CLOSED: "Session Tools", "Advanced Settings"

**Add domain-appropriate terminology**:
- Instead of "seed" → "Generation Seed"
- Instead of "quality" → "Output Quality" or "Render Quality"
- Instead of "size" → "Canvas Dimensions" or "Output Resolution"
- Model names: "FLUX Base", "FLUX Pro" (aspirational)

**Visual polish within Streamlit**:
- Use dividers (`st.divider()`) between sections.
- Use columns for aligned status indicators.
- Use `st.info()`, `st.warning()`, `st.error()` for feedback.
- Use spacing (`st.write("")`) for breathing room.
- Use markdown: `**Bold**` for primary, `_Italic_` for secondary.

#### Requirements
1. **No custom color systems**: Use Streamlit theme primitives only (primary, secondary, success, warning, error).
2. **Manage control density**: Sidebar ~300px max width, max 3 fields per row, labels wrappable at 250 chars.
3. **Domain-appropriate labels**: Test with architecture professionals; avoid jargon.
4. **Keyboard-friendly**: Avoid deep expander nesting; reasonable button sizes.

#### Implementation Notes
- Create constant file `ui/sidebar_copy.py` for centralized copy:
  ```python
  SECTION_LABELS = {
      'workflow_mode': '🎯 Design Workflow',
      'project_context': '📍 Active Project',
      'generation_controls': '✨ Generation Parameters',
      'operations': '⚙️ Generation Queue',
      'session_tools': '🛠 Session Control',
  }
  
  MODEL_PROFILES = {
      'flux-base': {
          'display_name': 'FLUX Base',
          'description': 'Fast, Balanced',
          'vram': '~8GB',
          'icon': '🚀',
      },
  }
  
  ACTION_LABELS = {
      'start_queue': '▶ Start Auto-Run',
      'pause_queue': '⏸ Pause',
      'resume_queue': '▶ Resume',
      'reset_session': '↻ Clear Session',
  }
  ```
- Manage expander state:
  ```python
  if 'sidebar_expander_state' not in st.session_state:
      st.session_state['sidebar_expander_state'] = {
          'generation_controls': True,
          'operations': False,
          'session_tools': False,
      }
  ```

#### Deliverables
1. **Polished section titles**: Renamed with emoji, consistent usage.
2. **Consistent button labels**: All similar actions use same icon + verb.
3. **Smart expander defaults**: Generation/Project open, Tools closed.
4. **Professional descriptions**: Model/preset show icon + name + 1-line summary.
5. **Clear visual hierarchy**: Disabled states marked, feedback uses color + icons, dividers between sections.

#### Validation Checklist
- ✅ Sidebar readable at 1280px–1440px (zoom 80–150%).
- ✅ Section titles concise (max 30 chars with emoji).
- ✅ No overwhelming long text (labels 1 line, descriptions 2 lines max).
- ✅ Buttons labeled consistently.
- ✅ Model/preset information scannable (icon + name + 1-line summary).
- ✅ Expander defaults intuitive.
- ✅ Error messages clear and actionable.
- ✅ Reduced repetitive copy.
- ✅ Emoji consistent and meaningful.
- ✅ Domain terminology correct (architect-approved).
- ✅ Colors match Streamlit theme (no custom CSS).

#### Common Pitfalls to Avoid
1. **Emoji overuse**: One per section, not per word.
2. **Truncated labels**: Let text wrap; don't abbreviate for space ("Seed: Rnd"?).
3. **Inconsistent status icons**: Use ✓, ✗, ⚠, ⧖ consistently.
4. **Color assumptions**: Test in light/dark mode; don't assume colors mean specific states.
5. **Disabled confusion**: Show why control is disabled ("Complete project setup first").

#### Non-Goals
- No feature-level logic changes.
- No control removal/reordering.
- No animations (Streamlit doesn't support well).
- No custom CSS.

---

## Phase 5 Prompt — Operational Sidebar Enhancements

### Prompt
Enhance sidebar operational usefulness by adding compact live summaries and safe quick actions tied to queue/progress/session health. Allow users to monitor and control run-state without switching tabs.

#### Goal
Enable real-time visibility and control of critical app systems (queue, generation, session) from sidebar without context-switching.

#### Detailed Context
Current limitations:
- Queue status only visible in queue tab (users unaware of auto-run state).
- No quick access to common recovery actions (undo, restore, reset).
- Generation progress scattered across multiple locations.
- High-frequency actions buried in page controls.

Phase 5 enhances sidebar with live monitoring and safe quick actions.

#### Scope
**Live Status Strip**:
- Queue counts (pending, running, completed, failed).
- Auto-run state indicator (ON/OFF with toggle).
- Last generation summary: model, seed, timestamp, duration.
- Estimation of next generation (if in queue).

**Safe Quick Actions**:
- "Apply Recommended" — applies production-ready defaults for current project type.
- "Restore Last Successful" — reverts to last generation that completed without errors.
- "Clear Session" — clears transient fields (temp images, results) but preserves project registry.
- "Undo Last" — reverts generation parameters to previous state.

**Session Health Indicators**:
- Project staleness warning (if no work done in X hours).
- Storage usage indicator (disk space consumed by current project).
- Error badge (if generation failed; linkt to error details).

#### Requirements
1. **All quick actions must be reversible or clearly scoped**.
   - "Clear Session" must not delete projects or permanent settings.
   - "Undo" should work across session_state limits (store undo history).

2. **Do not delete persistent user context unintentionally**.
   - Only clear: temp images, transient form fields, generation artifacts.
   - Preserve: project registry, project settings, saved presets, history.

3. **Keep queue actions synchronized** with `queue_auto_run` and `queue_paused`.
   - Live counts must reflect actual queue state (sync with history data).

4. **Surface blockers early**.
   - Missing project → "Complete project setup first" CTA.
   - Missing reference image (editor) → "Upload reference image" warning.
   - Insufficient disk space → "Insufficient disk space: X GB needed, Y GB available".

5. **Performance**: Live status must update without manual refresh.
   - Use Streamlit's reactive state (no polling if possible).
   - Cache queue counts if polling necessary (<1s refresh).

#### Implementation Notes
- Create `ui/sidebar_quick_actions.py` module:
  ```python
  def apply_recommended(project_type: str) -> dict:
      """Apply production defaults for project type."""
      
  def restore_last_successful() -> bool:
      """Restore last successful generation."""
      
  def clear_session_transients() -> None:
      """Clear temp files, preserve project context."""
      
  def undo_last_action() -> bool:
      """Revert to previous generation parameters."""
  ```
- Store action history: `session_state['action_history']` (list of dicts with timestamp, action, params).
- Implement session health checks: `check_session_health()` → dict of warnings.
- Live queue sync via: `update_queue_display()` function called on every sidebar render.

#### Deliverables
1. **Live Status Panel**:
   - Queue counts (pending, running, completed).
   - Auto-run toggle + status indicator.
   - Last generation card (model, seed, timestamp, duration).
   - Next generation ETA (if queue not empty).

2. **Quick Action Buttons**:
   - "🎨 Apply Recommended" (applies safe defaults).
   - "↩️ Restore Last Success" (undo-like behavior).
   - "↻ Clear Session" (temp cleanup).
   - "⬅️ Undo" (parameter rollback).

3. **Session Health Indicators**:
   - Warnings for staleness, disk space, errors.
   - Action history for undo/redo capability.

4. **Faster user workflows**:
   - Monitor queue without tab-switching.
   - Recover from errors with single click.
   - Apply safe defaults instantly.

#### Validation Checklist
- ✅ Queue status updates live without page refresh.
- ✅ Queue pending/running counts are accurate.
- ✅ Auto-run toggle syncs with session_state['queue_auto_run'].
- ✅ "Apply Recommended" sets correct defaults for project type (Residential, Commercial, etc.).
- ✅ "Restore Last Success" reverts exactly to last successful generation params.
- ✅ "Clear Session" deletes temp images but preserves project registry.
- ✅ "Undo" restores previous parameter state (works for ~5 actions back).
- ✅ All quick actions show success message ("✓ Action complete").
- ✅ All quick actions show error message if they fail ("✗ Action failed: [reason]").
- ✅ Session health warnings appear (staleness after 2+ hours, low disk after 80% usage).
- ✅ Error badge visible if last generation failed.
- ✅ Users can manage queue flow from sidebar only.
- ✅ No accidental destructive operations (Clear Session prompts confirm).
- ✅ Action history enables undo (min 3, max 10 states stored).

#### Common Pitfalls to Avoid
1. **Unclearity on "Clear"**: Specify exactly what gets deleted. Users fear losing projects.
2. **Missing undo history**: Store full state snapshots, not just deltas; allows true undo.
3. **Stale queue display**: Cache with <1s TTL; don't require manual refresh.
4. **Confusing "Recommended"**: Document what "recommended" means (e.g., "Production defaults for Mixed-Use Commercial").
5. **Destructive without confirm**: Always confirm before clearing, deleting, or undoing.
6. **Lost history**: If session crashes, action history lost; add persistent undo via checkpoint saving.

#### Non-Goals
- No backend queue redesign.
- No new generation models or parameters.
- No persistent undo across sessions (in-memory only; could be future phase).
- No advanced queue scheduling (job prioritization, batching).

---

## Phase 6 Prompt — Hardening, QA, and Rollout Strategy

### Prompt
Prepare the redesigned sidebar for stable rollout with targeted validation and backward compatibility checks. Ensure reliability after refactor and UX upgrades before production release.

#### Goal
Ensure zero regressions and confident rollout. All workflows remain stable; new features add value without introducing risk.

#### Detailed Context
After 5 phases of refactoring and enhancement, the sidebar is substantially different structurally and functionally. Before rollout:
- Confirm all widget keys work as expected (no state loss).
- Verify both navigation modes (Architecture + Legacy) function identically.
- Test all user journeys (project setup → generation → queue → progress).
- Document known limitations and defer to future phases.
- Create final validation checklist and rollout recommendation.

#### Scope
**Smoke Testing** (per navigation mode):
1. Fresh app start: No errors, sidebar renders, project CTA visible.
2. Project creation: Context card appears, generation controls activate.
3. Generation: Model/preset selection works, seed controls predictable.
4. Queue: Start auto-run, monitor queue, pause/resume, queue finishes.
5. History: Restore from history, undo last action, clear session.
6. Editor: Load reference image, enable smart crop, generation with reference.
7. Export: Generate image, export to various formats, verify file integrity.
8. Settings: Change advanced parameters, apply custom presets, theme toggle.

**Backward Compatibility Checks**:
- Session state keys unchanged (no renames, no deletions).
- Page components (generator, editor, queue, progress_monitor) unaffected.
- Existing user projects load without corruption.
- Existing saved presets still apply correctly.
- Existing history checkpoints still restore correctly.

**Key State Transitions** (verify no silent failures):
- `workflow_mode` toggle: Architecture ↔ Legacy (both modes render).
- `active_project_id` change: Project context card updates, controls enable/disable.
- `queue_auto_run` toggle: Queue starts/stops, operations panel updates.
- `queue_paused` toggle: Queue pauses/resumes seamlessly.
- Model/preset changes: Recommended sizes/seeds apply instantly.
- Seed toggle: Random ↔ Manual mode, values persist.

**Error Handling**:
- Missing project: Error gracefully, show CTA.
- Missing reference image (editor mode): Warn, block generation until uploaded.
- Insufficient disk space: Warn, prevent generation if <500MB free.
- Invalid seed input: Reject, show message ("Seed must be 0–" + max_seed).
- Network error during queue sync: Show retry button, use cached queue state.

#### Requirements
1. **No regressions in widget key behavior**.
   - All keys from Phases 1–5 function identically to original.
   - Session state persists across app reruns.

2. **No loss of existing controls**.
   - All sidebar controls from original implementation remain accessible.
   - No controls removed, only conditionally hidden (via expander).

3. **Clear fallback behavior when context incomplete**.
   - Missing project: Show CTA, block generation, explain why.
   - Missing reference image: Show inline warning, disable confirm button.
   - Queue error: Show error message, offer "Retry" button.

4. **Document known limitations and follow-up items**.
   - Create appendix listing minor issues deferred to future phases.
   - Example: "Undo history not persistent across sessions (Phase 6+)".
   - Example: "Queue estimation ETA accuracy ±30s (Phase 7 optimization)".

#### Implementation Notes
- Create `tests/test_sidebar_qa_checklist.md` (or append to this file).
- Assign QA tester to manually follow smoke test script per browser/OS combo.
- Use Streamlit's native logging to capture any key collisions or state warnings.
- Session state snapshots before/after each major action (project create, queue start, etc.); compare to detect drift.
- Test both light and dark Streamlit themes.

#### Deliverables
1. **QA Checklist Document**: Detailed smoke test script with pass/fail criteria.
2. **Backward Compatibility Report**: Test results for widget keys, state transitions, page components.
3. **Known Limitations Appendix**: List of deferred issues with justification.
4. **Final Rollout Recommendation**: "Safe to Release" or "Needs Patch" with reasoning.

#### Validation Manual (Smoke Test Script)

**Test Scenario 1: Fresh Start (Architecture Mode)**
```
1. [Clear browser cache / restart app]
2. Sidebar appears with 5 sections (Workflow, Project, Generation, Operations, Tools)
3. "Project Context" shows "→ Start New Project" CTA
4. Generation Controls are grayed out (no tooltip error)
5. "Operations" shows "No active queue" or similar empty state
6. Toggle workflow mode to "Legacy Tabs" → both modes visible
7. Toggle back to "Architecture" → no data loss
[PASS/FAIL: ___]
```

**Test Scenario 2: Create Project & Generate**
```
1. Click "Start New Project" CTA → Navigate to Project Setup
2. Fill: Name="Test Urban", Type="Mixed-Use", Phase="Design Dev", Location="NYC"
3. Click "Create Project"
4. Return to main app (should auto-navigate if workflow-aware)
5. Sidebar updates: "Project Context" card shows "Test Urban | Mixed-Use | Design Dev | NYC"
6. Generation Controls activate (no longer grayed)
7. Select Model → "FLUX Base"
8. Select Preset → "Standard" (1024×1024, Random Seed)
9. Click "Generate"
10. Queue status updates: "Queue: 1 pending"
11. Wait for generation complete (may take 30–120s depending on hardware)
12. "Last generation" summary appears: Model, Seed, Timestamp, Duration
[PASS/FAIL: ___]
```

**Test Scenario 3: Queue Control**
```
1. Start 3 generations (rapid "Generate" clicks)
2. "Operations" panel shows "Queue: 3 pending"
3. Click "⏸ Pause" → Queue pauses, button changes to "▶ Resume"
4. Wait 5 seconds, verify queue doesn't advance
5. Click "▶ Resume" → Queue resumes
6. Wait for all 3 to complete
7. "Operations" shows "Queue: 0 pending, 3 completed"
[PASS/FAIL: ___]
```

**Test Scenario 4: History & Restore**
```
1. Generate 3 images with different models (flux-base, flux-dev, flux-pro)
2. Expand "Session Tools" → History shows 3 entries
3. Click "Restore" on 2nd entry (flux-dev generation)
4. Model dropdown updates to "FLUX Dev", seed updates to saved value
5. Click "Generate" → Uses restored parameters
6. New generation uses FLUX Dev model (verify in output metadata or image style)
[PASS/FAIL: ___]
```

**Test Scenario 5: Error Handling**
```
1. Delete reference image from editor (if editor flow supports this)
2. Try to generate in editor mode
3. System shows: "✗ Missing reference image. Upload image to continue."
4. "Generate" button disabled with tooltip
5. Upload valid image
6. "Generate" button enables
7. Generate succeeds
[PASS/FAIL: ___]
```

**Test Scenario 6: Session Reset**
```
1. Generate 2 images, history shows 2 entries
2. Expand "Session Tools" → Click "↻ Clear Session"
3. Confirmation dialog: "Clear transient files? (History retained)" [Yes/No]
4. Click "Yes"
5. Temp images deleted, history still present
6. Queue status resets to empty
7. Last generation summary clears
[PASS/FAIL: ___]
```

**Test Scenario 7: Theme Toggle**
```
1. Open Streamlit settings menu (top-right)
2. Toggle between Light and Dark theme
3. Sidebar remains readable, no color/contrast issues
4. Emojis render correctly (✓, ⚠, ⏸, ▶, etc.)
5. Buttons and text remain legible
[PASS/FAIL: ___]
```

**Test Scenario 8: Legacy Mode Compatibility**
```
1. Toggle workflow mode to "Legacy Tabs"
2. All tabs (Project, Generator, Editor, Queue, Progress, etc.) appear as flat list
3. Sidebar still renders 5 sections identically
4. Model selection in sidebar affects "Generator" tab
5. Queue controls in sidebar affect "Queue" tab
6. Preset application works across both modes
[PASS/FAIL: ___]
```

#### Validation Checklist (Automated Checks)
- ✅ App starts cleanly with no sidebar key exceptions or warnings in console.
- ✅ No Streamlit logger errors (e.g., "Duplicate widget key").
- ✅ Sidebar controls persist values correctly across reruns (refresh page, state unchanged).
- ✅ Queue and history interactions remain stable (no data loss on navigation).
- ✅ Model sync logic triggers correctly (recommended sizes appear instantly).
- ✅ Seed mode toggle clears opposite mode's values.
- ✅ Expander state persists across reruns (if user expands Tools, stays expanded).
- ✅ All buttons have working `on_click` handlers (no mysterious unresponsive buttons).
- ✅ Error messages render without HTML breaking (no raw HTML in st.write()).
- ✅ No visual overflow on 1280px width (sidebar fits without horizontal scroll at zoom=100%).
- ✅ All action labels use consistent emojis and verbs.
- ✅ Disabled buttons show tooltip explaining why.

#### Known Limitations (Examples; Replace with Actual Findings)
1. **Undo history not persistent**: Action history stored in session_state; lost on app restart. Mitigation: Could add checkpoint-based undo in Phase 7.
2. **Queue ETA ±30s accuracy**: Estimation based on average hardware; actual time varies. Mitigation: Display as "est." and update as generation progresses (Phase 7).
3. **No visual undo/redo history**: Users can undo via "↩️ Restore", but no visual "undo stack". Mitigation: Minimal UI impact; could add undo breadcrumb in Phase 8.

#### Rollout Recommendation
- **Safe to Release** (if all test scenarios PASS)
- **Needs Patch** (if 1–2 scenarios fail; specify which and remediation)
- **Hold Release** (if 3+ scenarios fail; major rework needed)

[Place final recommendation and justification here after testing.]

---

## Suggested Execution Order

1. **Phase 1** (modularization) — Extract sidebar to `ui/sidebar.py`
2. **Phase 2** (state normalization) — Add helpers to `ui/state.py`
3. **Phase 3** (workflow-aware sections) — Restructure sidebar into 5 sections
4. **Phase 4** (visual polish) — Improve copy, labels, expander defaults
5. **Phase 5** (operational enhancements) — Add live status & quick actions
6. **Phase 6** (hardening & QA) — Full smoke testing & rollout

---

## Notes for Implementation Agents

- Preserve Streamlit widget keys unless migration is explicitly designed.
- Prioritize backward-compatible changes in early phases.
- Avoid changing page-level domain logic while restructuring sidebar.
- Keep architecture/urban terminology precise and concise.
- Favor incremental PR-sized changes per phase (one phase per PR recommended).
- Test both Architecture workflow and Legacy tab modes frequently.
- Document any deviations from this plan with clear reasoning.
- Escalate blockers early (missing dependencies, conflicting page logic, etc.).
