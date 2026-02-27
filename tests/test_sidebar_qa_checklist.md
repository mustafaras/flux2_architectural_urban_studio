<!-- markdownlint-disable MD022 MD032 MD035 MD060 -->

# Sidebar Phase 6 QA Checklist (Manual + Targeted Validation)

Purpose: Validate sidebar rollout readiness after refactor/polish phases with explicit backward-compatibility checks.

Owner: QA / Release Engineer  
Build under test: `ui_flux2_professional.py` + `ui/sidebar.py`  
Date: __________  
OS/Browser Matrix: __________

---

## 1) Preflight

- [ ] Launch app with clean session: `streamlit run ui_flux2_professional.py`
- [ ] Confirm app starts with no import/runtime errors.
- [ ] Open logs and verify no Streamlit duplicate-key warnings.
- [ ] Capture initial `st.session_state` snapshot (see Section 7 template).

Expected key baseline (must exist after init):
- `model_name`, `quality_preset`, `width`, `height`, `seed`, `use_random_seed`
- `queue_auto_run`, `queue_paused`, `generation_history`, `active_project_id`
- `use_architecture_workflow`, `show_legacy_navigation`

---

## 2) Smoke Scenarios (Architecture + Legacy)

### Scenario 1 — Fresh Start
- [ ] Sidebar renders all 5 sections.
- [ ] Project section shows new-project CTA when no active project.
- [ ] Generation controls show disabled guidance until project exists.
- [ ] Operations panel shows empty/idle queue state.
- [ ] Toggle workflow/legacy navigation controls and return to baseline.

PASS/FAIL: ____  Notes: ______________________

### Scenario 2 — Create Project + Generate
- [ ] Create project context from Project Setup.
- [ ] Sidebar Project card reflects created project.
- [ ] Generation controls activate.
- [ ] Select model + preset, generate once.
- [ ] Queue and last-generation summary update.

PASS/FAIL: ____  Notes: ______________________

### Scenario 3 — Queue Control
- [ ] Queue 3 generations.
- [ ] Pause and confirm queue does not progress.
- [ ] Resume and confirm queue drains to completion.
- [ ] Auto-run state chip and counts stay accurate.

PASS/FAIL: ____  Notes: ______________________

### Scenario 4 — History + Restore + Undo
- [ ] Generate multiple entries.
- [ ] Restore one entry from Session Tools history.
- [ ] Confirm model/seed settings restore.
- [ ] Trigger quick-action Undo and validate rollback.

PASS/FAIL: ____  Notes: ______________________

### Scenario 5 — Editor Missing Reference
- [ ] Enter editor mode without reference image.
- [ ] Confirm generation blocked with clear error.
- [ ] Upload image and confirm generation unblocks.

PASS/FAIL: ____  Notes: ______________________

### Scenario 6 — Clear Session (Scoped)
- [ ] Use clear-session confirmation checkbox.
- [ ] Execute clear session.
- [ ] Confirm transient fields clear; project context/history remain.

PASS/FAIL: ____  Notes: ______________________

### Scenario 7 — Theme Toggle
- [ ] Toggle Streamlit Light/Dark.
- [ ] Verify readability/contrast/icons in sidebar.

PASS/FAIL: ____  Notes: ______________________

### Scenario 8 — Legacy Mode Compatibility
- [ ] Navigate using Legacy Compatibility mode.
- [ ] Confirm sidebar controls affect generator/editor/queue tabs.
- [ ] Confirm parity with Architecture workflow outcomes.

PASS/FAIL: ____  Notes: ______________________

---

## 3) Backward-Compatibility Checks

### Session Key Compatibility (No regressions)
- [ ] No removal of active state keys relied upon by pages (`generator`, `editor`, `queue`, `progress_monitor`).
- [ ] Model sync remains centralized via `_state.sync_model_selection_state()`.
- [ ] Sidebar reruns do not reset generation parameter state unexpectedly.

### Component Compatibility
- [ ] `generator` still consumes sidebar model/preset/seed params.
- [ ] `editor` still enforces reference-image preconditions.
- [ ] `queue` still honors `queue_auto_run` and `queue_paused`.
- [ ] `progress_monitor` remains reachable/functional in both nav modes.

### Artifact Compatibility
- [ ] Existing projects load and activate successfully.
- [ ] Existing saved presets apply without errors.
- [ ] Existing history checkpoints restore parameters correctly.

---

## 4) Key State Transition Matrix

| Transition | Action | Expected Result | Pass/Fail | Notes |
|---|---|---|---|---|
| Navigation mode | Toggle architecture/legacy controls | Content routing changes, no sidebar state loss | ___ | ___ |
| Active project | Set/clear `active_project_id` | Context card + control enablement update correctly | ___ | ___ |
| Queue run-state | Toggle `queue_auto_run` | Operations state chip and behavior stay in sync | ___ | ___ |
| Queue pause-state | Toggle `queue_paused` | Queue pauses/resumes with accurate UI indicators | ___ | ___ |
| Model/preset | Change model and apply preset | Recommended defaults update immediately | ___ | ___ |
| Seed mode | Random/manual switch | Opposite mode value not leaked; seed persists as expected | ___ | ___ |

---

## 5) Error-Handling Validation

- [ ] Missing project: CTA shown, generation controls explain disabled state.
- [ ] Missing editor reference: clear blocking message, generation prevented.
- [ ] Low disk (<500MB): warning shown; generation prevented where enforced.
- [ ] Invalid seed input: value constrained/rejected with clear guidance.
- [ ] Queue sync failure simulation: sidebar shows cached state + `Retry Queue Sync` action.

PASS/FAIL: ____  Notes: ______________________

---

## 6) Automated/Static Validation Quick List

- [ ] `ui/sidebar.py` imports cleanly.
- [ ] No duplicate Streamlit widget key warnings at startup.
- [ ] No linter/type errors introduced by Phase 6 patch.
- [ ] Queue sync fallback path verified (exception path + retry button visible).

---

## 7) Session Snapshot Template (Drift Detection)

Capture before/after for these actions: project create, queue start, queue pause, restore history, clear session.

```text
Action: ______________________
Timestamp: ___________________
Before:
  model_name: __________
  quality_preset: ______
  width/height: ________
  seed/use_random_seed: _
  queue_auto_run: ______
  queue_paused: ________
  active_project_id: ___
After:
  model_name: __________
  quality_preset: ______
  width/height: ________
  seed/use_random_seed: _
  queue_auto_run: ______
  queue_paused: ________
  active_project_id: ___
Observed drift (if any): ______________________
```

---

## 8) Known Limitations (Record Actual Findings)

1. ____________________________________________
2. ____________________________________________
3. ____________________________________________

---

## 9) Final Recommendation

- [ ] Safe to Release
- [ ] Needs Patch
- [ ] Hold Release

Justification:

________________________________________________
________________________________________________
