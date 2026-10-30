# Changelog

All notable changes to the **HaDeS Anima Suite** are documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.1.8] - 2026-10-30

### Fixed
- **Suite Telemetry NAG Detection (Audit A2):**
  - Updated Attention Guidance parser in `HaDeSAnimaSuiteTelemetry` to detect `HaDeS Anima NAG Active` and `HaDeS NAG: Bypassed` without requiring legacy `"ATTENTION GUIDANCE"` substring or `Active: True`.
  - Accurately extracts `Scale` and `Tau` parameters directly from active attention guidance streams.
  - Eliminates false *"Low CFG used without Anima Attention Guidance (NAG)"* warnings on Turbo/Lightning generation at `CFG = 1.0` when NAG is active.
- **Prompt Studio Widget Alignment & Deserialization (Audit A1):**
  - Added dual positional/named schema auto-detection in `node.configure` for `HaDeSAnimaPromptStudio`, supporting both named restoration and positional fallback.
  - Reordered `widgets_values` across all 4 example workflows in `examples/` to match standard Python `INPUT_TYPES` (`positive_prompt` at index 6, `negative_prompt` at index 7, `header_mode` at index 8, `header_trigger` at index 9, `artwork_title` at index 10).
  - Guarantees 100% interoperability across standard ComfyUI, headless API runners, and external static workflow analyzers.
- **Workflow Usability Polish:**
  - Updated Stage 3 of `Triple Pass Generation Example.json` to standard community checkpoint `anima-base-v1.0.safetensors`.
- **Automated Verification:**
  - Added regression test assertions in `test_22_anima_suite_telemetry` verifying real production strings for Active vs Bypassed NAG states.

---

## [0.1.7] - 2026-10-30

### Added
- **Official Workflow Examples (`examples/`):**
  - `HaDeS Anima Suite - Simple Pass Generation Example.json`: Complete single-pass reference pipeline featuring native Loaders, Prompt Studio with Slop Guard, NAG Attention Guidance, Rectified Flow KSampler, Smart VAE Decode, and Telemetry Flight Recorder.
  - `HaDeS Anima Suite - Dual Pass Generation Example.json`: Two-stage HiRes generation pipeline with 16-channel Variance-Preserving Latent Bridge, 32px macroblock snapping, and hybrid model handoff.
  - `HaDeS Anima Suite - Triple Pass Generation Example.json`: Progressive 3-stage staircase pipeline (`1.0x` -> `1.35x` -> `1.75x`) for ultra-high-resolution poster rendering with zero anatomical drift.
  - `HaDeS - Anima Suite - Merge Workflow Example - Medium.json`: Advanced block-level model & LoRA merging workflow with QR-SVD rank adaptation, 24-rule Diffusers remapping, and `llm_adapter` golden rule protection.
- **HaDeS UX vs Vanilla ComfyUI Theme Switch Documentation:**
  - Added full user guide in `README.md` explaining how to toggle between the high-visibility **HaDeS Spectral Canvas UI** (cyberpunk badges, preset buttons, interactive HUDs, tooltips) and the **Native Classic ComfyUI Theme (Vanilla)** via the ComfyUI Settings menu (`HaDeS.ClassicUI`).
- Contextual hover tooltips across HUD banners, buttons, and custom widgets on all 18 nodes.
- Slop Guard expanded with `trending on artstation`.

### Fixed
- **Suite Telemetry ComfyUI Dispatch:**
  - Declared `OUTPUT_NODE = True` and formatted return as `{"ui": ..., "result": ...}` for real-time frontend execution event dispatch.
- **Suite Telemetry Regex Isolation:**
  - Restricted canvas resolution extraction to `SAMPLING` and `POST_PROCESSING` streams with explicit tags, preventing false 32px misalignment warnings caused by Corner Shield's `Box: WxHpx`.
  - Restricted VAE channel checks to VAE streams, preventing false errors on RGB images.
  - Fixed LoRA check so `Prompt Studio` alone no longer triggers false `External LoRAs detected` tips.
  - Fixed NAG scale parser to pull strictly from Attention Guidance streams rather than global text.
- **Output Slot Standardization:**
  - Renamed `scribble_mask` to `mask` in `Anima Corner Shield`.
  - Renamed `model_info`, `clip_info`, `vae_info` to `info` across all Loader nodes.
  - Implemented automatic, non-destructive client-side slot migration in JavaScript (`migrateOutputSlot`) preserving existing saved workflow wires.

### Changed
- Standardized package name in `pyproject.toml` to `hades-anima-suite` for Comfy Registry publication.
- `model_pass2` officially documented as primary input socket in KSampler (with backward-compatible fallback for legacy `model_pass2_optional`).

---

## [0.1.6] - 2026-10-29

### Added
- **`Anima Suite Telemetry` (Node #18):**
  - Centralized Flight Recorder & Pipeline Auditor for ComfyUI.
  - Automatically aggregates up to 8+ incoming `info` streams and semantically classifies them into pipeline phases (`ARCHITECTURE`, `CONDITIONING`, `SAMPLING`, `POST_PROCESSING`).
  - Automated Diagnostic & Health Engine: detects Flow Shift divergence, validates Turbo/Lightning CFG=1 NAG guidance, checks 32px macroblock canvas alignment, verifies 16ch Wan2.1 VAE integrity, and confirms `llm_adapter` text bridge protection.
  - Multi-format outputs: Markdown audit report, structured JSON telemetry, and compact single-line diagnostics string.
  - Interactive HUD console on canvas with real-time status monitor and one-click clipboard copying (`Copy Report`, `Copy JSON`).

---

## [0.1.5] - 2026-10-29

### Fixed
- **Corner Scribble Shield:** Resolved batch cross-talk edge case; verified 100% isolation across heterogeneous batches.
- **NAG Attention Guidance:** Fixed dynamic sigma shift propagation across cloned patchers and custom schedulers.
- **Prompt Studio:** Resolved semantic booru tag disambiguation (`hairstyle` vs `@artist` / `style`).
- **Frontend Canvas:** Implemented LiteGraph minimum width clamping (`minW = 440px`) across all single, dual, and triple pass samplers.

---

## [0.1.4] - 2026-10-28

### Added
- Standalone `Anima Corner Shield` node extracted from VAE Decode for modular composability and second-pass inpainting pipelines.

---

## [0.1.3] - 2026-10-27

### Added
- Native Diffusers-aware `Anima Diffusion Model Loader` with on-the-fly 24-rule remapping.
- `Anima Text Encoder Loader` with strict 1024-dim Qwen3-0.6B guard.
- `Anima VAE Loader` with pre-instantiation 16-channel Wan2.1 validation.
- Proactive `Anima VAE Decode` with auto-tile memory thresholding (>1.35 MP).

---

## [0.1.2] - 2026-10-26

### Added
- Rectified Flow KSampler Trilogy: `Anima KSampler`, `Anima Dual-Pass KSampler`, and `Anima Triple-Pass KSampler`.
- 16-channel latent variance restoration bridge for crisp lineart during multi-pass upscaling.

---

## [0.1.1] - 2026-10-25

### Added
- `Anima Prompt Studio` & Aesthetic Slop Guard.
- `Anima Attention Guidance (NAG)` with L1-norm anti-burn bounds.
- `Anima Latent` pipeline guard with 32px aspect ratio presets.
- `Anima Universal LoRA Remapper` supporting Diffusers, Kohya, OneTrainer, and Musubi.

---

## [0.1.0] - 2026-10-24

### Initial Release
- 34-band canonical Anima spectral block merger engine.
- Predictive Subspace Model Merger & Multi-LoRA Loader.
- QR-SVD high-speed orthogonal LoRA exporter.
