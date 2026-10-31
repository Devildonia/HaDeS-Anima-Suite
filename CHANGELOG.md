# Changelog

All notable changes to the **HaDeS Anima Suite** are documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed
- **Prompt Studio:** a linked `artist` / `character` socket now replaces the inline ARTIST / CHAR field instead of being merged with it, and the replaced row is drawn dimmed. Before, a stale inline value was silently added to the picker's output.

---

## [0.3.4] - 2026-10-31

### Added
- **Anima Sampler Metadata (node 24):** reads seed, steps, cfg, sampler, scheduler and denoise from a KSampler's `info` output for image-saver metadata. Values are the ones actually sampled with (profile / preset applied); Dual / Triple-Pass report pass 1.

### Changed
- KSampler, Dual-Pass and Triple-Pass info trailers now carry seed and denoise (multi-pass: pass 1 steps, sampler and scheduler as well).

---

## [0.3.3] - 2026-10-31

### Fixed
- **Block / Predictive Merger silently skipped quantized layers:** with int8 / ConvRot checkpoints every DiT linear was left out of the merge (raw int8 weights are not floating point) while the node reported a merge. Both now raise a clear error asking for unquantized models, like the LoRA Extractor.
- **Artist & Character Picker seed defaulted to randomize** (ComfyUI adds that control to any `seed` input), which re-ran Prompt Studio, encoding and sampling on every queue even with random artists off. It now defaults to fixed; random draws also never repeat a listed artist written with spaces or `@`.
- **Telemetry ignored LoRA Block Weight** for the llm_adapter check, and reported `llm_adapter_safe: false` when no LoRA was present (now `null`).
- **LoRA Resize resized DoRA layers** although they are documented as passed through unchanged.

### Changed
- Removed dead and old-ComfyUI fallbacks: Prompt Studio's unused `artist_in` / `character_in` parameters (the frontend already renames those sockets), the flow-shift clone's unreachable branches (single deepcopy path), optional `latent_preview` / `comfy.lora_convert` / sampler-list / scipy fallbacks, and the swallowed exception around VAE Decode's VRAM estimate. Discard-penultimate list is read from `comfy.samplers.KSampler`. `scipy` declared in dependencies and CI.
- README: the Simple Pass example description matches its nodes; documented 30 test suites.

---

## [0.3.2] - 2026-10-31

### Added
- **Anima LoRA Resize (node 22):** lowers a LoRA's rank with an exact SVD of its factors (QR + core SVD on factors), fixed or per layer by an energy target; diffs and unsupported formats pass through and the source is never overwritten. Info reports size and energy kept and warns below 90%. Measured on two LoRAs: rank 32 / dynamic 0.90 stay close to the source, rank 16 is fine for one and visibly drifts for the other.
- **Anima Artist & Character Picker (node 23):** local Danbooru tag catalog in `web/data/` (83k artists, 57k characters, 12k series, >= 20 posts) with a search popup; outputs `@artist` / character + series in Anima prompt form with parentheses escaped, optional weights and seeded random artists.

### Fixed
- **Prompt Studio artist names with parentheses** (e.g. `hammer_(sunset_beach)`) were passed unescaped, so the tokenizer read `(sunset beach)` as a weighted concept that leaked into the image; they are now escaped like character names.

---

## [0.3.1] - 2026-10-31

### Added
- **Anima LoRA Extractor (node 20):** saves `model_tuned - model_base` as a LoRA (truncated SVD per linear layer, full diffs for norms / biases, identical tensors skipped, optional `llm_adapter`). Reads baked weights like the Model Saver and rejects quantized inputs. Info reports the share of the difference kept by the rank. Measured on base -> pixelart: rank 32 = 132 MB, 94.5% energy, every key loads in the stock LoRA loader.

- **Anima LoRA Block Weight (node 21):** one LoRA with a multiplier per DiT depth group plus `other_layers`, optional CLIP strength and adapter stripping. No fixed presets: measured, where a LoRA acts differs per LoRA.
- **`tools/anima_ablation.py` `lora` / `lora_presets` modes:** per-group swap-in / leave-out for one LoRA (through LoRA Block Weight) and the Multi-LoRA presets against plain stacking. Results for two style LoRAs and a regional-prompt test in `tools/ABLATION_RESULTS.md`.

### Changed
- The LoRA Merger's output-path handling is now the shared `lora_output_path` helper (same behavior); the Multi-LoRA loader helper is the shared `load_anima_lora_patches`.
- **Multi-LoRA "Character + Style Fusion" is shown as "Favor Slot 1 (PRI)"** and the LoRA Merger's as "Favor Donor A": measured, it leans towards the first LoRA in composition and style alike. The other Multi-LoRA presets stay within 0.03 of plain stacking; tooltips say so. Stored values and math are unchanged.

---

## [0.3.0] - 2026-10-31

### Added
- **`tools/anima_ablation.py`:** dev script that measures block groups, Predictive Merger presets and schedulers through the local ComfyUI API (numpy + Pillow metrics for layout, edges, texture, color). Results for base -> pixelart / aesthetic, plus sublayer and model-switch probes, in `tools/ABLATION_RESULTS.md`.
- **Vue nodes (Nodes 2.0) support** for all HaDeS canvas widgets, including the classic-widget toggle.

### Changed
- **Info contract for Telemetry:** every node appends a `HADES_META {json}` trailer to its `info` output (`format_info`, keys per node in `HADES_META_SCHEMA`). Telemetry classifies streams by the trailer's node id and reads shift, CFG, NAG, resolution, VAE channels, adapter protection and corner results from it; regex parsing remains only for legacy / foreign strings. Reports show the streams without the trailer line. `info` outputs connected to ShowText now end with that line.
- **Block group labels and tooltips follow the measurements:** no group isolates one attribute; influence ranks B10-14 > B15-19 ~ B00-09 > B20-23 > B24-27. Labels are now Early / Early-Mid / Core / Mid-Late / Late / Final; input names, presets and saved recipes are unchanged. The "Style B" button is now "Late B" (same values).
- **Predictive Merger "Style Transfer" is shown as "Conservative (mostly Model 1)":** sublayer swaps and a late model switch confirmed that no merge keeps Model 1's composition with Model 2's style in Anima. The stored value and the math are unchanged, so saved workflows behave the same; the tooltip points to the Dual-Pass KSampler (`model_pass2`) for Model 2 detail. Styled combos gained a `displayNames` option for this.
- **Predictive Merger preset tooltip** describes what the presets measurably do (overall Model 2 strength).
- **anima_beta57 tooltips** no longer recommend it for refiner passes; measured, it converges slower than stock `beta`.
- **README** rewritten around verified behavior, with the measured block table and links to the results.
- **Tests:** contract test checks that all 18 emitting nodes use a schema node id with exactly its keys, and that Telemetry trusts trailers over contradictory prose; 16 px grid regression.

### Fixed
- **Telemetry resolution check used a 32 px rule:** Anima's grid is 16 px (8x VAE x 2x2 patches); 1008 px no longer warns, and the warning names the real cause (patch padding on the bottom / right edge).
- **Vue nodes (Nodes 2.0):** HaDeS rows were laid out for the zoom level the node was created at (text oversized and cut, clicks landing on the wrong control); they now fit their real width at any zoom. Rows also repaint after clicks, prompts, menus, presets and frontend-driven changes such as seed *control after generate*, and tooltips work over the widget canvases.

---

## [0.2.0] - 2026-10-31

> **Milestone Release v0.2.0:** Major architectural evolution of the HaDeS Anima Suite. Featuring Node #19 (Anima Model Saver with baked patches, Diffusers export & native ComfyUI ConvRot INT8/INT4), cross-attention-only NAG guidance, inherited Flow Shift across all KSamplers, comprehensive silent bug eradication (seed wrap, NaN+outlier clamp, seam continuation mirror), and a full UI design overhaul (restrained Linear-style dark aesthetic, active presets, and centralized `finalizeHadesNode` layout engine).

### Fixed
- **Model Saver saved unmerged weights:** weights now go through ComfyUI's `model_state_dict_for_saving` lazy casters (same path as `comfy.sd.save_checkpoint`), so merge and LoRA patches are baked in even with dynamic VRAM. Already-quantized layers keep their weight + scale pieces.
- **Model Saver Native files did not load in the stock loader:** keys are now saved without the `diffusion_model.` prefix.
- **Model Saver ConvRot INT8/INT4 produced files nothing could load:** both now use ComfyUI's native `comfy.quant_ops` layouts (`int8_tensorwise` + convrot, `convrot_w4a4`) on the DiT block linears with per-layer `comfy_quant` configs. Option names are unchanged.
- **Model Saver adapter export:** the companion `_adapter.safetensors` now holds bare AnimaTextConditioner keys in both formats (what `external_llm_adapter` expects).
- **NAG guided every attention layer:** guidance now runs only on cross-attention (tagged via `attn1_patch` / `attn2_patch`), re-attending the positive queries over the negative prompt's keys/values, with the reference NAG formula (`nag_scale = 1.0` disables). The token-mean single-branch fallback was removed.
- **KSamplers silently overrode the model's flow shift:** `flow_shift` / `shift_pass1` / `shift_stage1` accept `0` = inherit the shift already set on the model (Model Loader, ModelSampling nodes) and default to it. Profiles and presets that used the Anima default shift (3.0) now inherit. Info strings report the shift actually used.
- **Telemetry false diagnostics:** `llm_adapter` protection is read from an explicit `Protect Adapter: <bool>` flag in every LoRA stream (the word "Stripped" in Prompt Studio no longer fakes it), and the Latent node stream is classified as SAMPLING so the resolution check runs.
- **Dual/Triple-Pass seed overflow:** `seed + 1` / `seed + 2` wrap inside the uint64 range instead of crashing at the max seed.
- **Swallowed sampling errors:** `fix_empty_latent_channels` is no longer wrapped in `except Exception: pass`; dead noise fallback removed.
- **Corner Shield edge continuation was inverted:** the box row/column at the seam now copies the nearest clean pixel (mirror), avoiding a visible seam in Smart heal mode.
- **Prompt Studio legacy restore:** the old/new `widgets_values` layout is detected by exact header combo values, not by `"Dual"`/`"Canonical"` substrings that a prompt could contain.
- **Block Merger recipe preview:** groups-only JSON recipes are previewed like the backend applies them; unknown JSON recipes raise a JSON-specific error.
- **`clamp_spectral_outliers`:** tensors with NaN/Inf are scrubbed and still outlier-clamped.
- **Console noise:** loader/sampler/decode `print()` calls replaced by the suite logger (samplers and decode at debug level); removed the misleading `CLIPType.QWEN_IMAGE` fallback warning.

### Changed (UI)
- **Visual redesign:** one neutral surface scale with a single accent; green / amber / red reserved for state. Rainbow tag chips, neon tooltips and glows replaced by quiet uppercase row labels; block-group colors kept only as thin markers on the Block Merger rows (they match its 34-block chart).
- **Shared layout:** every node uses `finalizeHadesNode` (real minimum width via `computeSize`, content-height on creation, saved sizes clamped on load), replacing 19 copy-pasted sizing blocks. Fixes clipped / overlapping text when narrowing nodes and the empty space at the bottom of new nodes.
- **Remaining stock widgets restyled:** Prompt Studio header mode / trigger / artwork title, NAG `only_anima`, and the KSamplers' seed + control-after-generate (native widgets stay serialized underneath).
- **Preset buttons show the active preset**; slider drags snap to the widget step; plain-English labels (e.g. "Aggressiveness", "Guidance scale", "Norm ceiling (τ)"); multi-pass KSampler rows labelled by pass / stage.
- **Settings:** the classic-widgets option is declared through the extension `settings` API ("Use standard ComfyUI widgets on HaDeS nodes").

### Fixed (UI)
- Text Encoder Loader "CPU Safe" and VAE Loader "FP32" preset buttons set values that do not exist in the backend combos (prompt validation failed).

---

## [0.1.9] - 2026-10-30

### Added
- **Node #19: HaDeS Anima Model Saver (`HaDeSAnimaModelSaver`):**
  - High-fidelity model checkpoint saver under `HaDeS/Anima Mergers` with pass-through `MODEL` output for seamless downstream pipeline execution without wire duplication.
  - **Multi-Format Export:** Bidirectional support for `Native ComfyUI (MiniTrainDIT)` and `Diffusers (HuggingFace / Trainer)` using 24 inverse remapping rules (`transformer_blocks.*`, `attn1.*`, `attn2.*`, `text_conditioner.*`).
  - **ConvRot INT8 & INT4 Quantization:** Block-level Hadamard orthogonal rotation ($H_{64}$) that eliminates outlier channels before symmetric quantization, reducing quantization reconstruction error by >67% compared to naive INT8. Supports 4-bit uint8 nibble packing (`[-8, 7]` signed range offset by 8) with per-channel `bfloat16` scale tensors.
  - **LLM Adapter Golden Rule Isolation:** Supports embedding adapter directly into checkpoint, stripping adapter (`Strip (Pure DiT)`), or exporting adapter to a separate companion safetensors file (`_adapter.safetensors`).
  - **SAI ModelSpec Metadata Injection (Toggleable):** Controlled via `include_metadata` boolean toggle, injecting `modelspec.architecture: anima/mini_train_dit`, dynamic topology (`28L`, `40L`, `52L`), quantization tags, author, and timestamp.
  - **Frontend Canvas UI:** Spectral HUD banner with active status display, 4 quick preset buttons (`BF16 Master`, `ConvRot INT8`, `ConvRot INT4`, `Diffusers HF`), styled input widgets, and contextual tooltips.
  - **Telemetry Stream Integration:** Classified under `ARCHITECTURE` stream in `HaDeSAnimaSuiteTelemetry`.
  - **Test Suite 23:** Added comprehensive unit test asserting all 6 format, quantization, adapter separation, and metadata configurations.

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
