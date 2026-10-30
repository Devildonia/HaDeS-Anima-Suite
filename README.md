# HaDeS Anima Suite v0.1.9 — Complete 19-Node Anima Ecosystem for ComfyUI

**HaDeS Anima Suite** is a specialized **19-node toolkit** engineered specifically for the **Anima** generative architecture (*CircleStone Labs / Comfy Org*: `MiniTrainDIT` + `Qwen3-0.6B` / `Qwen3.5-4B` + `AnimaTextConditioner` + `Wan21 16-Channel VAE`).

It combines **native Diffusers-aware Anima Model/CLIP/16ch-VAE Loaders & Smart Tiled VAE Decode**, **subspace-aware block merging** (`28L / 40L / 52L` DiT + `6L` `llm_adapter`), **end-to-end Anima pipeline guards**, the **Anima Rectified Flow KSampler Trilogy (Single, Dual-Pass, and Triple-Pass)**, the **Multi-Format Anima Model Saver**, and the **Anima Suite Telemetry & Flight Auditor**—solving every major architectural pitfall of Anima in ComfyUI under a unified **HaDeS Spectral Canvas UI**.

---

## Key Architectural Innovations in v0.1.9

1. **Dynamic Multi-Variant Topology (`2.0B / 2.9B / 3.8B`):**
   - Automatically detects whether a checkpoint or LoRA belongs to **Anima-2.0B** (`28` DiT blocks), **Anima-2.9B** (`40` DiT blocks), or **Anima-3.8B** (`52` DiT blocks) plus the `6`-layer `llm_adapter` bridge, proportionally mapping any depth onto the canonical 34-band spectrum (`0..27` DiT + `28..33` Bridge).
2. **`llm_adapter` Golden Rule Protection (`llm_adapter_lr = 0`):**
   - Anima's `AnimaTextConditioner` (`llm_adapter.blocks.0..5`) is a delicate 6-layer translator between `Qwen3-0.6B` hidden states (`1024 dim`) and `T5XXL` query tokens (`32128 vocab`). Corrupting it destroys prompt adherence and `@artist` recognition.
   - All HaDeS mergers and LoRA loaders **lock or strip `llm_adapter` modifications by default (`0.0` / `protect_llm_adapter = True`)**, and automatically skip allocating RAM patches when tensors between two models are identical.
3. **24-Rule Universal Diffusers $\leftrightarrow$ ComfyUI Key Remapper:**
   - Native ComfyUI (`comfy/lora.py`) does not automatically translate Diffusers-trained Anima LoRAs (`transformer_blocks.X.attn1.to_q`, `to_out.0`, `ff.net.0.proj`, `norm1.linear_1`, `text_conditioner`).
   - HaDeS canonicalizes Diffusers, Kohya (`lora_unet_*`), OneTrainer, and Musubi keys into native `MiniTrainDIT` keys on the fly—allowing you to load Diffusers LoRAs directly or **merge a Diffusers LoRA with a Kohya LoRA in the same QR-SVD subspace**.
4. **Complete Anima Pipeline Guards:**
   - **Prompt Studio & Aesthetic Guard:** Prevents *Anima-Aesthetic* / *2.9B* latent saturation (*slop*) by auto-stripping `score_*` & `masterpiece` tags when appropriate, normalizing Booru grammar while protecting kaomojis (`^_^`) and whitelisted tokens (`safe_for_work`), extracting embedded dataset triggers (`ye-pop`, `deviantart`) into canonical Line 1 headers, preserving natural language sentences with grammatical colons intact, and scaling `T5XXL` query-token emphasis weights independently of `Qwen3`.
   - **NAG Attention Guidance (`CFG = 1.0` Negative Enabler & Anti-Burn):** Enables real negative guidance on **Anima-Turbo / Lightning** at `CFG = 1.0` and prevents high-CFG edge frying on Base/Aesthetic via L1-norm-bounded attention extrapolation (`nag_tau`) inside an exact sigma window (safely gated to Anima models via `only_anima = True`).
   - **16-Channel Latent & Pipeline Guard:** Blocks the common **64-channel Qwen-Image 2.1 VAE trap**, enforces `32px` grid alignment across 11 native Anima aspect ratios (up to `1:4` and `4:1`), and prevents the **Anima-2.9B `1280×1280` white-background dataset bias**.
   - **Corner Scribble Shield:** Detects and heals (or masks with `mask` output for second-pass inpainting) Anima's notorious **~20% corner pseudo-signature scribble artifact** with core ambient texture safeguards (peak-aware HF detection differentiating fine 1px strokes from uniform grain $\sigma=0.03$), telemetry sync, and adaptive expansion capped strictly at 50% for large marks.
5. **Centralized Suite Telemetry & Diagnostics Flight Auditor:**
   - Connects up to 8+ `info` streams across all HaDeS Suite nodes in the workflow, auto-classifies them into logical pipeline phases, and runs an automated **Diagnostics & Health Engine** (Flow Shift synchronization check, Turbo/CFG=1 NAG guidance validation, 32px macroblock alignment check, 16ch Wan2.1 VAE verification, and `llm_adapter` text bridge lock). Outputs an audit-grade Markdown report, a structured JSON telemetry payload, and a compact single-line status summary with one-click clipboard copying.
6. **Anima Model Saver (`bfloat16`, `fp8_e4m3fn`, native ConvRot `INT8`/`INT4`, `float32`, `Diffusers` export):**
   - High-fidelity export node with pass-through `MODEL` output for seamless downstream pipeline continuation.
   - Dual format: Native ComfyUI (`MiniTrainDIT`) or Diffusers (`transformer_blocks.*`, `text_conditioner.*`).
   - Merge and LoRA patches are baked into the saved weights (same lazy patching path as ComfyUI's own checkpoint saver); Native files load in the stock `Load Diffusion Model` node.
   - ConvRot INT8 / W4A4 use ComfyUI's own `comfy.quant_ops` layouts (`int8_tensorwise` + convrot, `convrot_w4a4`) on the DiT block linears, with per-layer `comfy_quant` configs.
   - Companion `llm_adapter` handling: Full checkpoint embedding, DiT stripping, or dedicated companion adapter export.
   - ModelSpec metadata injection with toggleable `include_metadata`.

---

## Overview of the 19 Nodes

```
HaDeS Anima Suite v0.1.9
├── HaDeS_Anima_Suite/Loaders (4 Nodes)
│   ├── 14. Anima Diffusion Model Loader
│   ├── 15. Anima Text Encoder Loader
│   ├── 16. Anima VAE Loader
│   └── 17. Anima VAE Decode
├── HaDeS/Anima Mergers (6 Nodes)
│   ├── 1. Anima Model Block Merger
│   ├── 2. Anima Predictive Model Merger
│   ├── 3. Anima Predictive Multi-LoRA
│   ├── 4. Anima LoRA Merger
│   ├── 5. Anima CLIP Block Merger
│   └── 19. Anima Model Saver
├── HaDeS/Anima Pipeline (6 Nodes)
│   ├── 6. Anima Prompt Studio
│   ├── 7. Anima Attention Guidance (NAG)
│   ├── 8. Anima Latent
│   ├── 9. Anima Universal LoRA Remapper
│   ├── 10. Anima Corner Shield
│   └── 18. Anima Suite Telemetry
└── HaDeS_Anima_Suite/Samplers (3 Nodes)
    ├── 11. Anima KSampler
    ├── 12. Anima Dual-Pass KSampler
    └── 13. Anima Triple-Pass KSampler
```

---

## Canonical 34-Block Anima Functional Map

| Canonical Index | Native Key Pattern (`28L` / `40L` / `52L`) | Group Widget Name | Functional Role |
| :--- | :--- | :--- | :--- |
| **Blocks `00–04`** | `blocks.0` .. `~18%` DiT depth | `BLOCKS_Layout_Geometry` | Camera framing, perspective grid, silhouette layout |
| **Blocks `05–09`** | `~18%` .. `~36%` DiT depth | `BLOCKS_Global_Composition` | Global pose, mass distribution, volumetric lighting |
| **Blocks `10–14`** | `~36%` .. `~54%` DiT depth | `BLOCKS_Subject_Identity` | Character anatomy, facial proportions, identity lock |
| **Blocks `15–19`** | `~54%` .. `~71%` DiT depth | `BLOCKS_Material_Substance` | Fabric folds, hair strands, skin & surface materials |
| **Blocks `20–23`** | `~71%` .. `~86%` DiT depth | `BLOCKS_Art_Style_Medium` | Lineart weight, brushstroke texture, color palette |
| **Blocks `24–27`** | `~86%` .. `100%` DiT depth | `BLOCKS_Lighting_Sharpness` | Micro-contrast, specular highlights, final sharpness |
| **Blocks `28–33`** | `llm_adapter.blocks.0..5` | `ADAPTER_Text_Alignment` | **Qwen3 $\to$ T5 Bridge** (Default `0.0` / Locked) |

---

## Node-by-Node Technical Reference

### Part A — Mergers & LoRA Subspace Engine (`HaDeS/Anima Mergers`)

#### 1. `Anima Model Block Merger`
- **Class:** `HaDeSAnimaModelBlockMerger`
- **Purpose:** Merges two Anima `MODEL` checkpoints across the 7 functional block groups with baked-patch awareness (`get_baked_weight`) and zero-RAM-overhead skipping for identical tensors.
- **Inputs:**
  - `model1`, `model2` (`MODEL`): Anchor and Secondary checkpoints.
  - `base_ratio` (`FLOAT`, default `0.50`): Fallback ratio for non-block layers (`x_embedder`, `t_embedder`, `final_layer`).
  - `BLOCKS_Layout_Geometry` .. `BLOCKS_Lighting_Sharpness` (`FLOAT`, default `0.50`).
  - `ADAPTER_Text_Alignment` (`FLOAT`, default **`0.00`**): Keeps Model 1's `llm_adapter` intact by default.
  - `recipe_override` (`STRING`, optional): Accepts a 34-float CSV string or full JSON export from the Predictive Merger.

#### 2. `Anima Predictive Model Merger`
- **Class:** `HaDeSAnimaPredictiveModelMerger`
- **Purpose:** Analyzes per-block angular similarity and Z-normalized inter-block Frobenius divergence (or exact Task Vectors $\Delta W_1 = W_1 - W_{\text{base}}$, $\Delta W_2 = W_2 - W_{\text{base}}$ when `base_model` is connected) to compute an optimal 34-block merge curve automatically.
- **Inputs:**
  - `model1`, `model2` (`MODEL`), `base_model` (`MODEL`, optional).
  - `preset`: `Style Transfer (Keep A identity, B style)`, `Interference Guard (Anti-Burn / Artifact Shield)`, `Complementary Richness (Fill gaps of A with B)`, or `Balanced Geometric Fusion`.
  - `target_influence` (`0.0..1.0`, default `0.50`), `aggressiveness` (`0.1..2.5`, default `1.00`).
  - `protect_llm_adapter` (`BOOLEAN`, default `True`): Locks blocks `28..33` (`llm_adapter`) to `0.0` so Model 1's semantic bridge is never degraded.

#### 3. `Anima Predictive Multi-LoRA`
- **Class:** `HaDeSAnimaPredictiveMultiLoraLoader`
- **Purpose:** Simultaneously loads up to 4 Anima LoRAs (`PRI`, `S2`, `S3`, `S4`), automatically remapping Diffusers/Kohya keys and computing **exact low-rank factor-space Grassmann overlap** (via reduced QR decomposition) between each secondary LoRA and the primary LoRA to prevent overcooking.
- **Key Features:**
  - Computes an independent 34-block modulation curve for every active slot.
  - Automatically strips dirty `llm_adapter` keys when `protect_llm_adapter = True`.
  - Applies square-root overlap damping ($1 / \sqrt{N}$) to shared `CLIP` layers.

#### 4. `Anima LoRA Merger`
- **Class:** `HaDeSAnimaStandaloneLoraMerger`
- **Purpose:** Merges two Anima LoRAs (`lora_a` and `lora_b`—even if one is in **Diffusers** format and the other is in **Kohya/ComfyUI** format) into a single `.safetensors` file using **Reduced QR-SVD Factorization**:
  $$\Delta W = U_{\text{cat}} D_{\text{cat}} = Q_U (R_U R_D^T) Q_D^T$$
  running SVD on the tiny $2r \times 2r$ core matrix in milliseconds instead of a full $M \times N$ matrix.
- **Modes:** `Additive (A + B, Multi-LoRA match)` (default) or `Interpolated (Lerp (1-r)A + rB)`, plus optional DARE pruning and automatic donor-file overwrite protection.

#### 5. `Anima CLIP Block Merger`
- **Class:** `HaDeSAnimaCLIPBlockMerger`
- **Purpose:** Merges two Anima text encoders (`Qwen3-0.6B` 28 layers or `Qwen3.5-4B` 36 layers) across 5 linguistic bands (`vocab_embeddings`, `syntax_rigidity` `[0–35%]`, `semantic_focus` `[35–72%]`, `style_abstraction` `[72%+]`, `final_projections`) using dynamic proportional layer depth detection.

#### 19. `Anima Model Saver`
- **Class:** `HaDeSAnimaModelSaver`
- **Category:** `HaDeS/Anima Mergers`
- **Purpose:** Dedicated high-fidelity model exporter for merged or fine-tuned Anima models supporting format conversion (Native ComfyUI $\leftrightarrow$ Diffusers), bf16 / fp8 / native ConvRot INT8-INT4 / fp32 storage with merge and LoRA patches baked in, granular `llm_adapter` isolation, and toggleable ModelSpec metadata. Features pass-through `MODEL` output so downstream KSamplers can execute directly without wire branching.
- **Inputs:**
  - `model`: Input `MODEL` patcher.
  - `filename_prefix`: Subfolder and filename prefix inside `ComfyUI/output` (default: `diffusion_models/Anima_HaDeS_Merged`).
  - `save_format`: `Native ComfyUI (MiniTrainDIT)` or `Diffusers (HuggingFace / Trainer)`.
  - `precision`: `bfloat16 (Original Master)`, `fp8_e4m3fn (ComfyUI Native FP8)`, `convrot_int8 (Hadamard Rotated INT8)`, `convrot_int4 (Hadamard Rotated INT4)`, `float32 (Uncompressed)`.
  - `llm_adapter_mode`: `Embed in Checkpoint (Full)`, `Strip (Pure DiT)`, `Export Adapter Separately`.
  - `include_metadata`: `True` / `False` (injects SAI ModelSpec `anima/mini_train_dit`, topology, author and recipe tags).
  - `custom_author` / `custom_model_title`: Custom metadata fields.
- **Outputs:**
  - `model`: Pass-through model for downstream samplers.
  - `saved_path`: Absolute file path on disk.
  - `info`: Detailed telemetry string for `HaDeSAnimaSuiteTelemetry`.

---

### Part B — Anima Pipeline & Guard Nodes (`HaDeS/Anima Pipeline`)

#### 6. `Anima Prompt Studio`
- **Class:** `HaDeSAnimaPromptStudio`
- **Purpose:** Automates CircleStone Labs' Booru/NL prompt rules and gives direct control over Anima's internal `T5XXL` query-token weight multiplier.
- **How it works:**
  - **Aesthetic Slop Guard:** When `model_variant` is set to `Anima Aesthetic (Strip score_* & Slop Guard)`, it automatically strips `score_9`, `score_8_up`, `score_7_up`, `masterpiece`, `best quality`, `ultra-detailed`, `8k`, `4k`, `extremely detailed`, `insanely detailed`, etc., which otherwise over-saturate Anima-Aesthetic latents. The `Anima Aesthetic v1.0b (Keep masterpiece, best quality)` profile preserves the optional `masterpiece, best quality` quality tags while still stripping all forbidden `score_*` tokens.
  - **Smart Booru Sanitizer:** Converts tags to lowercase and replaces `_` with spaces while preserving whitelisted tokens (`score_*`, `*_up`, `safe_for_work`) and kaomojis (`^_^`, `o_o`, `>_<`, `;_;`, etc.).
  - **Style Header Injection & Preservation:** In `Flat Vector / Graphic` mode, injects the canonical non-anime dataset header: a single trigger (`ye-pop` or `deviantart`, selectable) on Line 1, an optional artwork title on Line 2 (`artwork_title`), and the prompt body on Line 3, `\n`-separated (CircleStone Labs 3-line format). The legacy dual-trigger header (`ye-pop\ndeviantart\n…`) is preserved via `header_mode = "Dual Legacy"`. If the user inputs a prompt that already starts with `ye-pop` or `deviantart`, manual 2-line or 3-line headers are preserved verbatim (title casing and punctuation intact) and descriptive natural language sentences are kept intact without improper comma-splitting.
  - **`t5_weight_multiplier`:** In ComfyUI's Anima implementation (`comfy/text_encoders/anima.py`), `Qwen3-0.6B` token weights are forced to `1.0` while `(tag:weight)` multipliers are passed through `T5XXLTokenizer` to multiply the output of `llm_adapter`. This slider scales those `T5XXL` emphasis weights directly!

#### 7. `Anima Attention Guidance (NAG)`
- **Class:** `HaDeSAnimaAttentionGuidance`
- **Purpose:** Implements **Normalized Attention Guidance (NAG)** via `optimized_attention_override` within a precise denoising window (`start_percent` $\to$ `end_percent` converted via `percent_to_sigma`).
- **Why you need it:**
  - **Enables Negative Guidance at `CFG = 1.0`** on `Anima-Turbo` and `Anima-Lightning`.
  - **Prevents High-CFG Contrast Burn** on `Anima-Base` and `Anima-Aesthetic` by bounding the L1-norm ratio of extrapolated attention features to `nag_tau` ($\tau \le 2.5$).
  - **Anima Family Gate:** `only_anima = True` (default) safely bypasses non-Anima models by checking `unet_config.image_model`, matching the canonical NAG reference pattern.

#### 8. `Anima Latent`
- **Class:** `HaDeSAnimaLatentPipelineGuard`
- **Purpose:** Generates 16-channel Wan21-format latents (`[B, 16, H // 8, W // 8]`) while validating your connected `model` and `vae`.
- **Protections:**
  - **64-Channel VAE Trap Blocker:** Raises an immediate, descriptive error if you accidentally connect the 64-channel *Qwen-Image 2.1* VAE instead of the required 16-channel `qwen_image_vae.safetensors`.
  - **Anima-2.9B `1280×1280` White-BG Guard:** Automatically shifts `1280×1280` to `1216×1216` (`1.48 MP`, 32px-aligned) when `prevent_29b_white_bg_bias` is enabled, avoiding the white-background dataset bucket artifact in Anima-2.9B (`Preview 3`).
  - **11 Native Aspect Ratios:** Includes extreme `1:4` (`512×2048`) and `4:1` (`2048×512`) panoramas supported natively by Anima.

#### 9. `Anima Universal LoRA Remapper`
- **Class:** `HaDeSAnimaUniversalLoraLoader`
- **Purpose:** Loads any third-party Anima LoRA (Diffusers `transformer_blocks.*`, `ai-toolkit`, Kohya, OneTrainer, Musubi), applies the 24-rule `DIFFUSERS_TO_COMFY` key conversion, strips accidental `llm_adapter` weights (`strip_llm_adapter = True`), optionally scrubs `NaN`/`Inf` & $>10\sigma$ spectral outliers, and optionally exports a clean ComfyUI-native `.safetensors` copy to `models/loras/`.

#### 10. `Anima Corner Shield`
- **Class:** `HaDeSAnimaCornerScribbleShield`
- **Purpose:** Mitigates Anima's chronic **~20% corner signature scribble artifact** (which negative prompts like `signature, watermark` cannot suppress because training images had unlabeled corner signatures).
- **Tuning:** Detection sensitivity defaults to `0.75` (threshold auto-derives from it), and the inspection box defaults to `10%` of the shorter side (`0.03`–`0.50` allowed, with adaptive expansion capped strictly at 50% for large watermarks). The shield only touches corner pixels: clean corners are left 100% untouched in Smart mode.
- **Modes:**
  - `Mask Only (No Pixel Modification)`: Outputs a feathered `[B, H, W]` `mask` of affected corners to feed into a second-pass `VAEEncodeForInpaint` or `LatentCompositeMasked`.
  - `Smart Edge-Inward Blend (Auto-Heal)`: Detects high-frequency corner scribbles (tuned for realistic thin low-contrast strokes) and heals them by extending the clean **outer edge ring** (8–16 px outside the corner box) into the box, while an **Ambient Texture Guard** protects canvas grain, film noise ($\sigma=0.03$), and hatching textures from false positives. Safe default.
  - `Aggressive Corner Fill (Forced)`: Bypasses detection entirely and force-fills the corner box with spatially-falloff edge continuation. Use when the scribble is faint or Smart mode misses it — the strongest guaranteed removal.
  - `Soft Corner Blur`: Softens high-frequency corner artifacts with feathered local pooling (lightest touch, no content reconstruction).

---

### Part C — Anima Rectified Flow KSampler Trilogy (`HaDeS_Anima_Suite/Samplers`)

#### 11. `Anima KSampler`
- **Class:** `HaDeSAnimaKSampler`
- **Purpose:** Single-pass Rectified Flow sampling engine tailored to Anima's `MiniTrainDIT` (`multiplier = 1.0` Flow Shift with deepcopy buffer isolation, unlike `ModelSamplingSD3` which injects `multiplier = 1000` and breaks Anima's timestep scaling).
- **Key Features:**
  - **Built-in Custom Schedulers:** Adds `anima_beta57` ($\alpha = 0.50, \beta = 0.70$, allocating higher step density to low-noise texture/lineart refinement) and `anima_lightning_tdm` (exact 4-step Trajectory Distribution Matching schedule `999 → 749 → 499 → 249 → 0` for `Anima-Lightning`).
  - **1-Click Variant Profiles:** Pre-calibrated profiles for `Anima Base / Raehoshi`, `Anima Aesthetic Crisp`, `Anima Aesthetic Painterly (beta57)`, `Anima Turbo`, `Anima Lightning TDM`, and `Anima 2.9B / 3.8B`.
  - **Optional 16ch VAE Decode:** Automatically validates `vae.latent_channels == 16` when a VAE is connected and outputs both `latent` and `decoded_image`.

#### 12. `Anima Dual-Pass KSampler`
- **Class:** `HaDeSAnimaDualPassKSampler`
- **Purpose:** Two-stage `Composition (1.0 MP) → HiRes/Style Refiner (1.25x–2.0x)` KSampler with a **16-channel Variance-Preserving Latent Bridge** snapped strictly to the **32px pixel grid**.
- **Key Features:**
  - **16-Channel Variance Restoration:** Standard bicubic upscaling on `Wan21` 16-channel latents dampens high-frequency per-channel variance ($\text{Var}(z') < \text{Var}(z)$), causing soft/washed-out lineart at `denoise < 0.50`. HaDeS restores per-channel $(\mu_c, \sigma_c)$ across all 16 channels (or performs a `Pixel-Space VAE Round-Trip (Lanczos 32px)` when selected).
  - **Hybrid Model Handoff (`model_pass2`):** Run Pass 1 with `Raehoshi` or `Anima-Base + Character LoRA` for exact character anatomy, and hand off Pass 2 to `Anima-Aesthetic` or `Anima-Turbo` (8 steps, `CFG 1.0`) for high-resolution rendering (backward-compatible with legacy `model_pass2_optional`).

#### 13. `Anima Triple-Pass KSampler`
- **Class:** `HaDeSAnimaTriplePassKSampler`
- **Purpose:** Three-stage progressive studio pipeline (`Stage 1: Anatomy & Layout Lock 1.0x` $\to$ `Stage 2: Structural & Aesthetic Bridge ~1.35x` $\to$ `Stage 3: Micro-Lineart & Surface Polish ~1.75x–2.0x`).
- **Why 3 Stages:** Jumping directly from `1.0x` (`1024×1024`) to `2.0x` (`2048×2048`) in a single step forces a trade-off between patch-hallucination (duplicated fingers/locks of hair at `denoise >= 0.45`) and blurriness (`denoise <= 0.28`). Stepping through an intermediate 32px-aligned structural bridge (`1.35x–1.50x` at `denoise ~0.48`) followed by a low-denoise `anima_beta57` micro-polish (`denoise ~0.26`) achieves 3–4 MP studio posters with zero anatomical drift.

---

### Part D — Native Anima Loaders & Smart VAE Decode (`HaDeS_Anima_Suite/Loaders`)

#### 14. `Anima Diffusion Model Loader`
- **Class:** `HaDeSAnimaModelLoader`
- **Purpose:** Loads Anima checkpoints in either native ComfyUI format or raw **Diffusers** format (`transformer_blocks.*`, `patch_embed.proj.*`, `time_embed.t_embedder.*`, `text_conditioner.*`) by applying the 24-rule deterministic key converter in memory, optionally injects an external `llm_adapter` (`Anima-3.8B-expanded_adapter.safetensors`), auto-detects `28L / 40L / 52L` topology, and pre-configures Rectified Flow `flow_shift` (`multiplier = 1.0`).

#### 15. `Anima Text Encoder Loader`
- **Class:** `HaDeSAnimaCLIPLoader`
- **Purpose:** Dedicated loader for `qwen_3_06b_base.safetensors` (`AnimaTokenizer` + `AnimaTEModel`) with pre-load `1024d Qwen3-0.6B` architecture verification (blocking accidental `T5-XXL`, `CLIP-L`, or `3584d Qwen2.5-7B` loads) and a `cpu (Low-VRAM 6GB Safe)` placement mode.

#### 16. `Anima VAE Loader`
- **Class:** `HaDeSAnimaVAELoader`
- **Purpose:** Inspects the VAE `state_dict` header *before* instantiation to block the **64-channel Qwen-Image 2.1 VAE trap** (which shares the filename `qwen_image_vae.safetensors`) and enforces `bfloat16` precision for `WanVAE` RMSNorm stability.

#### 17. `Anima VAE Decode`
- **Class:** `HaDeSAnimaVAEDecode`
- **Purpose:** Decodes 16-channel Anima latents with proactive **Auto Smart-Tile** switching when resolution exceeds `auto_tile_mp_threshold` (default `1.35 MP`) or free VRAM is tight—avoiding `WanVAE` untiled OOM spikes on 6GB–12GB GPUs—and integrates an optional post-decode **Corner Scribble Shield** in a single node, with full user control over the shield: `corner_box_percent` (default `0.10`), `corner_sensitivity` (default `0.75`), `shield_heal_mode` (Smart Auto-Heal / Aggressive Forced Fill / Soft Blur) and `shield_feather_radius`.

#### 18. `Anima Suite Telemetry`
- **Class:** `HaDeSAnimaSuiteTelemetry`
- **Purpose:** Centralized **Flight Recorder & Pipeline Telemetry Auditor**. Aggregates up to 8+ `info` streams across all HaDeS Suite nodes in the workflow, semantically classifies them into pipeline phases (*Architecture, Conditioning & LoRAs, Sampling & Guidance, Post-Processing*), and executes an automated **Diagnostics & Health Engine** (validating Flow Shift synchronization, Turbo/CFG=1 NAG negative steering, 32px macroblock alignment, Wan2.1 16-channel integrity, and `llm_adapter` text bridge locks). Outputs an audit-grade Markdown report, a structured JSON telemetry payload, and a compact single-line status summary, with one-click clipboard copying.

---

## HaDeS Spectral Canvas UX vs. Vanilla ComfyUI Theme

HaDeS Anima Suite features a custom front-end architecture (`web/hades_widgets.js`, `web/hades_suite.js`) engineered for high clarity, rapid prototyping, and real-time diagnostic visibility:

- **HaDeS Spectral UX (Default):**
  - **Dynamic Visual Badges:** Real-time feedback badges (e.g., `16CH WAN2.1 ONLY`, `1024d LOCKED`, `24-RULE REMAP`, `DIT: 3.00`).
  - **Interactive Presets:** One-click recipe buttons (`Character + Style Fusion`, `HiRes Standard 1.5x`, `TDM Lightning`, `Progressive Staircase`).
  - **Telemetry HUD Monitors:** Visual spectral bars, real-time staircase flow monitors, and Corner Shield perimeter indicators directly embedded into node canvases.
  - **Contextual Hover Tooltips:** Granular, technical explanations when hovering over controls, buttons, and switches across all 18 nodes.

- **Vanilla ComfyUI Mode (Classic Theme Switch):**
  - Designed for minimalists or purists who prefer standard native ComfyUI controls and input widgets without custom canvas overlays.
  - **How to Toggle:**
    1. Open the **ComfyUI Settings** dialog (click the `⚙` **Gear Icon** in the top navigation panel).
    2. Scroll to the **HaDeS Anima Suite** section and locate:
       `HaDeS Suite: Force Classic ComfyUI Theme (Vanilla)`
    3. Toggle the switch:
       - **OFF (Default):** Full HaDeS Spectral Cyberpunk UX with interactive widgets, HUD monitors, and preset chips.
       - **ON (Vanilla):** Hides custom canvas widgets and restores standard ComfyUI text/number inputs across all 18 nodes.
    4. The setting updates immediately and persists automatically across sessions in browser storage (`HaDeS.ClassicUI`).

---

## Official Workflow Examples (`examples/`)

The repository includes four production-ready, pre-configured ComfyUI workflows in the [`examples/`](file:///examples) directory:

1. **`HaDeS Anima Suite - Simple Pass Generation Example.json`**
   - **Architecture:** Complete single-pass reference pipeline.
   - **Highlights:** Native Diffusers-aware `Model Loader` (Flow Shift 3.0), `Text Encoder Loader` (Qwen3 1024d safe), `VAE Loader` (16ch Wan2.1 verified), `Prompt Studio` (Aesthetic Slop Guard + Trigger extraction), `Attention Guidance (NAG)` (enabling CFG=1 negative steering on Lightning/Turbo), `Anima KSampler`, proactive `Smart Tiled VAE Decode`, and `Anima Suite Telemetry`.

2. **`HaDeS Anima Suite - Dual Pass Generation Example.json`**
   - **Architecture:** Two-stage HiRes pipeline (`1.0 MP Composition` $\to$ `1.5x Refiner`).
   - **Highlights:** Utilizes `Anima Dual-Pass KSampler` featuring the **16-Channel Variance-Preserving Latent Bridge** snapped strictly to the **32px pixel grid** (restoring per-channel variance to eliminate washed-out lines at low denoise) and hybrid model handoff.

3. **`HaDeS Anima Suite - Triple Pass Generation Example.json`**
   - **Architecture:** Progressive 3-stage staircase pipeline (`1.0x` $\to$ `1.35x` $\to$ `1.75x`).
   - **Highlights:** Utilizes `Anima Triple-Pass KSampler` for rendering pristine 3–4 MP poster-grade outputs. Distributes high-frequency generation smoothly across stages to eliminate patch-hallucinations (e.g., duplicated fingers/hair) without losing micro-detail.

4. **`HaDeS - Anima Suite - Merge Workflow Example - Medium.json`**
   - **Architecture:** Advanced block-level model and LoRA merging suite.
   - **Highlights:** Demonstrates `Anima Model Block Merger`, `Anima Predictive Multi-LoRA`, and `Anima LoRA Merger` with QR-SVD rank adaptation, 24-rule Diffusers $\leftrightarrow$ native key remapping, and default `llm_adapter` text conditioner bridge protection (`lr = 0`).

---

## Modular Project Structure

```
Comfyui-HaDeS-Anima-BlockMerger/
├── .github/
│   └── workflows/
│       └── ci.yml            # Multi-version CI (Python 3.10-3.12 + Node.js JS syntax checks)
├── __init__.py               # Core 34-block topology, 24-rule Diffusers remapper & 5 Merger nodes
├── anima_pipeline_nodes.py   # 6 Anima Pipeline, Guard & Telemetry nodes (Prompt, NAG, Latent, Remapper, Shield, Telemetry)
├── anima_ksampler_nodes.py   # 3 Anima Rectified Flow KSampler nodes (Single, Dual-Pass, Triple-Pass)
├── anima_loader_nodes.py     # 4 Native Anima Model/CLIP/16ch-VAE Loaders & Smart Tiled VAE Decode
├── web/
│   ├── icon.png              # 128x128 32-bit RGBA genuine PNG extension icon
│   ├── hades_suite.js        # Lightweight ComfyUI extension orchestrator (18 nodes)
│   ├── hades_widgets.js      # Reusable HaDeS Spectral Canvas UI components & theme tokens
│   ├── hades_mergers_ui.js   # UI installers for the 5 Merger & LoRA nodes
│   ├── hades_pipeline_ui.js  # UI installers & custom telemetry HUDs for the 6 Pipeline nodes
│   ├── hades_ksampler_ui.js  # UI installers & flow/staircase HUDs for the 3 Anima KSamplers
│   └── hades_loaders_ui.js   # UI installers & diagnostic badges for the 4 Loader/Decode nodes
├── examples/                 # Ready-to-use ComfyUI workflow JSONs (Single, Dual, Triple Pass, & Merging)
│   ├── HaDeS - Anima Suite - Merge Workflow Example - Medium.json
│   ├── HaDeS Anima Suite - Dual Pass Generation Example.json
│   ├── HaDeS Anima Suite - Simple Pass Generation Example.json
│   └── HaDeS Anima Suite - Triple Pass Generation Example.json
├── tests/
│   └── test_hades_suite.py   # 22 automated unit & integration test suites
├── CHANGELOG.md              # Historical version changelog
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Running Automated Tests

```bash
python tests/test_hades_suite.py
```

---

## License

MIT License — © 2026 **HaDeS**.
