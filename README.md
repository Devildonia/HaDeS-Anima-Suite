# HaDeS Anima Suite v0.3.3

24 ComfyUI nodes for the **Anima** model family (CircleStone Labs / Comfy Org: `MiniTrainDIT` DiT + `Qwen3-0.6B` text encoder + 6-layer `llm_adapter` + 16-channel Wan 2.1 VAE): loaders, block and LoRA mergers, a LoRA extractor and resizer, per-block LoRA strength, a model saver, prompt and pipeline helpers (including a local artist / character catalog), three samplers with a metadata reader for image savers and a telemetry node, with a shared canvas UI.

Claims about what block groups, presets and schedulers do are measured with [`tools/anima_ablation.py`](tools/anima_ablation.py); results are in [`tools/ABLATION_RESULTS.md`](tools/ABLATION_RESULTS.md).

---

## What the suite does

1. **Topology detection (`28L / 40L / 52L`).** Detects Anima-2.0B (28 DiT blocks), 2.9B (40) and 3.8B (52) plus the 6-layer `llm_adapter`, and maps any depth proportionally onto a 34-band index (`0..27` DiT, `28..33` adapter) used by every merger.
2. **`llm_adapter` protection.** Mergers keep Model 1's adapter (`0.0`) and LoRA loaders strip adapter keys by default, following the training rule `llm_adapter_lr = 0`. Most finetunes leave the adapter untouched, but not all: `anima-aesthetic-v1.1` retrained it (see the results file).
3. **Diffusers / Kohya / OneTrainer / Musubi key remapping.** ComfyUI does not translate Diffusers-format Anima LoRAs (`transformer_blocks.X.attn1.to_q`, `ff.net.0.proj`, `text_conditioner`, …). A 24-rule converter maps them to native `MiniTrainDIT` keys, so Diffusers and Kohya LoRAs can be loaded or merged together.
4. **Pipeline helpers.**
   - **Prompt Studio:** strips `score_*` (and optionally `masterpiece`) tags for Anima-Aesthetic, normalizes Booru tags while keeping kaomojis and whitelisted tokens, builds the `ye-pop` / `deviantart` header, and scales the `T5XXL` emphasis weights.
   - **NAG:** Normalized Attention Guidance on cross-attention, for negative prompts at `CFG = 1.0` (Turbo / Lightning).
   - **Latent:** 16-channel latents with VAE validation (blocks the 64-channel Qwen-Image 2.1 VAE), 11 aspect presets, sizes snapped to 32 px.
   - **Corner Shield:** detects and heals or masks scribble-like marks in image corners.
   - **Artist & Character Picker:** searchable local catalog of Danbooru tags (83k artists, 57k characters, 12k series), output in Anima prompt form.
5. **Telemetry.** Reads the `info` outputs of the other nodes and checks flow-shift consistency, CFG = 1 without NAG, the 16 px resolution grid, the 16-channel VAE and adapter protection. Outputs a Markdown report, JSON and a one-line summary.
6. **LoRA Extractor.** Turns a finetune or merge into a LoRA against its base (SVD of the weight difference). Rank 32 of `anima_pixelart_from_hades_v1.0` against `anima-base-v1.0` is 132 MB instead of 4 GB and renders close to the full model.
7. **LoRA Block Weight.** Applies one LoRA with a separate strength per DiT depth group. Where a LoRA acts differs per LoRA, so there are no fixed style / identity presets; `tools/anima_ablation.py --mode lora` measures a given LoRA.
8. **LoRA Resize.** Lowers a LoRA's rank (fixed, or per layer by an energy target). Measured: keeping about 90% of the energy or more stays visually close to the source.
9. **Model Saver.** Saves merged / LoRA-patched models with the patches baked in, in native ComfyUI or Diffusers layout, as `bfloat16`, `fp8_e4m3fn`, ComfyUI-native ConvRot `INT8` / `W4A4`, or `float32`. Native files load in the stock `Load Diffusion Model` node.

---

## Nodes

```
HaDeS Anima Suite v0.3.3
├── HaDeS_Anima_Suite/Loaders (4 Nodes)
│   ├── 14. Anima Diffusion Model Loader
│   ├── 15. Anima Text Encoder Loader
│   ├── 16. Anima VAE Loader
│   └── 17. Anima VAE Decode
├── HaDeS/Anima Mergers (9 Nodes)
│   ├── 1. Anima Model Block Merger
│   ├── 2. Anima Predictive Model Merger
│   ├── 3. Anima Predictive Multi-LoRA
│   ├── 4. Anima LoRA Merger
│   ├── 5. Anima CLIP Block Merger
│   ├── 19. Anima Model Saver
│   ├── 20. Anima LoRA Extractor
│   ├── 21. Anima LoRA Block Weight
│   └── 22. Anima LoRA Resize
├── HaDeS/Anima Pipeline (7 Nodes)
│   ├── 6. Anima Prompt Studio
│   ├── 23. Anima Artist & Character Picker
│   ├── 7. Anima Attention Guidance (NAG)
│   ├── 8. Anima Latent
│   ├── 9. Anima Universal LoRA Remapper
│   ├── 10. Anima Corner Shield
│   └── 18. Anima Suite Telemetry
└── HaDeS_Anima_Suite/Samplers (4 Nodes)
    ├── 11. Anima KSampler
    ├── 12. Anima Dual-Pass KSampler
    ├── 13. Anima Triple-Pass KSampler
    └── 24. Anima Sampler Metadata
```

---

## 34-Band Block Index (measured)

Input names (`BLOCKS_Layout_Geometry`, …) are kept for workflow compatibility; the UI shows depth-based labels. "Share" is how much of an A→B change the group alone carries (swap-in, two model pairs, see [`tools/ABLATION_RESULTS.md`](tools/ABLATION_RESULTS.md)).

| Index | Native keys (`28L` / `40L` / `52L`) | Input name | UI label | Share of an A→B change |
| :--- | :--- | :--- | :--- | :--- |
| `00–04` | `blocks.0` .. ~18% depth | `BLOCKS_Layout_Geometry` | Early Blocks | 0.22–0.84 |
| `05–09` | ~18% .. ~36% | `BLOCKS_Global_Composition` | Early-Mid Blocks | 0.18–0.57 |
| `10–14` | ~36% .. ~54% | `BLOCKS_Subject_Identity` | Core Blocks (strongest) | 0.44–0.75 |
| `15–19` | ~54% .. ~71% | `BLOCKS_Material_Substance` | Mid-Late Blocks | 0.38–0.62 |
| `20–23` | ~71% .. ~86% | `BLOCKS_Art_Style_Medium` | Late Blocks | 0.19–0.38 |
| `24–27` | ~86% .. 100% | `BLOCKS_Lighting_Sharpness` | Final Blocks (weakest) | 0.12–0.32 |
| `28–33` | `llm_adapter.blocks.0..5` | `ADAPTER_Text_Alignment` | LLM Adapter Bridge | 0 when the finetune kept the adapter |

No group isolates one attribute: each moves composition, structure, texture and color together. Neither per-sublayer merges nor a late model switch separate style from composition either.

---

## Node Reference

### Part A — Mergers & LoRA tools (`HaDeS/Anima Mergers`)

#### 1. `Anima Model Block Merger`
- **Class:** `HaDeSAnimaModelBlockMerger`
- **Purpose:** Merges two Anima `MODEL`s per block group, reading patched (merged / LoRA) weights and skipping identical tensors.
- **Inputs:**
  - `model1`, `model2` (`MODEL`).
  - `base_ratio` (`FLOAT`, default `0.50`): ratio for non-block layers (`x_embedder`, `t_embedder`, `final_layer`). `0.0` = Model 1, `1.0` = Model 2.
  - `BLOCKS_Layout_Geometry` .. `BLOCKS_Lighting_Sharpness` (`FLOAT`, default `0.50`).
  - `ADAPTER_Text_Alignment` (`FLOAT`, default `0.00`): keeps Model 1's `llm_adapter`.
  - `recipe_override` (`STRING`, optional): 34-float CSV or the JSON export of the Predictive Merger.

#### 2. `Anima Predictive Model Merger`
- **Class:** `HaDeSAnimaPredictiveModelMerger`
- **Purpose:** Derives per-block ratios from weight similarity and divergence between the two models (or task vectors against `base_model` when connected) and outputs the merged model plus a reusable recipe.
- **Inputs:**
  - `model1`, `model2` (`MODEL`), `base_model` (`MODEL`, optional).
  - `preset`: `Style Transfer (Keep A identity, B style)`, `Interference Guard (Anti-Burn / Artifact Shield)`, `Complementary Richness (Fill gaps of A with B)`, or `Balanced Geometric Fusion`. Measured, the presets mainly scale Model 2 strength; `Style Transfer` stays closest to Model 1 and is shown as **Conservative (mostly Model 1)** in the HaDeS UI, since no merge keeps Model 1's composition with Model 2's style. For Model 2 detail on Model 1's composition, use the Dual-Pass KSampler with `model_pass2` at low denoise.
  - `target_influence` (`0.0..1.0`, default `0.50`), `aggressiveness` (`0.1..2.5`, default `1.00`).
  - `protect_llm_adapter` (`BOOLEAN`, default `True`): locks blocks `28..33` to Model 1.

#### 3. `Anima Predictive Multi-LoRA`
- **Class:** `HaDeSAnimaPredictiveMultiLoraLoader`
- **Purpose:** Loads up to 4 Anima LoRAs (`PRI`, `S2`–`S4`), remaps Diffusers / Kohya keys, and damps secondary slots by their overlap with the primary LoRA (reduced QR on the low-rank factors).
- **Notes:** per-slot 34-band curve; strips `llm_adapter` keys when `protect_llm_adapter = True`; `1/√N` damping on shared CLIP layers.
- **Presets (measured with two style LoRAs):** `Character + Style Fusion (A identity, B style)` leans towards slot 1 in composition and style alike and is shown as **Favor Slot 1 (PRI)** in the HaDeS UI; `Anti-Overcook`, `Orthogonal Synthesis` and `Balanced 50/50 Adaptive` stay within 0.03 of stacking the LoRAs at 1.0.

#### 4. `Anima LoRA Merger`
- **Class:** `HaDeSAnimaStandaloneLoraMerger`
- **Purpose:** Merges two Anima LoRAs (Diffusers and Kohya / ComfyUI formats can be mixed) into one `.safetensors` file with reduced QR-SVD:
  $$\Delta W = U_{\text{cat}} D_{\text{cat}} = Q_U (R_U R_D^T) Q_D^T$$
  so the SVD runs on a $2r \times 2r$ core instead of the full $M \times N$ matrix.
- **Modes:** `Additive (A + B, Multi-LoRA match)` (default) or `Interpolated (Lerp (1-r)A + rB)`, optional DARE pruning, and protection against overwriting a donor file.
- **Presets:** same formula family as the Multi-LoRA presets; `Character + Style Fusion` is shown as **Favor Donor A**.

#### 5. `Anima CLIP Block Merger`
- **Class:** `HaDeSAnimaCLIPBlockMerger`
- **Purpose:** Merges two Anima text encoders (`Qwen3-0.6B` 28 layers or `Qwen3.5-4B` 36 layers) across 5 depth bands (`vocab_embeddings`, `syntax_rigidity` `[0–35%]`, `semantic_focus` `[35–72%]`, `style_abstraction` `[72%+]`, `final_projections`). Band names are not measured.

#### 19. `Anima Model Saver`
- **Class:** `HaDeSAnimaModelSaver`
- **Purpose:** Saves a `MODEL` with its merge / LoRA patches baked in (same lazy path as ComfyUI's checkpoint saver), as native ComfyUI or Diffusers layout, and passes the model through for downstream samplers.
- **Inputs:**
  - `model`: input `MODEL`.
  - `filename_prefix`: path under `ComfyUI/output` (default `diffusion_models/Anima_HaDeS_Merged`).
  - `save_format`: `Native ComfyUI (MiniTrainDIT)` or `Diffusers (HuggingFace / Trainer)`.
  - `precision`: `bfloat16 (Original Master)`, `fp8_e4m3fn (ComfyUI Native FP8)`, `convrot_int8 (Hadamard Rotated INT8)`, `convrot_int4 (Hadamard Rotated INT4)`, `float32 (Uncompressed)`. ConvRot uses ComfyUI's own `comfy.quant_ops` layouts (`int8_tensorwise` + convrot, `convrot_w4a4`) on the DiT block linears, with per-layer `comfy_quant` configs.
  - `llm_adapter_mode`: `Embed in Checkpoint (Full)`, `Strip (Pure DiT)`, `Export Adapter Separately` (bare `llm_adapter` keys, as `external_llm_adapter` expects).
  - `include_metadata`, `custom_author`, `custom_model_title`: SAI ModelSpec metadata.
- **Outputs:** `model` (pass-through), `saved_path`, `info`.

#### 20. `Anima LoRA Extractor`
- **Class:** `HaDeSAnimaLoraExtractor`
- **Purpose:** Saves `model_tuned − model_base` as a LoRA in `models/loras/`: truncated SVD per linear layer (`lora_up` / `lora_down` / `alpha`), full `.diff` / `.diff_b` for norm weights and biases, identical tensors skipped. Merge and LoRA patches on either input are included (same baked path as the Model Saver); quantized inputs are rejected with an error.
- **Inputs:**
  - `model_tuned`, `model_base` (`MODEL`).
  - `rank` (`4..256`, default `32`).
  - `export_filename` (`STRING`, saved under `models/loras/`).
  - `include_llm_adapter` (`BOOLEAN`, default `False`): also extract `llm_adapter` differences, for releases that retrained it (e.g. `anima-aesthetic-v1.1`).
  - `overwrite_existing` (`BOOLEAN`, default `False`).
- **Outputs:** `saved_lora_path`, `info` (includes the share of the difference kept by the rank, "energy kept").
- **Measured:** base → pixelart, rank 32: 132 MB, 94.5% energy kept, all keys load in the stock `LoraLoader`, renders at 0.87–0.96 of the way from base to the finetune on the suite's metrics; rank 128: 529 MB, 0.91–0.98.

#### 21. `Anima LoRA Block Weight`
- **Class:** `HaDeSAnimaLoraBlockWeight`
- **Purpose:** Applies one LoRA with a multiplier per DiT depth group (`blocks_00_04` … `blocks_24_27`, canonical 28-block index; 40L / 52L models are mapped proportionally) and `other_layers` for the embedders, final layer and unprotected `llm_adapter`. Keys are remapped like the other LoRA nodes.
- **Inputs:** `model`, `lora_name`, `strength_model`, the six group multipliers and `other_layers` (`-2..3`, default `1.0`); optional `clip`, `strength_clip`, `protect_llm_adapter` (default `True`).
- **Measured:** one style LoRA acted mostly in blocks 05–19, another in 00–04 (with most of its color change in 20–23); blocks 24–27 carried the least change for both. Fixed "style only" / "identity only" presets would be wrong for one of them, so the node has none; use `tools/anima_ablation.py --mode lora --lora <name>` to see where a LoRA acts.

#### 22. `Anima LoRA Resize`
- **Class:** `HaDeSAnimaLoraResize`
- **Purpose:** Lowers a LoRA's rank with an exact SVD of its low-rank factors (QR + small core SVD) and saves a new file in `models/loras/`; the source is never overwritten. Norm / bias diffs and unsupported formats (LoHa, LoKr, DoRA) are copied unchanged; keys are written in native Anima form.
- **Inputs:** `lora_name`, `target_rank` (max rank per layer, `1..256`, default `32`), `export_filename`; optional `dynamic_energy` (`0` = fixed rank; otherwise the smallest rank per layer that keeps that share of the layer's energy, capped at `target_rank`) and `overwrite_existing`.
- **Outputs:** `saved_lora_path`, `info` (size before / after, rank, mean and worst-layer energy kept).
- **Measured** (renders vs. the source LoRA, 0 = base model, 1 = source LoRA; every key loads in the stock LoRA loader):

  | LoRA | Variant | Size | Energy kept | layout / edges / texture / color |
  |---|---|---|---|---|
  | Niji_semi_realism_v5 (264 MB) | rank 16 | 66 MB | 86% | 0.79 / 0.80 / 0.88 / 0.92 |
  | | rank 32 | 132 MB | 94% | 0.85 / 0.87 / 0.91 / 0.95 |
  | | dynamic 0.90 | 85 MB | 93% | 0.82 / 0.83 / 0.89 / 0.94 |
  | dungeon_squad_style_anima_v1 (175 MB) | rank 16 | 44 MB | 75% | 0.57 / 0.59 / 0.65 / 0.61 |
  | | rank 32 | 88 MB | 89% | 0.68 / 0.72 / 0.79 / 0.80 |
  | | dynamic 0.90 | 80 MB | 90% | 0.69 / 0.73 / 0.74 / 0.77 |

  How far a LoRA compresses depends on the LoRA; the info output warns below 90% mean energy, where the dungeon rank-16 result visibly lost its pixel style.

---

### Part B — Pipeline nodes (`HaDeS/Anima Pipeline`)

#### 6. `Anima Prompt Studio`
- **Class:** `HaDeSAnimaPromptStudio`
- **Purpose:** Applies the CircleStone Labs prompt rules and exposes the `T5XXL` emphasis multiplier.
- **How it works:**
  - **Aesthetic tag guard:** with `Anima Aesthetic (Strip score_* & Slop Guard)` it strips `score_*`, `masterpiece`, `best quality`, `8k` and similar tags, which the Aesthetic models were trained without. `Anima Aesthetic v1.0b (Keep masterpiece, best quality)` keeps those two quality tags and still strips `score_*`.
  - **Booru sanitizer:** lowercases tags and replaces `_` with spaces, except whitelisted tokens (`score_*`, `*_up`, `safe_for_work`) and kaomojis (`^_^`, `o_o`, `>_<`, …).
  - **Style header:** in `Flat Vector / Graphic` mode, writes the non-anime header: one trigger (`ye-pop` or `deviantart`) on line 1, optional `artwork_title` on line 2, prompt on line 3. `header_mode = "Dual Legacy"` keeps the old two-trigger header; prompts that already start with a trigger are kept verbatim.
  - **Artist / character inputs:** names are normalized to Anima form (`@artist`, spaces, literal parentheses escaped). Unescaped, `@hammer (sunset beach)` turns `(sunset beach)` into a weighted concept that leaks into the image ([results](tools/ABLATION_RESULTS.md#catalog-tags-and-parenthesis-escaping)).
  - **`t5_weight_multiplier`:** ComfyUI's Anima tokenizer (`comfy/text_encoders/anima.py`) forces the Qwen3 token weights to `1.0` and passes `(tag:weight)` through the `T5XXL` tokenizer to the `llm_adapter`; this slider scales those emphasis weights.

#### 23. `Anima Artist & Character Picker`
- **Class:** `HaDeSAnimaArtistCharacterPicker`
- **Purpose:** Picks artists, a character and a series from a local Danbooru tag catalog shipped in `web/data/` (83,355 artists, 57,653 characters, 12,405 series with at least 20 posts; nothing is downloaded) and outputs them in Anima prompt form for Prompt Studio's `artist` / `character` inputs or any text node.
- **Inputs:** `artists` (comma-separated tags, optional `:weight`, e.g. `dairi:1.2, ebifurya`), `character`, `series`, `random_artists` (`0..4`, extra artists drawn by seed, never repeating a listed one), `random_min_posts` (default `200`), `seed`.
- **Outputs:** `artist` (e.g. `(@hammer \(sunset beach\):1.2), @dairi`), `character` (character + series, e.g. `firefly \(honkai: star rail\), honkai: star rail`), `info`.
- **UI:** click a row to open a search popup (prefix matches first, post counts shown; the catalog loads on first use); Shift+click edits the text; right-click clears.

#### 7. `Anima Attention Guidance (NAG)`
- **Class:** `HaDeSAnimaAttentionGuidance`
- **Purpose:** Normalized Attention Guidance on cross-attention only (tagged through `attn1_patch` / `attn2_patch`), inside a `start_percent` → `end_percent` window converted with `percent_to_sigma`. The positive queries re-attend over the negative prompt's keys / values and are blended with the reference NAG formula (scale, L1-norm ceiling `nag_tau`, `nag_alpha`).
- **Use:** negative prompts at `CFG = 1.0` on Anima-Turbo / Lightning. `nag_scale = 1.0` disables it. `only_anima = True` (default) bypasses non-Anima models via `unet_config.image_model`.

#### 8. `Anima Latent`
- **Class:** `HaDeSAnimaLatentPipelineGuard`
- **Purpose:** 16-channel latents (`[B, 16, H // 8, W // 8]`) with optional `model` / `vae` validation.
- **Checks:**
  - **64-channel VAE:** raises a clear error if the 64-channel Qwen-Image 2.1 VAE is connected instead of the 16-channel `qwen_image_vae.safetensors`.
  - **Anima-2.9B `1280×1280`:** with `prevent_29b_white_bg_bias`, shifts `1280×1280` to `1216×1216`, following the community report that this size biases 2.9B towards white backgrounds (not measured here).
  - **Sizes:** 11 aspect presets (including `1:4` and `4:1`); custom sizes snap to 32 px, which also satisfies Anima's 16 px grid.

#### 9. `Anima Universal LoRA Remapper`
- **Class:** `HaDeSAnimaUniversalLoraLoader`
- **Purpose:** Loads a third-party Anima LoRA (Diffusers `transformer_blocks.*`, `ai-toolkit`, Kohya, OneTrainer, Musubi), converts keys with the 24-rule `DIFFUSERS_TO_COMFY` map, strips `llm_adapter` weights (`strip_llm_adapter = True`), optionally scrubs `NaN` / `Inf` and $>10\sigma$ outliers, and optionally saves a ComfyUI-native copy to `models/loras/`.

#### 10. `Anima Corner Shield`
- **Class:** `HaDeSAnimaCornerScribbleShield`
- **Purpose:** Handles the signature-like scribbles Anima sometimes draws in corners (learned from signed training art; negative prompts only reduce them).
- **Tuning:** `corner_sensitivity` (default `0.75`) sets the detection threshold; the inspection box defaults to `10%` of the shorter side (`0.03`–`0.50`), with expansion for large marks capped at 50%. Clean corners are left untouched in Smart mode.
- **Modes:**
  - `Mask Only (No Pixel Modification)`: feathered `[B, H, W]` mask of flagged corners for a second inpainting pass.
  - `Smart Edge-Inward Blend (Auto-Heal)`: heals flagged corners by extending the clean edge ring (8–16 px outside the box) inwards; a texture guard avoids flagging uniform grain or hatching. Default.
  - `Aggressive Corner Fill (Forced)`: skips detection and always fills the corner boxes.
  - `Soft Corner Blur`: feathered local blur, no reconstruction.

---

### Part C — Samplers (`HaDeS_Anima_Suite/Samplers`)

#### 11. `Anima KSampler`
- **Class:** `HaDeSAnimaKSampler`
- **Purpose:** Single-pass sampler for Anima's rectified flow. Flow shift `0` inherits the shift already set on the model (from the loader); other values override it with `multiplier = 1.0` (Anima's own setting; `ModelSamplingSD3` defaults to `1000`).
- **Features:**
  - **Extra schedulers:** `anima_beta57` (Beta α = 0.50, β = 0.70; more steps at low noise — measured, it converges slower than stock `beta`) and `anima_lightning_tdm` (the documented 4-step schedule, sigmas `1.0 → 0.9 → 0.75 → 0.5 → 0` at shift 3; other step counts use an improvised spacing; not tested on a TDM model).
  - **Variant profiles:** `Anima Base / Raehoshi`, `Anima Aesthetic Crisp`, `Anima Aesthetic Painterly (beta57)`, `Anima Turbo`, `Anima Lightning TDM`, `Anima 2.9B / 3.8B`.
  - **Optional decode:** when a VAE is connected, checks `latent_channels == 16` and also outputs the decoded image.

#### 12. `Anima Dual-Pass KSampler`
- **Class:** `HaDeSAnimaDualPassKSampler`
- **Purpose:** Composition pass followed by an upscaled refinement pass (`1.25x`–`2.0x`), with the intermediate size snapped to 32 px.
- **Features:**
  - **Latent bridge:** bicubic latent upscale that restores per-channel mean / std on the 16 channels (bicubic lowers latent variance), or a `Pixel-Space VAE Round-Trip (Lanczos 32px)`.
  - **`model_pass2`:** optional different model for the refinement pass, e.g. composition with a character LoRA, detail with Anima-Aesthetic or Turbo. This is also the way to get Model 2 detail on Model 1's composition.

#### 13. `Anima Triple-Pass KSampler`
- **Class:** `HaDeSAnimaTriplePassKSampler`
- **Purpose:** Three passes: composition at `1.0x`, an intermediate pass at ~`1.35x` (denoise ~0.48) and a low-denoise pass at ~`1.75x`–`2.0x` (denoise ~0.26), as an alternative to one large upscale step that must trade duplicated details (high denoise) against blur (low denoise). Optional `model_pass2` / `model_pass3`.

#### 24. `Anima Sampler Metadata`
- **Class:** `HaDeSAnimaSamplerMetadata`
- **Purpose:** Connect a KSampler's `info` output; it outputs `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` (strings) and `denoise` for image-saver metadata nodes (e.g. Image Saver Metadata). The values are the ones actually sampled with, after the profile / preset is applied, so the metadata cannot drift from a separate settings node. Dual / Triple-Pass report pass 1, the pass that generates the image. Any other `info` string raises an error instead of saving made-up values.

---

### Part D — Loaders & VAE Decode (`HaDeS_Anima_Suite/Loaders`)

#### 14. `Anima Diffusion Model Loader`
- **Class:** `HaDeSAnimaModelLoader`
- **Purpose:** Loads Anima checkpoints in native ComfyUI or Diffusers layout (`transformer_blocks.*`, `patch_embed.proj.*`, `time_embed.t_embedder.*`, `text_conditioner.*`, converted in memory), optionally injects an external `llm_adapter` (e.g. `Anima-3.8B-expanded_adapter.safetensors`), detects `28L / 40L / 52L` topology, and sets the flow shift (`multiplier = 1.0`).

#### 15. `Anima Text Encoder Loader`
- **Class:** `HaDeSAnimaCLIPLoader`
- **Purpose:** Loads `qwen_3_06b_base.safetensors` after checking its header for the 1024-dim Qwen3-0.6B architecture (rejects T5-XXL, CLIP-L or 3584-dim Qwen2.5-7B files), with a `cpu (Low-VRAM 6GB Safe)` placement option. ComfyUI selects the Anima text encoder from the weights themselves, whatever `clip_type` is used.

#### 16. `Anima VAE Loader`
- **Class:** `HaDeSAnimaVAELoader`
- **Purpose:** Checks the VAE header before loading to reject the 64-channel Qwen-Image 2.1 VAE (which shares the filename `qwen_image_vae.safetensors`), with `bfloat16` as the default precision.

#### 17. `Anima VAE Decode`
- **Class:** `HaDeSAnimaVAEDecode`
- **Purpose:** Decodes 16-channel latents, switching to tiled decode above `auto_tile_mp_threshold` (default `1.35 MP`) or when free VRAM is low, with an optional Corner Shield pass (`corner_box_percent`, `corner_sensitivity`, `shield_heal_mode`, `shield_feather_radius`).

#### 18. `Anima Suite Telemetry`
- **Class:** `HaDeSAnimaSuiteTelemetry`
- **Purpose:** Collects the `info` outputs of HaDeS nodes, groups them by pipeline phase (architecture, conditioning & LoRAs, sampling & guidance, post-processing) and checks: flow-shift consistency between loader and sampler, `CFG = 1` without NAG, the 16 px resolution grid, the 16-channel VAE, and `llm_adapter` protection. Outputs a Markdown report, a JSON payload and a one-line summary.
- **Info contract:** every HaDeS node ends its `info` output with one machine-readable line, `HADES_META {"node": ..., ...}` (keys per node in `HADES_META_SCHEMA`). Telemetry reads values from that line; the prose above it is for humans only and can change freely. `info` strings without the line (older versions, other packs) still go through the legacy text parser.

---

## Regional Prompts

ComfyUI's stock `ConditioningSetArea` / `ConditioningSetMask` work on Anima. In a two-character test (left / right, 8 seeds) a single natural-language prompt ("the girl on the left has …, the girl on the right has …") placed both correctly 8/8 and separated them better than area or mask conditioning, so the suite has no regional node ([results](tools/ABLATION_RESULTS.md#regional-prompts)).

---

## HaDeS Node UI vs. Standard ComfyUI Widgets

The suite draws its own canvas widgets (`web/hades_widgets.js`): a neutral dark palette with one accent color (green / amber / red only signal state), quiet uppercase row labels, preset buttons that highlight the active preset, and a compact status panel per node. Every node enforces a minimum width so nothing clips or overlaps, and hover tooltips explain each control.

To use the plain ComfyUI inputs instead, open **Settings → HaDeS Anima Suite → Appearance** and enable `Use standard ComfyUI widgets on HaDeS nodes` (setting id `HaDeS.ClassicUI`). It applies immediately to all 24 nodes; a single node can also be switched from its right-click menu (**HaDeS Suite → Use standard ComfyUI widgets**).

Both the classic LiteGraph canvas and the Vue node renderer (*Nodes 2.0*, `Comfy.VueNodes.Enabled`) are supported, including the classic-widget toggle.

---

## Example Workflows (`examples/`)

1. **`HaDeS Anima Suite - Simple Pass Generation Example.json`:** loaders, Prompt Studio, Latent, `Anima KSampler` and VAE decode.
2. **`HaDeS Anima Suite - Dual Pass Generation Example.json`:** composition at 1.0 MP and a 1.5x refinement pass with the 16-channel latent bridge and `model_pass2`.
3. **`HaDeS Anima Suite - Triple Pass Generation Example.json`:** `1.0x → 1.35x → 1.75x` with `Anima Triple-Pass KSampler`.
4. **`HaDeS - Anima Suite - Merge Workflow Example - Medium.json`:** `Anima Model Block Merger`, `Anima Predictive Multi-LoRA` and `Anima LoRA Merger` with key remapping and adapter protection.

---

## Measuring Block Groups, Presets and Schedulers

[`tools/anima_ablation.py`](tools/anima_ablation.py) queues fixed prompt × seed grids on a running local ComfyUI (HTTP API only) and compares the results with numpy + Pillow image statistics (layout, edges, texture, color). Modes: `groups`, `presets`, `schedulers`, `lora` (one LoRA per block group, through LoRA Block Weight) and `lora_presets` (Multi-LoRA presets).

```bash
python tools/anima_ablation.py --mode groups --model-a anima-base-v1.0.safetensors --model-b <finetune>.safetensors --out <folder>
```

Re-running with the same `--out` resumes. Check the contact sheets as well as the numbers: noise artifacts can score as texture change.

---

## Project Structure

```
Comfyui-HaDeS-Anima-BlockMerger/
├── .github/workflows/ci.yml  # CI (Python 3.10-3.12 + JS syntax checks)
├── __init__.py               # 34-band topology, Diffusers remapper, info contract, 5 mergers, Model Saver, LoRA Extractor, LoRA Block Weight, LoRA Resize
├── anima_pipeline_nodes.py   # Prompt Studio, Artist & Character Picker, NAG, Latent, LoRA Remapper, Corner Shield, Telemetry
├── anima_ksampler_nodes.py   # Single, Dual-Pass and Triple-Pass KSamplers, Sampler Metadata, Anima schedulers
├── anima_loader_nodes.py     # Model / text encoder / VAE loaders and VAE Decode
├── web/
│   ├── icon.png
│   ├── hades_suite.js        # Extension entry point and settings (24 nodes)
│   ├── data/                 # Local Danbooru tag catalog (artists, characters, series) for the picker
│   ├── hades_widgets.js      # Shared canvas widgets, design tokens, Vue-node adapter
│   ├── hades_mergers_ui.js   # Merger, LoRA, Saver, Extractor, Block Weight and Resize node UIs
│   ├── hades_pipeline_ui.js  # Pipeline, Picker (catalog search) and Telemetry node UIs
│   ├── hades_ksampler_ui.js  # Sampler node UIs
│   └── hades_loaders_ui.js   # Loader and VAE Decode node UIs
├── examples/                 # Example workflows
├── tools/
│   ├── anima_ablation.py     # Measurement script (dev tool, not a node)
│   └── ABLATION_RESULTS.md   # Measured results
├── tests/
│   └── test_hades_suite.py   # 31 test suites (standalone stubs or real ComfyUI)
├── CHANGELOG.md
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Running Tests

```bash
python tests/test_hades_suite.py
```

Set `HADES_FORCE_STANDALONE_STUBS=1` to run against the bundled stubs instead of a ComfyUI install.

---

## License

MIT License — © 2026 **HaDeS**.
