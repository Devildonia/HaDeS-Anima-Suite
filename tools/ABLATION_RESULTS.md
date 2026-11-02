# Ablation results (2026-10-31)

Measured with `tools/anima_ablation.py` on an RTX 5090, ComfyUI local API. 4 prompts x 2 seeds per configuration,
1024x1024, er_sde / normal / 28 steps / CFG 4.5 / shift 3.0 (inherited), text encoder `qwen_3_06b_base`, VAE `qwen_image_vae`.
Model A = `anima-base-v1.0`. Two B models: `anima_pixelart_from_hades_v1.0` and `anima-aesthetic-v1.1`.

Metrics (see the script docstring): **layout** (32x32 lightness thumbnail), **edges** (Sobel correlation),
**texture** (high-frequency power spectrum), **color** (Lab mean / std).

## Block groups

`share` = distance(variant, reference) / distance(B, A). Swap-in = A with only that group taken from B (reference A).
Leave-out = B with only that group taken from A (reference B). Higher = the group carries more of the A->B change.

**Swap-in**

| Group | pixelart layout / edges / texture / color | aesthetic layout / edges / texture / color |
|---|---|---|
| B00-04 | 0.42 / 0.42 / 0.22 / 0.34 | 0.53 / 0.54 / 0.84 / 0.39 |
| B05-09 | 0.40 / 0.39 / 0.18 / 0.27 | 0.53 / 0.55 / 0.57 / 0.34 |
| B10-14 | **0.73 / 0.74 / 0.44 / 0.54** | **0.72 / 0.75 / 0.67 / 0.53** |
| B15-19 | 0.58 / 0.62 / 0.43 / 0.40 | 0.51 / 0.54 / 0.47 / 0.38 |
| B20-23 | 0.36 / 0.38 / 0.19 / 0.20 | 0.38 / 0.37 / 0.35 / 0.22 |
| B24-27 | 0.25 / 0.23 / 0.14 / 0.12 | 0.32 / 0.29 / 0.32 / 0.17 |
| B28-33 (llm_adapter) | 0.00 / 0.00 / 0.00 / 0.00 | 0.26 / 0.23 / 0.24 / 0.08 |

**Leave-out**

| Group | pixelart layout / edges / texture / color | aesthetic layout / edges / texture / color |
|---|---|---|
| B00-04 | 0.47 / 0.39 / 0.27 / 0.24 | 0.44 / 0.41 / 0.63 / 0.22 |
| B05-09 | 0.53 / 0.43 / 0.29 / 0.30 | 0.49 / 0.47 / 0.56 / 0.19 |
| B10-14 | **0.89 / 0.83 / 0.41 / 0.41** | **0.67 / 0.69 / 0.64 / 0.30** |
| B15-19 | 0.78 / 0.72 / **0.86 / 0.55** | 0.48 / 0.44 / 0.43 / 0.25 |
| B20-23 | 0.45 / 0.35 / 0.37 / 0.20 | 0.30 / 0.26 / 0.35 / 0.17 |
| B24-27 | 0.39 / 0.28 / 0.24 / 0.18 | 0.28 / 0.21 / 0.36 / 0.11 |
| B28-33 (llm_adapter) | 0.00 / 0.00 / 0.00 / 0.00 | 0.14 / 0.08 / 0.20 / 0.05 |

Findings:
- No group isolates one attribute. Every group moves layout, structure, texture and color together.
- Influence falls with depth after the middle: **B10-14 > B15-19 ~ B00-09 > B20-23 > B24-27** in both pairs.
- B20-23 ("Art Style & Medium") and B24-27 ("Lighting & Sharpness") carry the *least* texture and color change of all DiT groups.
- B00-04 ("Layout & Geometry") carries as much or more texture change than layout change (aesthetic: 0.84 texture).
- `llm_adapter` is untouched by the pixelart finetune but **was trained in anima-aesthetic-v1.1**, so "llm_adapter_lr = 0" does not hold for every Anima release.

## Predictive Merger presets

Position per metric between A (0.0) and B (1.0), `target_influence = 0.5`, aggressiveness 1.0, adapter protected, no base model.

| Preset | pixelart layout / edges / texture / color | aesthetic layout / edges / texture / color |
|---|---|---|
| Uniform 0.5 merge (reference) | 0.48 / 0.50 / 0.35 / 0.52 | 0.57 / 0.58 / 0.47 / 0.66 |
| Style Transfer (Keep A identity, B style) | 0.31 / 0.31 / 0.27 / 0.26 | 0.35 / 0.37 / 0.41 / 0.32 |
| Interference Guard | 0.41 / 0.42 / 0.29 / 0.39 | 0.42 / 0.43 / 0.44 / 0.49 |
| Complementary Richness | 0.53 / 0.56 / 0.46 / 0.61 | 0.64 / 0.68 / 0.60 / 0.75 |
| Balanced Geometric Fusion | 0.43 / 0.45 / 0.29 / 0.43 | 0.44 / 0.45 / 0.45 / 0.53 |

Findings:
- The presets behave as global strength scalings (Style Transfer < Interference Guard ~ Balanced < uniform < Complementary).
- **Style Transfer does not transfer style**: texture and color move no more towards B than layout does, and less than a plain 0.5 merge. It weights B on B20-27, the groups that carry the least style change.

## Schedulers

euler, model A, mean distance to an 80-step euler / normal reference (lower = closer to the converged image).

| Steps | normal layout / color | beta layout / color | anima_beta57 layout / color |
|---|---|---|---|
| 8 | 23.4 / 22.5 | **18.4 / 14.5** | 22.3 / 21.2 |
| 14 | 19.3 / 17.0 | **10.1 / 4.6** | 16.8 / 12.8 |
| 24 | 12.6 / 9.4 | **7.1 / 2.9** | 8.3 / 3.5 |

- `anima_beta57` (Beta 0.5 / 0.7) converges faster than `normal`, but slower than ComfyUI's stock `beta` at every step count.
  Its schedule does put more steps at low noise (8 of 28 below sigma 0.3, vs 5 for `beta`, 3 for `normal`); its scipy-free fallback is within 0.0008 of `scipy.stats.beta.ppf`.
- Convergence is not aesthetics; this only says beta57 reaches the same image later, not that it looks worse.
- `anima_lightning_tdm`: the 4-step schedule reproduces the documented [1.0, 0.9, 0.75, 0.5, 0.0] at shift 3. Not tested on images (no TDM-distilled model available); other step counts use an improvised spacing.

## Resolution grid

Base model, stock `EmptySD3LatentImage` at exact sizes (no snapping), 8 samples per size. Brightness jump between the last
8 px rows / columns and the 8 before them (ComfyUI pads the latent to the 2x2 patch grid at the bottom / right edges):

| Size | mod 16 | mod 32 | bottom jump | right jump | top jump (control) |
|---|---|---|---|---|---|
| 1024 | 0 | 0 | 5.1 | 2.5 | 5.0 |
| 1008 | 0 | 16 | 4.4 | 7.2 | 2.9 |
| 1032 | 8 | 8 | 6.8 | 3.2 | 12.2 |
| 1000 | 8 | 8 | 13.7 | 7.6 | 2.0 |
| 1016 | 8 | 24 | 12.9 | 5.4 | 0.8 |

- The native grid is **16 px** (VAE 8x * patch 2). Sizes off that grid tend to get a band on the padded bottom edge.
- Nothing separates 16-aligned from 32-aligned sizes; the "32 px macroblock" rule is not supported.

## Can style be separated from structure? (in-process probe)

Same prompts, seeds and metrics; position between A (0) and B (1). Run in-process with ComfyUI's own `KSampler`
(er_sde / normal / 28 steps), because neither probe is expressible with the suite's nodes:
- **sublayer**: A with one DiT sublayer type taken from B in all 28 blocks (`ModelPatcher.add_patches`, like `ModelMergeSimple`).
- **switch**: one sampling run that calls B instead of A once sigma drops below the threshold
  (`set_model_unet_function_wrapper` + `set_additional_models`).

| Variant | pixelart layout / edges / texture / color | aesthetic layout / edges / texture / color |
|---|---|---|
| self_attn from B | 0.34 / 0.35 / 0.21 / 0.31 | 0.35 / 0.34 / 0.43 / 0.37 |
| cross_attn from B | 0.24 / 0.25 / 0.14 / 0.12 | 0.26 / 0.25 / 0.37 / 0.14 |
| mlp from B | 0.47 / 0.50 / 0.34 / 0.46 | 0.43 / 0.43 / 0.52 / 0.48 |
| adaln_modulation from B | 0.37 / 0.38 / 0.34 / 0.33 | 0.45 / 0.47 / 0.53 / 0.54 |
| B below sigma 0.90 | 0.31 / 0.35 / 0.52 / 0.23 | 0.29 / 0.28 / 0.57 / 0.27 |
| B below sigma 0.75 | 0.12 / 0.10 / 0.36 / 0.09 | 0.10 / 0.06 / 0.20 / 0.09 |
| B below sigma 0.50 | 0.04 / 0.01 / 0.15 / 0.03 | 0.03 / 0.01 / 0.09 / 0.03 |

Findings:
- No sublayer type carries style without structure; results also flip between the two pairs.
- Switching to B late keeps A's composition **and** A's palette / lighting; B only re-renders fine detail (texture 0.52-0.57 at sigma 0.9).
  Palette, lighting and pixel grid are decided in the same high-noise steps as composition.
- Splitting the same schedule across two `KSamplerAdvanced` calls with er_sde produced heavy noise artifacts that the texture metric
  scored as "style"; always check the contact sheets, not only the numbers.

Conclusion: "A identity + B style" is not reachable by weight merging (per group or per sublayer) or by a late model switch
for these pairs. The closest measurable effect is "A composition and palette + B detail" via a late switch, which the
Dual-Pass / Triple-Pass KSamplers already offer through `model_pass2` / `model_pass3` at low denoise.

## LoRAs (`--mode lora` / `--mode lora_presets`)

Model `anima-base-v1.0`, LoRAs `dungeon_squad_style_anima_v1` and `Niji_semi_realism_v5` at strength 1.0, applied through
Anima LoRA Block Weight. A = no LoRA, B = full LoRA; same prompts, seeds and metrics as above.

**Swap-in (only that group of the LoRA)**

| Group | dungeon layout / edges / texture / color | niji layout / edges / texture / color |
|---|---|---|
| blocks 00-04 | 0.44 / 0.46 / 0.26 / 0.41 | **0.65 / 0.74 / 0.72** / 0.55 |
| blocks 05-09 | 0.64 / 0.67 / 0.57 / 0.57 | 0.39 / 0.42 / 0.43 / 0.17 |
| blocks 10-14 | **0.70 / 0.78 / 0.64 / 0.68** | 0.61 / 0.66 / 0.48 / 0.36 |
| blocks 15-19 | 0.59 / 0.66 / 0.44 / 0.48 | 0.51 / 0.62 / 0.52 / 0.48 |
| blocks 20-23 | 0.39 / 0.37 / 0.47 / 0.37 | 0.53 / 0.60 / 0.36 / **0.88** |
| blocks 24-27 | 0.26 / 0.24 / 0.18 / 0.17 | 0.23 / 0.24 / 0.19 / 0.14 |
| other layers | 0 (no such keys) | 0 (no such keys) |

**Leave-out (full LoRA minus that group)**

| Group | dungeon layout / edges / texture / color | niji layout / edges / texture / color |
|---|---|---|
| blocks 00-04 | 0.51 / 0.44 / 0.31 / 0.30 | 0.71 / 0.66 / 0.27 / 0.37 |
| blocks 05-09 | 0.73 / 0.70 / 0.45 / 0.38 | 0.30 / 0.32 / 0.11 / 0.07 |
| blocks 10-14 | 0.76 / 0.74 / 0.38 / 0.39 | 0.49 / 0.57 / 0.33 / 0.19 |
| blocks 15-19 | 0.77 / 0.78 / 0.66 / 0.64 | 0.57 / 0.62 / 0.34 / 0.37 |
| blocks 20-23 | 0.43 / 0.41 / 0.50 / 0.41 | 0.40 / 0.39 / 0.27 / 0.37 |
| blocks 24-27 | 0.31 / 0.26 / 0.37 / 0.18 | 0.17 / 0.16 / 0.10 / 0.07 |

Findings:
- Where a LoRA acts differs per LoRA: dungeon mostly in blocks 05-19, niji in 00-04 (and most of its color change in 20-23).
- Blocks 24-27 carry the least change for both LoRAs, as for the model merges.
- No group is "style" or "identity" for every LoRA, so Anima LoRA Block Weight ships without fixed style / identity presets.

**Predictive Multi-LoRA presets** (PRI = dungeon, S2 = niji; position between dungeon only (0) and niji only (1)):

| Variant | layout / edges / texture / color |
|---|---|
| Both stacked at 1.0 (reference) | 0.56 / 0.52 / 0.56 / 0.65 |
| Character + Style Fusion | 0.49 / 0.47 / 0.49 / 0.54 |
| Anti-Overcook | 0.56 / 0.52 / 0.54 / 0.65 |
| Orthogonal Synthesis | 0.56 / 0.52 / 0.58 / 0.67 |
| Balanced 50/50 Adaptive | 0.56 / 0.52 / 0.56 / 0.65 |

- "Character + Style Fusion" leans towards the PRI LoRA by about the same amount in every metric; it does not keep PRI's identity
  with S2's style. It is shown as "Favor Slot 1 (PRI)" in the HaDeS UI (stored value unchanged).
- The other three are within 0.03 of plain stacking. The LoRA Merger presets use the same formula family and were not measured separately.

**Can the weights predict where a LoRA acts? (2026-11-02)** Weight statistics per group of the same two LoRAs, compared with the
swap-in / leave-out means of the four metrics above. ΔW = up·down·alpha/rank; Spearman ρ over the 6 groups; top = strongest group.

| Predictor | dungeon ρ swap / leave, top | niji ρ swap / leave, top |
|---|---|---|
| Energy (Σ‖ΔW‖²) | 0.77 / 0.49, 05-09 | 0.09 / 0.43, 15-19 |
| ‖ΔW‖ / ‖W_base‖ (mean per layer) | 0.37 / 0.20, 00-04 | 0.43 / 0.60, 00-04 |
| Σ spectral norm | 0.54 / 0.31, 05-09 | 0.60 / 0.83, 00-04 |
| Measured | top 10-14 (swap) / 15-19 (leave) | top 00-04 (swap and leave) |

- No predictor works for both LoRAs, and none finds dungeon's strongest group. Niji's weight energy is spread evenly over blocks
  00-19 (05-09 has 98% of the peak) while 05-09 carries the least visual change; its color peak in 20-23 (0.88) sits in a low-energy group.
- The one thing every predictor gets right, blocks 24-27 being weakest, is already true for every LoRA and merge measured.
- No LoRA Inspector node: a per-block readout from the weights would show confident numbers that do not match the images.
  Rendering through `--mode lora` stays the way to find where a LoRA acts.

## Regional prompts

Base model, 1344x768, 8 seeds, two girls: red hair / dress on the left, blue on the right. Score = red pixels left minus right
plus blue pixels right minus left.

| Method | seeds with the right placement | separation |
|---|---|---|
| One natural-language prompt ("the girl on the left has red hair…") | 8/8 | **0.33** |
| `ConditioningSetArea` (left / right halves) + base prompt | 8/8 | 0.24 |
| `ConditioningSetMask` (left / right masks) + base prompt | 8/8 | 0.18 (one image grew a third character) |

- ComfyUI's stock area / mask conditioning works on Anima (it acts in the sampler, not in `attn2`).
- For this case Anima's natural-language prompt separates the two characters better than either, so no dedicated regional node was added.
  Harder scenes (named characters, interactions, more than two subjects) were not tested.

## Catalog tags and parenthesis escaping

Base model, 4 seeds, `best quality, safe, 1girl, solo, <tag>, upper body, simple background`.

- `hatsune miku, vocaloid`: recognized in every seed.
- `firefly \(honkai: star rail\), honkai: star rail`: recognized escaped and unescaped (the series tag helps; escaped vs. unescaped differ mostly in pose, color distance 2.4).
- `@hammer \(sunset beach\)` vs. `@hammer (sunset beach)`: very different images (color distance 17.3). Unescaped, the tokenizer reads
  `(sunset beach)` as a weighted concept and beach / swimsuit content leaks in; escaped, the artist's style is consistent.
  Prompt Studio now escapes literal parentheses in artist names too (it already did for characters), and the picker outputs escaped tags.

## CLIP Block Merger bands (`--mode clip`, 2026-11-02)

Model `anima-base-v1.0`, CLIP 1 = `qwen_3_06b_base`, CLIP 2 = `qwen_3_06b_abliterated` (same Qwen3-0.6B layout; weights differ by
2-12% per layer, most in layers 8-16, least in 24-27). Same 4 prompts x 2 seeds and metrics as the block groups. Bands on 28 layers:
early 0-9, middle 10-19, late 20-27.

`share` per metric (layout / edges / texture / color):

| Band | swap-in (CLIP 1 + band from 2) | leave-out (CLIP 2 + band from 1) | conditioning change, band alone |
|---|---|---|---|
| Token embeddings | 0.32 / 0.25 / 0.26 / 0.13 | 0.26 / 0.19 / 0.19 / 0.20 | 8% |
| Early layers | 0.46 / 0.39 / 0.51 / 0.27 | 0.57 / 0.52 / 0.62 / 0.37 | 22% |
| Middle layers | 0.49 / 0.43 / 0.53 / 0.30 | 0.67 / 0.61 / 0.51 / 0.53 | 17% |
| **Late layers** | **0.90 / 0.95 / 1.05 / 0.79** | **0.97 / 0.98 / 1.18 / 0.90** | **55%** |
| Final norm | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0.3% |

Conditioning change = relative L2 distance of the encoded prompt to CLIP 1's; CLIP 2 itself is at 53%.

- The late layers carry almost all of the difference, although their weights differ least: the output is read right after them,
  while changes in earlier layers are partly washed out on the way. All other bands together move the conditioning 29%.
- The bands are not additive at the image level (swap-in and leave-out of the small bands both land around 0.3-0.6): any change to
  the conditioning re-rolls composition and pose like a different seed.
- On the contact sheet every band changes layout, pose, texture and color together, with the same subject and drawing style.
  None isolates syntax, concepts or style / mood, so the UI now shows neutral depth names (Early / Middle / Late Layers).
- The old "A-Synt / B-Styl" button (0.15 early, 0.85 late) is close to CLIP 2; it is now labelled "Late B".
- The two final norms differ by at most 0.03, which does not change a single pixel; the band may matter more for other pairs.
- One encoder pair only.

## Corner Shield (`tools/corner_shield_measure.py`, 2026-11-02)

968 renders from the runs above (no real corner marks found in them; the negative prompt includes `watermark`). Clean corners:
3,872. Synthetic signatures: a cursive stroke (2-4 px, dark on light or light on dark) drawn into 240 corners that are clean
even at sensitivity 0.9. Corner error = mean abs difference to the clean original inside the 10% box.

| | sensitivity 0.5 | 0.75 (default) | 0.9 |
|---|---|---|---|
| Clean corners flagged, before | 10 | 11 | 14 |
| Signatures caught, before | 29% | 29% | 29% |
| Clean corners flagged, now | 14 (0.4%) | 33 (0.9%) | 67 (1.7%) |
| Signatures caught, now | 84% | 87% | 90% |

| Corner error at 0.75 | before | now |
|---|---|---|
| Untouched | 0.038 | 0.037 |
| Smart Edge-Inward Blend | 0.037 | **0.023** |
| Soft Corner Blur | 0.038 | 0.038 |

- Before, the anti-texture guard compared the corner's peak with the image center's peak. In any detailed image that peak is
  ~0.55, so a thin signature was nearly always classed as texture and the sensitivity threshold was never reached.
- Detection now compares the corner with the ring around it (99th vs 99.5th percentile of high-frequency energy). Of the
  statistics tried (corner vs center, corner vs ring ratio or difference, ring 99 / 99.5 / 99.9 / max), the difference to
  the ring's 99.5th percentile caught the most signatures at 1% false positives (88%; center-based guard: 59%).
- The heal mask is dilated and feathered outwards only; the old blur-only feather left a half-healed ghost of thin strokes.
- Remaining misses are signatures over busy backgrounds; false positives are hard edges at the corner (window frames
  against a bright sky, pixel art). Smart heal removes marks cleanly on flat backgrounds and leaves a soft patch on busy
  ones. Soft Corner Blur only softens marks.
- Real Anima scribbles were not in the sample; the recall figures are for the synthetic strokes.
