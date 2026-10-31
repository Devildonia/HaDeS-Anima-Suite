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
