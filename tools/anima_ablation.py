"""Empirical checks for the HaDeS Anima block groups, merge presets and schedulers (dev tool, not a node).

Queues fixed prompt x seed grids on a running local ComfyUI through its HTTP API, downloads the
results and measures what changed with image statistics (numpy + Pillow only):

  layout   - RMSE of a 32x32 lightness thumbnail (composition, silhouettes, light masses)
  edges    - 1 - correlation of Sobel edge maps at 128 px (line placement / structure)
  texture  - distance of the high-frequency log power spectrum (line weight, grain, detail)
  color    - distance of Lab mean / std (palette and contrast)

Modes:
  groups      A with one group taken from B (swap-in) and B with one group taken from A (leave-out).
              share = d(variant, reference) / d(B, A): how much of the A->B change that group carries.
  presets     Predictive Merger presets and a uniform 0.5 merge, placed between A (0) and B (1) per metric.
  schedulers  normal / beta / anima_beta57 at low step counts vs. an 80-step euler reference.

Example:
  python tools/anima_ablation.py --mode groups --model-a anima-base-v1.0.safetensors \
      --model-b anima_pixelart_from_hades_v1.0.safetensors --out H:/ablation/pixelart

Re-running with the same --out reuses downloaded images, so an interrupted run resumes and the groups and
presets modes share their A / B renders.
"""
import argparse
import io
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

GROUPS = {
    "BLOCKS_Layout_Geometry": "B00-04",
    "BLOCKS_Global_Composition": "B05-09",
    "BLOCKS_Subject_Identity": "B10-14",
    "BLOCKS_Material_Substance": "B15-19",
    "BLOCKS_Art_Style_Medium": "B20-23",
    "BLOCKS_Lighting_Sharpness": "B24-27",
    "ADAPTER_Text_Alignment": "B28-33",
}
PRESETS = [
    "Style Transfer (Keep A identity, B style)",
    "Interference Guard (Anti-Burn / Artifact Shield)",
    "Complementary Richness (Fill gaps of A with B)",
    "Balanced Geometric Fusion",
]
PROMPTS = [
    "masterpiece, best quality, safe, 1girl, solo, long silver hair, red eyes, school uniform, standing in a classroom, window light, looking at viewer",
    "best quality, safe, 1boy, knight, ornate armor, holding sword, castle courtyard, dramatic sky, full body",
    "best quality, safe, scenery, no humans, japanese street at night, neon signs, rain, reflections, wide shot",
    "best quality, safe, 2girls, sitting at a cafe table, laughing, warm afternoon light, from side, upper body",
]
NEGATIVE = "nsfw, worst quality, low quality, blurry, jpeg artifacts, watermark"
METRICS = ("layout", "edges", "texture", "color")


# ------------------------------------------------------------------------------------------------
# ComfyUI API
# ------------------------------------------------------------------------------------------------
def _request(server, path, data=None):
    req = urllib.request.Request(f"{server}{path}", data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def run_graph(server, graph):
    prompt_id = json.loads(_request(server, "/prompt", json.dumps({"prompt": graph}).encode()))["prompt_id"]
    while True:
        hist = json.loads(_request(server, f"/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"ComfyUI failed on prompt {prompt_id}: {entry['status'].get('messages')}")
            img = entry["outputs"]["out"]["images"][0]
            query = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
            return Image.open(io.BytesIO(_request(server, f"/view?{query}"))).convert("RGB")
        time.sleep(0.5)


def build_graph(args, prompt, seed, model_node, steps=None, sampler=None, scheduler=None):
    """API graph: HaDeS loaders -> model_node (dict of extra nodes ending in id "model") -> KSampler -> preview."""
    g = {
        "a": {"class_type": "HaDeSAnimaModelLoader", "inputs": {"unet_name": args.model_a, "weight_dtype": "default", "flow_shift": 3.0, "auto_remap_diffusers": True, "external_llm_adapter": "None"}},
        "clip": {"class_type": "HaDeSAnimaCLIPLoader", "inputs": {"clip_name": args.clip, "device_mode": "default (GPU Dynamic)", "precision": "auto", "strict_qwen3_guard": True}},
        "vae": {"class_type": "HaDeSAnimaVAELoader", "inputs": {"vae_name": args.vae, "vae_precision": "bfloat16 (Recommended Wan21)"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["clip", 0]}},
        "latent": {"class_type": "HaDeSAnimaLatentPipelineGuard", "inputs": {"resolution_preset": "Custom (Auto-Snap to 32px Grid)", "custom_width": args.size, "custom_height": args.size, "batch_size": 1, "prevent_29b_white_bg_bias": True}},
        "ks": {"class_type": "HaDeSAnimaKSampler", "inputs": {
            "model": ["model", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "variant_profile": "Manual", "seed": seed, "steps": steps or args.steps, "cfg": args.cfg,
            "sampler_name": sampler or args.sampler, "scheduler": scheduler or args.scheduler, "denoise": 1.0, "flow_shift": 0.0, "vae": ["vae", 0],
        }},
        "out": {"class_type": "PreviewImage", "inputs": {"images": ["ks", 1]}},
    }
    if args.model_b:
        g["b"] = {**g["a"], "inputs": {**g["a"]["inputs"], "unet_name": args.model_b}}
    g.update(model_node)
    return g


def block_merge(ratios, base_ratio):
    return {"model": {"class_type": "HaDeSAnimaModelBlockMerger", "inputs": {"model1": ["a", 0], "model2": ["b", 0], "base_ratio": base_ratio, "recipe_override": "", **ratios}}}


def configs_for(args):
    """name -> (model_node, sampling overrides)"""
    if args.mode == "groups":
        cfg = {
            "A": (block_merge({g: 0.0 for g in GROUPS}, 0.0), {}),
            "B": (block_merge({g: 1.0 for g in GROUPS}, 1.0), {}),
        }
        for g in GROUPS:
            cfg[f"in_{g}"] = (block_merge({k: float(k == g) for k in GROUPS}, 0.0), {})
            cfg[f"out_{g}"] = (block_merge({k: float(k != g) for k in GROUPS}, 1.0), {})
        return cfg
    if args.mode == "presets":
        cfg = {
            "A": (block_merge({g: 0.0 for g in GROUPS}, 0.0), {}),
            "B": (block_merge({g: 1.0 for g in GROUPS}, 1.0), {}),
            "uniform": (block_merge({g: (0.0 if g.startswith("ADAPTER") else args.influence) for g in GROUPS}, args.influence), {}),
        }
        for i, p in enumerate(PRESETS):
            cfg[f"preset{i}"] = ({"model": {"class_type": "HaDeSAnimaPredictiveModelMerger", "inputs": {
                "model1": ["a", 0], "model2": ["b", 0], "preset": p, "target_influence": args.influence, "aggressiveness": 1.0, "protect_llm_adapter": True,
            }}}, {})
        return cfg
    model_a = {"model": {"class_type": "HaDeSAnimaModelLoader", "inputs": {"unet_name": args.model_a, "weight_dtype": "default", "flow_shift": 3.0, "auto_remap_diffusers": True, "external_llm_adapter": "None"}}}
    cfg = {"ref": (model_a, {"steps": 80, "sampler": "euler", "scheduler": "normal"})}
    for sch in ("normal", "beta", "anima_beta57"):
        for st in (8, 14, 24):
            cfg[f"{sch}_{st}"] = (model_a, {"steps": st, "sampler": "euler", "scheduler": sch})
    return cfg


# ------------------------------------------------------------------------------------------------
# Image statistics
# ------------------------------------------------------------------------------------------------
def srgb_to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def features(img):
    lab = srgb_to_lab(np.asarray(img.resize((512, 512), Image.LANCZOS), dtype=np.float64) / 255.0)
    L = lab[..., 0]

    thumb = np.asarray(Image.fromarray(L.astype(np.float32)).resize((32, 32), Image.BOX))

    small = np.asarray(Image.fromarray(L.astype(np.float32)).resize((128, 128), Image.BOX))
    gx = small[1:-1, 2:] - small[1:-1, :-2] + 0.5 * (small[:-2, 2:] - small[:-2, :-2] + small[2:, 2:] - small[2:, :-2])
    gy = small[2:, 1:-1] - small[:-2, 1:-1] + 0.5 * (small[2:, :-2] - small[:-2, :-2] + small[2:, 2:] - small[:-2, 2:])
    edges = np.hypot(gx, gy).ravel()

    # Radially averaged log power spectrum, upper half of the frequency range only (line / grain / detail)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(L - L.mean()))) ** 2
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - 256, xx - 256).astype(int)
    radial = np.bincount(r.ravel(), spec.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    bands = np.array_split(np.arange(64, 256), 16)
    texture = np.log10(np.array([radial[b].mean() for b in bands]) + 1e-9)

    color = np.concatenate([lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)])
    return {"layout": thumb, "edges": edges, "texture": texture, "color": color}


def distance(fa, fb):
    e = np.corrcoef(fa["edges"], fb["edges"])[0, 1]
    return {
        "layout": float(np.sqrt(np.mean((fa["layout"] - fb["layout"]) ** 2))),
        "edges": float(1.0 - e),
        "texture": float(np.linalg.norm(fa["texture"] - fb["texture"])),
        "color": float(np.linalg.norm(fa["color"] - fb["color"])),
    }


# ------------------------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------------------------
def _sum(dists):
    return {m: sum(d[m] for d in dists) for m in METRICS}


def analyze(args, feats, samples):
    report = {"mode": args.mode, "model_a": args.model_a, "model_b": args.model_b, "samples": len(samples), "steps": args.steps}
    if args.mode == "schedulers":
        rows = {}
        for name in feats:
            if name != "ref":
                rows[name] = {m: v / len(samples) for m, v in _sum([distance(feats[name][s], feats["ref"][s]) for s in samples]).items()}
        report["distance_to_80_step_reference"] = rows
        return report

    ab = _sum([distance(feats["A"][s], feats["B"][s]) for s in samples])
    report["A_to_B_mean_distance"] = {m: v / len(samples) for m, v in ab.items()}
    if args.mode == "groups":
        for kind, prefix, ref in (("swap_in", "in", "A"), ("leave_out", "out", "B")):
            report[kind] = {
                GROUPS[g]: {m: round(v / ab[m], 3) for m, v in _sum([distance(feats[f"{prefix}_{g}"][s], feats[ref][s]) for s in samples]).items()}
                for g in GROUPS
            }
        return report

    # presets: position between A (0) and B (1) per metric
    rows = {}
    for name in feats:
        if name in ("A", "B"):
            continue
        da = _sum([distance(feats[name][s], feats["A"][s]) for s in samples])
        db = _sum([distance(feats[name][s], feats["B"][s]) for s in samples])
        label = "uniform" if name == "uniform" else PRESETS[int(name[6:])]
        rows[label] = {m: round(da[m] / (da[m] + db[m]), 3) for m in METRICS}
    report["position_A0_B1"] = rows
    return report


def contact_sheet(out_dir, mode, names, samples):
    tile = 192
    sheet = Image.new("RGB", (tile * len(names), (tile + 18) * len(samples)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for x, name in enumerate(names):
        for y, s in enumerate(samples):
            img = Image.open(os.path.join(out_dir, name, f"{s}.png")).resize((tile, tile), Image.LANCZOS)
            sheet.paste(img, (x * tile, y * (tile + 18) + 18))
            draw.text((x * tile + 4, y * (tile + 18) + 3), name[:30], fill=(220, 220, 220))
    sheet.save(os.path.join(out_dir, f"sheet_{mode}.jpg"), quality=88)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("groups", "presets", "schedulers"), default="groups")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--model-a", required=True, help="diffusion_models file name of Model 1 / A")
    p.add_argument("--model-b", help="diffusion_models file name of Model 2 / B (groups and presets modes)")
    p.add_argument("--clip", default="qwen_3_06b_base.safetensors")
    p.add_argument("--vae", default="qwen_image_vae.safetensors")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 98765])
    p.add_argument("--prompts", type=int, default=len(PROMPTS), help="use the first N built-in prompts")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.5)
    p.add_argument("--sampler", default="er_sde")
    p.add_argument("--scheduler", default="normal")
    p.add_argument("--influence", type=float, default=0.5, help="target_influence for the presets mode")
    args = p.parse_args()
    if args.mode != "schedulers" and not args.model_b:
        p.error("--model-b is required for groups and presets modes")

    os.makedirs(args.out, exist_ok=True)
    configs = configs_for(args)
    samples = [f"p{pi}_s{seed}" for pi in range(args.prompts) for seed in args.seeds]
    feats = {}
    total, done = len(configs) * len(samples), 0
    t0 = time.time()
    for name, (model_node, overrides) in configs.items():
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        feats[name] = {}
        for pi in range(args.prompts):
            for seed in args.seeds:
                s = f"p{pi}_s{seed}"
                path = os.path.join(args.out, name, f"{s}.png")
                if not os.path.exists(path):
                    run_graph(args.server, build_graph(args, PROMPTS[pi], seed, model_node, **overrides)).save(path)
                feats[name][s] = features(Image.open(path).convert("RGB"))
                done += 1
                print(f"[{done}/{total}] {name} {s} ({time.time() - t0:.0f}s)", flush=True)

    report = analyze(args, feats, samples)
    with open(os.path.join(args.out, f"report_{args.mode}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    contact_sheet(args.out, args.mode, list(configs), samples)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
