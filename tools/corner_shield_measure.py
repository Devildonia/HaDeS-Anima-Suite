"""Measures the Anima Corner Shield on real renders (dev tool, not a node).

1. Clean corners: how many corners of real Anima renders get flagged at each sensitivity (false positives).
2. Synthetic signatures: a cursive stroke is drawn into corners that are clean even at sensitivity 0.9, then
   recall per sensitivity and the corner error after healing (mean abs difference to the clean original).

Runs in-process with the ComfyUI venv, no server needed. Use renders without real corner marks, e.g. the
images written by tools/anima_ablation.py.

Example:
  python tools/corner_shield_measure.py --images "H:/ComfyUI/output/hades_ablation/**/*.png" --out H:/ablation/corner_shield
"""
import argparse
import glob
import importlib
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(PKG)))
_spec = importlib.util.spec_from_file_location("hades", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
sys.modules["hades"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sys.modules["hades"])
NODE = importlib.import_module("hades.anima_pipeline_nodes").HaDeSAnimaCornerScribbleShield()

SENS = (0.5, 0.75, 0.9)
CORNERS = ("TL", "TR", "BL", "BR")
MODES = {"smart": "Smart Edge-Inward Blend (Auto-Heal)", "blur": "Soft Corner Blur"}


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


def run(t, sens, mode="Mask Only (No Pixel Modification)"):
    img, _, info = NODE.shield_corners(t, 0.10, sens, "All 4 Corners", mode, 12)
    detected = info.split("Detected: ")[1].split("\n")[0]
    return img, {c for c in CORNERS if c + " (" in detected}


def corner_box(name, h, w, b):
    return {"TL": (0, b, 0, b), "TR": (0, b, w - b, w), "BL": (h - b, h, 0, b), "BR": (h - b, h, w - b, w)}[name]


def draw_signature(img, y0, x0, b, rng, color, width):
    sw, sh = rng.uniform(0.35, 0.7) * b, rng.uniform(0.15, 0.3) * b
    px, py = x0 + rng.uniform(0.08 * b, 0.92 * b - sw), y0 + rng.uniform(0.08 * b, 0.92 * b - sh)
    n, freq, phase = rng.randint(30, 60), rng.uniform(3, 7), rng.uniform(0, 6.28)
    pts = [(px + u * sw + 0.08 * sw * np.sin(u * freq * 2.1), py + sh * (0.5 + 0.45 * np.sin(u * freq * 6.28 + phase)))
           for u in np.linspace(0, 1, n)]
    draw = ImageDraw.Draw(img)
    draw.line(pts, fill=color, width=width, joint="curve")
    if rng.random() < 0.6:
        draw.line([(px, py + sh * 1.1), (px + sw, py + sh * 1.05)], fill=color, width=max(1, width - 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="glob of renders (recursive ** allowed)")
    p.add_argument("--out", required=True)
    p.add_argument("--synthetic", type=int, default=240)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.images, recursive=True))

    flagged = {s: 0 for s in SENS}
    clean_pool = []
    for f in files:
        t = to_tensor(Image.open(f).convert("RGB"))
        hits = {s: run(t, s)[1] for s in SENS}
        for s in SENS:
            flagged[s] += len(hits[s])
        clean_pool += [(f, c) for c in CORNERS if c not in hits[0.9]]
        print(f"[clean] {f}", flush=True)

    rng = random.Random(args.seed)
    picks = rng.sample(clean_pool, min(args.synthetic, len(clean_pool)))
    recall = {s: 0 for s in SENS}
    err = {"untouched": [], "smart": [], "blur": []}
    for j, (f, name) in enumerate(picks):
        clean = Image.open(f).convert("RGB")
        w, h = clean.size
        b = int(round(min(h, w) * 0.10))
        y0, y1, x0, x1 = corner_box(name, h, w, b)
        lum = np.asarray(clean.crop((x0, y0, x1, y1)).convert("L")).mean()
        marked = clean.copy()
        draw_signature(marked, y0, x0, b, random.Random(j), (20, 20, 25) if lum > 110 else (235, 235, 230), rng.choice((2, 3, 4)))
        tc, tm = to_tensor(clean), to_tensor(marked)
        for s in SENS:
            recall[s] += name in run(tm, s)[1]
        healed = {"untouched": tm, **{k: run(tm, 0.75, mode)[0] for k, mode in MODES.items()}}
        for k, im in healed.items():
            err[k].append(float((im[0, y0:y1, x0:x1] - tc[0, y0:y1, x0:x1]).abs().mean()))
        print(f"[synthetic {j + 1}/{len(picks)}]", flush=True)

    report = {
        "images": len(files),
        "clean_corners_flagged": {str(s): f"{flagged[s]}/{4 * len(files)}" for s in SENS},
        "synthetic_signatures": len(picks),
        "recall": {str(s): round(recall[s] / len(picks), 3) for s in SENS},
        "corner_mae_vs_clean_at_0.75": {k: round(float(np.mean(v)), 4) for k, v in err.items()},
    }
    with open(os.path.join(args.out, "report_corner_shield.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
