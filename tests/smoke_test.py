"""Smoke test against a real ComfyUI server (the unit suite in test_hades_suite.py runs on stubs).

Starts ComfyUI with only this pack loaded, checks that every node and web file registers, then runs graphs through the
real execution engine.

  python tests/smoke_test.py --comfy-dir /path/to/ComfyUI            # no model files needed (CI)
  python tests/smoke_test.py --comfy-dir /path/to/ComfyUI --models   # also merge / LoRA / sample / extract / save
                                                                     # with real Anima models (run before a release)

Model names default to the public Anima releases and are overridable (--unet, --unet2, --clip, --vae, --lora).
Files the graphs write go to a temporary output folder, or are deleted afterwards (the extracted LoRA).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_NAME = os.path.basename(PACK_DIR)
WEB_FILES = sorted(f for f in os.listdir(os.path.join(PACK_DIR, "web")) if f.endswith(".js"))


class Server:
    def __init__(self, comfy_dir, python, port, extra_args):
        self.url = f"http://127.0.0.1:{port}"
        self.work = tempfile.mkdtemp(prefix="hades_smoke_")
        self.output_dir = os.path.join(self.work, "output")
        self.log_path = os.path.join(self.work, "server.log")
        os.makedirs(os.path.join(self.work, "user"))
        cmd = [python, "main.py", "--port", str(port), "--listen", "127.0.0.1", "--disable-auto-launch",
               "--user-directory", os.path.join(self.work, "user"), "--output-directory", self.output_dir,
               "--temp-directory", os.path.join(self.work, "temp"),
               "--disable-all-custom-nodes", "--whitelist-custom-nodes", PACK_NAME, *extra_args]
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=comfy_dir, stdout=self.log, stderr=subprocess.STDOUT)

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return json.load(r)

    def wait_ready(self, timeout=600):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with code {self.proc.returncode}\n{self.tail()}")
            try:
                return self.get("/object_info")
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(2)
        raise TimeoutError(f"ComfyUI did not start within {timeout}s\n{self.tail()}")

    def run(self, prompt, timeout=1800):
        body = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(self.url + "/prompt", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                prompt_id = json.load(r)["prompt_id"]
        except urllib.error.HTTPError as e:
            raise AssertionError(f"prompt rejected: {e.read().decode(errors='replace')}") from None
        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = self.get(f"/history/{prompt_id}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
                if errors:
                    e = errors[0]
                    raise AssertionError(f"{e['node_type']} failed: {e['exception_message']}\n{''.join(e['traceback'])}")
                return entry["outputs"]
            time.sleep(1)
        raise TimeoutError(f"prompt did not finish within {timeout}s")

    def tail(self, n=40):
        self.log.flush()
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        shutil.rmtree(self.work, ignore_errors=True)


class Graph:
    """API-format prompt builder; widget inputs not given take the node's default (or first combo option)."""

    def __init__(self, object_info):
        self.info = object_info
        self.prompt = {}

    def add(self, class_type, **inputs):
        spec = self.info[class_type]["input"]
        for name, (kind, *opts) in spec.get("required", {}).items():
            if name in inputs:
                continue
            cfg = opts[0] if opts else {}
            if isinstance(kind, list):
                inputs[name] = cfg.get("default", kind[0])
            elif "default" in cfg:
                inputs[name] = cfg["default"]
        node_id = str(len(self.prompt) + 1)
        self.prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    @staticmethod
    def out(node_id, index=0):
        return [node_id, index]


def check_registration(object_info, server):
    hades = sorted(k for k in object_info if k.startswith("HaDeSAnima"))
    assert len(hades) == 24, f"expected 24 HaDeS nodes, got {len(hades)}: {hades}"
    assert "IMPORT FAILED" not in server.tail(400), server.tail()
    served = server.get("/extensions")
    missing = [f for f in WEB_FILES if not any(e.endswith(f"/{PACK_NAME}/{f}") for e in served)]
    assert not missing, f"web files not served: {missing}"
    print(f"[OK] registration: {len(hades)} nodes, {len(WEB_FILES)} web files")


def check_basic(object_info, server):
    g = Graph(object_info)
    picker = g.add("HaDeSAnimaArtistCharacterPicker", artists="ebifurya:1.2", character="hatsune_miku", series="vocaloid",
                   random_artists=1, random_min_posts=1000, seed=7)
    shield = g.add("HaDeSAnimaCornerScribbleShield", image=g.out(g.add("EmptyImage", width=256, height=256)))
    guard = g.add("HaDeSAnimaLatentPipelineGuard")
    g.add("HaDeSAnimaSuiteTelemetry", info_1=g.out(picker, 2), info_2=g.out(shield, 2), info_3=g.out(guard, 3))
    artist_view = g.add("PreviewAny", source=g.out(picker, 0))
    outputs = server.run(g.prompt)

    assert len(outputs[picker]["drawn"]) == 1, outputs[picker]
    artist = outputs[artist_view]["text"][0]
    assert artist.startswith("(@ebifurya:1.2), @"), artist
    report = next(o["report"][0] for o in outputs.values() if "report" in o)
    assert "Artist & Character Picker" in report and "Corner Scribble Shield" in report, report[:400]
    print(f"[OK] basic graph: picker -> {artist!r}, shield, latent guard, telemetry")


def check_models(object_info, server, args):
    g = Graph(object_info)
    m1 = g.add("HaDeSAnimaModelLoader", unet_name=args.unet)
    m2 = g.add("HaDeSAnimaModelLoader", unet_name=args.unet2)
    clip = g.add("HaDeSAnimaCLIPLoader", clip_name=args.clip)
    vae = g.add("HaDeSAnimaVAELoader", vae_name=args.vae)
    merged = g.add("HaDeSAnimaModelBlockMerger", model1=g.out(m1), model2=g.out(m2), BLOCKS_Art_Style_Medium=0.35)
    # A LoRA on top: the extractor and saver then read lazily patched weights (dynamic VRAM)
    tuned = g.add("HaDeSAnimaUniversalLoraLoader", model=g.out(merged), lora_name=args.lora, strength_model=0.8)
    studio = g.add("HaDeSAnimaPromptStudio", clip=g.out(clip))
    latent = g.add("HaDeSAnimaLatentPipelineGuard", resolution_preset="Custom (Auto-Snap to 32px Grid)",
                   custom_width=512, custom_height=512, model=g.out(tuned), vae=g.out(vae))
    guided = g.add("HaDeSAnimaAttentionGuidance", model=g.out(tuned))
    sampler = g.add("HaDeSAnimaKSampler", model=g.out(guided), positive=g.out(studio, 0), negative=g.out(studio, 1),
                    latent_image=g.out(latent), steps=4, seed=1, flow_shift=4.0)
    image = g.add("HaDeSAnimaVAEDecode", samples=g.out(sampler), vae=g.out(vae))
    g.add("PreviewImage", images=g.out(image))
    meta = g.add("HaDeSAnimaSamplerMetadata", info=g.out(sampler, 2))
    meta_view = g.add("PreviewAny", source=g.out(meta, 1))
    extractor = g.add("HaDeSAnimaLoraExtractor", model_tuned=g.out(tuned), model_base=g.out(m1), rank=4,
                      export_filename="hades_smoke_extract", overwrite_existing=True)
    extract_view = g.add("PreviewAny", source=g.out(extractor, 0))
    saver = g.add("HaDeSAnimaModelSaver", model=g.out(merged), filename_prefix="hades_smoke/merged",
                  precision="convrot_int8 (Hadamard Rotated INT8)")
    g.add("HaDeSAnimaSuiteTelemetry", info_1=g.out(merged, 1), info_2=g.out(sampler, 2), info_3=g.out(extractor, 1),
          info_4=g.out(saver, 2))

    t0 = time.time()
    outputs = server.run(g.prompt)
    lora_path = outputs[extract_view]["text"][0]
    try:
        assert outputs[meta_view]["text"][0] == "4", outputs[meta_view]
        assert any(o.get("images") for o in outputs.values()), "no preview image"
        saved = outputs[saver]["saved_path"][0]
        assert os.path.getsize(saved) > 1 << 30, saved
        assert os.path.getsize(lora_path) > 0, lora_path
    finally:
        if os.path.basename(lora_path) == "hades_smoke_extract.safetensors" and os.path.exists(lora_path):
            os.remove(lora_path)
    print(f"[OK] model graph ({time.time() - t0:.0f}s): merge + LoRA + NAG -> sample -> decode, extract, ConvRot INT8 save")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-dir", required=True, help="ComfyUI root (main.py); this pack must be in its custom_nodes")
    ap.add_argument("--python", default=sys.executable, help="interpreter of the ComfyUI environment")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--models", action="store_true", help="also run the graph that needs real model files")
    ap.add_argument("--unet", default="anima-base-v1.0.safetensors")
    ap.add_argument("--unet2", default="anima-aesthetic-v1.1.safetensors")
    ap.add_argument("--clip", default="qwen_3_06b_base.safetensors")
    ap.add_argument("--vae", default="qwen_image_vae.safetensors")
    ap.add_argument("--lora", default=None, help="an Anima LoRA in models/loras (required with --models)")
    args, extra = ap.parse_known_args()
    if args.models and not args.lora:
        ap.error("--models needs --lora")
    if not os.path.isdir(os.path.join(args.comfy_dir, "custom_nodes", PACK_NAME)):
        ap.error(f"{PACK_NAME} is not installed in {args.comfy_dir}/custom_nodes")

    server = Server(os.path.abspath(args.comfy_dir), args.python, args.port, extra)
    try:
        object_info = server.wait_ready()
        check_registration(object_info, server)
        check_basic(object_info, server)
        if args.models:
            check_models(object_info, server, args)
    except Exception:
        print(f"--- ComfyUI log ---\n{server.tail()}", file=sys.stderr)
        raise
    finally:
        server.stop()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
