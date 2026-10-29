import os
import sys
import time
import types
import tempfile
import importlib.util
import torch
import safetensors.torch

# Allow running both inside ComfyUI and in a standalone PyTorch environment without ComfyUI installed
COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if (
    not os.environ.get("HADES_FORCE_STANDALONE_STUBS")
    and os.path.isdir(os.path.join(COMFY_ROOT, "comfy"))
    and COMFY_ROOT not in sys.path
):
    sys.path.insert(0, COMFY_ROOT)


def _install_standalone_comfy_stubs():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = tempfile.gettempdir()
    folder_paths.get_filename_list = lambda _: ["None"]
    folder_paths.get_folder_paths = lambda _: [tempfile.gettempdir()]
    folder_paths.get_full_path_or_raise = lambda _, name: name
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    comfy.sd = types.ModuleType("comfy.sd")
    comfy.sd.load_lora_for_models = lambda m, c, sd, sm, sc: (m, c)
    comfy.sd.CLIPType = types.SimpleNamespace(QWEN_IMAGE=18, STABLE_DIFFUSION=1)
    comfy.sd.load_diffusion_model_state_dict = lambda sd, model_options=None, metadata=None: None
    comfy.sd.load_diffusion_model = lambda path, model_options=None: None
    comfy.sd.load_text_encoder_state_dicts = lambda sds, **kw: None
    comfy.sd.load_clip_model_patcher = lambda *a, **kw: None

    comfy.utils = types.ModuleType("comfy.utils")
    comfy.utils.PROGRESS_BAR_ENABLED = False
    comfy.utils.load_torch_file = lambda p, safe_load=True, return_metadata=False: (
        (safetensors.torch.load_file(p), {}) if return_metadata else safetensors.torch.load_file(p)
    )
    comfy.utils.common_upscale = lambda samples, w, h, method, crop: torch.nn.functional.interpolate(
        samples.float(), size=(h, w), mode="bilinear", align_corners=False
    ).to(samples.dtype)

    comfy.lora = types.ModuleType("comfy.lora")
    comfy.lora.calculate_weight = lambda patches, w, k: w
    comfy.lora.model_lora_keys_unet = lambda model, key_map=None: key_map or {}
    comfy.lora.model_lora_keys_clip = lambda model, key_map=None: key_map or {}
    comfy.lora.load_lora = lambda sd, key_map: sd

    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    comfy.model_management.intermediate_device = lambda: torch.device("cpu")
    comfy.model_management.get_free_memory = lambda dev=None: 8 * (1024 ** 3)

    comfy.sample = types.ModuleType("comfy.sample")
    comfy.sample.fix_empty_latent_channels = lambda model, lat, *a: lat
    comfy.sample.prepare_noise = lambda lat, seed, batch_inds=None: torch.zeros_like(lat)
    comfy.sample.sample = lambda model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kw: latent_image

    comfy.samplers = types.ModuleType("comfy.samplers")
    comfy.samplers.SAMPLER_NAMES = ["er_sde", "euler", "euler_ancestral", "dpmpp_2m_sde_gpu", "dpmpp_2m", "uni_pc"]
    comfy.samplers.SCHEDULER_NAMES = ["normal", "beta", "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform"]
    comfy.samplers.DISCARD_PENULTIMATE_SIGMA_SAMPLERS = {"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"}

    comfy.model_sampling = types.ModuleType("comfy.model_sampling")

    class _StubCONST:
        pass

    class _StubModelSamplingDiscreteFlow:
        def __init__(self, cfg=None):
            self.shift = 3.0
            self.multiplier = 1.0

        def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1.0):
            self.shift = shift
            self.multiplier = multiplier

        def percent_to_sigma(self, pct):
            if pct <= 0.0:
                return 1.0
            if pct >= 1.0:
                return 0.0
            t = 1.0 - float(pct)
            return (self.shift * t) / (1.0 + (self.shift - 1.0) * t)

    comfy.model_sampling.CONST = _StubCONST
    comfy.model_sampling.ModelSamplingDiscreteFlow = _StubModelSamplingDiscreteFlow

    sys.modules["comfy"] = comfy
    sys.modules["comfy.sd"] = comfy.sd
    sys.modules["comfy.utils"] = comfy.utils
    sys.modules["comfy.lora"] = comfy.lora
    sys.modules["comfy.model_management"] = comfy.model_management
    sys.modules["comfy.sample"] = comfy.sample
    sys.modules["comfy.samplers"] = comfy.samplers
    sys.modules["comfy.model_sampling"] = comfy.model_sampling


if os.environ.get("HADES_FORCE_STANDALONE_STUBS"):
    _install_standalone_comfy_stubs()
    import folder_paths
    import comfy.lora
    import comfy.sd
    import comfy.utils
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.model_sampling
else:
    try:
        import folder_paths
        import comfy.lora
        import comfy.sd
        import comfy.utils
        import comfy.model_management
        import comfy.sample
        import comfy.samplers
        import comfy.model_sampling
    except ImportError:
        _install_standalone_comfy_stubs()
        import folder_paths
        import comfy.lora
        import comfy.sd
        import comfy.utils
        import comfy.model_management
        import comfy.sample
        import comfy.samplers
        import comfy.model_sampling

PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INIT_PATH = os.path.join(PKG_DIR, "__init__.py")
spec = importlib.util.spec_from_file_location(
    "hades_anima_suite", INIT_PATH, submodule_search_locations=[PKG_DIR]
)
hades = importlib.util.module_from_spec(spec)
sys.modules["hades_anima_suite"] = hades
spec.loader.exec_module(hades)


class DummyInnerModel:
    def __init__(self, sd):
        self._sd = sd

    def state_dict(self):
        return self._sd


class DummyModelSampling:
    def __init__(self, shift=3.0, multiplier=1.0):
        self.shift = shift
        self.multiplier = multiplier
        self.sigmas = torch.linspace(10.0, 0.0, 100)
        self._buffers = {"sigmas": self.sigmas}

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1.0):
        self.shift = shift
        self.multiplier = multiplier

    def percent_to_sigma(self, pct):
        u = 1.0 - float(pct)
        return float(self.shift * u / (1.0 + (self.shift - 1.0) * u))


class DummyModelPatcher:
    def __init__(self, sd, image_model="anima"):
        inner = DummyInnerModel({k: v.clone() for k, v in sd.items()})
        inner.model_config = types.SimpleNamespace(unet_config={"image_model": image_model})
        self.model = inner
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.model_keys = set(sd.keys())
        self.model_options = {"transformer_options": {}}

    def clone(self):
        c = DummyModelPatcher(self.model.state_dict())
        # Preserve the Anima topology gate (real ComfyUI ModelPatcher clones keep model_config)
        if hasattr(self.model, "model_config"):
            c.model.model_config = self.model.model_config
        c.patches = {k: list(v) for k, v in self.patches.items()}
        c.backup = dict(self.backup)
        c.object_patches = dict(self.object_patches)
        c.model_options = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.model_options.items()}
        return c

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name == "model_sampling":
            return DummyModelSampling()
        return None

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        for k, v in patches.items():
            self.patches.setdefault(k, []).append((strength_patch, v, strength_model, None, None))

    def calculate_weight(self, patches, weight, key):
        out = weight.clone()
        for strength, v, _, _, _ in patches:
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "diff":
                out += strength * v[1][0].to(out.device, out.dtype)
            elif isinstance(v, tuple) and len(v) == 2 and v[0] == "set":
                out.copy_(v[1][0].to(out.device, out.dtype))
        return out


class DummyCLIP:
    def __init__(self, sd):
        self.patcher = DummyModelPatcher(sd)
        self.cond_stage_model = self.patcher.model
        self.last_tokens = None

    def clone(self):
        c = DummyCLIP(self.patcher.model.state_dict())
        c.patcher = self.patcher.clone()
        return c

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.patcher.add_patches(patches, strength_patch, strength_model)

    def tokenize(self, text):
        return {
            "qwen3_06b": [[(101, 1.0, 0), (102, 1.0, 1)]],
            "t5xxl": [[(201, 1.0, 0), (202, 1.5, 1)]],
        }

    def encode_from_tokens_scheduled(self, tokens):
        self.last_tokens = tokens
        return [[torch.zeros(1, 4, 16), {"pooled_output": torch.zeros(1, 16), "tokens": tokens}]]


class DummyVAE:
    def __init__(self, latent_channels=16, noisy_corner=False):
        self.latent_channels = latent_channels
        self.noisy_corner = noisy_corner
        self.last_decode_method = None

    def spacial_compression_decode(self):
        return 8

    def temporal_compression_decode(self):
        return 4

    def _make_img(self, z):
        b = z.shape[0]
        h = z.shape[-2] * 8
        w = z.shape[-1] * 8
        img = torch.full((b, h, w, 3), 0.5, dtype=torch.float32)
        if self.noisy_corner and h >= 32 and w >= 32:
            torch.manual_seed(42)
            img[:, h - 12 : h, w - 12 : w, :] = torch.randint(0, 2, (b, 12, 12, 3)).float()
        return img

    def decode(self, z):
        self.last_decode_method = "direct"
        return self._make_img(z)

    def decode_tiled(self, z, **kwargs):
        self.last_decode_method = "tiled"
        return self._make_img(z)

    def encode(self, img):
        b, h, w, _ = img.shape
        return torch.randn(b, self.latent_channels, h // 8, w // 8, dtype=torch.float32)


class DummyWan3DVAE(DummyVAE):
    """Exact mock of ComfyUI's Wan 2.1 VAE (latent_dim=3, not_video=False, 5D decode output [B, T, H, W, 3])."""

    def __init__(self, latent_channels=16, noisy_corner=False):
        super().__init__(latent_channels=latent_channels, noisy_corner=noisy_corner)
        self.latent_dim = 3
        self.not_video = False

    def _make_5d_img(self, z):
        if z.ndim != 5:
            raise IndexError(f"Wan 2.1 VAE expects 5D [B, C, T, H, W] latent, got {z.shape}")
        b, _, t, h_lat, w_lat = z.shape
        h, w = h_lat * 8, w_lat * 8
        img = torch.full((b, t, h, w, 3), 0.5, dtype=torch.float32)
        if self.noisy_corner and h >= 32 and w >= 32:
            torch.manual_seed(42)
            img[:, :, h - 12 : h, w - 12 : w, :] = torch.randint(0, 2, (b, t, 12, 12, 3)).float()
        return img

    def decode(self, z):
        self.last_decode_method = "direct"
        return self._make_5d_img(z)

    def decode_tiled(self, z, **kwargs):
        self.last_decode_method = "tiled"
        return self._make_5d_img(z)

    def encode(self, img):
        # Mimics comfy/sd.py line 1416: when latent_dim == 3 and not_video == False, 4D [B, H, W, 3]
        # treats B as video frames T of a single batch item (B_out=1, T_out=1 + (B-1)//4)!
        if img.ndim == 4:
            f, h, w, _ = img.shape
            t_out = 1 + max(0, f - 1) // 4
            return torch.randn(1, self.latent_channels, t_out, h // 8, w // 8, dtype=torch.float32)
        b, t, h, w, _ = img.shape
        return torch.randn(b, self.latent_channels, t, h // 8, w // 8, dtype=torch.float32)


class FakeLoRAAdapter:
    def __init__(self, up, down, alpha):
        self.weights = (up, down, alpha, None, None, None)


def test_1_dynamic_topology_and_diffusers_keys():
    # 1. Standard 28-block keys
    assert hades.get_anima_block_idx("diffusion_model.blocks.0.attn.qkv.weight") == 0
    assert hades.get_anima_block_idx("diffusion_model.blocks.27.mlp.fc2.weight") == 27
    assert hades.get_anima_block_idx("diffusion_model.llm_adapter.blocks.5.cross_attn.weight") == 33

    # 2. Kohya underscore keys
    assert hades.get_anima_block_idx("lora_unet_blocks_14_attn_qkv") == 14
    assert hades.get_anima_block_idx("lora_unet_llm_adapter_blocks_3_proj") == 31

    # 3. Diffusers transformer_blocks.* and text_conditioner.* keys
    assert hades.get_anima_block_idx("transformer_blocks.0.attn1.to_q.lora_A.weight") == 0
    assert hades.get_anima_block_idx("lora_transformer_blocks_21_ff_net_0_proj") == 21
    assert hades.get_anima_block_idx("text_conditioner.blocks.4.cross_attn.to_q") == 32

    # 4. 40-block (2.9B) and 52-block (3.8B) Anima variant detection and proportional mapping
    keys_40 = [f"diffusion_model.blocks.{i}.weight" for i in range(40)] + [
        f"diffusion_model.llm_adapter.blocks.{j}.weight" for j in range(6)
    ]
    assert hades.detect_anima_topology(keys_40) == (40, 6)
    assert hades.get_anima_block_idx("diffusion_model.blocks.39.weight", n_dit=40, n_adapter=6) == 27

    keys_52 = [f"diffusion_model.blocks.{i}.weight" for i in range(52)] + [
        f"diffusion_model.llm_adapter.blocks.{j}.weight" for j in range(6)
    ]
    assert hades.detect_anima_topology(keys_52) == (52, 6)
    assert hades.get_anima_block_idx("diffusion_model.blocks.51.weight", n_dit=52, n_adapter=6) == 27

    # 4b. Sparse partial LoRA topology snapping (e.g. partial 40L LoRA touching only block 34 -> snaps to 40L)
    sparse_40 = ["diffusion_model.blocks.2.weight", "diffusion_model.blocks.34.weight"]
    assert hades.detect_anima_topology(sparse_40) == (40, 6)
    sparse_52 = ["diffusion_model.blocks.5.weight", "diffusion_model.blocks.46.weight"]
    assert hades.detect_anima_topology(sparse_52) == (52, 6)

    # 4c. Non-block llm_adapter tensors (embed, in_proj, out_proj, norm) map to adapter bands 28 / 33 (never None!)
    assert hades.get_anima_block_idx("llm_adapter.embed.weight") == 28
    assert hades.get_anima_block_idx("llm_adapter.in_proj.weight") == 28
    assert hades.get_anima_block_idx("llm_adapter.out_proj.weight") == 33
    assert hades.get_anima_block_idx("llm_adapter.norm.weight") == 33
    assert hades.get_anima_block_idx("diffusion_model.llm_adapter.out_proj.weight") == 33

    # 4d. All 17 nodes share the unified "HaDeS/Anima ..." root category
    for node_name, node_cls in hades.NODE_CLASS_MAPPINGS.items():
        cat = getattr(node_cls, "CATEGORY", "")
        assert cat.startswith("HaDeS/Anima "), f"Node {node_name} has non-unified category: {cat}"

    # 5. Non-block Kohya / Diffusers / LyCORIS canonicalization
    assert (
        hades.canonicalize_anima_lora_key("lora_unet_final_layer_linear.lora_up.weight")
        == "diffusion_model.final_layer.linear.lora_up.weight"
    )
    assert (
        hades.canonicalize_anima_lora_key("lora_unet_t_embedder_1_linear_1.lora_down.weight")
        == "diffusion_model.t_embedder.1.linear_1.lora_down.weight"
    )
    assert (
        hades.canonicalize_anima_lora_key("base_model.model.transformer_blocks.3.attn2.to_v.lora_A.weight")
        == "diffusion_model.blocks.3.cross_attn.v_proj.lora_down.weight"
    )
    print("[OK] Test 1: Dynamic topology (28/40/52 blocks) + Kohya, Diffusers & non-block regex verified")


def test_2_qr_svd_accuracy_and_speed():
    merger = hades.HaDeSAnimaStandaloneLoraMerger()
    torch.manual_seed(42)
    m, n, r = 512, 1024, 16
    up_a = torch.randn(m, r)
    down_a = torch.randn(r, n)
    up_b = torch.randn(m, r)
    down_b = torch.randn(r, n)

    t0 = time.perf_counter()
    lora_up, lora_down, _ = merger._qr_svd_factor_merge(
        up_a, down_a, 0.7, up_b, down_b, 0.8, target_rank=32, device=torch.device("cpu")
    )
    qr_elapsed = time.perf_counter() - t0

    exact_delta = (0.7 * (up_a @ down_a)) + (0.8 * (up_b @ down_b))
    reconstructed = lora_up @ lora_down
    rel_err = (torch.norm(exact_delta - reconstructed) / torch.norm(exact_delta)).item()
    assert rel_err < 1e-5, f"QR-SVD reconstruction relative error too high: {rel_err}"
    print(f"[OK] Test 2: QR-SVD exact reconstruction verified (rel_err={rel_err:.2e}, time={qr_elapsed*1000:.1f}ms)")


def test_3_reactive_predictive_model_merger_and_llm_adapter_lock():
    torch.manual_seed(7)
    base_sd = {}
    sd1 = {}
    sd2 = {}
    for b in range(28):
        k = f"blocks.{b}.mlp.weight"
        w0 = torch.randn(32, 32)
        base_sd[k] = w0
        noise_scale = 0.01 if b < 15 else 0.08
        sd1[k] = w0 + torch.randn(32, 32) * 0.01
        sd2[k] = w0 + torch.randn(32, 32) * noise_scale
    for ab in range(6):
        ak = f"llm_adapter.blocks.{ab}.self_attn.q_proj.weight"
        w_ad = torch.randn(32, 32)
        base_sd[ak] = w_ad
        sd1[ak] = w_ad.clone()
        sd2[ak] = w_ad + torch.randn(32, 32) * 0.2
    for non_blk_ad in ("llm_adapter.embed.weight", "llm_adapter.in_proj.weight", "llm_adapter.out_proj.weight", "llm_adapter.norm.weight"):
        w_nb = torch.randn(32, 32)
        base_sd[non_blk_ad] = w_nb
        sd1[non_blk_ad] = w_nb.clone()
        sd2[non_blk_ad] = w_nb + torch.randn(32, 32) * 0.5

    m_base = DummyModelPatcher(base_sd)
    m1 = DummyModelPatcher(sd1)
    m2 = DummyModelPatcher(sd2)

    pred = hades.HaDeSAnimaPredictiveModelMerger()
    out_m, info = pred.predictive_merge(
        m1, m2, "Interference Guard (Anti-Burn / Artifact Shield)", 0.5, 1.5, protect_llm_adapter=True
    )
    vec_line = info.split("--- VECTORS OVERRIDE (34 BLOCKS) ---")[1].split("\n")[1].strip()
    ratios = [float(x) for x in vec_line.split(",")]
    assert sum(ratios[:10]) / 10 > sum(ratios[20:28]) / 8
    # Verify llm_adapter blocks 28..33 AND non-block adapter tensors (embed/in_proj/out_proj/norm) are locked at 0.0
    assert all(r == 0.0 for r in ratios[28:]), f"Expected llm_adapter blocks 28..33 to be 0.0, got {ratios[28:]}"
    for k_patch in out_m.patches.keys():
        assert "llm_adapter" not in k_patch, f"Non-block llm_adapter key '{k_patch}' leaked into patches despite protect_llm_adapter=True!"

    # Verify identical tensors produce 0 redundant patches in RAM
    m_same1 = DummyModelPatcher(sd1)
    m_same2 = DummyModelPatcher(sd1)
    _, merged_same, _ = hades.apply_model_block_merge(m_same1, m_same2, [0.5] * 34, 0.5)
    assert merged_same == 0, f"Expected 0 patches for identical model weights, got {merged_same}"

    # Task Vectors test
    _, info_tv = pred.predictive_merge(
        m1, m2, "Style Transfer (Keep A identity, B style)", 0.5, 1.2, base_model=m_base
    )
    assert "Task Vectors (vs base_model)" in info_tv
    print("[OK] Test 3: Predictive Merger + llm_adapter Bridge Lock (0.0) + Zero-RAM identical tensor skip verified")


def test_4_subspace_multi_lora_per_secondary_slot():
    torch.manual_seed(99)
    sd = {
        "diffusion_model.blocks.2.attn.weight": torch.ones(32, 32),
        "diffusion_model.blocks.22.mlp.weight": torch.ones(32, 32),
    }
    clip_sd = {"qwen.model.layers.0.self_attn.weight": torch.ones(32, 32)}
    m = DummyModelPatcher(sd)
    c = DummyCLIP(clip_sd)

    loader = hades.HaDeSAnimaPredictiveMultiLoraLoader()

    u_pri = torch.randn(32, 4)
    d_pri = torch.randn(4, 32)
    q_u, _ = torch.linalg.qr(torch.randn(32, 8), mode="reduced")
    q_d, _ = torch.linalg.qr(torch.randn(32, 8), mode="reduced")
    u_pri_orth = q_u[:, :4]
    d_pri_orth = q_d[:, :4].T
    u_s3_orth = q_u[:, 4:8]
    d_s3_orth = q_d[:, 4:8].T

    fake_loras = {
        "char_pri.safetensors": {
            "diffusion_model.blocks.2.attn.weight": FakeLoRAAdapter(u_pri_orth, d_pri_orth, 4.0),
            "diffusion_model.blocks.22.mlp.weight": FakeLoRAAdapter(u_pri, d_pri, 4.0),
        },
        "col_s2.safetensors": {
            "diffusion_model.blocks.2.attn.weight": FakeLoRAAdapter(u_pri_orth * 1.5, d_pri_orth, 4.0),
            "diffusion_model.blocks.22.mlp.weight": FakeLoRAAdapter(u_pri, d_pri, 4.0),
        },
        "orth_s3.safetensors": {
            "diffusion_model.blocks.2.attn.weight": FakeLoRAAdapter(u_s3_orth, d_s3_orth, 4.0),
            "diffusion_model.blocks.22.mlp.weight": FakeLoRAAdapter(u_s3_orth, d_s3_orth, 4.0),
        },
    }
    loader._load_raw_lora = lambda name, model, clip, protect_llm_adapter=True: fake_loras[name]

    _, _, info = loader.apply_predictive_multi_lora(
        m, c, True, "Anti-Overcook (Damped Safe Blend)", 1.5,
        "char_pri.safetensors", True, 1.0,
        "col_s2.safetensors", True, 1.0,
        "orth_s3.safetensors", True, 1.0,
        "None", False, 1.0,
    )
    s2_vec = [float(x) for x in info.split("SECONDARY SLOT S2")[1].split("\n")[1].split(",")]
    s3_vec = [float(x) for x in info.split("SECONDARY SLOT S3")[1].split("\n")[1].split(",")]
    assert s2_vec[2] < s3_vec[2]
    assert all(x == 0.0 for x in s2_vec[28:])
    print("[OK] Test 4: Subspace Multi-LoRA per-slot curves + Bridge protection verified")


def test_5_cross_format_diffusers_kohya_lora_svd_merge():
    with tempfile.TemporaryDirectory() as tmpdir:
        merger = hades.HaDeSAnimaStandaloneLoraMerger()
        merger.output_dir = tmpdir

        lora_a_path = os.path.join(tmpdir, "kohya_donor.safetensors")
        lora_b_path = os.path.join(tmpdir, "diffusers_donor.safetensors")

        # Donor A uses Kohya format; Donor B uses Diffusers transformer_blocks format + dirty text_conditioner key
        sd_a = {
            "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.randn(32, 8, dtype=torch.float16),
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(8, 32, dtype=torch.float16),
            "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(8.0, dtype=torch.float16),
            "lora_unet_blocks_0_attn1_norm_q.lora_up.weight": torch.randn(8, 4, dtype=torch.float16),
            "lora_unet_blocks_0_attn1_norm_q.lora_down.weight": torch.randn(4, 8, dtype=torch.float16),
        }
        sd_b = {
            "transformer.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.randn(32, 8, dtype=torch.float16),
            "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.randn(8, 32, dtype=torch.float16),
            "transformer.transformer_blocks.0.attn1.norm_q.lora_B.weight": torch.randn(8, 4, dtype=torch.float16),
            "transformer.transformer_blocks.0.attn1.norm_q.lora_A.weight": torch.randn(4, 8, dtype=torch.float16),
            "transformer.transformer_blocks.0.attn2.norm_k.lora_B.weight": torch.randn(8, 4, dtype=torch.float16),
            "transformer.transformer_blocks.0.attn2.norm_k.lora_A.weight": torch.randn(4, 8, dtype=torch.float16),
            "transformer.text_conditioner.blocks.0.attn1.to_q.lora_B.weight": torch.randn(32, 4, dtype=torch.float16),
            "transformer.text_conditioner.blocks.0.attn1.to_q.lora_A.weight": torch.randn(4, 32, dtype=torch.float16),
            # DoRA layer on block 1 must be excluded as a whole (never partially merged without its dora_scale)
            "transformer.transformer_blocks.1.attn1.to_q.dora_scale": torch.ones(32, 1, dtype=torch.float16),
            "transformer.transformer_blocks.1.attn1.to_q.lora_B.weight": torch.randn(32, 4, dtype=torch.float16),
            "transformer.transformer_blocks.1.attn1.to_q.lora_A.weight": torch.randn(4, 32, dtype=torch.float16),
        }
        safetensors.torch.save_file(sd_a, lora_a_path)
        safetensors.torch.save_file(sd_b, lora_b_path)

        orig_get_full_path = folder_paths.get_full_path_or_raise
        folder_paths.get_full_path_or_raise = lambda folder, name: os.path.join(tmpdir, name)
        try:
            saved_path, info = merger.merge_and_save(
                "kohya_donor.safetensors", "diffusers_donor.safetensors",
                "Character + Style Fusion (A identity, B style)", 0.5, 1.0, 16, "kohya_donor",
                merge_mode="Additive (A + B, Multi-LoRA match)", protect_llm_adapter=True, overwrite_existing=False,
                topology_override="Anima 2.9B (40 DiT + 6 Adapter)"
            )
            assert os.path.basename(saved_path) == "kohya_donor_001.safetensors"
            assert "Layers QR-SVD Merged: 2" in info, f"Expected Kohya + Diffusers q_proj & norm_q layers to collide and merge via QR-SVD! Info:\n{info}"
            assert "Adapter Stripped: 2" in info
            assert "Warning: Skipped" in info
            merged_sd = safetensors.torch.load_file(saved_path)
            assert "diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight" in merged_sd
            assert "diffusion_model.blocks.0.self_attn.q_norm.lora_up.weight" in merged_sd
            assert "diffusion_model.blocks.0.cross_attn.k_norm.lora_up.weight" in merged_sd
            assert "diffusion_model.blocks.1.self_attn.q_proj.lora_up.weight" not in merged_sd, (
                "DoRA layer should be completely excluded rather than merged without its dora_scale!"
            )
            assert not any("llm_adapter" in k for k in merged_sd.keys())
            print("[OK] Test 5: Cross-format Diffusers + Kohya QR-SVD LoRA merge, DoRA prefix guard & dirty llm_adapter stripping verified")
        finally:
            folder_paths.get_full_path_or_raise = orig_get_full_path


def test_6_clip_proportional_36_layer_depth():
    sd1 = {"embed_tokens.weight": torch.zeros(16, 16), "model.norm.weight": torch.zeros(16)}
    sd2 = {"embed_tokens.weight": torch.ones(16, 16), "model.norm.weight": torch.ones(16)}
    for i in range(36):
        sd1[f"model.layers.{i}.mlp.weight"] = torch.zeros(16, 16)
        sd2[f"model.layers.{i}.mlp.weight"] = torch.ones(16, 16)
    c1 = DummyCLIP(sd1)
    c2 = DummyCLIP(sd2)
    clip_merger = hades.HaDeSAnimaCLIPBlockMerger()
    _, info = clip_merger.merge_clips(c1, c2, 0.5, 0.2, 0.3, 0.6, 0.9, 0.4)
    assert "Depth: 36L" in info
    print(f"[OK] Test 6: CLIP Block Merger proportional 36-layer depth verified ({info})")


def test_7_prompt_studio_and_aesthetic_slop_guard():
    studio = hades.HaDeSAnimaPromptStudio()
    clip = DummyCLIP({})
    raw_pos = (
        "score_9, score_8_up, masterpiece, best quality, 1girl, hatsune_miku_(vocaloid), "
        "Blue_Hair, ^_^, safe_for_work, (cyan_eyes, score_7_up, Long_Hair:1.4), "
        "Digital artwork of Fern from Sousou no Frieren standing under a golden sky."
    )
    pos_cond, neg_cond, fmt_pos, fmt_neg, info = studio.process_prompt(
        "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Flat Vector / Graphic (Non-Anime Dataset Header)",
        True,
        1.5,
        "frieren, sousou no frieren",
        "by zocehuy, (@zomzomzomsauce:1.2)",
        raw_pos,
        "worst_quality, low_quality, score_1, score_2, score_3, artist_name",
        clip=clip,
        character_in="fern",
        artist_in="(@yalmyu:1.0)",
    )
    assert "score_9" not in fmt_pos and "score_7_up" not in fmt_pos and "masterpiece" not in fmt_pos
    assert "score_1" not in fmt_neg and "worst quality" in fmt_neg, f"Expected score_* stripped from Aesthetic negative prompt, got: {fmt_neg}"
    assert r"hatsune miku \(vocaloid\)" in fmt_pos, f"Expected unescaped Booru parens to be escaped, got: {fmt_pos}"

    # Verify canonical header: L1 = single dataset trigger (ye-pop); body is last line (no title -> 2 lines)
    lines = fmt_pos.split("\n")
    assert lines[0] == "ye-pop", f"Expected canonical header trigger 'ye-pop' on Line 1, got: {lines[:2]}"
    assert "deviantart" not in lines[0], "Canonical header must carry a single trigger, not Dual Legacy"
    body = lines[-1]

    # Verify Canonical Order in the body line:
    # [Quality/Safety: safe_for_work] -> [Subject: 1girl] -> [Characters: fern, frieren, sousou no frieren, hatsune miku...]
    # -> [Artists: @yalmyu, @zocehuy, @zomzomzomsauce] -> [General Tags] -> [Natural Language]
    assert "@zocehuy" in body and "(@zomzomzomsauce:1.2)" in body and "(@yalmyu:1.0)" in body
    assert "@by" not in body, "Leading 'by ' must be stripped from artist entries"
    assert "fern, frieren, sousou no frieren" in body

    # Verify Natural Language capitalization preservation
    assert "Digital artwork of Fern from Sousou no Frieren standing under a golden sky." in body, (
        f"Natural language sentence capitalization must be preserved! Got:\n{body}"
    )

    # Verify Anima 2.9B / 3.8B Expanded keeps score_* tags per SKILL.md
    _, _, fmt_pos_29b, fmt_neg_29b, _ = studio.process_prompt(
        "Anima 2.9B / 3.8B Expanded (Keep Quality Tags)",
        "Standard Anime / Illustration",
        True,
        1.0,
        "",
        "",
        "score_9, masterpiece, 1girl",
        "worst_quality, score_1",
        clip=None,
    )
    assert "score_9" in fmt_pos_29b and "masterpiece" in fmt_pos_29b and "score_1" in fmt_neg_29b

    # Verify Anima Aesthetic v1.0b keeps optional quality tags but still strips forbidden score_*
    _, _, fmt_pos_v10b, fmt_neg_v10b, _ = studio.process_prompt(
        "Anima Aesthetic v1.0b (Keep masterpiece, best quality)",
        "Standard Anime / Illustration",
        True,
        1.0,
        "",
        "",
        "score_9, masterpiece, best quality, 1girl",
        "worst_quality, score_1, masterpiece",
        clip=None,
    )
    assert "masterpiece" in fmt_pos_v10b and "best quality" in fmt_pos_v10b, f"v1.0b must keep optional quality tags: {fmt_pos_v10b}"
    assert "score_9" not in fmt_pos_v10b, f"v1.0b must still strip score_*: {fmt_pos_v10b}"
    assert "score_1" not in fmt_neg_v10b and "masterpiece" in fmt_neg_v10b, f"v1.0b negative: score_* stripped, quality kept: {fmt_neg_v10b}"

    assert pos_cond is not None and neg_cond is not None
    # Verify pos_cond token weight scaling: original 1.5 with multiplier 1.5 -> 1.0 + (0.5 * 1.5) = 1.75
    pos_tokens = pos_cond[0][1]["tokens"]["t5xxl"][0]
    assert abs(pos_tokens[1][1] - 1.75) < 1e-5, f"Expected T5 scaled weight 1.75, got {pos_tokens[1][1]}"

    # Error 1 Check: multi-word booru tags must not trick aesthetic guard into treating line as prose
    multi_tag_pos = "masterpiece, standing in the rain, wearing a red dress, holding a big umbrella, 1girl"
    _, _, fmt_multi, _, _ = studio.process_prompt(
        "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Standard Anime / Illustration",
        True,
        1.0,
        "",
        "",
        multi_tag_pos,
        "",
        clip=None,
    )
    assert "masterpiece" not in fmt_multi, f"Aesthetic guard must strip 'masterpiece' even when multi-word tags are present! Got: {fmt_multi}"

    # Error 4 Check: grammatical colon in natural language sentence must not be rejected as tag-weight
    prose_colon_pos = "Abstract, oil painting. Left: white, draped figure. Right: dark backdrop."
    _, _, fmt_colon, _, _ = studio.process_prompt(
        "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Standard Anime / Illustration",
        True,
        1.0,
        "",
        "",
        prose_colon_pos,
        "",
        clip=None,
    )
    assert "Left: white, draped figure. Right: dark backdrop." in fmt_colon, (
        f"Grammatical colon in natural language prose must be preserved! Got: {fmt_colon}"
    )

    print("[OK] Test 7: Prompt Studio (Canonical Ordering, Multi-Word Slop Scrubbing, NL Grammatical Colons, Character/Artist Sockets, Multi-Artist Weight Syntax, NL Capitalization & T5 Scaling) verified")


def test_8_nag_attention_guidance():
    m = DummyModelPatcher({"blocks.0.weight": torch.ones(8, 8)})
    nag_node = hades.HaDeSAnimaAttentionGuidance()
    m_guided, info = nag_node.apply_nag(
        m, "Turbo / Lightning Negative Enabler", 3.5, 0.45, 2.5, 0.0, 0.70
    )
    assert m_guided.model_options.get("disable_cfg1_optimization") is True
    override_fn = m_guided.model_options["transformer_options"]["optimized_attention_override"]

    def dummy_attn(q, k, v, heads, mask=None, **kwargs):
        return q + v

    # Test ComfyUI's reversed batch order: cond_or_uncond = [1, 0] -> index 0 is uncond, index 1 is cond
    q = torch.ones(2, 16, 64, dtype=torch.float16)
    k = torch.ones(2, 16, 64, dtype=torch.float16)
    v = torch.cat(
        [
            torch.ones(1, 16, 64, dtype=torch.float16) * 0.5,  # uncond at idx 0
            torch.ones(1, 16, 64, dtype=torch.float16) * 2.0,  # cond at idx 1
        ],
        dim=0,
    )

    out_active = override_fn(
        dummy_attn,
        q,
        k,
        v,
        8,
        transformer_options={"sigmas": torch.tensor([0.8]), "cond_or_uncond": [1, 0]},
    )
    assert out_active.dtype == torch.float16 and out_active.shape == q.shape
    # Uncond branch (index 0) must remain unchanged while cond branch (index 1) is extrapolated
    base_out = dummy_attn(q, k, v, 8)
    assert torch.allclose(out_active[0], base_out[0])
    assert not torch.allclose(out_active[1], base_out[1])

    # Test 4D q, k, v ([B, H, S, D] with skip_reshape=True) in single-branch CFG=1 mode
    def dummy_attn_4d(q, k, v, heads, mask=None, **kwargs):
        b, h, s, d = q.shape
        return (q + v).permute(0, 2, 1, 3).reshape(b, s, h * d)

    q4 = torch.randn(1, 8, 16, 8, dtype=torch.float16)
    k4 = torch.randn(1, 8, 16, 8, dtype=torch.float16)
    v4 = torch.randn(1, 8, 16, 8, dtype=torch.float16)
    out_4d = override_fn(
        dummy_attn_4d,
        q4,
        k4,
        v4,
        8,
        skip_reshape=True,
        transformer_options={"sigmas": torch.tensor([0.8]), "cond_or_uncond": [0]},
    )
    assert out_4d.shape == (1, 16, 64) and out_4d.dtype == torch.float16

    # When mask is present, must bypass cleanly
    out_masked = override_fn(
        dummy_attn, q, k, v, 8, mask=torch.ones(2, 16, 16), transformer_options={"sigmas": torch.tensor([0.8])}
    )
    assert torch.allclose(out_masked, dummy_attn(q, k, v, 8))

    # Verify chaining with an existing upstream optimized_attention_override
    m_chain = DummyModelPatcher({"blocks.0.weight": torch.ones(8, 8)})
    upstream_called = {"count": 0}

    def upstream_override(func, *a, **kw):
        upstream_called["count"] += 1
        return func(*a, **kw)

    m_chain.model_options["transformer_options"] = {"optimized_attention_override": upstream_override}
    m_chained, _ = nag_node.apply_nag(m_chain, "Balanced Quality & Anatomy", 2.5, 0.35, 2.5, 0.0, 0.65)
    chained_fn = m_chained.model_options["transformer_options"]["optimized_attention_override"]
    _ = chained_fn(dummy_attn, q, k, v, 8, transformer_options={"sigmas": torch.tensor([6.0]), "cond_or_uncond": [1, 0]})
    assert upstream_called["count"] == 1, "Expected upstream optimized_attention_override to be chained and invoked!"
    print("[OK] Test 8: NAG Attention Guidance (cond_or_uncond=[1,0], override chaining, 4D skip_reshape, L1 norm) verified")


def test_9_latent_pipeline_guard():
    guard = hades.HaDeSAnimaLatentPipelineGuard()
    vae_16 = DummyVAE(16)
    vae_64 = DummyVAE(64)

    # 1. Verify 1280x1280 white-background bias shift -> 1216x1216
    latent_dict, w, h, info = guard.generate_latent(
        "Custom (Auto-Snap to 32px Grid)", 1280, 1280, 1, prevent_29b_white_bg_bias=True, vae=vae_16
    )
    assert (w, h) == (1216, 1216)
    assert tuple(latent_dict["samples"].shape) == (1, 16, 1216 // 8, 1216 // 8)

    # 2. Verify 64-channel Qwen-Image 2.1 VAE trap raises clear ValueError
    raised = False
    try:
        guard.generate_latent("1024x1024 (1:1 Square - 1.05 MP)", 1024, 1024, 1, vae=vae_64)
    except ValueError as e:
        raised = "64" in str(e) and "16-channel" in str(e)
    assert raised, "Expected 64-channel VAE to be blocked by HaDeSAnimaLatentPipelineGuard!"
    print("[OK] Test 9: Latent Pipeline Guard (16ch validation, 64ch VAE block, 1280px 2.9B bias guard) verified")


def test_10_corner_scribble_shield():
    shield = hades.HaDeSAnimaCornerScribbleShield()
    assert shield.RETURN_NAMES == ("image", "mask", "info"), f"Expected RETURN_NAMES ('image', 'mask', 'info'), got {shield.RETURN_NAMES}"
    torch.manual_seed(123)
    # Realistic scene: smooth gradient background [1, 128, 128, 3] with low-contrast
    # thin signature strokes in the bottom-right corner (mimics real Anima watermarks,
    # far weaker than the old binary-noise test pattern).
    base = torch.linspace(0.3, 0.7, 128).view(1, 1, 128, 1).expand(1, 128, 128, 1)
    img = base.repeat(1, 1, 1, 3).clone()
    strokes = torch.full((1, 20, 20, 3), 0.5, dtype=torch.float32)
    for r in range(0, 20, 3):
        strokes[:, r, :] = 0.15          # thin horizontal strokes
        strokes[:, :, r] = 0.25          # thin vertical strokes
    img[:, 108:128, 108:128, :] = strokes

    healed, mask, info = shield.shield_corners(
        img, 0.12, 0.75, "Bottom-Right Only", "Smart Edge-Inward Blend (Auto-Heal)", 6
    )
    assert healed.shape == img.shape and mask.shape == (1, 128, 128)
    # Center of image must remain 100% untouched
    assert torch.equal(healed[:, 32:96, 32:96, :], img[:, 32:96, 32:96, :])
    # Bottom-right corner stroke variance must be reduced after healing
    orig_var = img[:, 108:128, 108:128, :].var().item()
    healed_var = healed[:, 108:128, 108:128, :].var().item()
    assert healed_var < orig_var, f"Expected corner stroke variance to decrease ({healed_var} vs {orig_var})"
    assert (healed[:, 108:128, 108:128, :] - img[:, 108:128, 108:128, :]).abs().mean().item() > 0.01, \
        "Expected Smart Auto-Heal to visibly modify the scribble corner"

    # Aggressive mode: forced fill, bypasses detection entirely; center stays untouched
    aggr, mask_a, info_a = shield.shield_corners(
        img, 0.12, 0.75, "Bottom-Right Only", "Aggressive Corner Fill (Forced)", 6
    )
    assert torch.equal(aggr[:, 32:96, 32:96, :], img[:, 32:96, 32:96, :])
    aggr_diff = (aggr[:, 108:128, 108:128, :] - img[:, 108:128, 108:128, :]).abs().mean().item()
    assert aggr_diff > 0.02, f"Aggressive mode should force a visible corner fill (diff={aggr_diff})"

    # Clean corner (no scribbles) must be left essentially untouched in Smart mode
    clean_img = base.repeat(1, 1, 1, 3).clone()  # [1, 128, 128, 3]
    clean_out, _, _ = shield.shield_corners(
        clean_img, 0.12, 0.75, "All 4 Corners", "Smart Edge-Inward Blend (Auto-Heal)", 6
    )
    clean_diff = (clean_out - clean_img).abs().max().item()
    assert clean_diff < 0.005, f"Smart mode must not alter clean corners (max diff={clean_diff})"
    # Large-logo support: a ~35% corner mark extends far beyond the default
    # base box (0.10); adaptive expansion must detect high-freq content at the
    # inner edge, grow the box, and remove the full mark. Center stays untouched.
    big = base.repeat(1, 1, 1, 3).clone()  # [1, 128, 128, 3]
    torch.manual_seed(7)
    logo = 0.5 + 0.45 * torch.randn(1, 45, 45, 3)  # noisy high-freq block logo
    big[:, 83:128, 83:128, :] = logo
    big_out, big_mask, big_info = shield.shield_corners(
        big, 0.10, 0.75, "Bottom-Right Only", "Aggressive Corner Fill (Forced)", 8
    )
    # (region well inside the untouched image center; box expansion stops at the mark)
    assert torch.equal(big_out[:, 16:56, 16:56, :], big[:, 16:56, 16:56, :]), "Center must remain untouched"
    logo_orig_var = big[:, 83:128, 83:128, :].var().item()
    logo_healed_var = big_out[:, 83:128, 83:128, :].var().item()
    # Feathered fill keeps a soft gradient at the outer ring; require >50% variance reduction
    assert logo_healed_var < 0.5 * logo_orig_var, \
        f"Large logo must be mostly removed via box expansion (var {logo_healed_var} vs {logo_orig_var})"
    assert big_mask[:, 83:128, 83:128].sum().item() > 0.5 * big_mask[:, 83:128, 83:128].numel(), \
        "Expansion mask must cover the bulk of the large logo region"

    # Clean corner must NOT expand (no-op on smooth images)
    clean2, clean_mask, clean_info = shield.shield_corners(
        base.repeat(1, 1, 1, 3).clone(), 0.10, 0.75, "Bottom-Right Only", "Aggressive Corner Fill (Forced)", 8
    )
    assert "Auto-Expanded" not in clean_info, f"Clean corner must not trigger expansion: {clean_info}"

    # Global texture / uniform grain guard: gaussian noise (sigma=0.03) or hatching across the whole image
    # Smart Auto-Heal must NOT mistake uniform texture for corner signatures.
    torch.manual_seed(999)
    grain_img = (base.repeat(1, 1, 1, 3) + torch.randn(1, 128, 128, 3) * 0.03).clamp(0.0, 1.0)
    grain_healed, grain_mask, grain_info = shield.shield_corners(
        grain_img, 0.12, 0.75, "All 4 Corners", "Smart Edge-Inward Blend (Auto-Heal)", 6
    )
    grain_diff = (grain_healed - grain_img).abs().max().item()
    assert grain_diff < 0.005, f"Smart mode must not alter uniform grain/textures (max diff={grain_diff})"
    assert grain_mask.sum().item() == 0, f"Expected zero masked pixels on uniform grain, got {grain_mask.sum().item()}"
    # Error 8 Check: Clean telemetry must not report contradictory Auto-Expanded corners
    assert "Auto-Expanded" not in grain_info, f"Clean/uniform texture must not claim Auto-Expanded in telemetry! Got: {grain_info}"

    # Error 2 Check: A thin 1px stroke in the corner on top of sigma=0.03 grain MUST be detected via peak HF
    grain_stroke_img = grain_img.clone()
    grain_stroke_img[:, 118:122, 118:126, :] = 0.0  # high-contrast dark stroke
    gs_healed, gs_mask, gs_info = shield.shield_corners(
        grain_stroke_img, 0.12, 0.75, "Bottom-Right Only", "Smart Edge-Inward Blend (Auto-Heal)", 6
    )
    assert gs_mask[:, 118:122, 118:126].max().item() > 0.5, (
        f"Thin 1px stroke on top of sigma=0.03 grain must be detected and masked via hf_max! Info: {gs_info}"
    )
    stroke_orig_diff = (grain_stroke_img[:, 118:122, 118:126, :] - base[:, 118:122, 118:126, :].repeat(1, 1, 1, 3)).abs().mean().item()
    stroke_healed_diff = (gs_healed[:, 118:122, 118:126, :] - base[:, 118:122, 118:126, :].repeat(1, 1, 1, 3)).abs().mean().item()
    assert stroke_healed_diff < stroke_orig_diff, f"Expected stroke to be healed towards baseline ({stroke_healed_diff} vs {stroke_orig_diff})"

    print("[OK] Test 10: Corner Scribble Shield (realistic thin-stroke detection, peak HF vs grain, Aggressive mode, clean-corner safety, large-logo auto-expansion, telemetry sync) verified")


def test_11_anima_single_pass_ksampler_and_schedulers():
    m = DummyModelPatcher({"blocks.0.weight": torch.ones(8, 8)})
    orig_ms = DummyModelSampling(shift=3.0, multiplier=1.0)
    m.add_object_patch("model_sampling", orig_ms)

    # 1. Verify multiplier=1.0 is preserved when patching flow_shift (unlike ModelSamplingSD3's multiplier=1000)
    m_shifted = hades.clone_model_with_anima_shift(m, 3.85)
    ms = m_shifted.get_model_object("model_sampling")
    assert abs(float(ms.shift) - 3.85) < 1e-4
    assert abs(float(ms.multiplier) - 1.0) < 1e-4, f"Expected multiplier=1.0, got {ms.multiplier}"

    # Verify deepcopy buffer isolation: mutating ms.sigmas does NOT affect orig_ms.sigmas
    ms.sigmas[0] = 999.0
    assert float(orig_ms.sigmas[0]) == 10.0, "Cloned model_sampling mutated original model_sampling.sigmas buffer!"

    # 2. Verify anima_lightning_tdm exact 4-step trajectory from distilled_generation_config.json
    #    raw_sigmas: [1.0, 0.75, 0.5, 0.25]
    #    effective (shift=3.0): [1.0, 0.9, 0.75, 0.5, 0.0]
    tdm_sigmas = hades.build_anima_sigmas(m, "anima_lightning_tdm", 4, denoise=1.0, flow_shift=3.0)
    assert len(tdm_sigmas) == 5 and float(tdm_sigmas[-1]) == 0.0
    assert abs(float(tdm_sigmas[0]) - 1.0) < 1e-6
    assert abs(float(tdm_sigmas[1]) - 0.9) < 1e-6
    assert abs(float(tdm_sigmas[2]) - 0.75) < 1e-6
    assert abs(float(tdm_sigmas[3]) - 0.5) < 1e-6

    # 3. Verify anima_beta57 schedule monotonically decreases to 0.0
    b57_sigmas = hades.build_anima_sigmas(m, "anima_beta57", 30, denoise=1.0, flow_shift=2.8)
    assert len(b57_sigmas) >= 20 and float(b57_sigmas[-1]) == 0.0
    for i in range(len(b57_sigmas) - 1):
        assert float(b57_sigmas[i]) >= float(b57_sigmas[i + 1])

    # 3b. Verify DISCARD_PENULTIMATE_SIGMA_SAMPLERS (e.g. uni_pc) still yields (steps + 1) sigmas after dropping penultimate
    unipc_sigmas = hades.build_anima_sigmas(m, "normal", 20, denoise=1.0, flow_shift=3.0, sampler_name="uni_pc")
    euler_sigmas = hades.build_anima_sigmas(m, "normal", 20, denoise=1.0, flow_shift=3.0, sampler_name="euler")
    assert len(unipc_sigmas) == 21 and len(euler_sigmas) == 21
    assert float(unipc_sigmas[-2]) > float(euler_sigmas[-2]), "Expected uni_pc penultimate sigma to be discarded!"

    # 4. Test HaDeSAnimaKSampler node execution with mock comfy.sample.sample
    orig_sample = comfy.sample.sample
    comfy.sample.sample = lambda model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kw: latent_image + 0.1
    try:
        ks = hades.HaDeSAnimaKSampler()
        lat_in = {"samples": torch.zeros(1, 16, 128, 128)}
        # Test both concise modern profile name and legacy string compatibility
        out_lat, dec_img, info = ks.sample_anima(
            m, [], [], lat_in,
            "Anima Lightning TDM",
            42, 32, 4.5, "er_sde", "normal", 1.0, 3.0, vae=DummyVAE(16)
        )
        assert tuple(out_lat["samples"].shape) == (1, 16, 128, 128)
        assert tuple(dec_img.shape) == (1, 1024, 1024, 3)
        assert "anima_lightning_tdm" in info and "Steps: 4" in info

        # Verify legacy workflow string
        out_lat2, dec_img2, info2 = ks.sample_anima(
            m, [], [], lat_in,
            "Anima Lightning TDM (euler | 4 steps | CFG 1.0 | anima_lightning_tdm)",
            42, 32, 4.5, "er_sde", "normal", 1.0, 3.0, vae=DummyVAE(16)
        )
        assert "Steps: 4" in info2
    finally:
        comfy.sample.sample = orig_sample
    print("[OK] Test 11: HaDeSAnimaKSampler (multiplier=1.0 shift, anima_beta57, anima_lightning_tdm) verified")


def test_12_anima_dual_pass_ksampler_16ch_upscale():
    torch.manual_seed(77)
    base_lat = {"samples": torch.randn(1, 16, 128, 128) * 1.35 + 0.25}

    # 1. Verify arbitrary scale factor (1.37x on 1024x1024 = 1402.88px) snaps strictly to 32px grid (1408x1408px = 176x176 latent)
    up_lat, (w1, h1), (w2, h2) = hades.upscale_anima_latent_32px(
        base_lat, 1.37, "Latent 16ch Variance-Preserving Bicubic"
    )
    assert (w1, h1) == (1024, 1024)
    assert (w2, h2) == (1408, 1408) and w2 % 32 == 0 and h2 % 32 == 0
    assert tuple(up_lat["samples"].shape) == (1, 16, 1408 // 8, 1408 // 8)

    # Verify per-channel variance preservation vs naive bicubic damping
    naive_up = torch.nn.functional.interpolate(base_lat["samples"], size=(176, 176), mode="bicubic", align_corners=False)
    orig_std = base_lat["samples"].std().item()
    naive_std = naive_up.std().item()
    preserved_std = up_lat["samples"].std().item()
    assert abs(preserved_std - orig_std) < abs(naive_std - orig_std), (
        f"Expected variance-preserving upscale std ({preserved_std:.3f}) to be closer to orig ({orig_std:.3f}) than naive ({naive_std:.3f})"
    )

    # 2. Test HaDeSAnimaDualPassKSampler execution
    orig_sample = comfy.sample.sample
    comfy.sample.sample = lambda model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kw: latent_image
    try:
        m1 = DummyModelPatcher({"blocks.0.weight": torch.ones(8, 8)})
        m2 = DummyModelPatcher({"blocks.0.weight": torch.ones(8, 8) * 2.0})
        dual = hades.HaDeSAnimaDualPassKSampler()
        assert dual.RETURN_NAMES == ("final_latent", "pass1_latent", "decoded_image", "info")
        final_lat, p1_lat, dec_img, info = dual.sample_dual_pass(
            m1, [], [], base_lat,
            "HiRes Standard 1.5x (er_sde 30s -> er_sde + beta57 16s @ 0.42 denoise)",
            100,
            30, 4.5, "er_sde", "normal", 1.0, 3.0,
            "Latent 16ch Variance-Preserving Bicubic", 1.50,
            16, 4.0, "er_sde", "anima_beta57", 0.42, 3.6,
            model_pass2=m2, vae=DummyVAE(16),
        )
        assert tuple(p1_lat["samples"].shape) == (1, 16, 128, 128)
        assert tuple(final_lat["samples"].shape) == (1, 16, 192, 192)
        assert tuple(dec_img.shape) == (1, 1536, 1536, 3)
        assert "HybridModel=Yes" in info and "1536x1536px" in info
    finally:
        comfy.sample.sample = orig_sample
    print("[OK] Test 12: HaDeSAnimaDualPassKSampler (32px grid snap, 16ch variance restoration & hybrid model handoff) verified")


def test_13_anima_triple_pass_progressive_ksampler():
    orig_sample = comfy.sample.sample
    comfy.sample.sample = lambda model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kw: latent_image
    try:
        m = DummyModelPatcher({"blocks.0.weight": torch.ones(8, 8)})
        triple = hades.HaDeSAnimaTriplePassKSampler()
        assert triple.RETURN_NAMES == ("final_latent", "pass1_latent", "pass2_latent", "decoded_image", "info")
        base_lat = {"samples": torch.randn(1, 16, 128, 128)}
        final_lat, s1_lat, s2_lat, dec_img, info = triple.sample_triple_pass(
            m, [], [], base_lat,
            "Progressive Staircase 1.0x -> 1.35x -> 1.75x (Ultra-Res Poster Pipeline)",
            777, "Latent 16ch Variance-Preserving Bicubic",
            28, 4.5, "er_sde", "normal", 1.0, 3.0,
            1.35, 16, 4.0, "er_sde", "beta", 0.48, 3.5,
            1.75, 12, 3.4, "euler", "anima_beta57", 0.26, 4.0,
            vae=DummyVAE(16),
        )
        # 1024 * 1.35 = 1382.4 -> snaps to 1376px (172 latent)
        # 1024 * 1.75 = 1792px (224 latent)
        assert tuple(s1_lat["samples"].shape) == (1, 16, 128, 128)
        assert tuple(s2_lat["samples"].shape) == (1, 16, 1376 // 8, 1376 // 8)
        assert tuple(final_lat["samples"].shape) == (1, 16, 1792 // 8, 1792 // 8)
        assert tuple(dec_img.shape) == (1, 1792, 1792, 3)

        # Verify zero cumulative 32px rounding drift on 1152x896 with scale_stage2=1.35, scale_stage3=2.00
        # Stage 1: 1152x896 -> Stage 2: 1568x1216 -> Stage 3 must be exact 2.0x of Stage 1 (2304x1792, NOT 2336x1792)
        lat_1152x896 = {"samples": torch.randn(1, 16, 896 // 8, 1152 // 8)}
        final_2x, _, s2_135, _, info_2x = triple.sample_triple_pass(
            m, [], [], lat_1152x896,
            "Manual (Use Widget Settings)",
            777, "Pixel-Space VAE Round-Trip (Lanczos 32px)",
            28, 4.5, "er_sde", "normal", 1.0, 3.0,
            1.35, 16, 4.0, "er_sde", "beta", 0.48, 3.5,
            2.00, 12, 3.4, "euler", "anima_beta57", 0.26, 4.0,
            vae=None,
        )
        assert tuple(s2_135["samples"].shape) == (1, 16, 1216 // 8, 1568 // 8)
        assert tuple(final_2x["samples"].shape) == (1, 16, 1792 // 8, 2304 // 8), (
            f"Expected exact 2304x1792px (288x224 latent), got {final_2x['samples'].shape}"
        )
        assert "Auto-Fallback: No VAE" in info_2x

        # Verify 64-channel VAE trap raises ValueError
        blocked = False
        try:
            triple.sample_triple_pass(
                m, [], [], base_lat,
                "Progressive Staircase 1.0x -> 1.35x -> 1.75x (Ultra-Res Poster Pipeline)",
                777, "Latent 16ch Variance-Preserving Bicubic",
                28, 4.5, "er_sde", "normal", 1.0, 3.0,
                1.35, 16, 4.0, "er_sde", "beta", 0.48, 3.5,
                1.75, 12, 3.4, "euler", "anima_beta57", 0.26, 4.0,
                vae=DummyVAE(64),
            )
        except ValueError as e:
            blocked = "64" in str(e) and "16-channel" in str(e)
        assert blocked, "Expected 64-channel VAE to be blocked by HaDeSAnimaTriplePassKSampler!"
    finally:
        comfy.sample.sample = orig_sample
    print("[OK] Test 13: HaDeSAnimaTriplePassKSampler (3-stage staircase 1024->1376->1792px & 64ch VAE block) verified")


def test_14_anima_model_loader_diffusers_and_external_adapter():
    diff_sd = {
        "transformer.transformer_blocks.0.attn1.to_q.weight": torch.ones(16, 16),
        "transformer.transformer_blocks.0.attn1.to_out.0.weight": torch.ones(16, 16),
        "transformer.transformer_blocks.0.attn2.norm_q.weight": torch.ones(16),
        "transformer.transformer_blocks.0.ff.net.0.proj.weight": torch.ones(32, 16),
        "transformer.transformer_blocks.0.norm1.linear_1.weight": torch.ones(16, 16),
        "transformer.time_embed.t_embedder.linear_1.weight": torch.ones(16, 16),
        "transformer.time_embed.emb.timestep_embedder.linear_2.weight": torch.ones(16, 16),
        "transformer.time_embed.norm.weight": torch.ones(16),
        "transformer.patch_embed.proj.weight": torch.ones(16, 68),
        "transformer.norm_out.linear_1.weight": torch.ones(16, 16),
        "transformer.proj_out.weight": torch.ones(64, 16),
    }
    ext_adapter_sd = {
        "blocks.0.cross_attn.q_proj.weight": torch.ones(16, 16),
        "blocks.0.self_attn.o_proj.weight": torch.ones(16, 16),
        "embed.weight": torch.ones(32, 16),
        "out_proj.weight": torch.ones(16, 16),
        "norm.weight": torch.ones(16),
    }

    remapped_sd, count_remap = hades.remap_anima_diffusers_model_sd(diff_sd)
    assert count_remap == len(diff_sd)
    assert "blocks.0.self_attn.q_proj.weight" in remapped_sd
    assert "blocks.0.self_attn.output_proj.weight" in remapped_sd
    assert "blocks.0.cross_attn.q_norm.weight" in remapped_sd
    assert "blocks.0.mlp.layer1.weight" in remapped_sd
    assert "blocks.0.adaln_modulation_self_attn.1.weight" in remapped_sd
    assert "t_embedder.1.linear_1.weight" in remapped_sd
    assert "t_embedder.1.linear_2.weight" in remapped_sd
    assert "t_embedding_norm.weight" in remapped_sd
    assert "x_embedder.proj.1.weight" in remapped_sd
    assert "final_layer.adaln_modulation.1.weight" in remapped_sd
    assert "final_layer.linear.weight" in remapped_sd

    merged_sd, count_inj = hades.inject_external_llm_adapter_sd(remapped_sd, ext_adapter_sd)
    assert count_inj == 5
    assert "llm_adapter.blocks.0.cross_attn.q_proj.weight" in merged_sd
    assert "llm_adapter.blocks.0.self_attn.o_proj.weight" in merged_sd
    assert "llm_adapter.embed.weight" in merged_sd
    assert hades.HaDeSAnimaModelLoader.RETURN_NAMES == ("model", "info")
    print("[OK] Test 14: HaDeSAnimaModelLoader (24-rule Diffusers checkpoint remap & external llm_adapter injection) verified")


def test_15_anima_clip_loader_qwen3_validation():
    # 1. Valid Qwen3-0.6B (1024-dim + q_norm)
    qwen3_sd = {
        "model.layers.0.post_attention_layernorm.weight": torch.ones(1024),
        "model.layers.0.input_layernorm.weight": torch.ones(1024),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(64),
    }
    arch, dim, layers = hades.validate_anima_te_state_dict(qwen3_sd, "qwen_3_06b_base.safetensors")
    assert dim == 1024 and "Qwen3-0.6B" in arch

    # 2. Block T5-XXL encoder
    t5_sd = {"encoder.block.0.layer.1.DenseReluDense.wi_1.weight": torch.ones(10240, 4096)}
    t5_blocked = False
    try:
        hades.validate_anima_te_state_dict(t5_sd, "t5xxl_fp16.safetensors")
    except ValueError as e:
        t5_blocked = "T5" in str(e) and "qwen_3_06b_base" in str(e)
    assert t5_blocked, "Expected T5 text encoder to be blocked by HaDeSAnimaCLIPLoader!"

    # 3. Block 3584-dim Qwen2.5-7B encoder
    qwen7b_sd = {
        "model.layers.0.post_attention_layernorm.weight": torch.ones(3584),
        "model.layers.0.input_layernorm.weight": torch.ones(3584),
    }
    q7b_blocked = False
    try:
        hades.validate_anima_te_state_dict(qwen7b_sd, "qwen_2.5_vl_7b.safetensors")
    except ValueError as e:
        q7b_blocked = "3584" in str(e)
    assert q7b_blocked, "Expected 3584-dim Qwen2.5-7B encoder to be blocked by HaDeSAnimaCLIPLoader!"
    assert hades.HaDeSAnimaCLIPLoader.RETURN_NAMES == ("clip", "info")
    print("[OK] Test 15: HaDeSAnimaCLIPLoader (Qwen3-0.6B 1024d validation & T5/Qwen2.5-7B block) verified")


def test_16_anima_vae_loader_64ch_pre_load_block():
    # 1. Valid 16-channel Wan 2.1 AutoencoderKLQwenImage state_dict
    wan21_16ch_sd = {
        "decoder.middle.0.residual.0.gamma": torch.ones(384, 1, 1, 1),
        "decoder.head.0.gamma": torch.ones(96, 1, 1, 1),
        "decoder.head.2.weight": torch.ones(3, 96, 3, 3, 3),
    }
    diag = hades._inspect_and_validate_anima_vae_sd(wan21_16ch_sd, "qwen_image_vae.safetensors")
    assert diag["latent_channels"] == 16 and diag["spatial_compression"] == 8

    # 2. Invalid 64-channel Qwen-Image 2.1 (Wan 2.2 layout + temporal_kernel=1)
    qwen21_64ch_sd = {
        "decoder.middle.0.residual.0.gamma": torch.ones(384, 1, 1, 1),
        "decoder.upsamples.0.upsamples.0.residual.2.weight": torch.ones(384, 384, 1, 3, 3),
        "decoder.head.2.weight": torch.ones(4, 96, 1, 3, 3),
    }
    trap_blocked = False
    try:
        hades._inspect_and_validate_anima_vae_sd(qwen21_64ch_sd, "qwen_image_vae_21.safetensors")
    except ValueError as e:
        trap_blocked = "64-channel" in str(e) and "CRITICAL TRAP BLOCKED" in str(e)
    assert trap_blocked, "Expected 64-channel Qwen-Image 2.1 VAE state_dict to be blocked pre-instantiation!"
    assert hades.HaDeSAnimaVAELoader.RETURN_NAMES == ("vae", "info")
    print("[OK] Test 16: HaDeSAnimaVAELoader (Pre-instantiation 16ch Wan21 validation & 64ch Qwen-Image 2.1 trap block) verified")


def test_17_anima_smart_vae_decode_and_scribble_shield():
    dec_node = hades.HaDeSAnimaVAEDecode()
    vae_16 = DummyVAE(16, noisy_corner=True)

    # 1. At 1024x1024 (1.05 MP <= 1.35 MP threshold), Auto Smart-Tile uses fast direct decode
    lat_1mp = {"samples": torch.zeros(1, 16, 128, 128)}
    img_1mp, info_1mp = dec_node.decode_anima(
        lat_1mp, vae_16, decode_mode="Auto Smart-Tile (VRAM & HiRes Aware)",
        tile_size_px=512, tile_overlap_px=64, auto_tile_mp_threshold=1.35
    )
    assert vae_16.last_decode_method == "direct"
    assert tuple(img_1mp.shape) == (1, 1024, 1024, 3)

    # 2. At 1536x1536 (2.36 MP > 1.35 MP threshold), Auto Smart-Tile proactively switches to tiled decode
    lat_hires = {"samples": torch.zeros(1, 16, 192, 192)}
    img_hi, info_hi = dec_node.decode_anima(
        lat_hires, vae_16, decode_mode="Auto Smart-Tile (VRAM & HiRes Aware)",
        tile_size_px=512, tile_overlap_px=64, auto_tile_mp_threshold=1.35
    )
    assert vae_16.last_decode_method == "tiled"
    assert tuple(img_hi.shape) == (1, 1536, 1536, 3)
    assert "Auto-Tile (2.36 MP > 1.35 MP threshold)" in info_hi

    # 3. Verify 3D WanVAE (latent_dim=3, 5D [B, T, H, W, 3] output) with 4D input & batch_size=2 round-trip upscale
    wan3d_vae = DummyWan3DVAE(16, noisy_corner=True)
    img_3d, _ = dec_node.decode_anima(
        {"samples": torch.zeros(2, 16, 64, 64)},  # 4D [B=2, 16, 64, 64] fed to 3D WanVAE
        wan3d_vae,
        decode_mode="Direct Full Decode (Max Speed)",
    )
    assert tuple(img_3d.shape) == (2, 512, 512, 3)

    rt_lat, (w0, h0), (w1, h1) = hades.upscale_anima_latent_32px(
        {"samples": torch.zeros(2, 16, 1, 64, 64)},
        1.5,
        "Pixel-Space VAE Round-Trip (Lanczos 32px)",
        vae=wan3d_vae,
    )
    assert (w0, h0) == (512, 512) and (w1, h1) == (768, 768)
    assert tuple(rt_lat["samples"].shape) == (2, 16, 1, 768 // 8, 768 // 8), (
        f"Expected batch_size=2 to be preserved on 3D WanVAE round-trip, got {rt_lat['samples'].shape}"
    )
    print("[OK] Test 17: HaDeSAnimaVAEDecode (Proactive HiRes Auto-Tile, 3D WanVAE 4D/5D & batch=2 round-trip) verified")


def test_18_nag_only_anima_gate():
    nag_node = hades.HaDeSAnimaAttentionGuidance()

    # 1. Non-Anima model + only_anima=True (default) -> safe bypass, no override installed
    m_sd = {"blocks.0.weight": torch.ones(8, 8)}
    m_non_anima = DummyModelPatcher(m_sd, image_model="stable_diffusion_3")
    m_out, info = nag_node.apply_nag(
        m_non_anima, "Turbo / Lightning Negative Enabler", 3.5, 0.45, 2.5, 0.0, 0.70
    )
    assert "Bypassed" in info and "not an Anima model" in info, f"Expected only_anima bypass info, got: {info}"
    assert "optimized_attention_override" not in m_out.model_options.get("transformer_options", {}), (
        "Expected no attention override installed for a non-Anima model!"
    )

    # 2. Non-Anima model + only_anima=False -> override IS installed (experimental force-apply)
    m_forced = DummyModelPatcher(m_sd, image_model="stable_diffusion_3")
    m_out2, info2 = nag_node.apply_nag(
        m_forced, "Turbo / Lightning Negative Enabler", 3.5, 0.45, 2.5, 0.0, 0.70, only_anima=False
    )
    assert "optimized_attention_override" in m_out2.model_options.get("transformer_options", {}), (
        f"Expected override installed with only_anima=False, info: {info2}"
    )

    # 3. Anima model + only_anima=True -> override installed (regression guard)
    m_anima = DummyModelPatcher(m_sd, image_model="anima")
    m_out3, info3 = nag_node.apply_nag(
        m_anima, "Turbo / Lightning Negative Enabler", 3.5, 0.45, 2.5, 0.0, 0.70
    )
    assert "optimized_attention_override" in m_out3.model_options.get("transformer_options", {}), (
        f"Expected override installed for Anima model, info: {info3}"
    )
    print("[OK] Test 18: NAG only_anima gate (default True, non-Anima bypass, force-apply escape hatch) verified")


def test_19_prompt_header_canonical_and_dual_legacy():
    body_prompt = "1girl, soft sunset light, watercolor background"

    # 1. Canonical 3-line header: L1 = single trigger, L2 = title, L3 = body
    canon = hades.format_anima_prompt_text(
        body_prompt, "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Flat Vector / Graphic (Non-Anime Dataset Header)",
        header_mode="Canonical (1 Trigger + Title)",
        header_trigger="ye-pop",
        artwork_title="Sunset Study",
    )
    lines = canon[0].split("\n")
    assert lines[0] == "ye-pop", f"Expected L1='ye-pop', got: {lines[:3]}"
    assert lines[1] == "Sunset Study", f"Expected L2='Sunset Study', got: {lines[:3]}"
    assert "1girl" in lines[2], f"Expected body on L3, got: {lines[:3]}"

    # 2. Canonical header with empty title: L2 omitted, body promoted to L2
    canon_notitle = hades.format_anima_prompt_text(
        body_prompt, "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Flat Vector / Graphic (Non-Anime Dataset Header)",
        header_trigger="deviantart",
        artwork_title="",
    )
    lines_nt = canon_notitle[0].split("\n")
    assert lines_nt[0] == "deviantart" and "1girl" in lines_nt[1], f"Expected [deviantart, body], got: {lines_nt[:3]}"
    assert len(lines_nt) == 2, f"Expected 2-line header without title, got {len(lines_nt)} lines"

    # 3. Dual Legacy mode: ye-pop + deviantart on lines 1-2, body on line 3
    dual = hades.format_anima_prompt_text(
        body_prompt, "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Flat Vector / Graphic (Non-Anime Dataset Header)",
        header_mode="Dual Legacy (ye-pop + deviantart)",
    )
    lines_dl = dual[0].split("\n")
    assert lines_dl[0] == "ye-pop" and lines_dl[1] == "deviantart" and "1girl" in lines_dl[2], (
        f"Expected dual-legacy [ye-pop, deviantart, body], got: {lines_dl[:3]}"
    )

    # 4. Dedup: body already starting with a dataset trigger must NOT be double-injected
    dedup = hades.format_anima_prompt_text(
        "ye-pop, 1girl, soft sunset light",
        "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Flat Vector / Graphic (Non-Anime Dataset Header)",
        header_mode="Canonical (1 Trigger + Title)",
        header_trigger="ye-pop",
        artwork_title="Sunset Study",
    )
    dedup_lines = dedup[0].split("\n")
    assert dedup_lines.count("ye-pop") == 1, f"Expected no double 'ye-pop' injection, got: {dedup_lines[:4]}"

    # 5. Non-header style mode: body untouched (no header lines)
    plain = hades.format_anima_prompt_text(
        body_prompt, "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Cinematic / Detailed Shading",
    )[0]
    assert "\n" not in plain.split(", ")[0] and plain.splitlines()[0].startswith("1girl"), (
        f"Expected header-less single line, got: {plain!r}"
    )

    # 6. Manual 3-line input preservation (case, punctuation, NL sentence untouched)
    manual_input = (
        "ye-pop\n"
        "Sunrise Over Neo-Tokyo (2025 Edition)!\n"
        "A breathtaking view of the futuristic cityscape at dawn, with neon reflections in the rain puddles."
    )
    manual_res = hades.format_anima_prompt_text(
        manual_input,
        "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Standard Anime / Illustration",
        header_mode="Canonical (1 Trigger + Title)",
    )
    manual_lines = manual_res[0].split("\n")
    assert manual_lines[0] == "ye-pop", f"Expected L1='ye-pop', got: {manual_lines[0]}"
    assert manual_lines[1] == "Sunrise Over Neo-Tokyo (2025 Edition)!", (
        f"Expected exact title preservation on L2, got: {manual_lines[1]}"
    )
    assert "A breathtaking view of the futuristic cityscape at dawn, with neon reflections in the rain puddles." in manual_lines[2], (
        f"Expected intact NL sentence on L3, got: {manual_lines[2]}"
    )

    # 7. Error 7 Check: Embedded triggers (e.g. deviantart) in the middle of tag list
    # must be extracted into Line 1 header and stripped from body without injecting duplicate ye-pop
    mid_prompt = "1girl, deviantart, silver hair"
    mid_res = hades.format_anima_prompt_text(
        mid_prompt,
        "Anima Aesthetic (Strip score_* & Slop Guard)",
        "Flat Vector / Graphic (Non-Anime Dataset Header)",
        header_mode="Canonical (1 Trigger + Title)",
        header_trigger="ye-pop",  # default selector, but deviantart is already in the tags!
        artwork_title="",
    )
    mid_lines = mid_res[0].split("\n")
    assert mid_lines[0] == "deviantart", f"Expected detected embedded trigger 'deviantart' on L1, got: {mid_lines[0]}"
    assert "ye-pop" not in mid_res[0], f"Expected redundant 'ye-pop' not to be injected when 'deviantart' was embedded! Got: {mid_res[0]}"
    assert "deviantart" not in mid_lines[-1], f"Expected 'deviantart' stripped from body line, got: {mid_lines[-1]}"
    assert "1girl" in mid_lines[-1] and "silver hair" in mid_lines[-1]

    print("[OK] Test 19: Prompt header (canonical 3-line, title omit, dual legacy, dedup, non-header passthrough, manual 3-line preservation, embedded trigger extraction) verified")


def test_20_corner_shield_batch_isolation():
    shield = hades.HaDeSAnimaCornerScribbleShield()
    # Batch with 2 images: Image 0 is clean; Image 1 has high-frequency watermark in top-left
    img0 = torch.full((1, 128, 128, 3), 0.5, dtype=torch.float32)
    img1 = torch.full((1, 128, 128, 3), 0.5, dtype=torch.float32)
    # Add high-frequency stroke to img1
    img1[0, 2:8, 2:8, :] = 0.0

    batch = torch.cat([img0, img1], dim=0)
    out_img, mask_out, info = shield.shield_corners(
        batch,
        corner_box_percent=0.10,
        detection_sensitivity=0.75,
        active_corners="All 4 Corners",
        heal_mode="Smart Edge-Inward Blend (Auto-Heal)",
        feather_radius=12,
    )

    # Batch isolation: Image 0 must be 100% untouched
    clean_diff = torch.max(torch.abs(out_img[0] - img0[0])).item()
    clean_mask_max = mask_out[0].max().item()
    assert clean_diff == 0.0, f"Expected 0.0 difference on clean batch item, got {clean_diff}"
    assert clean_mask_max == 0.0, f"Expected 0.0 mask on clean batch item, got {clean_mask_max}"

    # Image 1 should be healed
    dirty_mask_max = mask_out[1].max().item()
    assert dirty_mask_max > 0.0, f"Expected non-zero mask on dirty batch item, got {dirty_mask_max}"
    print("[OK] Test 20: Corner Scribble Shield batch independence & zero cross-talk verified")


def test_21_nag_dynamic_shift_propagation():
    nag_node = hades.HaDeSAnimaAttentionGuidance()
    m = DummyModelPatcher({"blocks.0.weight": torch.ones(8, 8)})
    m.add_object_patch("model_sampling", DummyModelSampling(shift=3.0, multiplier=1.0))

    # Apply NAG with 3.0 initial shift
    m_nag, info = nag_node.apply_nag(
        m,
        guidance_preset="Custom",
        nag_scale=3.0,
        nag_alpha=0.35,
        nag_tau=2.5,
        start_percent=0.0,
        end_percent=0.65,
        only_anima=False,
    )
    initial_sigma_end = m_nag.model_options["_hades_nag_state"]["sigma_end"]

    # Clone model with shifted flow (4.5) as done in KSampler
    m_shifted = hades.clone_model_with_anima_shift(m_nag, 4.5)
    shifted_sigma_end = m_shifted.model_options["_hades_nag_state"]["sigma_end"]
    assert shifted_sigma_end > initial_sigma_end, (
        f"Expected shifted sigma_end ({shifted_sigma_end}) > initial ({initial_sigma_end})"
    )

    # Verify override reads dynamically from transformer_options
    override = m_shifted.model_options["transformer_options"]["optimized_attention_override"]
    called = []
    def dummy_func(*args, **kwargs):
        called.append(True)
        return torch.zeros(1, 10, 64)

    # Within shifted window
    q = torch.zeros(1, 10, 64)
    k = torch.zeros(1, 10, 64)
    v = torch.zeros(1, 10, 64)
    t_opts = {
        "sigmas": torch.tensor([shifted_sigma_end + 0.05]),
        "_hades_nag_state": m_shifted.model_options["_hades_nag_state"],
    }
    override(dummy_func, q, k, v, 1, transformer_options=t_opts)
    assert len(called) > 0, "Expected attention override to execute within shifted sigma window"
    print("[OK] Test 21: NAG dynamic sigma shift propagation across cloned patchers verified")


def test_22_anima_suite_telemetry():
    telemetry_node = hades.HaDeSAnimaSuiteTelemetry()
    assert telemetry_node.RETURN_NAMES == ("report", "json_telemetry", "diagnostics"), \
        f"Expected RETURN_NAMES ('report', 'json_telemetry', 'diagnostics'), got {telemetry_node.RETURN_NAMES}"

    # 1. Idle state check
    rep_idle, json_idle, diag_idle = telemetry_node.audit_telemetry(
        "Full Audit Report (Markdown)", "Idle Report"
    )
    import json
    data_idle = json.loads(json_idle)
    assert data_idle["health_score"] == 100
    assert "STANDBY" in data_idle["status"]
    assert "STANDBY" in diag_idle

    # 2. Complete nominal pipeline telemetry
    t_model = "HaDeS Anima Model Loader | Model: Anima HaDeS v2.0 | Dtype: default | Precision: auto | Flow Shift: 3.00 | Remap: 24-Rule Active"
    t_clip = "HaDeS Anima CLIP Loader | CLIP: qwen3508b | Device: default (GPU Dynamic) | Precision: auto | Qwen3-0.6B: 1024d Guard Locked"
    t_vae = "HaDeS Anima VAE Loader | VAE: Wan2_1_VAE_bf16 | Precision: bfloat16 | Latent Channels: 16 (Wan2.1)"
    t_lora = "=== HADES ANIMA UNIVERSAL LORA REMAPPER ===\nSource LoRA: test_style.safetensors\nStrengths: Model +1.00 | CLIP +0.50\nDirty llm_adapter Keys Stripped: 6 | Outliers Clamped: 0"
    t_ksamp = "HaDeS Anima KSampler | Profile: Anima Lightning TDM | Resolution: 1024x1024px | Sampler: euler | Scheduler: anima_lightning_tdm | Steps: 4 | CFG: 1.00 | Denoise: 1.00 | Flow Shift (multiplier=1.0): 3.00"
    t_nag = "HaDeS Anima Attention Guidance | Preset: Turbo | Scale: 3.50 | Alpha: 0.45 | Tau: 2.50 | Window: 0.00 -> 0.70 (Active: True)"
    t_shield = "HaDeS Corner Scribble Shield | Box: 102x102px (10% base) | Mode: Smart Edge-Inward Blend (Auto-Heal) | Active: All 4 Corners | Detected: Clean (Low HF Energy)"
    t_decode = "HaDeS Anima VAE Decode | Output: 1024x1024px (1.05 MP) | Strategy: Direct Full Pass (Tile 512px / Overlap 64px)"

    rep_full, json_full, diag_full = telemetry_node.audit_telemetry(
        report_format="Full Audit Report (Markdown)",
        report_title="HaDeS Flight Test Nominal",
        user_notes="High quality anime render test",
        info_1=t_model,
        info_2=t_clip,
        info_3=t_vae,
        info_4=t_lora,
        info_5=t_ksamp,
        info_6=t_nag,
        info_7=t_shield,
        info_8=t_decode,
    )

    data_full = json.loads(json_full)
    assert data_full["health_score"] == 100, f"Expected 100 health score on nominal test, got {data_full['health_score']}"
    assert data_full["status"] == "100% NOMINAL"
    assert len(data_full["diagnostics"]["passed_checks"]) >= 5
    assert len(data_full["diagnostics"]["issues"]) == 0
    assert "Flow Shift Synchronized: 3.00" in rep_full
    assert "1024x1024px" in rep_full
    assert "Wan2.1 native 16-channel" in rep_full
    assert "text conditioner bridge protected" in rep_full
    assert "High quality anime render test" in rep_full

    # 3. Test divergence / warning detection (Mismatch shift + non-32px resolution)
    t_model_div = "HaDeS Anima Model Loader | Model: Anima HaDeS v2.0 | Flow Shift: 3.00"
    t_ksamp_div = "HaDeS Anima KSampler | Resolution: 1000x1000px | Steps: 20 | CFG: 4.50 | Flow Shift (multiplier=1.0): 1.00"
    rep_div, json_div, diag_div = telemetry_node.audit_telemetry(
        report_format="Compact Flight Summary",
        report_title="HaDeS Flight Test Warning",
        info_1=t_model_div,
        info_2=t_ksamp_div,
    )
    data_div = json.loads(json_div)
    assert data_div["health_score"] < 100, f"Expected degraded score on mismatch test, got {data_div['health_score']}"
    assert any("Flow Shift Divergence" in iss["message"] for iss in data_div["diagnostics"]["issues"])
    assert any("not 32px macroblock aligned" in iss["message"] for iss in data_div["diagnostics"]["issues"])
    assert "Flow Shift Divergence" in diag_div or "warnings" in diag_div

    # 4. Test format variations
    rep_diag, _, _ = telemetry_node.audit_telemetry("Diagnostics Only", "Diag Mode", info_1=t_model)
    assert "Passed Validations" in rep_diag

    rep_raw_json, _, _ = telemetry_node.audit_telemetry("JSON Raw Data", "JSON Mode", info_1=t_model)
    data_raw = json.loads(rep_raw_json)
    assert data_raw["report_title"] == "JSON Mode"

    print("[OK] Test 22: HaDeS Anima Suite Telemetry (Multi-Stream Aggregation, Diagnostics & Health Scoring, JSON & Formats) verified")


if __name__ == "__main__":
    test_1_dynamic_topology_and_diffusers_keys()
    test_2_qr_svd_accuracy_and_speed()
    test_3_reactive_predictive_model_merger_and_llm_adapter_lock()
    test_4_subspace_multi_lora_per_secondary_slot()
    test_5_cross_format_diffusers_kohya_lora_svd_merge()
    test_6_clip_proportional_36_layer_depth()
    test_7_prompt_studio_and_aesthetic_slop_guard()
    test_8_nag_attention_guidance()
    test_9_latent_pipeline_guard()
    test_10_corner_scribble_shield()
    test_11_anima_single_pass_ksampler_and_schedulers()
    test_12_anima_dual_pass_ksampler_16ch_upscale()
    test_13_anima_triple_pass_progressive_ksampler()
    test_14_anima_model_loader_diffusers_and_external_adapter()
    test_15_anima_clip_loader_qwen3_validation()
    test_16_anima_vae_loader_64ch_pre_load_block()
    test_17_anima_smart_vae_decode_and_scribble_shield()
    test_18_nag_only_anima_gate()
    test_19_prompt_header_canonical_and_dual_legacy()
    test_20_corner_shield_batch_isolation()
    test_21_nag_dynamic_shift_propagation()
    test_22_anima_suite_telemetry()
    print("\nALL 22 HADES ANIMA SUITE TEST SUITES PASSED!")
