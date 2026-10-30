import math
import torch
import torch.nn.functional as F
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_sampling
import comfy.model_management

try:
    import latent_preview
except ImportError:
    latent_preview = None

try:
    import scipy.stats
except ImportError:
    scipy = None


ANIMA_SAMPLER_LIST = list(getattr(
    comfy.samplers,
    "SAMPLER_NAMES",
    getattr(comfy.samplers, "KSAMPLER_NAMES", [
        "er_sde", "euler", "euler_ancestral", "dpmpp_2m_sde_gpu", "dpmpp_2m", "heun", "uni_pc"
    ]),
))
if "er_sde" in ANIMA_SAMPLER_LIST:
    # Place Anima's top recommended samplers at the top of the list for convenience
    preferred = ["er_sde", "euler", "euler_ancestral", "dpmpp_2m_sde_gpu", "dpmpp_2m"]
    ANIMA_SAMPLER_LIST = [s for s in preferred if s in ANIMA_SAMPLER_LIST] + [
        s for s in ANIMA_SAMPLER_LIST if s not in preferred
    ]

DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(getattr(
    comfy.samplers,
    "DISCARD_PENULTIMATE_SIGMA_SAMPLERS",
    {"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"},
))

BASE_SCHEDULER_LIST = list(getattr(comfy.samplers, "SCHEDULER_NAMES", [
    "normal", "beta", "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform"
]))
ANIMA_SCHEDULER_LIST = ["normal", "beta", "anima_beta57", "anima_lightning_tdm"] + [
    s for s in BASE_SCHEDULER_LIST if s not in ("normal", "beta", "anima_beta57", "anima_lightning_tdm")
]

ANIMA_UPSCALE_METHODS = [
    "Latent 16ch Variance-Preserving Bicubic",
    "Latent 16ch Sharp Bislerp",
    "Pixel-Space VAE Round-Trip (Lanczos 32px)",
    "Direct Latent Pass-Through (1.0x)",
]


def _validate_anima_16ch_vae_optional(vae):
    if vae is None:
        return
    latent_ch = getattr(vae, "latent_channels", None)
    if latent_ch is None and hasattr(vae, "first_stage_model"):
        latent_ch = getattr(vae.first_stage_model, "z_dim", None)
    if latent_ch is not None and int(latent_ch) != 16:
        raise ValueError(
            f"[HaDeS Anima KSampler] Invalid VAE detected ({latent_ch} latent channels). "
            "Anima strictly requires the 16-channel AutoencoderKLQwenImage (qwen_image_vae.safetensors, ~254 MB). "
            "Do NOT connect the 64-channel Qwen-Image 2.1 VAE."
        )


def _normalize_decoded_image_4d(decoded: torch.Tensor) -> torch.Tensor:
    """Normalize a VAE decoded tensor (4D or 5D from 3D WanVAE) into 4D [N, H, W, C]."""
    if not isinstance(decoded, torch.Tensor) or decoded.ndim != 5:
        return decoded
    # Standard ComfyUI 3D VAE.decode output after movedim(1, -1) is [B, T, H, W, C]
    if decoded.shape[-1] in (1, 3, 4):
        return decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    # Fallback if a custom/mock VAE returns [B, H, T, W, C] or [B, C, 1, H, W]
    if decoded.shape[2] == 1:
        return decoded[:, :, 0, :, :]
    return decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])


def _decode_or_placeholder(vae, latent_tensor):
    if vae is not None:
        _validate_anima_16ch_vae_optional(vae)
        lat_in = latent_tensor
        if getattr(vae, "latent_dim", None) == 3 and isinstance(lat_in, torch.Tensor) and lat_in.ndim == 4:
            lat_in = lat_in.unsqueeze(2)
        decoded = vae.decode(lat_in)
        return _normalize_decoded_image_4d(decoded)
    b = int(latent_tensor.shape[0]) if isinstance(latent_tensor, torch.Tensor) and latent_tensor.ndim >= 1 else 1
    return torch.zeros((b, 64, 64, 3), dtype=torch.float32, device="cpu")


def clone_model_with_anima_shift(model, flow_shift: float):
    """Clone model and apply Rectified Flow shift while preserving Anima's multiplier=1.0."""
    m = model.clone()
    shift_val = max(0.1, float(flow_shift))
    orig_ms = m.get_model_object("model_sampling")

    if orig_ms is not None and hasattr(orig_ms, "shift"):
        orig_shift = float(getattr(orig_ms, "shift", 3.0))
        if abs(orig_shift - shift_val) < 1e-4:
            return m

    multiplier = float(getattr(orig_ms, "multiplier", 1.0)) if orig_ms is not None else 1.0
    inner_cfg = getattr(getattr(m, "model", None), "model_config", None)

    if (
        inner_cfg is not None
        and hasattr(inner_cfg, "sampling_settings")
        and hasattr(comfy.model_sampling, "ModelSamplingDiscreteFlow")
        and hasattr(comfy.model_sampling, "CONST")
    ):
        class _AnimaFlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass

        new_ms = _AnimaFlowSampling(inner_cfg)
        new_ms.set_parameters(shift=shift_val, multiplier=multiplier)
        if orig_ms is not None and hasattr(orig_ms, "noise_scale"):
            new_ms.set_noise_scale(orig_ms.noise_scale)
        if hasattr(m, "add_object_patch"):
            m.add_object_patch("model_sampling", new_ms)
        else:
            m._hades_model_sampling = new_ms
    elif orig_ms is not None:
        import copy
        new_ms = copy.deepcopy(orig_ms)
        if hasattr(new_ms, "set_parameters"):
            new_ms.set_parameters(shift=shift_val, multiplier=multiplier)
        else:
            new_ms.shift = shift_val
            new_ms.multiplier = multiplier
        if hasattr(m, "add_object_patch"):
            m.add_object_patch("model_sampling", new_ms)
        else:
            m._hades_model_sampling = new_ms

    # Synchronize any upstream HaDeS NAG sigma window with the newly shifted model_sampling
    nag_state = getattr(m, "model_options", {}).get("_hades_nag_state")
    if isinstance(nag_state, dict):
        active_ms = m.get_model_object("model_sampling")
        if active_ms is not None and hasattr(active_ms, "percent_to_sigma"):
            nag_state = nag_state.copy()
            nag_state["sigma_start"] = float(active_ms.percent_to_sigma(nag_state["start_percent"]))
            nag_state["sigma_end"] = float(active_ms.percent_to_sigma(nag_state["end_percent"]))
            m.model_options["_hades_nag_state"] = nag_state
            if "transformer_options" in m.model_options:
                m.model_options["transformer_options"] = m.model_options["transformer_options"].copy()
                m.model_options["transformer_options"]["_hades_nag_state"] = nag_state

    return m


def _resolve_flow_shift(model, flow_shift: float) -> float:
    """Shift a pass will actually sample with: the widget value, or the model's own shift when the widget is 0 (inherit)."""
    if float(flow_shift) > 0.0:
        return float(flow_shift)
    return float(getattr(model.get_model_object("model_sampling"), "shift", 3.0))


def _beta_ppf_approx(q: float, alpha: float = 0.5, beta: float = 0.7) -> float:
    """Fast and accurate numerical inverse of Regularized Incomplete Beta I_x(alpha, beta) = q."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    if scipy is not None and hasattr(scipy, "stats"):
        return float(scipy.stats.beta.ppf(q, alpha, beta))

    # Singularity-free trigonometric quadrature t = sin^2(theta) on [0, pi/2]
    n_grid = 1024
    theta = (torch.arange(n_grid, dtype=torch.float64) + 0.5) * (math.pi / (2.0 * n_grid))
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    w = (sin_t ** (2.0 * alpha - 1.0)) * (cos_t ** (2.0 * beta - 1.0))
    cdf = torch.cumsum(w, dim=0)
    cdf = cdf / cdf[-1]
    idx = int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)).clamp(0, n_grid - 1).item())
    return float((sin_t[idx] ** 2).item())


def _time_snr_shift_scalar(shift: float, t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if abs(shift - 1.0) < 1e-6:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def build_anima_sigmas(
    model,
    scheduler_name: str,
    steps: int,
    denoise: float = 1.0,
    flow_shift: float = 3.0,
    sampler_name: str = None,
) -> torch.Tensor:
    """Compute sigma schedule supporting Anima's custom anima_beta57 (0.5, 0.7) and anima_lightning_tdm."""
    steps = max(1, int(steps))
    denoise = max(0.0, min(1.0, float(denoise)))
    if denoise <= 0.0:
        return torch.FloatTensor([])

    discard_penultimate = sampler_name in DISCARD_PENULTIMATE_SIGMA_SAMPLERS
    req_steps = (steps + 1) if discard_penultimate else steps

    total_steps = req_steps if denoise >= 0.9999 else max(req_steps, int(round(req_steps / denoise)))
    ms = model.get_model_object("model_sampling")
    shift = float(getattr(ms, "shift", flow_shift)) if ms is not None else float(flow_shift)

    if scheduler_name == "anima_lightning_tdm":
        # Exact TDM schedule from distilled_generation_config.json:
        #   raw_sigmas: [1.0, 0.75, 0.5, 0.25]
        #   effective (shift=3.0): [1.0, 0.9, 0.75, 0.5, 0.0]
        if total_steps == 4:
            raw_t = [1.0, 0.75, 0.5, 0.25]
        else:
            raw_t = [max(0.0, min(0.999, 1.0 - (i / float(total_steps)) - 0.001)) for i in range(total_steps)]
        sigs = [_time_snr_shift_scalar(shift, t) for t in raw_t] + [0.0]
        full_sigmas = torch.tensor(sigs, dtype=torch.float32)

    elif scheduler_name == "anima_beta57":
        # Beta(alpha=0.5, beta=0.7) schedule dedicating higher step density to low-noise texture refinement
        alpha_b, beta_b = 0.50, 0.70
        sigs = []
        last_s = -1.0
        for i in range(total_steps):
            u = 1.0 - (i / float(total_steps))
            t_beta = _beta_ppf_approx(u, alpha_b, beta_b)
            s_val = _time_snr_shift_scalar(shift, t_beta)
            if len(sigs) == 0 or abs(s_val - last_s) > 1e-6:
                sigs.append(s_val)
                last_s = s_val
        sigs.append(0.0)
        full_sigmas = torch.tensor(sigs, dtype=torch.float32)

    else:
        if (
            ms is not None
            and hasattr(ms, "timestep")
            and hasattr(ms, "sigmas")
            and hasattr(comfy.samplers, "calculate_sigmas")
            and scheduler_name in BASE_SCHEDULER_LIST
        ):
            full_sigmas = comfy.samplers.calculate_sigmas(ms, scheduler_name, total_steps).float().cpu()
        else:
            raw_t = [1.0 - (i / float(total_steps)) for i in range(total_steps)]
            sigs = [_time_snr_shift_scalar(shift, t) for t in raw_t] + [0.0]
            full_sigmas = torch.tensor(sigs, dtype=torch.float32)

    if denoise < 0.9999 and full_sigmas.numel() > (req_steps + 1):
        full_sigmas = full_sigmas[-(req_steps + 1):]

    if discard_penultimate and full_sigmas.numel() > 2:
        full_sigmas = torch.cat([full_sigmas[:-2], full_sigmas[-1:]])

    return full_sigmas


def _bislerp_latent_2d(samples: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """High-sharpness bicubic + spherical norm restoration for 16-channel Wan21 latents."""
    orig_ndim = samples.ndim
    if orig_ndim == 5:
        b, c, t, h, w = samples.shape
        x = samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    else:
        x = samples

    x_f32 = x.float()
    # Preserve radial magnitude map across the 16 channels
    norms = torch.linalg.vector_norm(x_f32, dim=1, keepdim=True).clamp(min=1e-6)
    unit_dirs = x_f32 / norms

    up_dirs = F.interpolate(unit_dirs, size=(target_h, target_w), mode="bicubic", align_corners=False)
    up_dirs = up_dirs / torch.linalg.vector_norm(up_dirs, dim=1, keepdim=True).clamp(min=1e-6)
    up_norms = F.interpolate(norms, size=(target_h, target_w), mode="bicubic", align_corners=False)
    out = (up_dirs * up_norms).to(dtype=samples.dtype)

    if orig_ndim == 5:
        out = out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)
    return out


def upscale_anima_latent_32px(
    latent_dict: dict,
    scale_factor: float,
    upscale_method: str,
    vae=None,
    base_px_dims: tuple[int, int] | None = None,
) -> tuple[dict, tuple[int, int], tuple[int, int]]:
    """
    Upscale a 16-channel Anima latent while snapping target pixel resolution strictly to the 32px grid
    (meaning latent H and W are strictly multiples of 4) and preserving per-channel variance.
    If base_px_dims=(w1, h1) is supplied, scale_factor is applied directly to (w1, h1) to prevent
    cumulative multi-stage 32px rounding drift.
    Returns: (new_latent_dict, (orig_w_px, orig_h_px), (new_w_px, new_h_px))
    """
    samples = latent_dict["samples"]
    orig_h_lat = int(samples.shape[-2])
    orig_w_lat = int(samples.shape[-1])
    orig_h_px = orig_h_lat * 8
    orig_w_px = orig_w_lat * 8

    ref_w_px, ref_h_px = base_px_dims if base_px_dims is not None else (orig_w_px, orig_h_px)
    scale = max(0.5, float(scale_factor))

    # Snap target pixel dimensions to exact multiples of 32 (minimum 256px)
    target_w_px = max(256, int(round((ref_w_px * scale) / 32.0)) * 32)
    target_h_px = max(256, int(round((ref_h_px * scale) / 32.0)) * 32)

    if (
        upscale_method == "Direct Latent Pass-Through (1.0x)"
        or (target_w_px == orig_w_px and target_h_px == orig_h_px)
    ):
        out_dict = latent_dict.copy()
        out_dict["samples"] = samples.clone()
        return out_dict, (orig_w_px, orig_h_px), (orig_w_px, orig_h_px)

    target_w_lat = target_w_px // 8
    target_h_lat = target_h_px // 8

    if upscale_method == "Pixel-Space VAE Round-Trip (Lanczos 32px)" and vae is not None:
        _validate_anima_16ch_vae_optional(vae)
        orig_ndim = samples.ndim
        lat_in = samples.unsqueeze(2) if (getattr(vae, "latent_dim", None) == 3 and orig_ndim == 4) else samples
        decoded = _normalize_decoded_image_4d(vae.decode(lat_in))
        # decoded is [N, H, W, C] -> [N, C, H, W] for common_upscale
        img_bchw = decoded.movedim(-1, 1)
        upscaled_bchw = comfy.utils.common_upscale(img_bchw, target_w_px, target_h_px, "lanczos", "disabled")
        upscaled_bhwc = upscaled_bchw.movedim(1, -1)[:, :, :, :3]
        if (
            upscaled_bhwc.shape[0] > 1
            and getattr(vae, "latent_dim", None) == 3
            and not getattr(vae, "not_video", False)
        ):
            encoded = torch.cat([vae.encode(upscaled_bhwc[i : i + 1]) for i in range(upscaled_bhwc.shape[0])], dim=0)
        else:
            encoded = vae.encode(upscaled_bhwc)
        if orig_ndim == 5 and encoded.ndim == 4:
            encoded = encoded.unsqueeze(2)
        elif orig_ndim == 4 and encoded.ndim == 5 and encoded.shape[2] == 1 and getattr(vae, "latent_dim", None) != 3:
            encoded = encoded.squeeze(2)
        out_dict = latent_dict.copy()
        out_dict["samples"] = encoded
        out_dict.pop("noise_mask", None)
        return out_dict, (orig_w_px, orig_h_px), (target_w_px, target_h_px)

    if upscale_method == "Latent 16ch Sharp Bislerp":
        up_samples = _bislerp_latent_2d(samples, target_h_lat, target_w_lat)
    else:
        # Latent 16ch Variance-Preserving Bicubic
        orig_ndim = samples.ndim
        if orig_ndim == 5:
            b, c, t, h, w = samples.shape
            x_in = samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        else:
            x_in = samples

        x_f32 = x_in.float()
        spatial_dims = (-2, -1)
        orig_mean = x_f32.mean(dim=spatial_dims, keepdim=True)
        orig_std = x_f32.std(dim=spatial_dims, keepdim=True, unbiased=False).clamp(min=1e-6)

        up_f32 = F.interpolate(x_f32, size=(target_h_lat, target_w_lat), mode="bicubic", align_corners=False)
        up_mean = up_f32.mean(dim=spatial_dims, keepdim=True)
        up_std = up_f32.std(dim=spatial_dims, keepdim=True, unbiased=False).clamp(min=1e-6)

        # Restore 92% of original per-channel variance to eliminate bicubic high-frequency damping
        tau = 0.92
        scale_gain = tau * (orig_std / up_std) + (1.0 - tau)
        restored = orig_mean + (up_f32 - up_mean) * scale_gain
        up_samples = restored.to(dtype=samples.dtype)

        if orig_ndim == 5:
            up_samples = up_samples.reshape(b, t, c, target_h_lat, target_w_lat).permute(0, 2, 1, 3, 4)

    out_dict = latent_dict.copy()
    out_dict["samples"] = up_samples
    if "noise_mask" in out_dict and isinstance(out_dict["noise_mask"], torch.Tensor):
        nm = out_dict["noise_mask"]
        if nm.ndim == 3:
            nm = nm.unsqueeze(1)
        out_dict["noise_mask"] = F.interpolate(nm.float(), size=(target_h_lat, target_w_lat), mode="bilinear", align_corners=False)

    return out_dict, (orig_w_px, orig_h_px), (target_w_px, target_h_px)


def run_anima_ksampler_pass(
    model,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent_dict: dict,
    denoise: float = 1.0,
    flow_shift: float = 3.0,
    disable_noise: bool = False,
) -> tuple[dict, torch.Tensor]:
    """Execute a single Anima Rectified Flow sampling pass with multiplier=1.0 shift and custom sigmas.
    flow_shift <= 0 keeps the shift already set on the model (Model Loader, ModelSampling* nodes)."""
    patched_model = clone_model_with_anima_shift(model, flow_shift) if float(flow_shift) > 0.0 else model

    latent_image = latent_dict["samples"]
    if hasattr(comfy.sample, "fix_empty_latent_channels"):
        try:
            latent_image = comfy.sample.fix_empty_latent_channels(
                patched_model,
                latent_image,
                latent_dict.get("downscale_ratio_spacial", None),
                latent_dict.get("downscale_ratio_temporal", None),
            )
        except Exception:
            pass

    if disable_noise:
        noise = torch.zeros_like(latent_image, device="cpu")
    else:
        batch_inds = latent_dict.get("batch_index", None)
        if hasattr(comfy.sample, "prepare_noise"):
            noise = comfy.sample.prepare_noise(latent_image, int(seed), batch_inds)
        else:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            noise = torch.randn(latent_image.shape, generator=gen, dtype=torch.float32, device="cpu").to(dtype=latent_image.dtype)

    noise_mask = latent_dict.get("noise_mask", None)
    sigmas = build_anima_sigmas(
        patched_model, scheduler, steps, denoise=denoise, flow_shift=flow_shift, sampler_name=sampler_name
    )

    # Map custom scheduler name to a valid base scheduler name for KSampler.__init__, while passing our exact sigmas
    fallback_sched = scheduler if scheduler in BASE_SCHEDULER_LIST else "normal"
    callback = None
    if latent_preview is not None and hasattr(latent_preview, "prepare_callback"):
        try:
            callback = latent_preview.prepare_callback(patched_model, max(1, len(sigmas) - 1))
        except Exception:
            callback = None

    disable_pbar = not getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True)

    samples = comfy.sample.sample(
        patched_model,
        noise,
        int(steps),
        float(cfg),
        sampler_name,
        fallback_sched,
        positive,
        negative,
        latent_image,
        denoise=float(denoise),
        disable_noise=disable_noise,
        force_full_denoise=True,
        noise_mask=noise_mask,
        sigmas=sigmas,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=int(seed),
    )

    out_latent = latent_dict.copy()
    out_latent.pop("downscale_ratio_spacial", None)
    out_latent.pop("downscale_ratio_temporal", None)
    out_latent["samples"] = samples
    return out_latent, sigmas


# ==============================================================================
# NODE 11: HaDeS Anima KSampler (Single-Pass Master)
# ==============================================================================
class HaDeSAnimaKSampler:
    """
    Single-Pass Master KSampler tailored to Anima's Cosmos-Predict2 Rectified Flow
    architecture (multiplier=1.0 Flow Shift, anima_beta57 & anima_lightning_tdm schedulers).
    """

    VARIANT_PROFILES = {
        "Manual": None,
        "Anima Base / Raehoshi": (32, 4.5, "er_sde", "normal", 1.0, 0.0),
        "Anima Aesthetic Crisp": (30, 4.0, "er_sde", "beta", 1.0, 0.0),
        "Anima Aesthetic Painterly": (35, 4.2, "euler_ancestral", "anima_beta57", 1.0, 2.8),
        "Anima Turbo": (10, 1.0, "euler", "normal", 1.0, 0.0),
        "Anima Lightning TDM": (4, 1.0, "euler", "anima_lightning_tdm", 1.0, 0.0),
        "Anima 2.9B / 3.8B Deep DiT": (36, 4.5, "er_sde", "normal", 1.0, 0.0),
    }

    _LEGACY_PROFILE_MAP = {
        "Manual (Use Widget Settings)": "Manual",
        "Anima Base / Raehoshi (er_sde | 32 steps | CFG 4.5 | Shift 3.0)": "Anima Base / Raehoshi",
        "Anima Aesthetic Crisp (er_sde | 30 steps | CFG 4.0 | beta | Shift 3.0)": "Anima Aesthetic Crisp",
        "Anima Aesthetic Painterly (euler_a | 35 steps | CFG 4.2 | anima_beta57 | Shift 2.8)": "Anima Aesthetic Painterly",
        "Anima Turbo (euler | 10 steps | CFG 1.0 | Shift 3.0)": "Anima Turbo",
        "Anima Lightning TDM (euler | 4 steps | CFG 1.0 | anima_lightning_tdm)": "Anima Lightning TDM",
        "Anima 2.9B / 3.8B Deep DiT (er_sde | 36 steps | CFG 4.5 | Shift 3.0)": "Anima 2.9B / 3.8B Deep DiT",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "variant_profile": (list(cls.VARIANT_PROFILES.keys()), {
                    "default": "Manual"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 32, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (ANIMA_SAMPLER_LIST, {"default": "er_sde"}),
                "scheduler": (ANIMA_SCHEDULER_LIST, {"default": "normal"}),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "1.0 for txt2img. For img2img: 0.50–0.60 for minor detail refinement, 0.75–0.85 for creative compositional guidance."
                }),
                "flow_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Rectified Flow shift for this pass. 0 = inherit the shift already set on the model (Model Loader flow_shift or ModelSampling nodes)."}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "info")
    FUNCTION = "sample_anima"
    CATEGORY = "HaDeS/Anima Samplers"

    def sample_anima(
        self,
        model,
        positive,
        negative,
        latent_image,
        variant_profile,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        flow_shift,
        vae=None,
    ):
        canonical_profile = self._LEGACY_PROFILE_MAP.get(variant_profile, variant_profile)
        prof = self.VARIANT_PROFILES.get(canonical_profile)
        if prof is not None:
            steps, cfg, sampler_name, scheduler, denoise, flow_shift = prof

        out_latent, sigmas = run_anima_ksampler_pass(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_dict=latent_image,
            denoise=denoise,
            flow_shift=flow_shift,
        )

        decoded_img = _decode_or_placeholder(vae, out_latent["samples"])
        h_px = int(out_latent["samples"].shape[-2]) * 8
        w_px = int(out_latent["samples"].shape[-1]) * 8
        sig_head = ", ".join(f"{float(s):.3f}" for s in sigmas[: min(5, len(sigmas))])

        info = (
            f"HaDeS Anima KSampler | Profile: {canonical_profile} | Resolution: {w_px}x{h_px}px | "
            f"Sampler: {sampler_name} | Scheduler: {scheduler} | Steps: {steps} | CFG: {cfg:.2f} | "
            f"Denoise: {denoise:.2f} | Flow Shift (multiplier=1.0): {_resolve_flow_shift(model, flow_shift):.2f}{' (inherited)' if flow_shift <= 0 else ''} | "
            f"Sigmas[0..]: [{sig_head}]"
        )
        print(f"\n[HaDeS Anima KSampler] {info}")
        return (out_latent, decoded_img, info)


# ==============================================================================
# NODE 12: HaDeS Anima Dual-Pass KSampler (HiRes & Hybrid Refiner)
# ==============================================================================
class HaDeSAnimaDualPassKSampler:
    """
    Two-Stage Composition -> HiRes/Style Refiner KSampler for Anima.
    Includes a 16-channel variance-preserving latent bridge snapped strictly to the 32px grid
    and optional hybrid model handoff (e.g. Base/Raehoshi -> Aesthetic/Turbo).
    """

    DUAL_PRESETS = {
        "Manual (Use Widget Settings)": None,
        "HiRes Standard 1.5x (er_sde 30s -> er_sde + beta57 16s @ 0.42 denoise)": (
            30, 4.5, "er_sde", "normal", 1.00, 0.0,
            "Latent 16ch Variance-Preserving Bicubic", 1.50,
            16, 4.0, "er_sde", "anima_beta57", 0.42, 3.60,
        ),
        "Aesthetic Painterly 1.5x (er_sde 32s -> euler_a + beta57 18s @ 0.45 denoise)": (
            32, 4.2, "er_sde", "beta", 1.00, 0.0,
            "Latent 16ch Variance-Preserving Bicubic", 1.50,
            18, 3.8, "euler_ancestral", "anima_beta57", 0.45, 3.50,
        ),
        "Base-to-Turbo Fast HiRes 1.5x (er_sde 30s CFG 4.5 -> euler 8s CFG 1.0 @ 0.38 denoise)": (
            30, 4.5, "er_sde", "normal", 1.00, 0.0,
            "Latent 16ch Variance-Preserving Bicubic", 1.50,
            8, 1.0, "euler", "normal", 0.38, 3.50,
        ),
        "Same-Res Texture & Lineart Refiner 1.0x (30s -> 12s @ 0.28 denoise)": (
            30, 4.5, "er_sde", "normal", 1.00, 0.0,
            "Direct Latent Pass-Through (1.0x)", 1.00,
            12, 3.6, "euler", "anima_beta57", 0.28, 3.20,
        ),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "dual_preset": (list(cls.DUAL_PRESETS.keys()), {
                    "default": "Manual (Use Widget Settings)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps_pass1": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
                "cfg_pass1": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_pass1": (ANIMA_SAMPLER_LIST, {"default": "er_sde"}),
                "scheduler_pass1": (ANIMA_SCHEDULER_LIST, {"default": "normal"}),
                "denoise_pass1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift_pass1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Rectified Flow shift for this pass. 0 = inherit the shift already set on the model (Model Loader flow_shift or ModelSampling nodes)."}),
                "upscale_method": (ANIMA_UPSCALE_METHODS, {
                    "default": "Latent 16ch Variance-Preserving Bicubic"
                }),
                "upscale_factor": ("FLOAT", {"default": 1.50, "min": 1.0, "max": 2.5, "step": 0.05}),
                "steps_pass2": ("INT", {"default": 16, "min": 1, "max": 200, "step": 1}),
                "cfg_pass2": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_pass2": (ANIMA_SAMPLER_LIST, {"default": "er_sde"}),
                "scheduler_pass2": (ANIMA_SCHEDULER_LIST, {"default": "anima_beta57"}),
                "denoise_pass2": ("FLOAT", {"default": 0.42, "min": 0.05, "max": 1.0, "step": 0.01}),
                "shift_pass2": ("FLOAT", {"default": 3.6, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Rectified Flow shift for this pass. 0 = inherit the shift already set on the model (Model Loader flow_shift or ModelSampling nodes)."}),
            },
            "optional": {
                "model_pass2": ("MODEL",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("final_latent", "pass1_latent", "decoded_image", "info")
    FUNCTION = "sample_dual_pass"
    CATEGORY = "HaDeS/Anima Samplers"

    def sample_dual_pass(
        self,
        model,
        positive,
        negative,
        latent_image,
        dual_preset,
        seed,
        steps_pass1,
        cfg_pass1,
        sampler_pass1,
        scheduler_pass1,
        denoise_pass1,
        shift_pass1,
        upscale_method,
        upscale_factor,
        steps_pass2,
        cfg_pass2,
        sampler_pass2,
        scheduler_pass2,
        denoise_pass2,
        shift_pass2,
        model_pass2=None,
        vae=None,
        **kwargs,
    ):
        if model_pass2 is None:
            model_pass2 = kwargs.get("model_pass2_optional")

        preset = self.DUAL_PRESETS.get(dual_preset)
        if preset is not None:
            (
                steps_pass1, cfg_pass1, sampler_pass1, scheduler_pass1, denoise_pass1, shift_pass1,
                upscale_method, upscale_factor,
                steps_pass2, cfg_pass2, sampler_pass2, scheduler_pass2, denoise_pass2, shift_pass2,
            ) = preset

        # Pass 1: Base Composition
        pass1_latent, _ = run_anima_ksampler_pass(
            model=model,
            seed=seed,
            steps=steps_pass1,
            cfg=cfg_pass1,
            sampler_name=sampler_pass1,
            scheduler=scheduler_pass1,
            positive=positive,
            negative=negative,
            latent_dict=latent_image,
            denoise=denoise_pass1,
            flow_shift=shift_pass1,
        )

        # Inter-Pass 16ch 32px Bridge
        bridged_latent, (w1, h1), (w2, h2) = upscale_anima_latent_32px(
            pass1_latent, upscale_factor, upscale_method, vae=vae
        )

        # Pass 2: HiRes / Style Refinement (uses seed + 1 so noise pattern does not reinforce Pass 1 artifacts)
        model_p2 = model_pass2 if model_pass2 is not None else model
        final_latent, _ = run_anima_ksampler_pass(
            model=model_p2,
            seed=int(seed) + 1,
            steps=steps_pass2,
            cfg=cfg_pass2,
            sampler_name=sampler_pass2,
            scheduler=scheduler_pass2,
            positive=positive,
            negative=negative,
            latent_dict=bridged_latent,
            denoise=denoise_pass2,
            flow_shift=shift_pass2,
        )

        decoded_img = _decode_or_placeholder(vae, final_latent["samples"])
        effective_bridge = (
            "Latent 16ch Variance-Preserving Bicubic (Auto-Fallback: No VAE)"
            if upscale_method == "Pixel-Space VAE Round-Trip (Lanczos 32px)" and vae is None
            else upscale_method
        )
        info = (
            f"HaDeS Anima Dual-Pass KSampler | Preset: {dual_preset} | "
            f"Pass 1: {w1}x{h1}px ({sampler_pass1}/{scheduler_pass1}, {steps_pass1}s, CFG {cfg_pass1:.1f}, Shift {_resolve_flow_shift(model, shift_pass1):.2f}) -> "
            f"Bridge: {effective_bridge} ({upscale_factor:.2f}x -> {w2}x{h2}px) -> "
            f"Pass 2: {w2}x{h2}px ({sampler_pass2}/{scheduler_pass2}, {steps_pass2}s, CFG {cfg_pass2:.1f}, "
            f"Denoise {denoise_pass2:.2f}, Shift {_resolve_flow_shift(model_p2, shift_pass2):.2f}, HybridModel={'Yes' if model_pass2 is not None else 'No'})"
        )
        print(f"\n[HaDeS Anima Dual-Pass KSampler] {info}")
        return (final_latent, pass1_latent, decoded_img, info)


# ==============================================================================
# NODE 13: HaDeS Anima Triple-Pass KSampler (Progressive Studio Pipeline)
# ==============================================================================
class HaDeSAnimaTriplePassKSampler:
    """
    Three-Stage Progressive Studio Pipeline (Draft/Anatomy Lock -> Structural/Aesthetic Bridge ->
    Micro-Lineart & Surface Polish). Prevents patch-hallucination and anatomy distortion when
    pushing Anima to 1.75x - 2.0x ultra-high resolutions.
    """

    TRIPLE_PRESETS = {
        "Manual (Use Widget Settings)": None,
        "Progressive Staircase 1.0x -> 1.35x -> 1.75x (Ultra-Res Poster Pipeline)": (
            "Latent 16ch Variance-Preserving Bicubic",
            28, 4.5, "er_sde", "normal", 1.00, 0.0,
            1.35, 16, 4.0, "er_sde", "beta", 0.48, 3.50,
            1.75, 12, 3.4, "euler", "anima_beta57", 0.26, 4.00,
        ),
        "Progressive 2x Master 1.0x -> 1.5x -> 2.0x (4 MP Studio Illustration)": (
            "Latent 16ch Variance-Preserving Bicubic",
            30, 4.5, "er_sde", "normal", 1.00, 0.0,
            1.50, 18, 4.0, "er_sde", "beta", 0.46, 3.60,
            2.00, 14, 3.5, "euler", "anima_beta57", 0.28, 4.20,
        ),
        "Concept -> Aesthetic Bridge 1.5x -> Micro-Polish 1.5x (Lock Res at Stage 2)": (
            "Latent 16ch Variance-Preserving Bicubic",
            28, 4.5, "er_sde", "normal", 1.00, 0.0,
            1.50, 16, 4.0, "er_sde", "beta", 0.45, 3.60,
            1.50, 10, 3.2, "euler", "anima_beta57", 0.22, 3.60,
        ),
        "Hybrid Base -> Aesthetic -> Turbo Polish (Fast 3-Stage Studio)": (
            "Latent 16ch Variance-Preserving Bicubic",
            28, 4.5, "er_sde", "normal", 1.00, 0.0,
            1.40, 14, 4.0, "er_sde", "beta", 0.46, 3.50,
            1.75, 6, 1.0, "euler", "normal", 0.28, 3.90,
        ),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "triple_preset": (list(cls.TRIPLE_PRESETS.keys()), {
                    "default": "Manual (Use Widget Settings)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "upscale_method": (ANIMA_UPSCALE_METHODS, {
                    "default": "Latent 16ch Variance-Preserving Bicubic"
                }),
                # Stage 1: Anatomy & Layout Lock (1.0x)
                "steps_stage1": ("INT", {"default": 28, "min": 1, "max": 200, "step": 1}),
                "cfg_stage1": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_stage1": (ANIMA_SAMPLER_LIST, {"default": "er_sde"}),
                "scheduler_stage1": (ANIMA_SCHEDULER_LIST, {"default": "normal"}),
                "denoise_stage1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift_stage1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Rectified Flow shift for this pass. 0 = inherit the shift already set on the model (Model Loader flow_shift or ModelSampling nodes)."}),
                # Stage 2: Structural & Aesthetic Bridge
                "scale_stage2": ("FLOAT", {"default": 1.35, "min": 1.0, "max": 2.5, "step": 0.05}),
                "steps_stage2": ("INT", {"default": 16, "min": 1, "max": 200, "step": 1}),
                "cfg_stage2": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_stage2": (ANIMA_SAMPLER_LIST, {"default": "er_sde"}),
                "scheduler_stage2": (ANIMA_SCHEDULER_LIST, {"default": "beta"}),
                "denoise_stage2": ("FLOAT", {"default": 0.48, "min": 0.05, "max": 1.0, "step": 0.01}),
                "shift_stage2": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Rectified Flow shift for this pass. 0 = inherit the shift already set on the model (Model Loader flow_shift or ModelSampling nodes)."}),
                # Stage 3: Micro-Lineart & Surface Polish (scale_stage3 is relative to Stage 1 base)
                "scale_stage3": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 3.0, "step": 0.05}),
                "steps_stage3": ("INT", {"default": 12, "min": 1, "max": 200, "step": 1}),
                "cfg_stage3": ("FLOAT", {"default": 3.4, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_stage3": (ANIMA_SAMPLER_LIST, {"default": "euler"}),
                "scheduler_stage3": (ANIMA_SCHEDULER_LIST, {"default": "anima_beta57"}),
                "denoise_stage3": ("FLOAT", {"default": 0.26, "min": 0.05, "max": 1.0, "step": 0.01}),
                "shift_stage3": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Rectified Flow shift for this pass. 0 = inherit the shift already set on the model (Model Loader flow_shift or ModelSampling nodes)."}),
            },
            "optional": {
                "model_pass2": ("MODEL",),
                "model_pass3": ("MODEL",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("final_latent", "pass1_latent", "pass2_latent", "decoded_image", "info")
    FUNCTION = "sample_triple_pass"
    CATEGORY = "HaDeS/Anima Samplers"

    def sample_triple_pass(
        self,
        model,
        positive,
        negative,
        latent_image,
        triple_preset,
        seed,
        upscale_method,
        steps_stage1,
        cfg_stage1,
        sampler_stage1,
        scheduler_stage1,
        denoise_stage1,
        shift_stage1,
        scale_stage2,
        steps_stage2,
        cfg_stage2,
        sampler_stage2,
        scheduler_stage2,
        denoise_stage2,
        shift_stage2,
        scale_stage3,
        steps_stage3,
        cfg_stage3,
        sampler_stage3,
        scheduler_stage3,
        denoise_stage3,
        shift_stage3,
        model_pass2=None,
        model_pass3=None,
        vae=None,
        **kwargs,
    ):
        if model_pass2 is None:
            model_pass2 = kwargs.get("model_stage2_optional")
        if model_pass3 is None:
            model_pass3 = kwargs.get("model_stage3_optional")

        preset = self.TRIPLE_PRESETS.get(triple_preset)
        if preset is not None:
            (
                upscale_method,
                steps_stage1, cfg_stage1, sampler_stage1, scheduler_stage1, denoise_stage1, shift_stage1,
                scale_stage2, steps_stage2, cfg_stage2, sampler_stage2, scheduler_stage2, denoise_stage2, shift_stage2,
                scale_stage3, steps_stage3, cfg_stage3, sampler_stage3, scheduler_stage3, denoise_stage3, shift_stage3,
            ) = preset

        # Stage 1: Base Composition & Anatomy Lock (1.0x)
        stage1_latent, _ = run_anima_ksampler_pass(
            model=model,
            seed=seed,
            steps=steps_stage1,
            cfg=cfg_stage1,
            sampler_name=sampler_stage1,
            scheduler=scheduler_stage1,
            positive=positive,
            negative=negative,
            latent_dict=latent_image,
            denoise=denoise_stage1,
            flow_shift=shift_stage1,
        )

        # Bridge 1 -> 2
        bridged_s2, (w1, h1), (w2, h2) = upscale_anima_latent_32px(
            stage1_latent, scale_stage2, upscale_method, vae=vae
        )

        # Stage 2: Structural & Aesthetic Bridge
        model_s2 = model_pass2 if model_pass2 is not None else model
        stage2_latent, _ = run_anima_ksampler_pass(
            model=model_s2,
            seed=int(seed) + 1,
            steps=steps_stage2,
            cfg=cfg_stage2,
            sampler_name=sampler_stage2,
            scheduler=scheduler_stage2,
            positive=positive,
            negative=negative,
            latent_dict=bridged_s2,
            denoise=denoise_stage2,
            flow_shift=shift_stage2,
        )

        # Bridge 2 -> 3 (scale_stage3 is relative to Stage 1 base resolution (w1, h1) to prevent cumulative 32px rounding drift)
        bridged_s3, _, (w3, h3) = upscale_anima_latent_32px(
            stage2_latent, scale_stage3, upscale_method, vae=vae, base_px_dims=(w1, h1)
        )

        # Stage 3: Micro-Lineart & Surface Polish
        model_s3 = model_pass3 if model_pass3 is not None else model_s2
        final_latent, _ = run_anima_ksampler_pass(
            model=model_s3,
            seed=int(seed) + 2,
            steps=steps_stage3,
            cfg=cfg_stage3,
            sampler_name=sampler_stage3,
            scheduler=scheduler_stage3,
            positive=positive,
            negative=negative,
            latent_dict=bridged_s3,
            denoise=denoise_stage3,
            flow_shift=shift_stage3,
        )

        decoded_img = _decode_or_placeholder(vae, final_latent["samples"])
        effective_bridge = (
            "Latent 16ch Variance-Preserving Bicubic (Auto-Fallback: No VAE)"
            if upscale_method == "Pixel-Space VAE Round-Trip (Lanczos 32px)" and vae is None
            else upscale_method
        )
        info = (
            f"HaDeS Anima Triple-Pass KSampler | Preset: {triple_preset} | Bridge: {effective_bridge} | "
            f"S1: {w1}x{h1}px ({sampler_stage1}/{scheduler_stage1}, {steps_stage1}s, CFG {cfg_stage1:.1f}, Shift {_resolve_flow_shift(model, shift_stage1):.2f}) -> "
            f"S2 ({scale_stage2:.2f}x): {w2}x{h2}px ({sampler_stage2}/{scheduler_stage2}, {steps_stage2}s, CFG {cfg_stage2:.1f}, Denoise {denoise_stage2:.2f}, Shift {_resolve_flow_shift(model_s2, shift_stage2):.2f}) -> "
            f"S3 ({scale_stage3:.2f}x): {w3}x{h3}px ({sampler_stage3}/{scheduler_stage3}, {steps_stage3}s, CFG {cfg_stage3:.1f}, Denoise {denoise_stage3:.2f}, Shift {_resolve_flow_shift(model_s3, shift_stage3):.2f})"
        )
        print(f"\n[HaDeS Anima Triple-Pass KSampler] {info}")
        return (final_latent, stage1_latent, stage2_latent, decoded_img, info)
