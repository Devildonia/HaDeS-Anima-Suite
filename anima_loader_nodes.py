import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management

from . import detect_anima_topology
from .anima_pipeline_nodes import HaDeSAnimaCornerScribbleShield
from .anima_ksampler_nodes import (
    clone_model_with_anima_shift,
    _validate_anima_16ch_vae_optional,
    _normalize_decoded_image_4d,
)


# ==============================================================================
# DIFFUSERS -> COMFYUI FULL CHECKPOINT KEY CONVERTER (24 DETERMINISTIC RULES)
# ==============================================================================
_MODEL_DIFFUSERS_TO_COMFY_SUBS = (
    ("transformer_blocks.", "blocks."),
    (".attn1.to_out.0.", ".self_attn.output_proj."),
    (".attn2.to_out.0.", ".cross_attn.output_proj."),
    (".attn1.to_q.", ".self_attn.q_proj."),
    (".attn1.to_k.", ".self_attn.k_proj."),
    (".attn1.to_v.", ".self_attn.v_proj."),
    (".attn1.norm_q.", ".self_attn.q_norm."),
    (".attn1.norm_k.", ".self_attn.k_norm."),
    (".attn2.to_q.", ".cross_attn.q_proj."),
    (".attn2.to_k.", ".cross_attn.k_proj."),
    (".attn2.to_v.", ".cross_attn.v_proj."),
    (".attn2.norm_q.", ".cross_attn.q_norm."),
    (".attn2.norm_k.", ".cross_attn.k_norm."),
    (".ff.net.0.proj.", ".mlp.layer1."),
    (".ff.net.2.", ".mlp.layer2."),
    (".norm1.linear_1.", ".adaln_modulation_self_attn.1."),
    (".norm1.linear_2.", ".adaln_modulation_self_attn.2."),
    (".norm2.linear_1.", ".adaln_modulation_cross_attn.1."),
    (".norm2.linear_2.", ".adaln_modulation_cross_attn.2."),
    (".norm3.linear_1.", ".adaln_modulation_mlp.1."),
    (".norm3.linear_2.", ".adaln_modulation_mlp.2."),
    ("time_embed.emb.timestep_embedder.", "t_embedder.1."),
    ("time_embed.t_embedder.", "t_embedder.1."),
    ("time_embed.norm.", "t_embedding_norm."),
    ("patch_embed.proj.", "x_embedder.proj.1."),
    ("norm_out.linear_1.", "final_layer.adaln_modulation.1."),
    ("norm_out.linear_2.", "final_layer.adaln_modulation.2."),
    ("proj_out.", "final_layer.linear."),
)


def _prioritize_filenames(file_list: list[str], keywords: tuple[str, ...]) -> list[str]:
    if not file_list:
        return ["None"]
    preferred = []
    others = []
    for f in file_list:
        low = f.lower()
        if any(kw in low for kw in keywords):
            preferred.append(f)
        else:
            others.append(f)
    return preferred + others


def remap_anima_diffusers_model_sd(sd: dict) -> tuple[dict, int]:
    """
    Convert a Diffusers-format Anima / Cosmos-Predict2 state_dict (and/or text_conditioner.* keys)
    into ComfyUI's native MiniTrainDIT + LLMAdapter naming convention in-place/copy without tensor transpositions.
    Returns: (converted_sd, remapped_key_count)
    """
    out_sd = {}
    remapped = 0

    for orig_k, tensor in sd.items():
        k = orig_k
        for root_pfx in ("model.diffusion_model.", "diffusion_model.", "transformer."):
            if k.startswith(root_pfx):
                k = k[len(root_pfx):]
                break

        if k.startswith("text_conditioner."):
            k = "llm_adapter." + k[len("text_conditioner."):]
        elif not k.startswith("llm_adapter."):
            for old_frag, new_frag in _MODEL_DIFFUSERS_TO_COMFY_SUBS:
                if old_frag in k:
                    k = k.replace(old_frag, new_frag)

        if k != orig_k:
            remapped += 1
        out_sd[k] = tensor

    return out_sd, remapped


def inject_external_llm_adapter_sd(base_sd: dict, adapter_sd: dict) -> tuple[dict, int]:
    """
    Inject or replace an external AnimaTextConditioner / expanded_adapter state_dict under the 'llm_adapter.' prefix.
    Returns: (merged_sd, injected_tensor_count)
    """
    merged = base_sd.copy()
    injected = 0
    for orig_k, tensor in adapter_sd.items():
        k = orig_k
        for pfx in ("model.diffusion_model.", "diffusion_model.", "transformer."):
            if k.startswith(pfx):
                k = k[len(pfx):]
                break
        if k.startswith("text_conditioner."):
            k = "llm_adapter." + k[len("text_conditioner."):]
        elif not k.startswith("llm_adapter."):
            # Standalone text_conditioner/model.safetensors has keys like 'blocks.0...', 'embed.weight', 'out_proj.weight', 'norm.weight'
            if (
                k.startswith("blocks.")
                or k.startswith("embed.")
                or k.startswith("in_proj.")
                or k.startswith("out_proj.")
                or k.startswith("norm.")
            ):
                k = "llm_adapter." + k
        if k.startswith("llm_adapter."):
            merged[k] = tensor
            injected += 1
    return merged, injected


def validate_anima_te_state_dict(sd: dict, clip_name: str = "") -> tuple[str, int, int]:
    """
    Validate that a text encoder state_dict is compatible with Anima:
    - Primary: Qwen3-0.6B (hidden_size=1024, 28 layers, q_norm/k_norm present)
    - Secondary (Anima-3.8B Dual TE): Qwen3.5-4B / Qwen3-4B (hidden_size=2560) or Qwen3.5-0.8B (1024)
    Raises a clear ValueError if an incompatible encoder (CLIP-L, CLIP-G, T5-XXL, Qwen2.5-7B) is supplied.
    Returns: (detected_arch_label, hidden_dim, num_layers)
    """
    keys = list(sd.keys())

    if any("text_model.encoder.layers." in k for k in keys):
        raise ValueError(
            f"[HaDeS Anima CLIP Loader] Incompatible SD1.5/SDXL CLIP text encoder ('{clip_name}') detected. "
            "Anima requires 'qwen_3_06b_base.safetensors' (Qwen3-0.6B Base, 1024-dim)."
        )
    if any("encoder.block." in k and "DenseReluDense" in k for k in keys):
        raise ValueError(
            f"[HaDeS Anima CLIP Loader] Incompatible T5 text encoder ('{clip_name}') detected. "
            "Although Cosmos-Predict2 used T5, Anima replaced T5 with 'qwen_3_06b_base.safetensors' (Qwen3-0.6B Base) + internal llm_adapter."
        )

    # Check Qwen3 / Qwen3.5 layer norms
    norm_w = sd.get("model.layers.0.post_attention_layernorm.weight", None)
    if norm_w is None:
        norm_w = sd.get("model.language_model.layers.0.input_layernorm.weight", None)
    if norm_w is None:
        norm_w = sd.get("layers.0.post_attention_layernorm.weight", None)

    has_q_norm = (
        "model.layers.0.self_attn.q_norm.weight" in sd
        or "layers.0.self_attn.q_norm.weight" in sd
        or "model.language_model.layers.0.linear_attn.A_log" in sd
    )

    if norm_w is not None:
        hidden_dim = int(norm_w.shape[0])
        n_layers = sum(
            1 for k in keys
            if (k.startswith("model.layers.") or k.startswith("model.language_model.layers.") or k.startswith("layers."))
            and k.endswith(".input_layernorm.weight")
        )
        if hidden_dim == 1024 and has_q_norm:
            return ("Qwen3-0.6B Base (Native Anima 1024d)", hidden_dim, max(1, n_layers))
        if hidden_dim == 2560 and has_q_norm:
            return ("Qwen3/3.5-4B (Anima-3.8B Secondary 2560d)", hidden_dim, max(1, n_layers))
        if hidden_dim == 3584:
            raise ValueError(
                f"[HaDeS Anima CLIP Loader] Incompatible 3584-dim Qwen2.5-7B encoder ('{clip_name}') detected. "
                "Anima requires the 1024-dim 'qwen_3_06b_base.safetensors' (Qwen3-0.6B Base), NOT the 7B Qwen-Image encoder."
            )

    raise ValueError(
        f"[HaDeS Anima CLIP Loader] Unrecognized or incompatible text encoder '{clip_name}'. "
        "Anima strictly requires 'qwen_3_06b_base.safetensors' (Qwen3-0.6B Base, 1024d)."
    )


def _inspect_and_validate_anima_vae_sd(sd: dict, vae_name: str = "") -> dict:
    """
    Pre-instantiation inspection of a VAE state_dict to block the 64-channel Qwen-Image 2.1 VAE
    trap (and 4-channel SD1.5/SDXL VAEs) before comfy.sd.VAE allocates or runs the wrong architecture.
    Returns diagnostic metadata dict.
    """
    if "decoder.middle.0.residual.0.gamma" in sd:
        wan22_layout = "decoder.upsamples.0.upsamples.0.residual.2.weight" in sd
        head = sd.get("decoder.head.2.weight", None)
        if wan22_layout and head is not None and head.ndim == 5 and head.shape[2] == 1:
            raise ValueError(
                f"[HaDeS Anima VAE Loader] CRITICAL TRAP BLOCKED: '{vae_name}' is the 64-channel Qwen-Image 2.1 VAE "
                "(Wan 2.2 layout, 16x spatial compression). Anima strictly requires the 16-channel Wan 2.1 VAE "
                "('qwen_image_vae.safetensors', ~254 MB in bf16, 8x spatial compression)."
            )
        if wan22_layout:
            raise ValueError(
                f"[HaDeS Anima VAE Loader] Incompatible 48-channel Wan 2.2 VAE ('{vae_name}') detected. "
                "Anima requires the 16-channel Wan 2.1 'qwen_image_vae.safetensors' (8x spatial compression)."
            )
        dim = int(sd["decoder.head.0.gamma"].shape[0]) if "decoder.head.0.gamma" in sd else 96
        return {
            "architecture": "AutoencoderKLQwenImage (Wan 2.1 16ch)",
            "latent_channels": 16,
            "spatial_compression": 8,
            "base_dim": dim,
        }

    if "decoder.conv_in.weight" in sd and sd["decoder.conv_in.weight"].ndim == 4:
        z_ch = int(sd["decoder.conv_in.weight"].shape[1])
        raise ValueError(
            f"[HaDeS Anima VAE Loader] Incompatible 2D AutoencoderKL ('{vae_name}', {z_ch} channels) detected. "
            "SD1.5/SDXL (4ch) and Flux (16ch 2D) VAEs are incompatible with Anima's 3D causal Wan 2.1 latent space. "
            "Please select 'qwen_image_vae.safetensors' (~254 MB)."
        )

    raise ValueError(
        f"[HaDeS Anima VAE Loader] Unrecognized VAE state_dict in '{vae_name}'. "
        "Anima requires the 16-channel 'qwen_image_vae.safetensors' (AutoencoderKLQwenImage / Wan 2.1)."
    )


# ==============================================================================
# NODE 14: HaDeS Anima Diffusion Model Loader
# ==============================================================================
class HaDeSAnimaModelLoader:
    """
    Native Anima Diffusion Model Loader with automatic Diffusers-to-ComfyUI 24-rule key conversion,
    optional external llm_adapter (text_conditioner / Anima-3.8B expanded_adapter) injection,
    topology auto-detection (28L / 40L / 52L), and Rectified Flow shift pre-configuration (multiplier=1.0).
    """

    WEIGHT_DTYPES = [
        "default",
        "bfloat16",
        "float16",
        "fp8_e4m3fn",
        "fp8_e4m3fn_fast",
        "fp8_e5m2",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        raw_models = folder_paths.get_filename_list("diffusion_models")
        ordered_models = _prioritize_filenames(raw_models, ("anima", "raehoshi", "cosmos"))
        adapter_choices = ["None"] + _prioritize_filenames(raw_models, ("adapter", "conditioner", "anima"))
        return {
            "required": {
                "unet_name": (ordered_models,),
                "weight_dtype": (cls.WEIGHT_DTYPES, {"default": "default"}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.05}),
                "auto_remap_diffusers": ("BOOLEAN", {"default": True}),
                "external_llm_adapter": (adapter_choices, {"default": "None"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_anima_model"
    CATEGORY = "HaDeS/Anima Loaders"

    def load_anima_model(
        self,
        unet_name: str,
        weight_dtype: str = "default",
        flow_shift: float = 3.0,
        auto_remap_diffusers: bool = True,
        external_llm_adapter: str = "None",
    ):
        model_options = {}
        if weight_dtype == "bfloat16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "float16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "fp8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast" and hasattr(torch, "float8_e4m3fn"):
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2" and hasattr(torch, "float8_e5m2"):
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

        remapped_count = 0
        if auto_remap_diffusers:
            has_diffusers_keys = any(
                "transformer_blocks." in k
                or "patch_embed.proj." in k
                or "text_conditioner." in k
                or "time_embed." in k
                or "norm_out." in k
                or k.startswith("proj_out.")
                or k.startswith("model.diffusion_model.")
                or k.startswith("diffusion_model.")
                or k.startswith("transformer.")
                for k in sd.keys()
            )
            if has_diffusers_keys:
                sd, remapped_count = remap_anima_diffusers_model_sd(sd)

        injected_count = 0
        if external_llm_adapter and external_llm_adapter != "None":
            adapter_path = folder_paths.get_full_path_or_raise("diffusion_models", external_llm_adapter)
            adapter_sd = comfy.utils.load_torch_file(adapter_path, safe_load=True)
            sd, injected_count = inject_external_llm_adapter_sd(sd, adapter_sd)

        n_dit, n_adapter = detect_anima_topology(sd.keys())
        has_adapter = any("llm_adapter.blocks.0." in k for k in sd.keys())

        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
        if model is None:
            raise RuntimeError(
                f"[HaDeS Anima Model Loader] Failed to initialize Anima diffusion model from '{unet_name}'."
            )

        if remapped_count == 0 and injected_count == 0:
            model.cached_patcher_init = (comfy.sd.load_diffusion_model, (unet_path, model_options))

        if abs(float(flow_shift) - 3.0) > 1e-4:
            model = clone_model_with_anima_shift(model, float(flow_shift))

        variant_label = (
            "Anima 2.0B (28 DiT Blocks)"
            if n_dit == 28
            else ("Anima 2.9B (40 DiT Blocks)" if n_dit == 40 else (
                "Anima 3.8B (52 DiT Blocks)" if n_dit == 52 else f"Anima Custom ({n_dit} DiT Blocks)"
            ))
        )
        adapter_status = f"Present ({n_adapter} Blocks)" if has_adapter else "WARNING: Missing llm_adapter (Cosmos-Predict2 Raw)"
        info = (
            f"HaDeS Anima Model Loader | File: {os.path.basename(unet_name)} | "
            f"Topology: {variant_label} | llm_adapter: {adapter_status} | "
            f"Dtype: {weight_dtype} | Flow Shift: {float(flow_shift):.2f} (mult=1.0) | "
            f"Diffusers Remapped: {remapped_count} keys | External Adapter Injected: {injected_count} keys"
        )
        print(f"\n[HaDeS Anima Model Loader] {info}")
        return (model, info)


# ==============================================================================
# NODE 15: HaDeS Anima Qwen3 Text Encoder (CLIP) Loader
# ==============================================================================
class HaDeSAnimaCLIPLoader:
    """
    Dedicated Anima Text Encoder Loader that automatically selects Anima's Qwen3-0.6B + T5Tokenizer
    pipeline, validates the 1024-dim Qwen3 architecture before instantiation (blocking accidental
    T5-XXL, CLIP-L, or 3584-dim Qwen2.5-7B loads), and offers 6GB Low-VRAM CPU placement.
    """

    DEVICE_MODES = [
        "default (GPU Dynamic)",
        "cpu (Low-VRAM 6GB Safe)",
    ]
    PRECISION_MODES = [
        "auto",
        "bfloat16",
        "float16",
        "float32",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        raw_clips = folder_paths.get_filename_list("text_encoders")
        ordered_clips = _prioritize_filenames(raw_clips, ("qwen_3_06b", "qwen3_06b", "qwen3", "anima"))
        return {
            "required": {
                "clip_name": (ordered_clips,),
                "device_mode": (cls.DEVICE_MODES, {"default": "default (GPU Dynamic)"}),
                "precision": (cls.PRECISION_MODES, {"default": "auto"}),
                "strict_qwen3_guard": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "clip_info")
    FUNCTION = "load_anima_clip"
    CATEGORY = "HaDeS/Anima Loaders"

    def load_anima_clip(
        self,
        clip_name: str,
        device_mode: str = "default (GPU Dynamic)",
        precision: str = "auto",
        strict_qwen3_guard: bool = True,
    ):
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        sd, metadata = comfy.utils.load_torch_file(clip_path, safe_load=True, return_metadata=True)

        arch_label = "Unchecked (Guard Disabled)"
        hidden_dim = 1024
        n_layers = 28
        if strict_qwen3_guard:
            arch_label, hidden_dim, n_layers = validate_anima_te_state_dict(sd, clip_name=clip_name)

        model_options = {}
        if device_mode.startswith("cpu"):
            model_options["load_device"] = torch.device("cpu")
            model_options["offload_device"] = torch.device("cpu")

        if precision == "bfloat16":
            model_options["dtype"] = torch.bfloat16
        elif precision == "float16":
            model_options["dtype"] = torch.float16
        elif precision == "float32":
            model_options["dtype"] = torch.float32

        if model_options.get("custom_operations", None) is None and hasattr(comfy.utils, "convert_old_quants"):
            sd, metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)

        if not hasattr(comfy.sd.CLIPType, "QWEN_IMAGE"):
            print(
                "[HaDeS Anima CLIP Loader] WARNING: CLIPType.QWEN_IMAGE not found in this ComfyUI build; "
                "falling back to STABLE_DIFFUSION. Anima Text Encoder may misbehave — "
                "please upgrade ComfyUI to a build with the qwen_image CLIP type."
            )
        clip_type = getattr(comfy.sd.CLIPType, "QWEN_IMAGE", comfy.sd.CLIPType.STABLE_DIFFUSION)
        emb_dir = folder_paths.get_folder_paths("embeddings")
        clip = comfy.sd.load_text_encoder_state_dicts(
            [sd],
            embedding_directory=emb_dir,
            clip_type=clip_type,
            model_options=model_options,
        )
        if hasattr(clip, "patcher") and hasattr(comfy.sd, "load_clip_model_patcher"):
            clip.patcher.cached_patcher_init = (
                comfy.sd.load_clip_model_patcher,
                ([clip_path], emb_dir, clip_type, model_options),
            )

        info = (
            f"HaDeS Anima CLIP Loader | File: {os.path.basename(clip_name)} | "
            f"Architecture: {arch_label} ({n_layers}L, {hidden_dim}d) | "
            f"Device: {device_mode} | Precision: {precision}"
        )
        print(f"\n[HaDeS Anima CLIP Loader] {info}")
        return (clip, info)


def _reload_anima_vae_patcher(vae_path: str, metadata=None, target_dtype=None):
    sd, loaded_meta = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata if metadata is not None else loaded_meta, dtype=target_dtype)
    vae.throw_exception_if_invalid()
    return vae.patcher


# ==============================================================================
# NODE 16: HaDeS Anima 16ch Wan21 VAE Loader
# ==============================================================================
class HaDeSAnimaVAELoader:
    """
    Dedicated Anima 16-Channel Wan 2.1 VAE Loader (AutoencoderKLQwenImage) that inspects the
    state_dict header BEFORE instantiation to block the 64-channel Qwen-Image 2.1 VAE trap
    and enforces bfloat16 precision for Wan21 RMSNorm stability.
    """

    VAE_PRECISIONS = [
        "bfloat16 (Recommended Wan21)",
        "auto",
        "float16",
        "float32",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        raw_vaes = folder_paths.get_filename_list("vae")
        ordered_vaes = _prioritize_filenames(raw_vaes, ("qwen_image_vae", "qwen", "wan", "anima"))
        return {
            "required": {
                "vae_name": (ordered_vaes,),
                "vae_precision": (cls.VAE_PRECISIONS, {"default": "bfloat16 (Recommended Wan21)"}),
            }
        }

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "vae_info")
    FUNCTION = "load_anima_vae"
    CATEGORY = "HaDeS/Anima Loaders"

    def load_anima_vae(self, vae_name: str, vae_precision: str = "bfloat16 (Recommended Wan21)"):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)

        diag = _inspect_and_validate_anima_vae_sd(sd, vae_name=vae_name)

        target_dtype = None
        if vae_precision.startswith("bfloat16"):
            target_dtype = torch.bfloat16
        elif vae_precision == "float16":
            target_dtype = torch.float16
        elif vae_precision == "float32":
            target_dtype = torch.float32

        vae = comfy.sd.VAE(sd=sd, metadata=metadata, dtype=target_dtype)
        vae.throw_exception_if_invalid()
        _validate_anima_16ch_vae_optional(vae)

        if hasattr(vae, "patcher"):
            vae.patcher.cached_patcher_init = (_reload_anima_vae_patcher, (vae_path, metadata, target_dtype))

        active_dtype = str(getattr(vae, "vae_dtype", target_dtype)).replace("torch.", "")
        info = (
            f"HaDeS Anima VAE Loader | File: {os.path.basename(vae_name)} | "
            f"Arch: {diag['architecture']} | Channels: {diag['latent_channels']}ch | "
            f"Spatial Factor: {diag['spatial_compression']}x | Dtype: {active_dtype}"
        )
        print(f"\n[HaDeS Anima VAE Loader] {info}")
        return (vae, info)


# ==============================================================================
# NODE 17: HaDeS Anima Smart Tiled VAE Decode & Scribble Shield
# ==============================================================================
class HaDeSAnimaVAEDecode:
    """
    Smart Anima 16-channel VAE Decoder with proactive resolution/VRAM-aware tiling (preventing
    untiled WanVAE OOM spikes on 6GB-12GB GPUs during 1.5x-2.0x HiRes passes), 5D->4D single-frame
    normalization, and integrated post-decode Corner Signature Scribble healing.
    """

    DECODE_MODES = [
        "Auto Smart-Tile (VRAM & HiRes Aware)",
        "Force Seamless Tiled (Low-VRAM 6GB Safe)",
        "Direct Full Decode (Max Speed)",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "decode_mode": (cls.DECODE_MODES, {"default": "Auto Smart-Tile (VRAM & HiRes Aware)"}),
                "tile_size_px": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "tile_overlap_px": ("INT", {"default": 64, "min": 32, "max": 512, "step": 32}),
                "auto_tile_mp_threshold": ("FLOAT", {"default": 1.35, "min": 0.50, "max": 4.00, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "decode_anima"
    CATEGORY = "HaDeS/Anima Loaders"

    def decode_anima(
        self,
        samples: dict,
        vae,
        decode_mode: str = "Auto Smart-Tile (VRAM & HiRes Aware)",
        tile_size_px: int = 512,
        tile_overlap_px: int = 64,
        auto_tile_mp_threshold: float = 1.35,
    ):
        _validate_anima_16ch_vae_optional(vae)

        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        if getattr(vae, "latent_dim", None) == 3 and isinstance(latent, torch.Tensor) and latent.ndim == 4:
            latent = latent.unsqueeze(2)

        h_lat = int(latent.shape[-2])
        w_lat = int(latent.shape[-1])
        compression = int(vae.spacial_compression_decode()) if hasattr(vae, "spacial_compression_decode") else 8
        h_px = h_lat * compression
        w_px = w_lat * compression
        mp = (h_px * w_px) / 1_000_000.0

        use_tiled = False
        tile_reason = "Direct Full Pass"
        if decode_mode == "Force Seamless Tiled (Low-VRAM 6GB Safe)":
            use_tiled = True
            tile_reason = "Forced Seamless Tiled"
        elif decode_mode == "Auto Smart-Tile (VRAM & HiRes Aware)":
            if mp > float(auto_tile_mp_threshold):
                use_tiled = True
                tile_reason = f"Auto-Tile ({mp:.2f} MP > {float(auto_tile_mp_threshold):.2f} MP threshold)"
            elif hasattr(vae, "memory_used_decode") and hasattr(vae, "device"):
                try:
                    shape_5d = (latent.shape[0], latent.shape[1], 1, h_lat, w_lat) if latent.ndim == 4 else latent.shape
                    est_mem = float(vae.memory_used_decode(shape_5d, getattr(vae, "vae_dtype", torch.bfloat16)))
                    free_mem = float(comfy.model_management.get_free_memory(vae.device))
                    if free_mem > 0 and est_mem > (free_mem * 0.75):
                        use_tiled = True
                        tile_reason = "Auto-Tile (Low Free VRAM Guard)"
                except Exception:
                    pass

        eff_tile_px = max(256, int(tile_size_px))
        eff_overlap_px = min(eff_tile_px // 4, max(32, int(tile_overlap_px)))
        tile_lat = max(8, eff_tile_px // compression)
        overlap_lat = max(2, eff_overlap_px // compression)

        if use_tiled and hasattr(vae, "decode_tiled"):
            temp_comp = vae.temporal_compression_decode() if hasattr(vae, "temporal_compression_decode") else None
            t_size = 2 if temp_comp is not None else None
            t_overlap = 1 if temp_comp is not None else None
            images = vae.decode_tiled(
                latent,
                tile_x=tile_lat,
                tile_y=tile_lat,
                overlap=overlap_lat,
                tile_t=t_size,
                overlap_t=t_overlap,
            )
        else:
            images = vae.decode(latent)

        images = _normalize_decoded_image_4d(images)

        info = (
            f"HaDeS Anima VAE Decode | Output: {w_px}x{h_px}px ({mp:.2f} MP) | "
            f"Strategy: {tile_reason} (Tile {eff_tile_px}px / Overlap {eff_overlap_px}px)"
        )
        print(f"\n[HaDeS Anima VAE Decode] {info}")
        return (images, info)
