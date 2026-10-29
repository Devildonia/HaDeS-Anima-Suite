import os
import re
import json
import math
import logging
import torch
import folder_paths
import safetensors.torch
import comfy.sd
import comfy.utils
import comfy.lora
import comfy.model_management

try:
    import comfy.lora_convert
    _convert_lora_fn = comfy.lora_convert.convert_lora
except ImportError:
    def _convert_lora_fn(sd):
        return sd

__version__ = "0.1.4"

logger = logging.getLogger("HaDeSAnimaSuite")

# ==============================================================================
# HADES ANIMA ENGINE - TOPOLOGY, DIFFUSERS REMAPPING & SUBSPACE MATH
# ==============================================================================
_LLM_BLOCK_RE = re.compile(
    r"(?:llm_adapter\.blocks\.|llm_adapter_blocks_|text_conditioner\.blocks\.|text_conditioner_blocks_)(\d+)(?:\.|_|$)"
)
_DIT_BLOCK_RE = re.compile(
    r"(?:^|\.|_|transformer_)blocks(?:\.|_)(\d+)(?:\.|_|$)"
)
_CLIP_LAYER_RE = re.compile(r"(?:\bmodel\.layers\.|layers_)(\d+)(?:\.|_|$)")
_SAFE_FILENAME_RE = re.compile(r"[^\w\-. ]+")
_FP8_DTYPES = tuple(
    getattr(torch, attr) for attr in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    if hasattr(torch, attr)
)

# 24-Rule Diffusers-to-ComfyUI Anima (MiniTrainDIT + AnimaTextConditioner) Mapping
DIFFUSERS_TO_COMFY_RULES = (
    ("transformer_blocks.", "blocks."),
    ("transformer_blocks_", "blocks_"),
    (".attn1.to_q", ".self_attn.q_proj"),
    (".attn1.to_k", ".self_attn.k_proj"),
    (".attn1.to_v", ".self_attn.v_proj"),
    (".attn1.to_out.0", ".self_attn.output_proj"),
    (".attn1.norm_q", ".self_attn.q_norm"),
    (".attn1.norm_k", ".self_attn.k_norm"),
    ("_attn1_to_q", "_self_attn_q_proj"),
    ("_attn1_to_k", "_self_attn_k_proj"),
    ("_attn1_to_v", "_self_attn_v_proj"),
    ("_attn1_to_out_0", "_self_attn_output_proj"),
    ("_attn1_norm_q", "_self_attn_q_norm"),
    ("_attn1_norm_k", "_self_attn_k_norm"),
    (".attn2.to_q", ".cross_attn.q_proj"),
    (".attn2.to_k", ".cross_attn.k_proj"),
    (".attn2.to_v", ".cross_attn.v_proj"),
    (".attn2.to_out.0", ".cross_attn.output_proj"),
    (".attn2.norm_q", ".cross_attn.q_norm"),
    (".attn2.norm_k", ".cross_attn.k_norm"),
    ("_attn2_to_q", "_cross_attn_q_proj"),
    ("_attn2_to_k", "_cross_attn_k_proj"),
    ("_attn2_to_v", "_cross_attn_v_proj"),
    ("_attn2_to_out_0", "_cross_attn_output_proj"),
    ("_attn2_norm_q", "_cross_attn_q_norm"),
    ("_attn2_norm_k", "_cross_attn_k_norm"),
    (".ff.net.0.proj", ".mlp.layer1"),
    (".ff.net.2", ".mlp.layer2"),
    ("_ff_net_0_proj", "_mlp_layer1"),
    ("_ff_net_2", "_mlp_layer2"),
    (".norm1.linear_1", ".adaln_modulation_self_attn.1"),
    (".norm1.linear_2", ".adaln_modulation_self_attn.2"),
    (".norm2.linear_1", ".adaln_modulation_cross_attn.1"),
    (".norm2.linear_2", ".adaln_modulation_cross_attn.2"),
    (".norm3.linear_1", ".adaln_modulation_mlp.1"),
    (".norm3.linear_2", ".adaln_modulation_mlp.2"),
    ("time_embed.emb.timestep_embedder.linear_1", "t_embedder.1.linear_1"),
    ("time_embed.emb.timestep_embedder.linear_2", "t_embedder.1.linear_2"),
    ("time_embed.norm", "t_embedding_norm"),
    ("patch_embed.proj", "x_embedder.proj.1"),
    ("proj_out", "final_layer.linear"),
    ("norm_out.linear_1", "final_layer.adaln_modulation.1"),
    ("norm_out.linear_2", "final_layer.adaln_modulation.2"),
    ("text_conditioner.", "llm_adapter."),
    ("text_conditioner_", "llm_adapter_"),
)

_KOHYA_COMFY_SUBS = (
    ("_self_attn_q_proj", ".self_attn.q_proj"),
    ("_self_attn_k_proj", ".self_attn.k_proj"),
    ("_self_attn_v_proj", ".self_attn.v_proj"),
    ("_self_attn_output_proj", ".self_attn.output_proj"),
    ("_self_attn_q_norm", ".self_attn.q_norm"),
    ("_self_attn_k_norm", ".self_attn.k_norm"),
    ("_cross_attn_q_proj", ".cross_attn.q_proj"),
    ("_cross_attn_k_proj", ".cross_attn.k_proj"),
    ("_cross_attn_v_proj", ".cross_attn.v_proj"),
    ("_cross_attn_output_proj", ".cross_attn.output_proj"),
    ("_cross_attn_q_norm", ".cross_attn.q_norm"),
    ("_cross_attn_k_norm", ".cross_attn.k_norm"),
    ("_mlp_layer1", ".mlp.layer1"),
    ("_mlp_layer2", ".mlp.layer2"),
    ("_adaln_modulation_self_attn_1", ".adaln_modulation_self_attn.1"),
    ("_adaln_modulation_self_attn_2", ".adaln_modulation_self_attn.2"),
    ("_adaln_modulation_cross_attn_1", ".adaln_modulation_cross_attn.1"),
    ("_adaln_modulation_cross_attn_2", ".adaln_modulation_cross_attn.2"),
    ("_adaln_modulation_mlp_1", ".adaln_modulation_mlp.1"),
    ("_adaln_modulation_mlp_2", ".adaln_modulation_mlp.2"),
    ("_self_attn_o_proj", ".self_attn.o_proj"),
    ("_cross_attn_o_proj", ".cross_attn.o_proj"),
    ("_norm_self_attn", ".norm_self_attn"),
    ("_norm_cross_attn", ".norm_cross_attn"),
    ("_norm_mlp", ".norm_mlp"),
    ("_mlp_0", ".mlp.0"),
    ("_mlp_2", ".mlp.2"),
    ("x_embedder_proj_1", "x_embedder.proj.1"),
    ("x_embedder_proj", "x_embedder.proj"),
    ("t_embedder_1_linear_1", "t_embedder.1.linear_1"),
    ("t_embedder_1_linear_2", "t_embedder.1.linear_2"),
    ("final_layer_adaln_modulation_1", "final_layer.adaln_modulation.1"),
    ("final_layer_adaln_modulation_2", "final_layer.adaln_modulation.2"),
    ("final_layer_linear", "final_layer.linear"),
    ("llm_adapter_in_proj", "llm_adapter.in_proj"),
    ("llm_adapter_out_proj", "llm_adapter.out_proj"),
    ("llm_adapter_embed", "llm_adapter.embed"),
    ("llm_adapter_norm", "llm_adapter.norm"),
)


def canonicalize_anima_lora_key(key):
    k = key
    for old_pat, new_pat in DIFFUSERS_TO_COMFY_RULES:
        if old_pat in k:
            k = k.replace(old_pat, new_pat)

    if k.startswith("text_encoder.model."):
        k = "text_encoders.transformer.model." + k[len("text_encoder.model."):]

    # Normalize Kohya / Musubi / Diffusers / PEFT / LyCORIS prefix into standard ComfyUI diffusion_model.*
    for root_pfx in (
        "base_model.model.transformer.",
        "base_model.model.",
        "lora_unet_",
        "lora_transformer_",
        "lycoris_",
        "transformer.",
        "model.diffusion_model.",
    ):
        if k.startswith(root_pfx):
            k = "diffusion_model." + k[len(root_pfx):]
            break

    if k.startswith("blocks.") or k.startswith("llm_adapter.") or k.startswith("x_embedder.") or k.startswith("t_embedder.") or k.startswith("final_layer."):
        k = "diffusion_model." + k

    if k.startswith("diffusion_model."):
        for old_sub, new_sub in _KOHYA_COMFY_SUBS:
            if old_sub in k:
                k = k.replace(old_sub, new_sub)

    if k.startswith("diffusion_model.blocks_"):
        k = re.sub(r"^diffusion_model\.blocks_(\d+)(?:_|\.)", r"diffusion_model.blocks.\1.", k)
    elif k.startswith("diffusion_model.llm_adapter_blocks_"):
        k = re.sub(r"^diffusion_model\.llm_adapter_blocks_(\d+)(?:_|\.)", r"diffusion_model.llm_adapter.blocks.\1.", k)

    for old_sfx, new_sfx in (
        (".lora_A.default.weight", ".lora_down.weight"),
        (".lora_B.default.weight", ".lora_up.weight"),
        (".lora_A.weight", ".lora_down.weight"),
        (".lora_B.weight", ".lora_up.weight"),
        ("_lora.down.weight", ".lora_down.weight"),
        ("_lora.up.weight", ".lora_up.weight"),
    ):
        if k.endswith(old_sfx):
            k = k[:-len(old_sfx)] + new_sfx
            break

    return k


def canonicalize_anima_lora_sd(sd, strip_llm_adapter=False, clamp_outliers=False):
    out = {}
    remapped_count = 0
    stripped_adapter_count = 0
    clamped_count = 0

    for orig_k, tensor in sd.items():
        new_k = canonicalize_anima_lora_key(orig_k)
        if new_k != orig_k:
            remapped_count += 1

        if strip_llm_adapter and ("llm_adapter" in new_k or "text_conditioner" in new_k):
            stripped_adapter_count += 1
            continue

        t_out = tensor
        if clamp_outliers and isinstance(t_out, torch.Tensor) and t_out.is_floating_point():
            if torch.isnan(t_out).any() or torch.isinf(t_out).any():
                t_out = torch.nan_to_num(t_out, nan=0.0, posinf=0.0, neginf=0.0)
                clamped_count += 1
            elif t_out.numel() > 16:
                std = float(t_out.float().std().item())
                if std > 0.0:
                    limit = 10.0 * std
                    if float(t_out.abs().max().item()) > limit:
                        t_out = t_out.clamp(-limit, limit)
                        clamped_count += 1

        out[new_k] = t_out

    return out, {
        "remapped_keys": remapped_count,
        "stripped_adapter_keys": stripped_adapter_count,
        "clamped_tensors": clamped_count,
    }


def parse_anima_block_key(key):
    if "llm_adapter" in key or "text_conditioner" in key:
        match = _LLM_BLOCK_RE.search(key)
        if match:
            return "adapter", int(match.group(1))
    else:
        match = _DIT_BLOCK_RE.search(key)
        if match:
            return "dit", int(match.group(1))
    return None, None


def detect_anima_topology(keys):
    max_dit = -1
    max_adapter = -1
    dit_seen = set()
    for k in keys:
        target_k = k[0] if isinstance(k, tuple) else k
        kind, idx = parse_anima_block_key(target_k)
        if kind == "dit" and idx is not None:
            dit_seen.add(idx)
            if idx > max_dit:
                max_dit = idx
        elif kind == "adapter" and idx is not None and idx > max_adapter:
            max_adapter = idx
    if max_dit < 28:
        n_dit = 28
    elif len(dit_seen) < (max_dit + 1):
        # Partial/sparse LoRA key set: snap to canonical Anima tier (40L for 2.9B, 52L for 3.8B)
        n_dit = 40 if max_dit < 40 else (52 if max_dit < 52 else (max_dit + 1))
    else:
        n_dit = max_dit + 1
    n_adapter = (max_adapter + 1) if max_adapter >= 6 else 6
    return n_dit, n_adapter


def map_to_canonical_34(kind, idx, n_dit=28, n_adapter=6):
    if kind == "dit" and idx is not None and 0 <= idx < n_dit:
        if n_dit == 28:
            return idx
        return min(27, max(0, int((idx / float(n_dit)) * 28)))
    if kind == "adapter" and idx is not None and 0 <= idx < n_adapter:
        if n_adapter == 6:
            return 28 + idx
        return 28 + min(5, max(0, int((idx / float(n_adapter)) * 6)))
    return None


def get_anima_block_idx(key, n_dit=28, n_adapter=6):
    kind, idx = parse_anima_block_key(key)
    mapped = map_to_canonical_34(kind, idx, n_dit=n_dit, n_adapter=n_adapter)
    if mapped is not None:
        return mapped
    if "llm_adapter" in key or "text_conditioner" in key:
        low_k = key.lower()
        if "out_proj" in low_k or ".norm" in low_k or "_norm" in low_k:
            return 33
        return 28
    return None


def get_clean_weight(model_or_patcher, key, current_weight):
    patcher = model_or_patcher.patcher if hasattr(model_or_patcher, "patcher") else model_or_patcher
    if hasattr(patcher, "backup") and patcher.backup:
        if key in patcher.backup:
            return patcher.backup[key].weight
        if key.startswith("diffusion_model."):
            clean_key = key[len("diffusion_model."):]
            if clean_key in patcher.backup:
                return patcher.backup[clean_key].weight
        else:
            diff_key = f"diffusion_model.{key}"
            if diff_key in patcher.backup:
                return patcher.backup[diff_key].weight
    return current_weight


def get_baked_weight(model_or_patcher, patch_key, base_weight):
    patcher = model_or_patcher.patcher if hasattr(model_or_patcher, "patcher") else model_or_patcher
    patches = patcher.patches.get(patch_key, [])
    clean_w = get_clean_weight(patcher, patch_key, base_weight)
    if not patches:
        return clean_w
    orig_dtype = clean_w.dtype
    calc_fn = patcher.calculate_weight if hasattr(patcher, "calculate_weight") else comfy.lora.calculate_weight
    return calc_fn(patches, clean_w.clone().float(), patch_key).to(orig_dtype)


def deterministic_sample_pair(w1, w2, max_samples=4096):
    f1 = w1.flatten()
    f2 = w2.flatten()
    numel = f1.numel()
    if numel <= max_samples:
        return f1.float(), f2.to(f1.device).float()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42 + (numel % 997))
    idx = torch.randint(0, numel, (max_samples,), generator=gen, device="cpu")
    if f1.device.type != "cpu":
        idx = idx.to(f1.device)
    s1 = f1.index_select(0, idx).float()
    s2 = f2.to(f1.device).index_select(0, idx).float()
    return s1, s2


def normalize_block_scores(raw_dict, default_val=0.0):
    vals = [v for v in raw_dict.values() if v is not None]
    if not vals:
        return {b: default_val for b in range(34)}
    v_min = min(vals)
    v_max = max(vals)
    span = v_max - v_min
    if span < 1e-7:
        return {b: (0.5 if raw_dict.get(b) is not None else default_val) for b in range(34)}
    return {
        b: ((raw_dict[b] - v_min) / span if raw_dict.get(b) is not None else default_val)
        for b in range(34)
    }


def parse_recipe_override(recipe_str, group_map):
    if not recipe_str or not str(recipe_str).strip():
        return None
    text = str(recipe_str).strip()
    if text.startswith("{"):
        data = json.loads(text)
        if "predicted_ratios_34" in data and len(data["predicted_ratios_34"]) == 34:
            return [float(x) for x in data["predicted_ratios_34"]]
        if "groups" in data and isinstance(data["groups"], dict):
            ratios = [0.5] * 28 + [0.0] * 6
            for g_name, indices in group_map.items():
                if g_name in data["groups"]:
                    val = float(data["groups"][g_name])
                    for idx in indices:
                        ratios[idx] = val
            return ratios
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 34:
        return [float(p) for p in parts]
    raise ValueError(f"Invalid recipe_override: expected 34 comma-separated floats or HaDeS JSON recipe, got {len(parts)} values.")


def compute_factor_energy(up, down, scale):
    if up.ndim == 4 and down.ndim == 4 and up.shape[2] == 1 and up.shape[3] == 1:
        u = up.reshape(up.shape[0], up.shape[1]).float()
        d = down.reshape(down.shape[0], -1).float()
    elif up.ndim == 2 and down.ndim == 2:
        u = up.float()
        d = down.float()
    else:
        return None
    gu = torch.matmul(u.T, u)
    gd = torch.matmul(d, d.T)
    tr = torch.sum(gu * gd.T).item()
    return abs(float(scale)) * math.sqrt(max(0.0, tr))


def compute_subspace_metrics(up1, down1, scale1, up2, down2, scale2):
    if up1.ndim == 4 and down1.ndim == 4 and up1.shape[2] == 1 and up1.shape[3] == 1:
        u1 = up1.reshape(up1.shape[0], up1.shape[1]).float()
        d1 = down1.reshape(down1.shape[0], -1).float()
    elif up1.ndim == 2 and down1.ndim == 2:
        u1, d1 = up1.float(), down1.float()
    else:
        return None

    if up2.ndim == 4 and down2.ndim == 4 and up2.shape[2] == 1 and up2.shape[3] == 1:
        u2 = up2.reshape(up2.shape[0], up2.shape[1]).to(u1.device).float()
        d2 = down2.reshape(down2.shape[0], -1).to(u1.device).float()
    elif up2.ndim == 2 and down2.ndim == 2:
        u2, d2 = up2.to(u1.device).float(), down2.to(u1.device).float()
    else:
        return None

    if u1.shape[0] != u2.shape[0] or d1.shape[1] != d2.shape[1]:
        return None

    norm1 = compute_factor_energy(u1, d1, scale1)
    norm2 = compute_factor_energy(u2, d2, scale2)
    if norm1 is None or norm2 is None or (norm1 < 1e-8 and norm2 < 1e-8):
        return None

    # Exact Frobenius inner product Tr((U1^T U2)(D2 D1^T))
    u12 = torch.matmul(u1.T, u2)
    d21 = torch.matmul(d1, d2.T)
    frob_dot = float(scale1) * float(scale2) * torch.sum(u12 * d21).item()
    cos_frob = max(-1.0, min(1.0, frob_dot / ((norm1 * norm2) + 1e-8)))

    # Principal angles / Grassmann subspace overlap via reduced QR with rank-baseline correction
    qu1, _ = torch.linalg.qr(u1, mode="reduced")
    qu2, _ = torch.linalg.qr(u2, mode="reduced")
    qd1, _ = torch.linalg.qr(d1.T, mode="reduced")
    qd2, _ = torch.linalg.qr(d2.T, mode="reduced")

    ru1, ru2 = qu1.shape[1], qu2.shape[1]
    rd1, rd2 = qd1.shape[1], qd2.shape[1]
    ru_min, ru_max = max(1, min(ru1, ru2)), max(1, max(ru1, ru2))
    rd_min, rd_max = max(1, min(rd1, rd2)), max(1, max(rd1, rd2))

    raw_ov_u = torch.norm(torch.matmul(qu1.T, qu2)).item() / math.sqrt(ru_min)
    raw_ov_d = torch.norm(torch.matmul(qd1.T, qd2)).item() / math.sqrt(rd_min)

    base_u = min(0.95, math.sqrt(ru_max / float(max(1, u1.shape[0]))))
    base_d = min(0.95, math.sqrt(rd_max / float(max(1, d1.shape[1]))))
    ov_u = max(0.0, (raw_ov_u - base_u) / (1.0 - base_u))
    ov_d = max(0.0, (raw_ov_d - base_d) / (1.0 - base_d))
    subspace_overlap = max(0.0, min(1.0, math.sqrt(ov_u * ov_d)))
    energy_ratio = norm2 / (norm1 + norm2 + 1e-8)

    return {
        "cos_sim": cos_frob,
        "subspace_overlap": subspace_overlap,
        "energy_ratio": energy_ratio,
        "norm1": norm1,
        "norm2": norm2,
    }


def apply_model_block_merge(model1, model2, block_ratios_34, fallback_ratio):
    m = model1.clone()
    base_sd1 = m.model.state_dict()
    base_sd2 = model2.model.state_dict()
    valid_keys = m.model_keys if hasattr(m, "model_keys") else None

    n_dit, n_adapter = detect_anima_topology(base_sd1.keys())
    patches_to_add = {}
    merged_tensors = 0
    compatible_keys = 0
    warned_fp8 = False

    for k, base_w1 in base_sd1.items():
        if k not in base_sd2:
            continue
        if valid_keys is not None and k not in valid_keys:
            continue

        base_w2 = base_sd2[k]
        if base_w1.shape != base_w2.shape:
            raise ValueError(f"HaDeS Anima Merger: Tensor shape mismatch on key '{k}': {tuple(base_w1.shape)} vs {tuple(base_w2.shape)}.")

        if k.endswith(".scale_weight") or not (base_w1.is_floating_point() or base_w1.dtype in _FP8_DTYPES):
            continue

        if not warned_fp8 and (base_w1.dtype in _FP8_DTYPES or base_w2.dtype in _FP8_DTYPES):
            logger.warning("HaDeS Anima Merger: FP8 quantized weights detected. Merging will cast back to FP8 and may lose scale_weight precision.")
            warned_fp8 = True

        compatible_keys += 1
        block_idx = get_anima_block_idx(k, n_dit=n_dit, n_adapter=n_adapter)
        target_ratio = block_ratios_34[block_idx] if block_idx is not None else fallback_ratio

        if abs(target_ratio) > 1e-5:
            patch_key = f"diffusion_model.{k}" if not k.startswith("diffusion_model.") else k
            baked1_float = get_baked_weight(m, patch_key, base_w1).float()
            baked2_float = get_baked_weight(model2, patch_key, base_w2).to(baked1_float.device).float()

            if torch.equal(baked1_float, baked2_float):
                continue

            v_modified = baked1_float + ((baked2_float - baked1_float) * target_ratio)
            v_target = v_modified.to(base_w1.dtype)

            patches_to_add[patch_key] = ("set", (v_target.cpu(),))
            merged_tensors += 1

    if compatible_keys == 0:
        raise ValueError("HaDeS Anima Merger: model1 and model2 share 0 compatible floating-point tensors.")

    if patches_to_add:
        m.add_patches(patches_to_add, 1.0)

    return m, merged_tensors, (n_dit, n_adapter)


# ==============================================================================
# 1. HADES ANIMA MODEL BLOCK MERGER (MANUAL)
# ==============================================================================
class HaDeSAnimaModelBlockMerger:
    GROUP_MAP = {
        "BLOCKS_Layout_Geometry": [0, 1, 2, 3, 4],
        "BLOCKS_Global_Composition": [5, 6, 7, 8, 9],
        "BLOCKS_Subject_Identity": [10, 11, 12, 13, 14],
        "BLOCKS_Material_Substance": [15, 16, 17, 18, 19],
        "BLOCKS_Art_Style_Medium": [20, 21, 22, 23],
        "BLOCKS_Lighting_Sharpness": [24, 25, 26, 27],
        "ADAPTER_Text_Alignment": [28, 29, 30, 31, 32, 33]
    }
    ORDERED_KEYS = list(GROUP_MAP.keys())

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "base_ratio": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "0.0 = Model 1, 1.0 = Model 2 (Fallback for non-block layers)"}),
            },
            "optional": {
                "recipe_override": ("STRING", {"default": "", "tooltip": "Optional 34-float CSV vector or JSON recipe from Predictive Merger"}),
            }
        }
        tooltips = {
            "BLOCKS_Layout_Geometry": "Canonical Blocks 0-4: Camera geometry, perspective, and framing",
            "BLOCKS_Global_Composition": "Canonical Blocks 5-9: Global composition, pose, and light masses",
            "BLOCKS_Subject_Identity": "Canonical Blocks 10-14: Subject identity, face features, and anatomy",
            "BLOCKS_Material_Substance": "Canonical Blocks 15-19: Materials, clothing folds, and surface textures",
            "BLOCKS_Art_Style_Medium": "Canonical Blocks 20-23: Art style, lineart weight, brushwork, and palette",
            "BLOCKS_Lighting_Sharpness": "Canonical Blocks 24-27: Final lighting, micro-contrast, and sharpness",
            "ADAPTER_Text_Alignment": "Canonical Blocks 28-33: LLM Adapter (Qwen3->T5 Bridge). Default 0.0 preserves Anchor Model 1 text alignment."
        }
        for name in s.ORDERED_KEYS:
            default_val = 0.0 if name == "ADAPTER_Text_Alignment" else 0.5
            inputs["required"][name] = ("FLOAT", {"default": default_val, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": tooltips.get(name, "")})
        return inputs

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "merge_models"
    CATEGORY = "HaDeS/Anima Mergers"

    def merge_models(self, model1, model2, base_ratio, recipe_override="", **kwargs):
        override_ratios = parse_recipe_override(recipe_override, self.GROUP_MAP)
        if override_ratios is not None:
            final_ratios = override_ratios
        else:
            final_ratios = [base_ratio] * 34
            for group_name, indices in self.GROUP_MAP.items():
                fallback = 0.0 if group_name == "ADAPTER_Text_Alignment" else base_ratio
                ratio_val = kwargs.get(group_name, fallback)
                for idx in indices:
                    final_ratios[idx] = ratio_val

        m, merged_blocks, (n_dit, n_adapter) = apply_model_block_merge(model1, model2, final_ratios, base_ratio)
        mode_str = "Recipe Override" if override_ratios is not None else "Manual Groups"
        return (
            m,
            f"HaDeS Anima Model Merged ({mode_str}) | Topology: {n_dit}+{n_adapter} | Modified Tensors: {merged_blocks}"
        )


# ==============================================================================
# 2. HADES ANIMA PREDICTIVE MODEL MERGER (AUTO SPECTRAL / TASK VECTORS)
# ==============================================================================
class HaDeSAnimaPredictiveModelMerger:
    GROUP_MAP = HaDeSAnimaModelBlockMerger.GROUP_MAP

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "preset": ([
                    "Style Transfer (Keep A identity, B style)",
                    "Interference Guard (Anti-Burn / Artifact Shield)",
                    "Complementary Richness (Fill gaps of A with B)",
                    "Balanced Geometric Fusion"
                ],),
                "target_influence": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Global baseline influence for Model 2"
                }),
                "aggressiveness": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.5, "step": 0.05,
                    "tooltip": "Sensitivity to inter-block spectral divergence and task-vector angles"
                }),
            },
            "optional": {
                "base_model": ("MODEL", {"tooltip": "Optional common base checkpoint to compute exact Task Vectors (W1 - Wbase, W2 - Wbase)"}),
                "protect_llm_adapter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep llm_adapter (Blocks 28-33) locked to Model 1 (0.0) per Anima's llm_adapter_lr=0 rule."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "predictive_merge"
    CATEGORY = "HaDeS/Anima Mergers"

    def predictive_merge(self, model1, model2, preset, target_influence, aggressiveness, base_model=None, protect_llm_adapter=True):
        base_sd1 = model1.model.state_dict()
        base_sd2 = model2.model.state_dict()
        base_sd0 = base_model.model.state_dict() if base_model is not None else None
        valid_keys = model1.model_keys if hasattr(model1, "model_keys") else None

        n_dit, n_adapter = detect_anima_topology(base_sd1.keys())
        block_metrics = {i: {"cos_sim": [], "diff_norm": []} for i in range(34)}

        for k, base_w1 in base_sd1.items():
            if k not in base_sd2:
                continue
            if valid_keys is not None and k not in valid_keys:
                continue
            if k.endswith(".scale_weight") or not (base_w1.is_floating_point() or base_w1.dtype in _FP8_DTYPES):
                continue

            block_idx = get_anima_block_idx(k, n_dit=n_dit, n_adapter=n_adapter)
            if block_idx is not None:
                patch_key = f"diffusion_model.{k}" if not k.startswith("diffusion_model.") else k
                w1 = get_baked_weight(model1, patch_key, base_w1)
                w2 = get_baked_weight(model2, patch_key, base_sd2[k])
                if w1.shape != w2.shape:
                    raise ValueError(f"HaDeS Predictive Merger: Shape mismatch on '{k}': {tuple(w1.shape)} vs {tuple(w2.shape)}.")

                t1, t2 = deterministic_sample_pair(w1, w2, max_samples=4096)

                if base_sd0 is not None and k in base_sd0 and base_sd0[k].shape == base_w1.shape:
                    w0 = get_baked_weight(base_model, patch_key, base_sd0[k])
                    t0, _ = deterministic_sample_pair(w0, w0, max_samples=4096)
                    t0 = t0.to(t1.device)
                    d1 = t1 - t0
                    d2 = t2 - t0
                    n1 = torch.norm(d1).item()
                    n2 = torch.norm(d2).item()
                    if n1 > 1e-6 or n2 > 1e-6:
                        cos_sim = torch.nn.functional.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
                        norm_diff = n2 / (n1 + n2 + 1e-7)
                    else:
                        cos_sim = 1.0
                        norm_diff = 0.0
                else:
                    c1 = t1 - t1.mean()
                    c2 = t2 - t2.mean()
                    cos_sim = torch.nn.functional.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()
                    norm1 = torch.norm(t1).item()
                    norm_diff = (torch.norm(t2 - t1).item()) / (norm1 + 1e-7)

                block_metrics[block_idx]["cos_sim"].append(cos_sim)
                block_metrics[block_idx]["diff_norm"].append(norm_diff)

        raw_cos = {}
        raw_diff = {}
        for b, m_data in block_metrics.items():
            raw_cos[b] = (sum(m_data["cos_sim"]) / len(m_data["cos_sim"])) if m_data["cos_sim"] else None
            raw_diff[b] = (sum(m_data["diff_norm"]) / len(m_data["diff_norm"])) if m_data["diff_norm"] else None

        norm_divergence = normalize_block_scores(raw_diff, default_val=0.0)
        avg_cos = {b: (raw_cos[b] if raw_cos[b] is not None else 1.0) for b in range(34)}

        predicted_ratios = [target_influence] * 34

        for b in range(34):
            if protect_llm_adapter and b >= 28:
                predicted_ratios[b] = 0.0
                continue

            c_sim = avg_cos[b]
            rel_div = norm_divergence[b]
            eff_div = max(1.0 - c_sim, rel_div * 0.65) if base_model is None else (1.0 - c_sim)

            if preset == "Style Transfer (Keep A identity, B style)":
                if b in range(0, 15):
                    ratio = target_influence * 0.15 * max(0.05, 1.0 - (eff_div * 0.7 * aggressiveness))
                elif b in range(15, 20):
                    ratio = target_influence * (0.65 + (rel_div - 0.5) * 0.15 * aggressiveness)
                elif b in range(20, 28):
                    ratio = target_influence * (1.0 + (eff_div * 0.65 * aggressiveness))
                else:
                    ratio = target_influence * (0.70 + (rel_div - 0.5) * 0.15 * aggressiveness)

            elif preset == "Interference Guard (Anti-Burn / Artifact Shield)":
                safety_damping = max(0.15, 1.0 - (eff_div * 0.55 * aggressiveness))
                ratio = target_influence * safety_damping

            elif preset == "Complementary Richness (Fill gaps of A with B)":
                divergence_boost = 0.75 + (rel_div * 0.65 * aggressiveness)
                ratio = target_influence * divergence_boost

            else:  # Balanced Geometric Fusion
                norm_balance = 1.0 - (eff_div * 0.35 * aggressiveness)
                ratio = target_influence * max(0.2, norm_balance)

            predicted_ratios[b] = round(max(-0.2, min(1.2, float(ratio))), 3)

        m, merged_blocks, _ = apply_model_block_merge(model1, model2, predicted_ratios, target_influence)

        group_summary = {}
        for group_name, indices in self.GROUP_MAP.items():
            sub_ratios = [predicted_ratios[idx] for idx in indices]
            sub_cossim = [avg_cos[idx] for idx in indices]
            sub_div = [norm_divergence[idx] for idx in indices]
            group_summary[group_name] = {
                "avg_ratio": round(sum(sub_ratios) / len(sub_ratios), 3),
                "avg_cos_similarity": round(sum(sub_cossim) / len(sub_cossim), 4),
                "rel_divergence": round(sum(sub_div) / len(sub_div), 3),
                "block_ratios": sub_ratios
            }

        recipe_vector = ",".join(str(r) for r in predicted_ratios)
        analysis_mode = "Task Vectors (vs base_model)" if base_model is not None else "Relative Block Divergence (Z-Normalized)"
        bridge_status = "Locked (0.0)" if protect_llm_adapter else "Adaptive"

        info_lines = [
            "=== HADES ANIMA PREDICTIVE MERGE RECIPE ===",
            f"Preset: {preset} | Analysis: {analysis_mode}",
            f"Topology: {n_dit} DiT + {n_adapter} Adapter (Bridge: {bridge_status}) | Tensors merged: {merged_blocks}",
            f"Target Influence: {target_influence} | Aggressiveness: {aggressiveness}\n",
            "--- GROUP PREDICTIONS ---"
        ]
        for grp, data in group_summary.items():
            info_lines.append(
                f"  • {grp:<26}: Ratio {data['avg_ratio']:+.3f} (CosSim: {data['avg_cos_similarity']:.4f}, RelDiv: {data['rel_divergence']:.2f})"
            )

        info_lines.append("\n--- VECTORS OVERRIDE (34 BLOCKS) ---")
        info_lines.append(recipe_vector)

        info_lines.append("\n--- JSON PRESET EXPORT ---")
        info_lines.append(json.dumps({
            "author": "HaDeS",
            "preset_name": preset,
            "topology": {"dit_blocks": n_dit, "adapter_blocks": n_adapter},
            "protect_llm_adapter": protect_llm_adapter,
            "target_influence": target_influence,
            "aggressiveness": aggressiveness,
            "groups": {g: data["avg_ratio"] for g, data in group_summary.items()},
            "predicted_ratios_34": predicted_ratios
        }, indent=2))

        return (m, "\n".join(info_lines))


# ==============================================================================
# 3. HADES ANIMA PREDICTIVE MULTI LORA LOADER (SUBSPACE & ENERGY MODULATION)
# ==============================================================================
class HaDeSAnimaPredictiveMultiLoraLoader:
    GROUP_MAP = HaDeSAnimaModelBlockMerger.GROUP_MAP

    @classmethod
    def INPUT_TYPES(s):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "enable_all": ("BOOLEAN", {"default": True, "label_on": "Enable All", "label_off": "Mute All"}),
                "preset": ([
                    "Character + Style Fusion (A identity, B style)",
                    "Anti-Overcook (Damped Safe Blend)",
                    "Orthogonal Synthesis (Complementary features)",
                    "Balanced 50/50 Adaptive"
                ],),
                "aggressiveness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.5, "step": 0.05}),

                # Slot 1
                "lora_1": (lora_list, {"default": "None"}),
                "on_1": ("BOOLEAN", {"default": True}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # Slot 2
                "lora_2": (lora_list, {"default": "None"}),
                "on_2": ("BOOLEAN", {"default": True}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # Slot 3
                "lora_3": (lora_list, {"default": "None"}),
                "on_3": ("BOOLEAN", {"default": True}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # Slot 4
                "lora_4": (lora_list, {"default": "None"}),
                "on_4": ("BOOLEAN", {"default": True}),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "protect_llm_adapter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Strip dirty llm_adapter keys from LoRAs to protect Qwen3->T5 prompt adherence."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "apply_predictive_multi_lora"
    CATEGORY = "HaDeS/Anima Mergers"

    def _load_raw_lora(self, lora_name, model, clip, protect_llm_adapter=True):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        raw = comfy.utils.load_torch_file(path, safe_load=True)
        converted = _convert_lora_fn(raw)
        canonical_sd, _ = canonicalize_anima_lora_sd(converted, strip_llm_adapter=protect_llm_adapter)
        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        return comfy.lora.load_lora(canonical_sd, key_map)

    def _extract_adapter_factors(self, patch_data):
        weights = getattr(patch_data, "weights", patch_data)
        if isinstance(weights, tuple) and len(weights) >= 2:
            if len(weights) == 2 and weights[0] == "diff" and isinstance(weights[1], tuple):
                return None
            up, down = weights[0], weights[1]
            if isinstance(up, torch.Tensor) and isinstance(down, torch.Tensor):
                rank = down.shape[0] if down.ndim >= 1 else up.shape[1]
                alpha = weights[2] if len(weights) > 2 and weights[2] is not None else rank
                scale = float(alpha) / float(rank) if rank > 0 else 1.0
                return up, down, scale
        return None

    def _reconstruct_delta(self, patch_data):
        weights = getattr(patch_data, "weights", patch_data)
        if isinstance(weights, tuple):
            if len(weights) == 2 and weights[0] == "diff" and isinstance(weights[1], tuple):
                return weights[1][0].float()
            factors = self._extract_adapter_factors(patch_data)
            if factors is not None:
                up, down, scale = factors
                if up.ndim == 4 and down.ndim == 4:
                    return torch.nn.functional.conv2d(down.float().permute(1, 0, 2, 3), up.float()).permute(1, 0, 2, 3) * scale
                return torch.matmul(up.flatten(1).float(), down.flatten(1).float()) * scale
        return None

    def _compute_slot_curve(self, base_lora, sec_lora, preset, aggressiveness, n_dit, n_adapter, protect_llm_adapter=True):
        block_stats = {i: {"overlap": [], "energy": [], "cos": []} for i in range(34)}
        sec_block_energy = {i: [] for i in range(34)}

        for k, sec_patch in sec_lora.items():
            target_k = k[0] if isinstance(k, tuple) else k
            b_idx = get_anima_block_idx(target_k, n_dit=n_dit, n_adapter=n_adapter)
            if b_idx is None:
                continue

            f_sec = self._extract_adapter_factors(sec_patch)
            if f_sec is not None:
                e_sec = compute_factor_energy(*f_sec)
                if e_sec is not None:
                    sec_block_energy[b_idx].append(e_sec)

            if k in base_lora:
                f_base = self._extract_adapter_factors(base_lora[k])
                if f_base is not None and f_sec is not None:
                    m = compute_subspace_metrics(*f_base, *f_sec)
                    if m is not None:
                        block_stats[b_idx]["overlap"].append(m["subspace_overlap"])
                        block_stats[b_idx]["energy"].append(m["energy_ratio"])
                        block_stats[b_idx]["cos"].append(m["cos_sim"])
                        continue

                d1 = self._reconstruct_delta(base_lora[k])
                d2 = self._reconstruct_delta(sec_patch)
                if d1 is not None and d2 is not None and d1.shape == d2.shape:
                    s1, s2 = deterministic_sample_pair(d1, d2, max_samples=2048)
                    n1 = torch.norm(s1).item()
                    n2 = torch.norm(s2).item()
                    cos = torch.nn.functional.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
                    block_stats[b_idx]["overlap"].append(abs(cos))
                    block_stats[b_idx]["energy"].append(n2 / (n1 + n2 + 1e-8))
                    block_stats[b_idx]["cos"].append(cos)

        raw_sec_e = {b: (sum(v) / len(v) if v else None) for b, v in sec_block_energy.items()}
        norm_sec_e = normalize_block_scores(raw_sec_e, default_val=0.5)

        sec_mults = [1.0] * 34
        prim_collision = [0.0] * 34

        for b in range(34):
            if protect_llm_adapter and b >= 28:
                sec_mults[b] = 0.0
                continue

            st = block_stats[b]
            ov = (sum(st["overlap"]) / len(st["overlap"])) if st["overlap"] else 0.25
            er = (sum(st["energy"]) / len(st["energy"])) if st["energy"] else norm_sec_e[b]
            collision = max(ov, abs(sum(st["cos"]) / len(st["cos"])) if st["cos"] else 0.0)
            prim_collision[b] = collision

            if preset == "Character + Style Fusion (A identity, B style)":
                if b in range(0, 15):
                    mult = 0.15 * max(0.05, 1.0 - (collision * 0.85 * aggressiveness))
                elif b in range(15, 20):
                    mult = 0.60 * (1.0 - (collision * 0.25 * aggressiveness))
                elif b in range(20, 28):
                    orth_gain = (1.0 - collision * 0.5) * (0.5 + 0.5 * er)
                    mult = 1.0 + (orth_gain * 0.55 * aggressiveness)
                else:
                    mult = 0.80 * (1.0 - (collision * 0.2 * aggressiveness))
            elif preset == "Anti-Overcook (Damped Safe Blend)":
                mult = max(0.2, 1.0 - ((collision * 0.55 + er * 0.25) * aggressiveness))
            elif preset == "Orthogonal Synthesis (Complementary features)":
                orth = 1.0 - collision
                mult = 0.45 + ((orth * 0.6 + er * 0.25) * aggressiveness)
            else:
                mult = 1.0 - (collision * 0.35 * aggressiveness)

            sec_mults[b] = round(max(0.0, min(1.5, float(mult))), 3)

        return sec_mults, prim_collision

    def apply_predictive_multi_lora(self, model, clip, enable_all, preset, aggressiveness,
                                    lora_1, on_1, strength_1,
                                    lora_2, on_2, strength_2,
                                    lora_3, on_3, strength_3,
                                    lora_4, on_4, strength_4,
                                    protect_llm_adapter=True):
        if not enable_all:
            return (model, clip, "HaDeS Predictive Multi-LoRA: Master bypass enabled (Mute All).")

        raw_slots = [
            (lora_1, on_1, strength_1),
            (lora_2, on_2, strength_2),
            (lora_3, on_3, strength_3),
            (lora_4, on_4, strength_4),
        ]

        active_slots = [(l_name, s_val) for l_name, is_on, s_val in raw_slots if is_on and l_name != "None" and s_val != 0.0]

        if not active_slots:
            return (model, clip, "HaDeS Predictive Multi-LoRA: No active LoRA slots.")

        new_model = model.clone()
        new_clip = clip.clone() if clip is not None else None

        if len(active_slots) == 1:
            l_name, str_val = active_slots[0]
            l_path = folder_paths.get_full_path_or_raise("loras", l_name)
            raw_sd = _convert_lora_fn(comfy.utils.load_torch_file(l_path, safe_load=True))
            canon_sd, stats = canonicalize_anima_lora_sd(raw_sd, strip_llm_adapter=protect_llm_adapter)
            m_res, c_res = comfy.sd.load_lora_for_models(new_model, new_clip, canon_sd, str_val, str_val)
            return (
                m_res,
                c_res,
                f"HaDeS Multi-LoRA: Single active LoRA '{l_name}' applied at strength {str_val:+.2f} "
                f"(Remapped: {stats['remapped_keys']}, Adapter Stripped: {stats['stripped_adapter_keys']})"
            )

        model_sd = new_model.model.state_dict()
        n_dit, n_adapter = detect_anima_topology(model_sd.keys())

        loaded_loras = [self._load_raw_lora(name, model, clip, protect_llm_adapter=protect_llm_adapter) for name, _ in active_slots]
        base_lora = loaded_loras[0]

        secondary_curves = []
        max_collision_per_block = [0.0] * 34

        for sec_lora in loaded_loras[1:]:
            sec_mults, prim_col = self._compute_slot_curve(
                base_lora, sec_lora, preset, aggressiveness, n_dit, n_adapter, protect_llm_adapter=protect_llm_adapter
            )
            secondary_curves.append(sec_mults)
            for b in range(34):
                if prim_col[b] > max_collision_per_block[b]:
                    max_collision_per_block[b] = prim_col[b]

        primary_multipliers = [1.0] * 34
        for b in range(34):
            if protect_llm_adapter and b >= 28:
                primary_multipliers[b] = 0.0
                continue

            col = max_collision_per_block[b]
            if preset == "Character + Style Fusion (A identity, B style)":
                if b in range(0, 15):
                    prim = 1.0
                elif b in range(15, 20):
                    prim = 0.90
                elif b in range(20, 28):
                    prim = max(0.35, 0.75 - (col * 0.25 * aggressiveness))
                else:
                    prim = 0.85
            elif preset == "Anti-Overcook (Damped Safe Blend)":
                prim = max(0.5, 1.0 - (col * 0.35 * aggressiveness))
            elif preset == "Orthogonal Synthesis (Complementary features)":
                prim = max(0.6, 1.0 - (col * 0.25 * aggressiveness))
            else:
                prim = max(0.65, 1.0 - (col * 0.2 * aggressiveness))
            primary_multipliers[b] = round(max(0.2, min(1.2, float(prim))), 3)

        all_slot_curves = [primary_multipliers] + secondary_curves

        for idx, (l_dict, (l_name, str_val), slot_curve) in enumerate(zip(loaded_loras, active_slots, all_slot_curves)):
            dit_slice = slot_curve[:28]
            avg_curve_mult = round(sum(dit_slice) / len(dit_slice), 3)
            lora_patches_by_strength = {}
            for k, patch_data in l_dict.items():
                target_key = k[0] if isinstance(k, tuple) else k
                check_key = target_key[len("diffusion_model."):] if target_key.startswith("diffusion_model.") else target_key
                if target_key in model_sd or check_key in model_sd:
                    block_idx = get_anima_block_idx(target_key, n_dit=n_dit, n_adapter=n_adapter)
                    block_weight = slot_curve[block_idx] if block_idx is not None else (1.0 if idx == 0 else avg_curve_mult)
                    eff_strength = round(str_val * block_weight, 4)
                    if eff_strength != 0.0:
                        lora_patches_by_strength.setdefault(eff_strength, {})[k] = patch_data

            for s_val, group in lora_patches_by_strength.items():
                new_model.add_patches(group, strength_patch=s_val)

        if new_clip is not None and hasattr(new_clip, "patcher"):
            clip_sd = new_clip.patcher.model.state_dict()
            clip_key_counts = {}
            for l_dict in loaded_loras:
                for k in l_dict.keys():
                    target_key = k[0] if isinstance(k, tuple) else k
                    if target_key in clip_sd:
                        clip_key_counts[target_key] = clip_key_counts.get(target_key, 0) + 1

            for idx, (l_dict, (l_name, str_val)) in enumerate(zip(loaded_loras, active_slots)):
                if str_val == 0.0:
                    continue
                for k, patch_data in l_dict.items():
                    target_key = k[0] if isinstance(k, tuple) else k
                    if target_key in clip_sd:
                        overlap = clip_key_counts.get(target_key, 1)
                        clip_damping = 1.0 if (idx == 0 and overlap == 1) else (1.0 / (overlap ** 0.5))
                        new_clip.add_patches({k: patch_data}, round(str_val * clip_damping, 4))

        info_lines = [
            "=== HADES PREDICTIVE MULTI-LORA APPLIED ===",
            f"Preset: {preset} | Aggressiveness: {aggressiveness} | Topology: {n_dit}+{n_adapter} | Protect Adapter: {protect_llm_adapter}",
            f"Active LoRAs ({len(active_slots)}): " + ", ".join([f"{name} ({s:+.2f})" for name, s in active_slots]),
            "\n--- PRIMARY SLOT (PRI) MULTIPLIERS (34 BLOCKS) ---",
            ",".join(str(r) for r in primary_multipliers),
        ]
        for s_idx, curve in enumerate(secondary_curves, start=2):
            s_name, _ = active_slots[s_idx - 1]
            info_lines.append(f"\n--- SECONDARY SLOT S{s_idx} ({s_name}) MULTIPLIERS (34 BLOCKS) ---")
            info_lines.append(",".join(str(r) for r in curve))

        return (new_model, new_clip, "\n".join(info_lines))


# ==============================================================================
# 4. HADES ANIMA STANDALONE LORA MERGER (FAST QR-SVD EXPORTER)
# ==============================================================================
class HaDeSAnimaStandaloneLoraMerger:
    GROUP_MAP = HaDeSAnimaModelBlockMerger.GROUP_MAP
    _UP_DOWN_SUFFIXES = (
        (".lora_up.weight", ".lora_down.weight"),
        (".lora_B.weight", ".lora_A.weight"),
        ("_lora.up.weight", "_lora.down.weight"),
        (".lora.up.weight", ".lora.down.weight"),
        (".lora_B.default.weight", ".lora_A.default.weight"),
    )
    _UNSUPPORTED_MARKERS = (".hada_w1", ".hada_w2", ".lokr_w1", ".lokr_w2", ".dora_scale", "dora_scale")
    TOPOLOGY_CHOICES = [
        "Auto-Detect from Keys",
        "Anima 2.0B (28 DiT + 6 Adapter)",
        "Anima 2.9B (40 DiT + 6 Adapter)",
        "Anima 3.8B (52 DiT + 6 Adapter)",
    ]

    def __init__(self):
        lora_dirs = folder_paths.get_folder_paths("loras") if hasattr(folder_paths, "get_folder_paths") else []
        self.output_dir = lora_dirs[0] if lora_dirs else os.path.join(folder_paths.models_dir, "loras")

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_a": (lora_list,),
                "lora_b": (lora_list,),
                "preset": ([
                    "Character + Style Fusion (A identity, B style)",
                    "Anti-Overcook & Clean Blend",
                    "Orthogonal Synthesis (DARE / SVD)",
                    "Geometric Harmonic Fusion"
                ],),
                "target_influence_b": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "aggressiveness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.5, "step": 0.05}),
                "target_rank": ("INT", {"default": 32, "min": 4, "max": 128, "step": 4}),
                "export_filename": ("STRING", {"default": "HaDeS_Merged_LoRA"}),
            },
            "optional": {
                "merge_mode": ([
                    "Additive (A + B, Multi-LoRA match)",
                    "Interpolated (Lerp (1-r)A + rB)"
                ], {"default": "Additive (A + B, Multi-LoRA match)"}),
                "protect_llm_adapter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Strip dirty llm_adapter keys before SVD synthesis."
                }),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
                "topology_override": (s.TOPOLOGY_CHOICES, {
                    "default": "Auto-Detect from Keys",
                    "tooltip": "Explicitly set DiT block count (28L/40L/52L) when merging partial LoRAs that only touch a subset of blocks."
                }),
                "reference_model": ("MODEL", {
                    "tooltip": "Optional base model to read exact DiT + Adapter block topology."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_lora_path", "info")
    FUNCTION = "merge_and_save"
    CATEGORY = "HaDeS/Anima Mergers"
    OUTPUT_NODE = True

    def _collect_prefixes(self, sd):
        prefix_map = {}
        diff_biases = {}
        unsupported_prefixes = set()

        for k in sd.keys():
            for m in self._UNSUPPORTED_MARKERS:
                if m in k:
                    unsupported_prefixes.add(k.split(m)[0].rstrip("."))
                    break

        for k in sd.keys():
            if any(m in k for m in self._UNSUPPORTED_MARKERS):
                continue
            if k.endswith(".diff_b") or k.endswith(".lora_B.bias"):
                pfx = k[:-len(".diff_b")] if k.endswith(".diff_b") else k[:-len(".lora_B.bias")]
                if pfx in unsupported_prefixes:
                    continue
                diff_biases[pfx] = k
                continue
            for up_sfx, down_sfx in self._UP_DOWN_SUFFIXES:
                if k.endswith(up_sfx):
                    pfx = k[:-len(up_sfx)]
                    if pfx in unsupported_prefixes:
                        break
                    down_k = f"{pfx}{down_sfx}"
                    if down_k in sd:
                        prefix_map[pfx] = (k, down_k)
                    break

        # Count unique non-standard layer prefixes (adapters), not raw keys:
        # a single DoRA adapter exposes several keys (lora_down, lora_magnitude_vector, ...).
        skipped_unsupported = len(unsupported_prefixes)
        if unsupported_prefixes:
            logger.warning(
                f"HaDeS Standalone LoRA Merger: Excluded {skipped_unsupported} non-standard adapter layer(s) (DoRA/LoHa/LoKr)."
            )
        return prefix_map, diff_biases, skipped_unsupported

    def _extract_components(self, sd, pfx, pair):
        up_k, down_k = pair
        alpha_k = f"{pfx}.alpha"
        up = sd[up_k]
        down = sd[down_k]
        rank = down.shape[0] if down.ndim >= 1 else up.shape[1]
        alpha = float(sd[alpha_k].item()) if alpha_k in sd else float(rank)
        scale = alpha / float(rank) if rank > 0 else 1.0
        return up, down, alpha, rank, scale

    def _apply_dare_to_factor(self, d_tensor, drop_prob, seed_offset=0):
        if drop_prob <= 0.0:
            return d_tensor
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1337 + seed_offset + (d_tensor.numel() % 997))
        mask = (torch.rand(d_tensor.shape, generator=gen, device="cpu") >= drop_prob).to(
            device=d_tensor.device, dtype=d_tensor.dtype
        )
        return (d_tensor * mask) / (1.0 - drop_prob)

    def _qr_svd_factor_merge(self, up_a, down_a, eff_scale_a, up_b, down_b, eff_scale_b, target_rank, device, dare_drop=0.0):
        is_conv_1x1 = (
            up_a.ndim == 4 and down_a.ndim == 4 and up_a.shape[2] == 1 and up_a.shape[3] == 1
            and up_b.ndim == 4 and down_b.ndim == 4 and up_b.shape[2] == 1 and up_b.shape[3] == 1
        )
        orig_down_shape = down_a.shape if is_conv_1x1 else None

        if is_conv_1x1 or (up_a.ndim == 2 and down_a.ndim == 2 and up_b.ndim == 2 and down_b.ndim == 2):
            ua = up_a.to(device=device, dtype=torch.float32).reshape(up_a.shape[0], up_a.shape[1]) * eff_scale_a
            da = down_a.to(device=device, dtype=torch.float32).reshape(down_a.shape[0], -1)
            ub = up_b.to(device=device, dtype=torch.float32).reshape(up_b.shape[0], up_b.shape[1]) * eff_scale_b
            db = down_b.to(device=device, dtype=torch.float32).reshape(down_b.shape[0], -1)

            if dare_drop > 0.0:
                db = self._apply_dare_to_factor(db, dare_drop, seed_offset=ua.shape[0])

            u_cat = torch.cat([ua, ub], dim=1)
            d_cat = torch.cat([da, db], dim=0)

            qu, ru = torch.linalg.qr(u_cat, mode="reduced")
            qd, rd = torch.linalg.qr(d_cat.T, mode="reduced")
            core = torch.matmul(ru, rd.T)

            u_hat, s_val, vh_hat = torch.linalg.svd(core, full_matrices=False)
            rank = min(target_rank, s_val.shape[0], ua.shape[0], da.shape[1])

            sqrt_s = torch.sqrt(s_val[:rank].clamp(min=0.0))
            lora_up = torch.matmul(qu, u_hat[:, :rank]) * sqrt_s.unsqueeze(0)
            lora_down = torch.matmul(sqrt_s.unsqueeze(1) * vh_hat[:rank, :], qd.T)

            if is_conv_1x1:
                lora_up = lora_up.reshape(up_a.shape[0], rank, 1, 1)
                lora_down = lora_down.reshape(rank, orig_down_shape[1], orig_down_shape[2], orig_down_shape[3])

            return lora_up, lora_down, rank

        da_full = torch.nn.functional.conv2d(
            down_a.to(device=device, dtype=torch.float32).permute(1, 0, 2, 3),
            up_a.to(device=device, dtype=torch.float32)
        ).permute(1, 0, 2, 3) * eff_scale_a
        db_full = torch.nn.functional.conv2d(
            down_b.to(device=device, dtype=torch.float32).permute(1, 0, 2, 3),
            up_b.to(device=device, dtype=torch.float32)
        ).permute(1, 0, 2, 3) * eff_scale_b
        if dare_drop > 0.0:
            db_full = self._apply_dare_to_factor(db_full, dare_drop)
        d_merged = da_full + db_full
        orig_shape = d_merged.shape
        matrix = d_merged.reshape(orig_shape[0], -1)
        rank = min(target_rank, matrix.shape[0], matrix.shape[1])
        u_m, s_m, vh_m = torch.linalg.svd(matrix, full_matrices=False)
        sqrt_s = torch.sqrt(s_m[:rank].clamp(min=0.0))
        lora_up = (u_m[:, :rank] * sqrt_s.unsqueeze(0)).reshape(orig_shape[0], rank, 1, 1)
        lora_down = (sqrt_s.unsqueeze(1) * vh_m[:rank, :]).reshape(rank, orig_shape[1], orig_shape[2], orig_shape[3])
        return lora_up, lora_down, rank

    def _compute_block_weights(self, preset, block_idx, collision, energy_b, target_influence_b, aggressiveness, is_additive, protect_llm_adapter=True):
        if protect_llm_adapter and block_idx >= 28:
            return 0.0, 0.0

        if preset == "Character + Style Fusion (A identity, B style)":
            if block_idx in range(0, 15):
                wb = target_influence_b * 0.15 * max(0.05, 1.0 - (collision * 0.85 * aggressiveness))
                wa = 1.0 if is_additive else (1.0 - wb)
            elif block_idx in range(15, 20):
                wb = target_influence_b * 0.55
                wa = 0.90 if is_additive else (1.0 - wb)
            elif block_idx in range(20, 28):
                orth_boost = (1.0 - collision * 0.5) * (0.5 + 0.5 * energy_b)
                wb = min(1.35 if is_additive else 1.0, target_influence_b * (1.0 + orth_boost * 0.65 * aggressiveness))
                wa = max(0.35, 0.75 - (collision * 0.25 * aggressiveness)) if is_additive else (1.0 - wb)
            else:
                wb = target_influence_b * 0.75
                wa = 0.85 if is_additive else (1.0 - wb)
        elif preset == "Anti-Overcook & Clean Blend":
            wb = target_influence_b * max(0.15, 1.0 - ((collision * 0.55 + energy_b * 0.2) * aggressiveness))
            wa = max(0.5, 1.0 - (collision * 0.35 * aggressiveness)) if is_additive else (1.0 - wb)
        elif preset == "Orthogonal Synthesis (DARE / SVD)":
            orth = 1.0 - collision
            wb = target_influence_b * (0.5 + ((orth * 0.55 + energy_b * 0.25) * aggressiveness))
            wa = max(0.6, 1.0 - (collision * 0.25 * aggressiveness)) if is_additive else (1.0 - min(1.0, wb))
        else:
            wb = target_influence_b * max(0.2, 1.0 - (collision * 0.35 * aggressiveness))
            wa = max(0.65, 1.0 - (collision * 0.2 * aggressiveness)) if is_additive else (1.0 - wb)

        max_b = 1.5 if is_additive else 1.0
        return round(max(0.0, min(1.2, float(wa))), 3), round(max(0.0, min(max_b, float(wb))), 3)

    def _sanitize_output_path(self, export_filename, donor_paths=(), overwrite_existing=False):
        os.makedirs(self.output_dir, exist_ok=True)
        raw_name = os.path.basename(export_filename.strip()) if export_filename else ""
        if raw_name.lower().endswith(".safetensors"):
            raw_name = raw_name[:-len(".safetensors")]
        clean_name = _SAFE_FILENAME_RE.sub("_", raw_name).strip(" .")
        if not clean_name:
            clean_name = "HaDeS_Merged_LoRA"
        base_dir = os.path.abspath(self.output_dir)
        candidate = os.path.abspath(os.path.join(base_dir, f"{clean_name}.safetensors"))
        if os.path.commonpath([base_dir, candidate]) != base_dir:
            raise ValueError("Invalid export filename path.")

        abs_donors = {os.path.abspath(p) for p in donor_paths if p}
        if candidate in abs_donors or (not overwrite_existing and os.path.exists(candidate)):
            counter = 1
            while True:
                alt = os.path.abspath(os.path.join(base_dir, f"{clean_name}_{counter:03d}.safetensors"))
                if alt not in abs_donors and (overwrite_existing or not os.path.exists(alt)):
                    candidate = alt
                    break
                counter += 1
        return candidate

    def merge_and_save(self, lora_a, lora_b, preset, target_influence_b, aggressiveness, target_rank, export_filename,
                       merge_mode="Additive (A + B, Multi-LoRA match)", protect_llm_adapter=True, overwrite_existing=False,
                       topology_override="Auto-Detect from Keys", reference_model=None):
        path_a = folder_paths.get_full_path_or_raise("loras", lora_a)
        path_b = folder_paths.get_full_path_or_raise("loras", lora_b)

        raw_a = _convert_lora_fn(comfy.utils.load_torch_file(path_a, safe_load=True))
        raw_b = _convert_lora_fn(comfy.utils.load_torch_file(path_b, safe_load=True))
        sd_a, rem_a = canonicalize_anima_lora_sd(raw_a, strip_llm_adapter=protect_llm_adapter)
        sd_b, rem_b = canonicalize_anima_lora_sd(raw_b, strip_llm_adapter=protect_llm_adapter)

        map_a, bias_a, skip_a = self._collect_prefixes(sd_a)
        map_b, bias_b, skip_b = self._collect_prefixes(sd_b)
        all_prefixes = sorted(set(map_a.keys()).union(set(map_b.keys())))
        if not all_prefixes:
            raise ValueError("HaDeS Standalone LoRA Merger: Neither donor LoRA contains supported low-rank factor keys.")

        if reference_model is not None and hasattr(reference_model, "model"):
            n_dit, n_adapter = detect_anima_topology(reference_model.model.state_dict().keys())
        elif "28 DiT" in str(topology_override):
            n_dit, n_adapter = 28, 6
        elif "40 DiT" in str(topology_override):
            n_dit, n_adapter = 40, 6
        elif "52 DiT" in str(topology_override):
            n_dit, n_adapter = 52, 6
        else:
            n_dit, n_adapter = detect_anima_topology(all_prefixes)
            # P2 guard: Auto mode can mis-detect partial LoRAs (e.g. blocks 0-20) as the 28L tier.
            max_seen_dit = max((parse_anima_block_key(p)[1] for p in all_prefixes
                               if parse_anima_block_key(p)[0] == "dit"), default=-1)
            if 0 <= max_seen_dit < (n_dit - 1):
                logger.warning(
                    f"HaDeS Standalone LoRA Merger: Auto topology detected {n_dit} DiT layers, but the LoRAs only reach "
                    f"block {max_seen_dit}. If these are partial Anima 2.9B (40L) or 3.8B (52L) LoRAs, set "
                    f"topology_override or pass reference_model to avoid silent band misalignment."
                )
        is_additive = str(merge_mode).startswith("Additive")
        dare_drop = min(0.45, 0.12 * aggressiveness) if preset == "Orthogonal Synthesis (DARE / SVD)" else 0.0

        prefixes_by_block = {i: [] for i in range(34)}
        non_block_prefixes = []
        for pfx in all_prefixes:
            b_idx = get_anima_block_idx(pfx, n_dit=n_dit, n_adapter=n_adapter)
            if b_idx is not None:
                prefixes_by_block[b_idx].append(pfx)
            else:
                non_block_prefixes.append(pfx)

        device = comfy.model_management.get_torch_device()
        block_ratios_a = [1.0 if is_additive else (1.0 - target_influence_b)] * 34
        block_ratios_b = [target_influence_b] * 34
        merged_sd = {}
        stats = {
            "qr_svd": 0,
            "fast_copy": 0,
            "cpu_retried": 0,
            "donor_fallback": 0,
            "skipped_keys": skip_a + skip_b,
            "remapped": rem_a["remapped_keys"] + rem_b["remapped_keys"],
            "stripped_adapter": rem_a["stripped_adapter_keys"] + rem_b["stripped_adapter_keys"],
        }

        def synthesize_prefix(pfx, wa, wb):
            if abs(wa) <= 1e-6 and abs(wb) <= 1e-6:
                return
            in_a = pfx in map_a
            in_b = pfx in map_b

            if in_a and not in_b:
                up, down, alpha, rank, scale = self._extract_components(sd_a, pfx, map_a[pfx])
                if rank <= target_rank:
                    merged_sd[f"{pfx}.lora_up.weight"] = up.cpu().half().contiguous()
                    merged_sd[f"{pfx}.lora_down.weight"] = down.cpu().half().contiguous()
                    merged_sd[f"{pfx}.alpha"] = torch.tensor(float(alpha * wa), dtype=torch.float16)
                    stats["fast_copy"] += 1
                    return
                up_b_zero = torch.zeros_like(up)
                down_b_zero = torch.zeros_like(down)
                up_a, down_a, scale_a = up, down, scale
                up_b, down_b, scale_b = up_b_zero, down_b_zero, 0.0
            elif in_b and not in_a:
                up, down, alpha, rank, scale = self._extract_components(sd_b, pfx, map_b[pfx])
                if rank <= target_rank and dare_drop == 0.0:
                    merged_sd[f"{pfx}.lora_up.weight"] = up.cpu().half().contiguous()
                    merged_sd[f"{pfx}.lora_down.weight"] = down.cpu().half().contiguous()
                    merged_sd[f"{pfx}.alpha"] = torch.tensor(float(alpha * wb), dtype=torch.float16)
                    stats["fast_copy"] += 1
                    return
                up_a_zero = torch.zeros_like(up)
                down_a_zero = torch.zeros_like(down)
                up_a, down_a, scale_a = up_a_zero, down_a_zero, 0.0
                up_b, down_b, scale_b = up, down, scale
            else:
                up_a, down_a, _, _, scale_a = self._extract_components(sd_a, pfx, map_a[pfx])
                up_b, down_b, _, _, scale_b = self._extract_components(sd_b, pfx, map_b[pfx])

            try:
                lora_up, lora_down, rank = self._qr_svd_factor_merge(
                    up_a, down_a, scale_a * wa, up_b, down_b, scale_b * wb, target_rank, device, dare_drop=dare_drop
                )
                stats["qr_svd"] += 1
            except RuntimeError:
                try:
                    lora_up, lora_down, rank = self._qr_svd_factor_merge(
                        up_a, down_a, scale_a * wa, up_b, down_b, scale_b * wb, target_rank, torch.device("cpu"), dare_drop=dare_drop
                    )
                    stats["qr_svd"] += 1
                    stats["cpu_retried"] += 1
                except RuntimeError:
                    donor_sd, donor_map = (sd_a, map_a) if in_a else (sd_b, map_b)
                    up_k, down_k = donor_map[pfx]
                    merged_sd[f"{pfx}.lora_up.weight"] = donor_sd[up_k].cpu().half().contiguous()
                    merged_sd[f"{pfx}.lora_down.weight"] = donor_sd[down_k].cpu().half().contiguous()
                    if f"{pfx}.alpha" in donor_sd:
                        merged_sd[f"{pfx}.alpha"] = donor_sd[f"{pfx}.alpha"].cpu().half()
                    stats["donor_fallback"] += 1
                    return

            merged_sd[f"{pfx}.lora_up.weight"] = lora_up.cpu().half().contiguous()
            merged_sd[f"{pfx}.lora_down.weight"] = lora_down.cpu().half().contiguous()
            merged_sd[f"{pfx}.alpha"] = torch.tensor(float(rank), dtype=torch.float16)

        for b in range(34):
            b_prefixes = prefixes_by_block[b]
            overlaps = []
            energies = []

            for pfx in b_prefixes:
                if pfx in map_a and pfx in map_b:
                    up_a, down_a, _, _, scale_a = self._extract_components(sd_a, pfx, map_a[pfx])
                    up_b, down_b, _, _, scale_b = self._extract_components(sd_b, pfx, map_b[pfx])
                    m = compute_subspace_metrics(up_a, down_a, scale_a, up_b, down_b, scale_b)
                    if m is not None:
                        overlaps.append(max(m["subspace_overlap"], abs(m["cos_sim"])))
                        energies.append(m["energy_ratio"])

            col_b = (sum(overlaps) / len(overlaps)) if overlaps else 0.25
            en_b = (sum(energies) / len(energies)) if energies else 0.5
            wa, wb = self._compute_block_weights(
                preset, b, col_b, en_b, target_influence_b, aggressiveness, is_additive, protect_llm_adapter=protect_llm_adapter
            )
            block_ratios_a[b] = wa
            block_ratios_b[b] = wb

            for pfx in b_prefixes:
                synthesize_prefix(pfx, wa, wb)

        dit_a = block_ratios_a[:28]
        dit_b = block_ratios_b[:28]
        avg_wa = round(sum(dit_a) / len(dit_a), 3)
        avg_wb = round(sum(dit_b) / len(dit_b), 3)
        for pfx in non_block_prefixes:
            synthesize_prefix(pfx, avg_wa, avg_wb)

        all_biases = set(bias_a.keys()).union(set(bias_b.keys()))
        for pfx in all_biases:
            b_idx = get_anima_block_idx(pfx, n_dit=n_dit, n_adapter=n_adapter)
            wa = block_ratios_a[b_idx] if b_idx is not None else avg_wa
            wb = block_ratios_b[b_idx] if b_idx is not None else avg_wb
            if abs(wa) <= 1e-6 and abs(wb) <= 1e-6:
                continue
            ba = sd_a[bias_a[pfx]].float() * wa if pfx in bias_a else None
            bb = sd_b[bias_b[pfx]].float() * wb if pfx in bias_b else None
            if ba is not None and bb is not None and ba.shape == bb.shape:
                merged_sd[f"{pfx}.diff_b"] = (ba + bb.to(ba.device)).cpu().half().contiguous()
            elif ba is not None:
                merged_sd[f"{pfx}.diff_b"] = ba.cpu().half().contiguous()
            elif bb is not None:
                merged_sd[f"{pfx}.diff_b"] = bb.cpu().half().contiguous()

        output_filepath = self._sanitize_output_path(
            export_filename, donor_paths=(path_a, path_b), overwrite_existing=overwrite_existing
        )

        metadata = {
            "author": "HaDeS",
            "format": "pt",
            "donor_lora_a": lora_a,
            "donor_lora_b": lora_b,
            "preset": preset,
            "merge_mode": merge_mode,
            "protect_llm_adapter": str(protect_llm_adapter),
            "target_rank": str(target_rank),
            "weights_a_34": ",".join(str(r) for r in block_ratios_a),
            "weights_b_34": ",".join(str(r) for r in block_ratios_b),
        }
        safetensors.torch.save_file(merged_sd, output_filepath, metadata=metadata)

        info_lines = [
            "=== HADES ANIMA STANDALONE LORA MERGER (QR-SVD) ===",
            f"Saved to: {output_filepath}",
            f"Preset: {preset} | Mode: {merge_mode} | Target SVD Rank: {target_rank}",
            f"Layers QR-SVD Merged: {stats['qr_svd']} | Fast-Copied: {stats['fast_copy']} | Diffusers Keys Remapped: {stats['remapped']} | Adapter Stripped: {stats['stripped_adapter']}",
        ]
        if stats["skipped_keys"] > 0:
            info_lines.append(f"Warning: Skipped {stats['skipped_keys']} non-standard adapter(s) (DoRA/LoHa/LoKr).")
        info_lines.extend([
            "\n--- DONOR A WEIGHTS (34 BLOCKS) ---",
            ",".join(str(r) for r in block_ratios_a),
            "\n--- DONOR B WEIGHTS (34 BLOCKS) ---",
            ",".join(str(r) for r in block_ratios_b),
        ])

        return (output_filepath, "\n".join(info_lines))


# ==============================================================================
# 5. HADES ANIMA CLIP BLOCK MERGER (PROPORTIONAL 28L / 36L QWEN3 DEPTH)
# ==============================================================================
class HaDeSAnimaCLIPBlockMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip1": ("CLIP",),
                "clip2": ("CLIP",),
                "base_ratio": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "0.0 = CLIP 1, 1.0 = CLIP 2"}),
                "vocab_embeddings": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Lexical token embeddings (embed_tokens)"}),
                "syntax_rigidity": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Early transformer layers (~0-35% depth): Syntax and Booru tag binding"}),
                "semantic_focus": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Middle transformer layers (~35-72% depth): Concepts and entity relations"}),
                "style_abstraction": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Deep transformer layers (~72-100% depth): Aesthetic nuance and mood"}),
                "final_projections": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Final normalization layer (norm.weight / final_layer_norm)"}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "merge_clips"
    CATEGORY = "HaDeS/Anima Mergers"

    def merge_clips(self, clip1, clip2, base_ratio, vocab_embeddings,
                    syntax_rigidity, semantic_focus, style_abstraction, final_projections):
        c = clip1.clone()
        base_sd1 = c.patcher.model.state_dict()
        base_sd2 = clip2.patcher.model.state_dict()
        patches_to_add = {}
        merged_blocks = 0
        compatible_keys = 0
        valid_keys = c.patcher.model_keys if hasattr(c.patcher, "model_keys") else None

        max_layer_idx = -1
        for k in base_sd1.keys():
            match = _CLIP_LAYER_RE.search(k)
            if match:
                idx = int(match.group(1))
                if idx > max_layer_idx:
                    max_layer_idx = idx
        n_clip_layers = (max_layer_idx + 1) if max_layer_idx >= 12 else 28
        early_cutoff = max(1, round(0.357 * n_clip_layers))
        mid_cutoff = max(early_cutoff + 1, round(0.714 * n_clip_layers))

        for k, base_w1 in base_sd1.items():
            if k not in base_sd2:
                continue
            if k.endswith(".position_ids") or k.endswith(".logit_scale") or k.endswith(".scale_weight"):
                continue
            if valid_keys is not None and k not in valid_keys:
                continue

            base_w2 = base_sd2[k]
            if base_w1.shape != base_w2.shape:
                raise ValueError(f"HaDeS CLIP Merger: Shape mismatch on key '{k}': {tuple(base_w1.shape)} vs {tuple(base_w2.shape)}.")
            if not (base_w1.is_floating_point() or base_w1.dtype in _FP8_DTYPES):
                continue

            compatible_keys += 1
            target_ratio = base_ratio
            if "embed_tokens" in k or "token_embedding" in k:
                target_ratio = vocab_embeddings
            elif k.endswith("model.norm.weight") or "final_layer_norm" in k or (k.endswith("norm.weight") and not _CLIP_LAYER_RE.search(k)):
                target_ratio = final_projections
            else:
                match = _CLIP_LAYER_RE.search(k)
                if match:
                    idx = int(match.group(1))
                    if idx < early_cutoff:
                        target_ratio = syntax_rigidity
                    elif idx < mid_cutoff:
                        target_ratio = semantic_focus
                    else:
                        target_ratio = style_abstraction

            if abs(target_ratio) > 1e-5:
                patch_key = k
                baked1_float = get_baked_weight(c.patcher, patch_key, base_w1).float()
                baked2_float = get_baked_weight(clip2.patcher, patch_key, base_w2).to(baked1_float.device).float()

                if torch.equal(baked1_float, baked2_float):
                    continue

                v_modified = baked1_float + ((baked2_float - baked1_float) * target_ratio)
                v_target = v_modified.to(base_w1.dtype)

                patches_to_add[patch_key] = ("set", (v_target.cpu(),))
                merged_blocks += 1

        if compatible_keys == 0:
            raise ValueError("HaDeS CLIP Merger: clip1 and clip2 share 0 compatible tensors.")

        if patches_to_add:
            c.patcher.add_patches(patches_to_add, 1.0)
        return (c, f"HaDeS Anima CLIP Merged | Depth: {n_clip_layers}L (0..{early_cutoff-1} / {early_cutoff}..{mid_cutoff-1} / {mid_cutoff}+) | Modified Tensors: {merged_blocks}")


# ==============================================================================
# NODE REGISTRATION (MERGERS + ANIMA PIPELINE + ANIMA KSAMPLER + LOADER NODES)
# ==============================================================================
from .anima_pipeline_nodes import (
    HaDeSAnimaPromptStudio,
    HaDeSAnimaAttentionGuidance,
    HaDeSAnimaLatentPipelineGuard,
    HaDeSAnimaUniversalLoraLoader,
    HaDeSAnimaCornerScribbleShield,
    format_anima_prompt_text,
)
from .anima_ksampler_nodes import (
    HaDeSAnimaKSampler,
    HaDeSAnimaDualPassKSampler,
    HaDeSAnimaTriplePassKSampler,
    clone_model_with_anima_shift,
    build_anima_sigmas,
    upscale_anima_latent_32px,
    run_anima_ksampler_pass,
)
from .anima_loader_nodes import (
    HaDeSAnimaModelLoader,
    HaDeSAnimaCLIPLoader,
    HaDeSAnimaVAELoader,
    HaDeSAnimaVAEDecode,
    remap_anima_diffusers_model_sd,
    inject_external_llm_adapter_sd,
    validate_anima_te_state_dict,
    _inspect_and_validate_anima_vae_sd,
)

NODE_CLASS_MAPPINGS = {
    "HaDeSAnimaModelBlockMerger": HaDeSAnimaModelBlockMerger,
    "HaDeSAnimaPredictiveModelMerger": HaDeSAnimaPredictiveModelMerger,
    "HaDeSAnimaPredictiveMultiLoraLoader": HaDeSAnimaPredictiveMultiLoraLoader,
    "HaDeSAnimaStandaloneLoraMerger": HaDeSAnimaStandaloneLoraMerger,
    "HaDeSAnimaCLIPBlockMerger": HaDeSAnimaCLIPBlockMerger,
    "HaDeSAnimaPromptStudio": HaDeSAnimaPromptStudio,
    "HaDeSAnimaAttentionGuidance": HaDeSAnimaAttentionGuidance,
    "HaDeSAnimaLatentPipelineGuard": HaDeSAnimaLatentPipelineGuard,
    "HaDeSAnimaUniversalLoraLoader": HaDeSAnimaUniversalLoraLoader,
    "HaDeSAnimaCornerScribbleShield": HaDeSAnimaCornerScribbleShield,
    "HaDeSAnimaKSampler": HaDeSAnimaKSampler,
    "HaDeSAnimaDualPassKSampler": HaDeSAnimaDualPassKSampler,
    "HaDeSAnimaTriplePassKSampler": HaDeSAnimaTriplePassKSampler,
    "HaDeSAnimaModelLoader": HaDeSAnimaModelLoader,
    "HaDeSAnimaCLIPLoader": HaDeSAnimaCLIPLoader,
    "HaDeSAnimaVAELoader": HaDeSAnimaVAELoader,
    "HaDeSAnimaVAEDecode": HaDeSAnimaVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HaDeSAnimaModelBlockMerger": "Anima Model Block Merger",
    "HaDeSAnimaPredictiveModelMerger": "Anima Predictive Model Merger",
    "HaDeSAnimaPredictiveMultiLoraLoader": "Anima Predictive Multi-LoRA",
    "HaDeSAnimaStandaloneLoraMerger": "Anima LoRA Merger",
    "HaDeSAnimaCLIPBlockMerger": "Anima CLIP Block Merger",
    "HaDeSAnimaPromptStudio": "Anima Prompt Studio",
    "HaDeSAnimaAttentionGuidance": "Anima Attention Guidance (NAG)",
    "HaDeSAnimaLatentPipelineGuard": "Anima Latent",
    "HaDeSAnimaUniversalLoraLoader": "Anima Universal LoRA Remapper",
    "HaDeSAnimaCornerScribbleShield": "Anima Corner Shield",
    "HaDeSAnimaKSampler": "Anima KSampler (Single-Pass)",
    "HaDeSAnimaDualPassKSampler": "Anima Dual-Pass KSampler",
    "HaDeSAnimaTriplePassKSampler": "Anima Triple-Pass KSampler",
    "HaDeSAnimaModelLoader": "Anima Diffusion Model Loader",
    "HaDeSAnimaCLIPLoader": "Anima Text Encoder Loader",
    "HaDeSAnimaVAELoader": "Anima VAE Loader",
    "HaDeSAnimaVAEDecode": "Anima VAE Decode",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "DIFFUSERS_TO_COMFY_RULES",
    "canonicalize_anima_lora_key",
    "canonicalize_anima_lora_sd",
    "parse_anima_block_key",
    "detect_anima_topology",
    "map_to_canonical_34",
    "get_anima_block_idx",
    "clone_model_with_anima_shift",
    "build_anima_sigmas",
    "upscale_anima_latent_32px",
    "run_anima_ksampler_pass",
    "remap_anima_diffusers_model_sd",
    "inject_external_llm_adapter_sd",
    "validate_anima_te_state_dict",
    "_inspect_and_validate_anima_vae_sd",
    "format_anima_prompt_text",
]
