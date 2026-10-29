import os
import re
import math
import torch
import torch.nn.functional as F
import folder_paths
import safetensors.torch
import comfy.sd
import comfy.utils
import comfy.lora
import comfy.model_management

from . import (
    _convert_lora_fn,
    _SAFE_FILENAME_RE,
    canonicalize_anima_lora_sd,
    detect_anima_topology,
)

# ==============================================================================
# 6. HADES ANIMA PROMPT STUDIO & AESTHETIC GUARD
# ==============================================================================
_SCORE_WHITELIST_RE = re.compile(r"^(?:score_\d+|.*_up|safe_for_work)$", re.IGNORECASE)
_SCORE_TAG_RE = re.compile(r"^score_\d+(?:_up)?$", re.IGNORECASE)
_UNESCAPED_PAREN_RE = re.compile(r"(?<!\\)([()])")
_KAOMOJI_SET = {
    "^_^", "o_o", "0_0", ">_<", "u_u", "t_t", "=_=", ";_;", "x_x", "-_-", "v_v", "q_q", "^o^", ">_o", "o_<"
}
# Only score_* tags are strictly forbidden on all Anima-Aesthetic models.
# Generic quality tags ("masterpiece, best quality") are allowed on v1.0b ("Keep" profile) but stripped on v1.0/v1.1.
_AESTHETIC_SLOP_TAGS = {
    "score_9", "score_8_up", "score_7_up", "score_6_up", "score_5_up", "score_4_up",
    "masterpiece", "best quality", "ultra-detailed", "ultra detailed", "8k", "4k",
    "extremely detailed", "insanely detailed"
}
_WEIGHTED_TAG_RE = re.compile(r"^\((.+):([+-]?\d+(?:\.\d+)?)\)$")
_WEIGHTED_PARENS_ONLY_RE = re.compile(r"^\((.+)\)$")

# Canonical Bucket Classifier Sets
_BUCKET_QUALITY_META_RE = re.compile(
    r"^(?:score_\d+(?:_up)?|masterpiece|best quality|good quality|normal quality|low quality|worst quality|"
    r"year 20\d\d|newest|recent|mid|early|old|"
    r"highres|absurdres|anime screenshot|official art|"
    r"safe|sensitive|nsfw|explicit)$",
    re.IGNORECASE
)
_BUCKET_SUBJECT_COUNT_RE = re.compile(
    r"^(?:1girl|1boy|1other|2girls|2boys|3girls|3boys|4girls|4boys|solo|multiple girls|multiple boys|male focus)$",
    re.IGNORECASE
)
_BUILTIN_STYLE_TRIGGERS = (
    "studio ghibli style", "pixar style", "spider-verse style", "akira toriyama style",
    "arknights style", "art nouveau style", "vaporwave style", "gothic style", "golden age comic style"
)


def _split_tags_preserving_parens(text):
    parts = []
    curr = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            curr.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            curr.append(ch)
        elif ch == "," and depth == 0:
            s = "".join(curr).strip()
            if s:
                parts.append(s)
            curr = []
        else:
            curr.append(ch)
    tail = "".join(curr).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_artist_entry(raw_art):
    """
    Normalize an artist entry into Anima's canonical '@artist' format.
    Preserves weight syntax like '(@zocehuy:1.2)' and strips unwanted 'by ' prefixes.
    """
    raw = str(raw_art or "").strip()
    if not raw:
        return ""

    weight_suffix = ""
    m_weighted = _WEIGHTED_TAG_RE.match(raw)
    if m_weighted:
        inner = m_weighted.group(1).strip()
        weight_suffix = f":{m_weighted.group(2)}"
    else:
        m_parens = _WEIGHTED_PARENS_ONLY_RE.match(raw)
        if m_parens:
            inner = m_parens.group(1).strip()
            weight_suffix = ""
        else:
            inner = raw

    # Strip leading "by " if present
    inner = re.sub(r"^by\s+", "", inner, flags=re.IGNORECASE).strip()
    # Normalize underscores to spaces, lowercase
    inner = inner.lower().replace("_", " ")

    if not inner.startswith("@"):
        inner = "@" + inner.lstrip("@").strip()

    if weight_suffix:
        return f"({inner}{weight_suffix})"
    return inner


def _normalize_character_entry(raw_char):
    """
    Normalize a character tag entry (e.g. 'frieren, sousou no frieren')
    preserving spaces instead of underscores and escaping literal parentheses.
    """
    raw = str(raw_char or "").strip()
    if not raw:
        return []
    results = []
    for part in _split_tags_preserving_parens(raw):
        p = part.strip().lower().replace("_", " ")
        if not p:
            continue
        if "(" in p or ")" in p:
            p = _UNESCAPED_PAREN_RE.sub(r"\\\1", p)
        results.append(p)
    return results


def _is_natural_language_sentence(tag):
    """
    Detect whether a tag is a natural language descriptive sentence
    (which should preserve capitalization and punctuation) vs a discrete Booru tag.
    """
    stripped = tag.strip()
    words = stripped.split()
    if len(words) >= 4 and any(stripped.endswith(punct) for punct in (".", "!", "?")):
        return True
    if len(words) >= 5 and any(w.lower() in ("with", "wearing", "holding", "standing", "sitting", "under", "against", "in", "on", "from") for w in words):
        return True
    return False


_TAG_WEIGHT_DIGIT_RE = re.compile(r":\d+(?:\.\d+)?")
_PROSE_DISQUALIFY_PREFIX_RE = re.compile(
    r"^(?:masterpiece|best quality|score_\d+(?:_up)?|1girl|1boy|2girls|2boys|3girls|3boys|solo|safe_for_work|nsfw)\s*,\s*",
    re.IGNORECASE,
)


def _is_pure_natural_sentence_line(line: str) -> bool:
    """
    Detect whether an entire line is a single natural language sentence
    (which may contain descriptive commas) rather than a comma-separated list of booru tags.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # If it starts with typical discrete booru tags like 'masterpiece,' or '1girl,', it's a tag list!
    if _PROSE_DISQUALIFY_PREFIX_RE.match(stripped):
        return False
    # If it contains booru tag weight syntax with digits like ':1.2', it's tag syntax
    if _TAG_WEIGHT_DIGIT_RE.search(stripped):
        return False
    # If it contains underscores (except kaomoji), it's booru tags
    if "_" in stripped and not any(k in stripped for k in _KAOMOJI_SET):
        return False

    words = stripped.split()
    if len(words) < 3:
        return False

    has_prose_punct = any(stripped.endswith(p) for p in (".", "!", "?"))
    has_preposition = any(
        w.lower() in ("with", "wearing", "holding", "standing", "sitting", "under", "against", "in", "on", "from", "at", "by", "of")
        for w in words
    )
    has_prose_opener = any(
        stripped.lower().startswith(op)
        for op in ("digital ", "a ", "an ", "the ", "oil painting", "abstract", "portrait of", "illustration of")
    )

    if has_prose_punct and (has_preposition or has_prose_opener):
        return True
    if len(words) >= 6 and has_preposition and has_prose_punct:
        return True
    return False


def _clean_single_anima_tag(raw_tag, booru_sanitizer=True, strip_aesthetic_slop=False, strip_score_only=False):
    tag = raw_tag.strip()
    if not tag:
        return None, 0

    # Natural language sentences bypass tag-lowercasing and aesthetic slop stripping
    if _is_natural_language_sentence(tag):
        return tag, 0

    weight_suffix = ""
    m = _WEIGHTED_TAG_RE.match(tag)
    if m:
        inner = m.group(1).strip()
        weight_suffix = f":{m.group(2)}"
    else:
        inner = tag

    # Handle multi-tag weighted groups such as (blue_hair, score_9, cyan_eyes:1.3)
    if weight_suffix and "," in inner:
        sub_parts = [p.strip() for p in inner.split(",") if p.strip()]
        cleaned_subs = []
        stripped_subs = 0
        for sp in sub_parts:
            c_sub, s_cnt = _clean_single_anima_tag(
                sp,
                booru_sanitizer=booru_sanitizer,
                strip_aesthetic_slop=strip_aesthetic_slop,
                strip_score_only=strip_score_only,
            )
            stripped_subs += s_cnt
            if c_sub:
                cleaned_subs.append(c_sub)
        if not cleaned_subs:
            return None, stripped_subs
        return f"({', '.join(cleaned_subs)}{weight_suffix})", stripped_subs

    if booru_sanitizer:
        inner_lower = inner.lower()
        if inner_lower not in _KAOMOJI_SET and not _SCORE_WHITELIST_RE.match(inner_lower):
            if inner_lower.startswith("@"):
                inner = "@" + inner_lower[1:].replace("_", " ").strip()
            else:
                if "_" in inner_lower and len(inner_lower) < 48:
                    inner = inner_lower.replace("_", " ")
                else:
                    inner = inner_lower
        else:
            inner = inner_lower

    check_key = inner.lower().strip()
    if strip_aesthetic_slop and (check_key in _AESTHETIC_SLOP_TAGS or _SCORE_TAG_RE.match(check_key)):
        return None, 1
    if strip_score_only and _SCORE_TAG_RE.match(check_key):
        return None, 1

    if booru_sanitizer and ("(" in inner or ")" in inner):
        inner = _UNESCAPED_PAREN_RE.sub(r"\\\1", inner)

    if weight_suffix:
        return f"({inner}{weight_suffix})", 0
    return inner, 0


def format_anima_prompt_text(raw_prompt, model_variant, style_mode, booru_sanitizer=True,
                             artist_tag="", character_tag="", is_negative=False,
                             header_mode="Canonical (1 Trigger + Title)",
                             header_trigger="ye-pop",
                             artwork_title=""):
    if style_mode == "Raw Pass-Through" and not artist_tag.strip() and not character_tag.strip():
        return raw_prompt, {"stripped_slop": 0, "tag_count": 0, "header_injected": False}

    is_aesthetic = "Aesthetic" in str(model_variant)
    keep_quality = is_aesthetic and "v1.0b" in str(model_variant)
    strip_slop = is_aesthetic and not keep_quality and not is_negative
    strip_score_only = is_aesthetic and (keep_quality or is_negative)

    # 6 Canonical Buckets for Positive Prompts:
    # 0: Quality / Score / Meta / Year / Safety
    # 1: Subject Count (1girl, solo...)
    # 2: Character & Series
    # 3: Artists (@...) and Built-in Style Triggers
    # 4: General Booru Tags (appearance, clothing, pose, background)
    # 5: Natural Language Descriptive Sentences
    b_quality = []
    b_subject = []
    b_character = []
    b_artist = []
    b_general = []
    b_natural = []

    seen_tags = set()
    stripped_count = 0
    total_tags = 0

    # 1. Normalize Character Injection (Bucket 2)
    if character_tag and str(character_tag).strip():
        norm_chars = _normalize_character_entry(character_tag)
        for c in norm_chars:
            dedup = c.lower()
            if dedup not in seen_tags:
                seen_tags.add(dedup)
                b_character.append(c)
                total_tags += 1

    # 2. Normalize Artist Injection (Bucket 3)
    if artist_tag and str(artist_tag).strip():
        for raw_art in str(artist_tag).split(","):
            norm_art = _normalize_artist_entry(raw_art)
            if not norm_art:
                continue
            dedup = norm_art.lower()
            if dedup not in seen_tags:
                seen_tags.add(dedup)
                b_artist.append(norm_art)
                total_tags += 1

    # 3. Process prompt body line by line
    raw_lines = [ln.strip() for ln in str(raw_prompt or "").splitlines() if ln.strip()]
    manual_header_lines = []
    lines_to_process = raw_lines
    embedded_trigger = None

    if raw_lines and raw_lines[0].lower() in ("ye-pop", "deviantart"):
        trigger_line = raw_lines[0].lower()
        if len(raw_lines) >= 3:
            title_line = raw_lines[1]
            manual_header_lines = [trigger_line, title_line]
            lines_to_process = raw_lines[2:]
        elif len(raw_lines) == 2:
            if _is_natural_language_sentence(raw_lines[1]) or "," in raw_lines[1]:
                manual_header_lines = [trigger_line]
                lines_to_process = raw_lines[1:]
            else:
                manual_header_lines = [trigger_line, raw_lines[1]]
                lines_to_process = []
        else:
            manual_header_lines = [trigger_line]
            lines_to_process = []

    for line in lines_to_process:
        if _is_pure_natural_sentence_line(line):
            clean_nl = line.strip()
            if strip_score_only or strip_slop:
                clean_nl = re.sub(r"\bscore_\d+(?:_up)?\b[, ]*", "", clean_nl, flags=re.IGNORECASE).strip(" ,")
            if strip_slop:
                for slop in ("masterpiece", "best quality", "ultra-detailed", "ultra detailed", "8k", "4k", "extremely detailed", "insanely detailed"):
                    clean_nl = re.sub(rf"\b{re.escape(slop)}\b[, ]*", "", clean_nl, flags=re.IGNORECASE).strip(" ,")
            if clean_nl and clean_nl.lower() not in seen_tags:
                seen_tags.add(clean_nl.lower())
                b_natural.append(clean_nl)
                total_tags += 1
            continue

        parts = _split_tags_preserving_parens(line)
        for p in parts:
            p_strip = p.strip()
            p_lower = p_strip.lower()
            if p_lower in ("deviantart", "ye-pop"):
                if embedded_trigger is None:
                    embedded_trigger = p_lower
                continue

            if _is_natural_language_sentence(p):
                # Preserve natural language sentence without lowercasing
                clean_nl = p.strip()
                if strip_score_only or strip_slop:
                    clean_nl = re.sub(r"\bscore_\d+(?:_up)?\b[, ]*", "", clean_nl, flags=re.IGNORECASE).strip(" ,")
                if strip_slop:
                    for slop in ("masterpiece", "best quality", "ultra-detailed", "ultra detailed", "8k", "4k", "extremely detailed", "insanely detailed"):
                        clean_nl = re.sub(rf"\b{re.escape(slop)}\b[, ]*", "", clean_nl, flags=re.IGNORECASE).strip(" ,")
                if clean_nl and clean_nl.lower() not in seen_tags:
                    seen_tags.add(clean_nl.lower())
                    b_natural.append(clean_nl)
                    total_tags += 1
                continue

            cleaned, n_stripped = _clean_single_anima_tag(
                p,
                booru_sanitizer=booru_sanitizer,
                strip_aesthetic_slop=strip_slop,
                strip_score_only=strip_score_only,
            )
            stripped_count += n_stripped
            if not cleaned:
                continue

            dedup_key = cleaned.lower()
            if dedup_key in seen_tags:
                continue
            seen_tags.add(dedup_key)
            total_tags += 1

            if is_negative:
                b_general.append(cleaned)
                continue

            # Classify into canonical bucket for positive prompts
            raw_clean = re.sub(r"^\((.+):[+-]?\d+(?:\.\d+)?\)$", r"\1", cleaned).strip()
            if _BUCKET_QUALITY_META_RE.match(raw_clean):
                b_quality.append(cleaned)
            elif _BUCKET_SUBJECT_COUNT_RE.match(raw_clean):
                b_subject.append(cleaned)
            elif raw_clean.startswith("@") or any(raw_clean.endswith(st) or raw_clean == st for st in _BUILTIN_STYLE_TRIGGERS) or "style" in raw_clean:
                b_artist.append(cleaned)
            else:
                b_general.append(cleaned)

    # 4. Assemble in Canonical Order
    if is_negative:
        body = ", ".join(b_general)
    else:
        canonical_parts = []
        if b_quality:
            canonical_parts.append(", ".join(b_quality))
        if b_subject:
            canonical_parts.append(", ".join(b_subject))
        if b_character:
            canonical_parts.append(", ".join(b_character))
        if b_artist:
            canonical_parts.append(", ".join(b_artist))
        if b_general:
            canonical_parts.append(", ".join(b_general))
        if b_natural:
            canonical_parts.append(" ".join(b_natural))
        body = ", ".join(canonical_parts)

    header_injected = False
    if manual_header_lines:
        if body.strip():
            body = "\n".join(manual_header_lines) + "\n" + body.strip()
        else:
            body = "\n".join(manual_header_lines)
        header_injected = True
    elif style_mode == "Flat Vector / Graphic (Non-Anime Dataset Header)":
        chosen_trigger = embedded_trigger if embedded_trigger else ("deviantart" if "deviantart" in str(header_trigger) else "ye-pop")
        trigger = chosen_trigger
        title = str(artwork_title or "").strip().strip("\n")
        first_line = body.splitlines()[0].strip().lower() if body else ""
        already_has_trigger = first_line.startswith("ye-pop") or first_line.startswith("deviantart")
        if already_has_trigger:
            # The body already carries its own dataset trigger header; do not double-inject.
            pass
        elif "Dual Legacy" in str(header_mode):
            # Legacy dual trigger header (both dataset triggers on lines 1-2).
            body = f"ye-pop\ndeviantart\n{body}".strip()
            header_injected = True
        else:
            # CircleStone Labs canonical 3-line header format:
            # Line 1: single dataset trigger (ye-pop OR deviantart) + \n
            # Line 2: artwork title / alt-text (optional) + \n
            # Line 3: natural language / booru body
            header_lines = [trigger]
            if title:
                header_lines.append(title)
            header_lines.append(body.strip())
            body = "\n".join(header_lines)
            header_injected = True
    elif style_mode == "Cinematic / Detailed Shading":
        lower_body = body.lower()
        if "cinematic lighting" not in lower_body:
            body = f"{body}, cinematic lighting, volumetric atmosphere".strip(", ")

    return body, {
        "stripped_slop": stripped_count,
        "tag_count": total_tags,
        "header_injected": header_injected,
    }


def _encode_anima_clip_with_t5_scale(clip, text, t5_weight_multiplier=1.0):
    if clip is None:
        return None
    tokens = clip.tokenize(text)
    if isinstance(tokens, dict) and "t5xxl" in tokens and abs(float(t5_weight_multiplier) - 1.0) > 1e-4:
        mult = float(t5_weight_multiplier)
        scaled_batches = []
        for batch in tokens["t5xxl"]:
            new_batch = []
            for item in batch:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    tok_id = item[0]
                    orig_w = float(item[1])
                    new_w = 1.0 + (orig_w - 1.0) * mult if abs(orig_w - 1.0) > 1e-6 else 1.0
                    if len(item) >= 3:
                        new_batch.append((tok_id, new_w, item[2]))
                    else:
                        new_batch.append((tok_id, new_w))
                else:
                    new_batch.append(item)
            scaled_batches.append(new_batch)
        tokens["t5xxl"] = scaled_batches
    return clip.encode_from_tokens_scheduled(tokens)


class HaDeSAnimaPromptStudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_variant": ([
                    "Anima Aesthetic (Strip score_* & Slop Guard)",
                    "Anima Aesthetic v1.0b (Keep masterpiece, best quality)",
                    "Anima 2.9B / 3.8B Expanded (Keep Quality Tags)",
                    "Anima Base 2B (Keep Quality Tags)",
                    "Anima Turbo / Lightning (CFG 1.0 Optimized)",
                ],),
                "style_mode": ([
                    "Standard Anime / Illustration",
                    "Flat Vector / Graphic (Non-Anime Dataset Header)",
                    "Cinematic / Detailed Shading",
                    "Raw Pass-Through",
                ],),
                "booru_sanitizer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize Booru tags to lowercase, replace underscores with spaces (except score_*, *_up, safe_for_work, kaomojis), escape literal parens, order into canonical buckets, and preserve natural language capitalization."
                }),
                "t5_weight_multiplier": ("FLOAT", {
                    "default": 1.00, "min": 0.0, "max": 2.5, "step": 0.05,
                    "tooltip": "Scales T5XXL query-token emphasis weights (tag:weight) exiting Anima's llm_adapter bridge."
                }),
                "character_tag": ("STRING", {
                    "default": "",
                    "tooltip": "Optional character and series tag(s) (e.g. frieren, sousou no frieren). Automatically normalized and placed in canonical character position."
                }),
                "artist_tag": ("STRING", {
                    "default": "",
                    "tooltip": "Optional artist tag(s) with optional weight (e.g. @zocehuy, (@zocehuy:1.2)). Automatically prefixed with '@' and placed in canonical artist position."
                }),
                "positive_prompt": ("STRING", {
                    "default": "1girl, solo, silver hair, cyan eyes, detailed anime illustration, soft rim lighting",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "worst quality, low quality, bad anatomy, bad hands, blurry, jpeg artifacts, artist name, watermark, signature",
                    "multiline": True,
                }),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional Qwen3/Anima CLIP to encode CONDITIONING directly with T5 weight scaling."}),
                "artist": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional external artist tag(s) input from selector or text nodes."
                }),
                "header_mode": ([
                    "Canonical (1 Trigger + Title)",
                    "Dual Legacy (ye-pop + deviantart)",
                ], {
                    "tooltip": "Flat Vector / Graphic header format. Canonical = CircleStone Labs 3-line format: trigger on Line 1, optional artwork title on Line 2, prompt on Line 3. Dual Legacy = legacy ye-pop\ndeviantart header."
                }),
                "header_trigger": (["ye-pop", "deviantart"], {
                    "tooltip": "Dataset trigger for Line 1 of the canonical header."
                }),
                "artwork_title": ("STRING", {
                    "default": "",
                    "tooltip": "Optional artwork title / alt-text for Line 2 of the canonical header. Leave empty to omit Line 2."
                }),
                "character": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional external character tag(s) input from selector or text nodes."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_string", "negative_string", "info")
    FUNCTION = "process_prompt"
    CATEGORY = "HaDeS/Anima Pipeline"

    def process_prompt(self, model_variant, style_mode, booru_sanitizer, t5_weight_multiplier,
                       character_tag, artist_tag, positive_prompt, negative_prompt,
                       clip=None, artist=None, character=None,
                       header_mode="Canonical (1 Trigger + Title)",
                       header_trigger="ye-pop",
                       artwork_title="",
                       artist_in=None, character_in=None):
        # Backwards-compatibility fallback for legacy socket names
        if artist is None and artist_in is not None:
            artist = artist_in
        if character is None and character_in is not None:
            character = character_in

        # Merge external sockets with inline widgets without duplicates
        merged_chars = []
        for src in (character, character_tag):
            if src and str(src).strip():
                merged_chars.append(str(src).strip())
        effective_char = ", ".join(merged_chars)

        merged_artists = []
        for src in (artist, artist_tag):
            if src and str(src).strip():
                merged_artists.append(str(src).strip())
        effective_artist = ", ".join(merged_artists)

        fmt_pos, pos_stats = format_anima_prompt_text(
            positive_prompt,
            model_variant,
            style_mode,
            booru_sanitizer=booru_sanitizer,
            artist_tag=effective_artist,
            character_tag=effective_char,
            is_negative=False,
            header_mode=header_mode,
            header_trigger=header_trigger,
            artwork_title=artwork_title,
        )
        fmt_neg, neg_stats = format_anima_prompt_text(
            negative_prompt,
            model_variant,
            "Raw Pass-Through" if not booru_sanitizer else "Standard Anime / Illustration",
            booru_sanitizer=booru_sanitizer,
            artist_tag="",
            character_tag="",
            is_negative=True,
        )

        pos_cond = _encode_anima_clip_with_t5_scale(clip, fmt_pos, t5_weight_multiplier=t5_weight_multiplier)
        neg_cond = _encode_anima_clip_with_t5_scale(clip, fmt_neg, t5_weight_multiplier=1.0)
        total_stripped = pos_stats["stripped_slop"] + neg_stats["stripped_slop"]

        info_lines = [
            "=== HADES ANIMA PROMPT STUDIO ===",
            f"Variant Profile: {model_variant}",
            f"Style Mode: {style_mode} | Booru Sanitizer: {booru_sanitizer}",
            f"Active Positive Tags: {pos_stats['tag_count']} | Stripped Slop Tags: {total_stripped} | Header Injected: {pos_stats['header_injected']} (mode: {header_mode}, trigger: {header_trigger})",
            f"Character: {effective_char or 'None'} (Socket: {character is not None}) | Artist: {effective_artist or 'None'} (Socket: {artist is not None})",
            f"T5 Query Emphasis Scale: {t5_weight_multiplier:.2f}x | CLIP Connected: {clip is not None}",
            "\n--- FORMATTED POSITIVE PROMPT (CANONICAL ORDER) ---",
            fmt_pos,
        ]
        return (pos_cond, neg_cond, fmt_pos, fmt_neg, "\n".join(info_lines))


# ==============================================================================
# 7. HADES ANIMA ATTENTION GUIDANCE (NAG / CFG-1.0 BOOSTER)
# ==============================================================================
class HaDeSAnimaAttentionGuidance:
    PRESET_CONFIGS = {
        "Turbo / Lightning Negative Enabler": (3.5, 0.45, 2.5, 0.0, 0.70),
        "Aesthetic Crispness (Anti-Burn)": (2.2, 0.30, 2.0, 0.0, 0.60),
        "High Adherence Extrapolation": (4.5, 0.50, 3.0, 0.0, 0.75),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "guidance_preset": ([
                    "Custom",
                    "Turbo / Lightning Negative Enabler",
                    "Aesthetic Crispness (Anti-Burn)",
                    "High Adherence Extrapolation",
                ],),
                "nag_scale": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Attention extrapolation factor (1.0 = disabled)."
                }),
                "nag_alpha": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend ratio between L1-normalized guided attention and base conditional attention."
                }),
                "nag_tau": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "L1-norm ratio ceiling (prevents high-CFG contrast burn)."
                }),
                "start_percent": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "only_anima": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Only apply guidance to Anima models (unet_config.image_model == 'anima'). Non-Anima models are safely bypassed. Disable to force-apply to any model (experimental)."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "apply_nag"
    CATEGORY = "HaDeS/Anima Pipeline"

    def apply_nag(self, model, guidance_preset, nag_scale, nag_alpha, nag_tau, start_percent, end_percent, only_anima=True):
        if guidance_preset in self.PRESET_CONFIGS:
            nag_scale, nag_alpha, nag_tau, start_percent, end_percent = self.PRESET_CONFIGS[guidance_preset]

        m = model.clone()
        if only_anima:
            unet_config = getattr(getattr(m.model, "model_config", None), "unet_config", None)
            image_model = None
            if isinstance(unet_config, dict):
                image_model = unet_config.get("image_model")
            else:
                image_model = getattr(unet_config, "image_model", None)
            if image_model != "anima":
                return (m, "HaDeS NAG: Bypassed (not an Anima model; only_anima=True).")
        if nag_scale <= 1.0 or nag_alpha <= 0.0 or start_percent >= end_percent:
            return (m, "HaDeS NAG: Bypassed (nag_scale <= 1.0 or nag_alpha == 0.0).")

        model_sampling = m.get_model_object("model_sampling")
        if model_sampling is not None and hasattr(model_sampling, "percent_to_sigma"):
            sigma_start = float(model_sampling.percent_to_sigma(start_percent))
            sigma_end = float(model_sampling.percent_to_sigma(end_percent))
        else:
            sigma_start = float("inf")
            sigma_end = 0.0

        nag_state = {
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }
        m.model_options["_hades_nag_state"] = nag_state

        prev_override = m.model_options.get("transformer_options", {}).get("optimized_attention_override", None)

        def _call_base_attn(func, *args, **kwargs):
            if callable(prev_override):
                return prev_override(func, *args, **kwargs)
            return func(*args, **kwargs)

        def _apply_l1_nag(z_cond, z_baseline):
            orig_dtype = z_cond.dtype
            zc = z_cond.float()
            zb = z_baseline.float()
            z_guided = zc + float(nag_scale) * (zc - zb)

            norm_c = torch.norm(zc, p=1, dim=-1, keepdim=True).clamp(min=1e-6)
            norm_g = torch.norm(z_guided, p=1, dim=-1, keepdim=True).clamp(min=1e-6)
            ratio = (norm_g / norm_c).clamp(max=float(nag_tau))
            z_normed = (z_guided / norm_g) * (norm_c * ratio)

            z_final = float(nag_alpha) * z_normed + (1.0 - float(nag_alpha)) * zc
            return z_final.to(orig_dtype)

        def hades_nag_attention_override(func, *args, **kwargs):
            q, k, v, heads = args[0], args[1], args[2], args[3]
            mask = kwargs.get("mask", None)

            if mask is not None:
                return _call_base_attn(func, *args, **kwargs)

            t_opts = kwargs.get("transformer_options", {})
            sigmas = t_opts.get("sigmas", None)
            if sigmas is not None:
                active_state = m.model_options.get("_hades_nag_state", nag_state)
                s_start = active_state["sigma_start"]
                s_end = active_state["sigma_end"]
                curr_sigma = float(sigmas[0].item()) if isinstance(sigmas, torch.Tensor) else float(sigmas)
                if not (s_end <= curr_sigma <= s_start):
                    return _call_base_attn(func, *args, **kwargs)

            total_batch = q.shape[0]
            cond_or_uncond = t_opts.get("cond_or_uncond", [0, 1])
            n_chunks = len(cond_or_uncond)

            if n_chunks >= 2 and total_batch >= n_chunks and total_batch % n_chunks == 0 and (0 in cond_or_uncond and 1 in cond_or_uncond):
                out_all = _call_base_attn(func, *args, **kwargs)
                chunks = list(out_all.chunk(n_chunks, dim=0))
                uncond_idx = cond_or_uncond.index(1)
                z_uncond = chunks[uncond_idx]
                for idx_c, flag in enumerate(cond_or_uncond):
                    if flag == 0:
                        chunks[idx_c] = _apply_l1_nag(chunks[idx_c], z_uncond)
                return torch.cat(chunks, dim=0)

            # Single-branch fallback: extrapolate cross-attention against token-mean context baseline
            out_cond = _call_base_attn(func, *args, **kwargs)
            if v.ndim == 4 and out_cond.ndim == 3 and k.shape[2] != q.shape[2] and v.shape[2] > 1:
                v_mean = v.mean(dim=2, keepdim=True).permute(0, 2, 1, 3).reshape(out_cond.shape[0], 1, -1)
                if v_mean.shape[-1] == out_cond.shape[-1]:
                    return _apply_l1_nag(out_cond, v_mean.expand_as(out_cond))
            elif v.ndim == 3 and out_cond.ndim == 3 and k.shape[1] != q.shape[1] and v.shape[1] > 1:
                v_mean = v.mean(dim=1, keepdim=True)
                if v_mean.shape[-1] == out_cond.shape[-1]:
                    return _apply_l1_nag(out_cond, v_mean.expand_as(out_cond))
            return out_cond

        if "transformer_options" not in m.model_options:
            m.model_options["transformer_options"] = {}
        else:
            m.model_options["transformer_options"] = m.model_options["transformer_options"].copy()
        m.model_options["transformer_options"]["optimized_attention_override"] = hades_nag_attention_override
        m.model_options["disable_cfg1_optimization"] = True

        info = (
            f"HaDeS Anima NAG Active | Preset: {guidance_preset} | Scale: {nag_scale:.2f} | "
            f"Alpha: {nag_alpha:.2f} | Tau: {nag_tau:.2f} | Window: {start_percent*100:.0f}%..{end_percent*100:.0f}% "
            f"(Sigma {sigma_start:.3f} -> {sigma_end:.3f})"
        )
        return (m, info)


# ==============================================================================
# 8. HADES ANIMA LATENT & PIPELINE GUARD
# ==============================================================================
class HaDeSAnimaLatentPipelineGuard:
    RESOLUTION_MAP = {
        "1024x1024 (1:1 Square - 1.05 MP)": (1024, 1024),
        "896x1152 (7:9 Portrait - 1.03 MP)": (896, 1152),
        "832x1216 (13:19 Portrait - 1.01 MP)": (832, 1216),
        "1152x896 (9:7 Landscape - 1.03 MP)": (1152, 896),
        "1216x832 (19:13 Landscape - 1.01 MP)": (1216, 832),
        "1344x768 (7:4 Widescreen - 1.03 MP)": (1344, 768),
        "1536x640 (12:5 Ultrawide - 0.98 MP)": (1536, 640),
        "704x1408 (1:2 Tall Banner - 0.99 MP)": (704, 1408),
        "1408x704 (2:1 Wide Banner - 0.99 MP)": (1408, 704),
        "512x2048 (1:4 Extreme Vertical - 1.05 MP)": (512, 2048),
        "2048x512 (4:1 Extreme Panorama - 1.05 MP)": (2048, 512),
    }

    @classmethod
    def INPUT_TYPES(s):
        presets = list(s.RESOLUTION_MAP.keys()) + ["Custom (Auto-Snap to 32px Grid)"]
        return {
            "required": {
                "resolution_preset": (presets, {"default": "1024x1024 (1:1 Square - 1.05 MP)"}),
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "prevent_29b_white_bg_bias": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Shifts 1280x1280 to 1216x1216 to avoid the Anima-2.9B (Preview 3) white-background dataset bucket bias."
                }),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional Anima model to inspect topology (28L / 40L / 52L)."}),
                "vae": ("VAE", {"tooltip": "Optional VAE to validate 16-channel Wan21/Qwen-Image compatibility (blocks 64ch VAE error)."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "info")
    FUNCTION = "generate_latent"
    CATEGORY = "HaDeS/Anima Pipeline"

    def generate_latent(self, resolution_preset, custom_width, custom_height, batch_size,
                        prevent_29b_white_bg_bias=True, model=None, vae=None):
        if vae is not None:
            ch = getattr(vae, "latent_channels", None)
            if ch is None and hasattr(vae, "first_stage_model"):
                ch = getattr(vae.first_stage_model, "z_dim", None)
            if ch is not None and int(ch) != 16:
                raise ValueError(
                    f"[HaDeS Anima Pipeline Guard] Incompatible VAE detected ({ch} latent channels). "
                    "Anima requires the 16-channel 'qwen_image_vae.safetensors' (Wan21 latent space), "
                    "NOT the 64-channel Qwen-Image 2.1 VAE."
                )

        if resolution_preset in self.RESOLUTION_MAP:
            w, h = self.RESOLUTION_MAP[resolution_preset]
        else:
            w = max(256, int(round(custom_width / 32.0)) * 32)
            h = max(256, int(round(custom_height / 32.0)) * 32)

        bias_adjusted = False
        if prevent_29b_white_bg_bias and w == 1280 and h == 1280:
            w, h = 1216, 1216
            bias_adjusted = True

        topo_str = "Not connected"
        if model is not None:
            sd_keys = getattr(model, "model_keys", None)
            if not sd_keys and hasattr(model, "model"):
                sd_keys = model.model.state_dict().keys()
            if sd_keys:
                n_dit, n_adapter = detect_anima_topology(sd_keys)
                variant_label = "2.0B" if n_dit == 28 else ("2.9B" if n_dit == 40 else ("3.8B" if n_dit == 52 else "Custom"))
                topo_str = f"Anima {variant_label} ({n_dit} DiT + {n_adapter} Adapter)"

        mp = (w * h) / 1_000_000.0
        mp_status = "Optimal (0.5-1.5 MP)" if 0.5 <= mp <= 1.55 else "Outside Optimal 1MP Window"
        vae_str = "Validated (16ch Wan21)" if vae is not None else "Unchecked (16ch Latent Allocated)"

        latent = torch.zeros(
            [batch_size, 16, h // 8, w // 8],
            device=comfy.model_management.intermediate_device()
        )

        info_lines = [
            "=== HADES ANIMA LATENT & PIPELINE GUARD ===",
            f"Resolution: {w}x{h} ({mp:.2f} MP - {mp_status}) | Batch: {batch_size}",
            f"Latent Shape: [{batch_size}, 16, {h // 8}, {w // 8}] (32px Grid Aligned)",
            f"Detected Model: {topo_str}",
            f"VAE Check: {vae_str}",
        ]
        if bias_adjusted:
            info_lines.append("Note: Auto-shifted 1280x1280 -> 1216x1216 to prevent Anima-2.9B white background bias.")

        return ({"samples": latent}, w, h, "\n".join(info_lines))


# ==============================================================================
# 9. HADES ANIMA UNIVERSAL LORA REMAPPER & SANITIZER
# ==============================================================================
class HaDeSAnimaUniversalLoraLoader:
    def __init__(self):
        lora_dirs = folder_paths.get_folder_paths("loras") if hasattr(folder_paths, "get_folder_paths") else []
        self.output_dir = lora_dirs[0] if lora_dirs else os.path.join(folder_paths.models_dir, "loras")

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "strip_llm_adapter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove llm_adapter / text_conditioner weights from improperly trained LoRAs (enforces llm_adapter_lr = 0)."
                }),
                "clamp_spectral_outliers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Scrub NaN/Inf values and clamp extreme (>10 sigma) tensor outliers."
                }),
                "save_converted_copy": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Export a clean ComfyUI-native .safetensors copy to models/loras/."
                }),
                "output_filename": ("STRING", {"default": "Anima_ComfyNative_LoRA"}),
            },
            "optional": {
                "clip": ("CLIP",),
                "strength_clip": ("FLOAT", {"default": 0.5, "min": -4.0, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "remap_and_load"
    CATEGORY = "HaDeS/Anima Pipeline"

    def _safe_export_path(self, filename, source_path):
        os.makedirs(self.output_dir, exist_ok=True)
        raw_name = os.path.basename(str(filename or "").strip())
        if raw_name.lower().endswith(".safetensors"):
            raw_name = raw_name[:-len(".safetensors")]
        clean_name = _SAFE_FILENAME_RE.sub("_", raw_name).strip(" .") or "Anima_ComfyNative_LoRA"
        base_dir = os.path.abspath(self.output_dir)
        candidate = os.path.abspath(os.path.join(base_dir, f"{clean_name}.safetensors"))
        if os.path.commonpath([base_dir, candidate]) != base_dir:
            raise ValueError("Invalid output filename path.")
        src_abs = os.path.abspath(source_path) if source_path else ""
        if candidate == src_abs or os.path.exists(candidate):
            counter = 1
            while True:
                alt = os.path.abspath(os.path.join(base_dir, f"{clean_name}_{counter:03d}.safetensors"))
                if alt != src_abs and not os.path.exists(alt):
                    candidate = alt
                    break
                counter += 1
        return candidate

    def remap_and_load(self, model, lora_name, strength_model, strip_llm_adapter,
                       clamp_spectral_outliers, save_converted_copy, output_filename,
                       clip=None, strength_clip=0.5):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        raw_sd = _convert_lora_fn(comfy.utils.load_torch_file(lora_path, safe_load=True))
        canon_sd, stats = canonicalize_anima_lora_sd(
            raw_sd,
            strip_llm_adapter=strip_llm_adapter,
            clamp_outliers=clamp_spectral_outliers,
        )

        saved_msg = "Not exported"
        if save_converted_copy and canon_sd:
            out_path = self._safe_export_path(output_filename, lora_path)
            safetensors.torch.save_file(
                canon_sd,
                out_path,
                metadata={
                    "author": "HaDeS_Universal_Remapper",
                    "source_lora": str(lora_name),
                    "strip_llm_adapter": str(strip_llm_adapter),
                },
            )
            saved_msg = out_path

        new_model = model.clone() if model is not None else None
        new_clip = clip.clone() if clip is not None else None

        if (strength_model != 0.0 or (new_clip is not None and strength_clip != 0.0)) and canon_sd:
            eff_clip_str = strength_clip if new_clip is not None else 0.0
            new_model, new_clip = comfy.sd.load_lora_for_models(
                new_model, new_clip, canon_sd, strength_model, eff_clip_str
            )

        info_lines = [
            "=== HADES ANIMA UNIVERSAL LORA REMAPPER ===",
            f"Source LoRA: {lora_name}",
            f"Strengths: Model {strength_model:+.2f} | CLIP {strength_clip:+.2f}",
            f"Diffusers/Kohya Keys Remapped: {stats['remapped_keys']} | Total Output Keys: {len(canon_sd)}",
            f"Dirty llm_adapter Keys Stripped: {stats['stripped_adapter_keys']} | Outliers Clamped: {stats['clamped_tensors']}",
            f"Exported Copy: {saved_msg}",
        ]
        return (new_model, new_clip, "\n".join(info_lines))


# ==============================================================================
# 10. HADES ANIMA CORNER SCRIBBLE SHIELD
# ==============================================================================
class HaDeSAnimaCornerScribbleShield:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "corner_box_percent": ("FLOAT", {
                    "default": 0.10, "min": 0.03, "max": 0.50, "step": 0.01,
                    "tooltip": "Size of corner inspection regions as a fraction of image width/height (default 10%). Large logos can be covered up to 50% — if high-frequency content is still present at the box edge, the box auto-expands up to this cap."
                }),
                "detection_sensitivity": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Sensitivity to high-frequency corner signature scribbles."
                }),
                "active_corners": ([
                    "All 4 Corners",
                    "Bottom Corners Only (Left & Right)",
                    "Bottom-Right Only",
                    "Bottom-Left Only",
                ],),
                "heal_mode": ([
                    "Mask Only (No Pixel Modification)",
                    "Smart Edge-Inward Blend (Auto-Heal)",
                    "Soft Corner Blur",
                    "Aggressive Corner Fill (Forced)",
                ],),
                "feather_radius": ("INT", {"default": 12, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "scribble_mask", "info")
    FUNCTION = "shield_corners"
    CATEGORY = "HaDeS/Anima Pipeline"

    def _get_corner_slices(self, h, w, bh, bw, active_corners):
        coords = []
        if active_corners == "All 4 Corners":
            coords.append(("TL", 0, bh, 0, bw))
            coords.append(("TR", 0, bh, w - bw, w))
            coords.append(("BL", h - bh, h, 0, bw))
            coords.append(("BR", h - bh, h, w - bw, w))
        elif active_corners == "Bottom Corners Only (Left & Right)":
            coords.append(("BL", h - bh, h, 0, bw))
            coords.append(("BR", h - bh, h, w - bw, w))
        elif active_corners == "Bottom-Right Only":
            coords.append(("BR", h - bh, h, w - bw, w))
        else:
            coords.append(("BL", h - bh, h, 0, bw))
        return coords

    def _high_freq_map(self, image, y0, y1, x0, x1):
        """High-frequency energy map [B, h_box, w_box] for one rectangular region.
        Reflect padding: corner patches touch the image border, and zero padding
        would artificially inflate high-frequency energy at that edge (false
        positives on perfectly clean images)."""
        patch_chw = image[:, y0:y1, x0:x1, :3].permute(0, 3, 1, 2)
        smooth_chw = F.avg_pool2d(F.pad(patch_chw, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
        return (patch_chw - smooth_chw).abs().mean(dim=1)  # [B, y1-y0, x1-x0]

    def _ambient_hf(self, image):
        """Measures ambient / background high-frequency energy (mean and peak) from the central core of the image
        to distinguish global texture (grain, noise, canvas pattern) from localized corner watermarks."""
        b, h, w, _ = image.shape
        cy0, cy1 = max(0, int(round(h * 0.35))), min(h, int(round(h * 0.65)))
        cx0, cx1 = max(0, int(round(w * 0.35))), min(w, int(round(w * 0.65)))
        amb_map = self._high_freq_map(image, cy0, cy1, cx0, cx1)
        return float(amb_map.mean().item()), float(amb_map.max().item())

    def _expand_corner_box(self, name, y0, y1, x0, x1, image, thresh, h, w, ambient_mean=0.0, ambient_max=0.0, max_iter=3):
        """Adaptive box expansion: if high-frequency (logo) content is still present
        at the box edge that faces the image center, the mark extends beyond the
        box — grow it (x1.5 per step) and re-measure, up to 3 times.
        Strictly capped at 50% of image dimensions (max_bh, max_bw).
        No-op on clean corners or dense global textures."""
        max_bh = int(round(h * 0.50))
        max_bw = int(round(w * 0.50))
        eff_thresh = max(thresh, ambient_mean * 1.35 + 0.01) if ambient_mean >= 0.005 else thresh
        for _ in range(max_iter):
            curr_bh = y1 - y0
            curr_bw = x1 - x0
            if curr_bh >= max_bh and curr_bw >= max_bw:
                break
            hf = self._high_freq_map(image, y0, y1, x0, x1)

            band = max(2, min(6, hf.shape[1] // 4, hf.shape[2] // 4))
            if band < 2:
                break
            # Inner edge = the box side that does NOT touch the image border.
            edge_y = hf[:, :band, :].mean().item() if y0 > 0 else hf[:, -band:, :].mean().item()
            edge_x = hf[:, :, :band].mean().item() if x0 > 0 else hf[:, :, -band:].mean().item()
            if max(float(edge_y), float(edge_x)) < eff_thresh:
                break

            gy = min(max(16, curr_bh // 2), max(0, max_bh - curr_bh))
            gx = min(max(16, curr_bw // 2), max(0, max_bw - curr_bw))
            if gy == 0 and gx == 0:
                break
            if y0 > 0:
                y0 = max(0, y0 - gy)
            else:
                y1 = min(h, y1 + gy)
            if x0 > 0:
                x0 = max(0, x0 - gx)
            else:
                x1 = min(w, x1 + gx)
        return y0, y1, x0, x1

    def shield_corners(self, image, corner_box_percent, detection_sensitivity, active_corners, heal_mode, feather_radius):
        b_sz, h, w, c = image.shape
        # Box may now cover up to 50% of the dimension (large logo support);
        # the widget max (0.50) is the effective cap.
        bh = max(8, min(h, int(round(h * float(corner_box_percent)))))
        bw = max(8, min(w, int(round(w * float(corner_box_percent)))))

        out_img = image.clone()
        mask_out = torch.zeros((b_sz, h, w), dtype=image.dtype, device=image.device)

        corners = self._get_corner_slices(h, w, bh, bw, active_corners)
        triggered_corners = []

        ambient_mean, ambient_max = self._ambient_hf(image)
        has_ambient_texture = (ambient_mean >= 0.005)

        # Threshold scales inversely with sensitivity (higher sensitivity -> lower threshold).
        # Calibrated against real Anima watermark signatures (hf_score ~0.03-0.04).
        thresh = max(0.008, 0.06 - 0.05 * float(detection_sensitivity))

        actually_expanded = []
        for name, y0, y1, x0, x1 in corners:
            # Large-logo support: auto-grow the box while logo content is still
            # detected at its inner edge (capped at 50%, no-op on clean/texture).
            y0e, y1e, x0e, x1e = self._expand_corner_box(
                name, y0, y1, x0, x1, image, thresh, h, w, ambient_mean=ambient_mean, ambient_max=ambient_max
            )
            was_expanded = (y0e, y1e, x0e, x1e) != (y0, y1, x0, x1)
            y0, y1, x0, x1 = y0e, y1e, x0e, x1e

            curr_bh = y1 - y0
            curr_bw = x1 - x0
            patch = image[:, y0:y1, x0:x1, :3]  # [B, curr_bh, curr_bw, 3]
            patch_chw = patch.permute(0, 3, 1, 2)
            high_freq = self._high_freq_map(image, y0, y1, x0, x1)  # [B, curr_bh, curr_bw]

            # Build spatial falloff weight so the outer corner softly feathers inward
            vy = torch.linspace(1.0, 0.0, curr_bh, device=image.device, dtype=image.dtype) if y0 == 0 else torch.linspace(0.0, 1.0, curr_bh, device=image.device, dtype=image.dtype)
            vx = torch.linspace(1.0, 0.0, curr_bw, device=image.device, dtype=image.dtype) if x0 == 0 else torch.linspace(0.0, 1.0, curr_bw, device=image.device, dtype=image.dtype)
            spatial_env = (vy.unsqueeze(1) * vx.unsqueeze(0)).unsqueeze(0)  # [1, curr_bh, curr_bw]

            hf_score = float(high_freq.mean().item())
            hf_max = float(high_freq.max().item())

            # Anti-Texture Guard: If ambient texture is detected AND neither the mean nor the local peak
            # exceeds the ambient texture level, this corner is uniform canvas/grain/hatching.
            is_uniform_texture = (
                has_ambient_texture
                and (hf_score <= ambient_mean * 1.45 + 0.01)
                and (hf_max <= ambient_max * 1.40 + 0.02)
                and heal_mode != "Aggressive Corner Fill (Forced)"
            )

            if is_uniform_texture or (hf_max < thresh and heal_mode != "Aggressive Corner Fill (Forced)"):
                stroke_mask = torch.zeros_like(high_freq)
                full_fill = False
            else:
                eff_thresh = max(thresh, ambient_max * 0.85) if has_ambient_texture else thresh
                stroke_mask = ((high_freq - eff_thresh).clamp(min=0.0) / 0.04).clamp(max=1.0)
                full_fill = (hf_score >= eff_thresh * 0.8)

            if heal_mode == "Aggressive Corner Fill (Forced)":
                corner_mask = torch.maximum(stroke_mask * spatial_env.sqrt(), spatial_env)
            else:
                corner_mask = torch.maximum(stroke_mask * spatial_env.sqrt(), spatial_env * (1.0 if full_fill else 0.0))

            if int(feather_radius) > 1:
                k_sz = max(3, (int(feather_radius) // 2) * 2 + 1)
                pad_sz = k_sz // 2
                corner_mask = F.avg_pool2d(corner_mask.unsqueeze(1), kernel_size=k_sz, stride=1, padding=pad_sz).squeeze(1)

            is_active_corner = (full_fill or float(corner_mask.max().item()) > 0.05) and not is_uniform_texture
            if is_active_corner:
                triggered_corners.append(f"{name} (HF={hf_score:.3f})")
                if was_expanded:
                    actually_expanded.append(name)

            mask_out[:, y0:y1, x0:x1] = torch.maximum(mask_out[:, y0:y1, x0:x1], corner_mask)

            if heal_mode in ("Smart Edge-Inward Blend (Auto-Heal)", "Aggressive Corner Fill (Forced)"):
                # Robust reference: sample the guaranteed-clean outer ring (8-16px)
                # just OUTSIDE the corner box and continue its content inward
                # (edge continuation), instead of the old 2px inner strips.
                ring = max(8, min(16, max(curr_bh, curr_bw) // 2))
                if y0 > 0:
                    v_rows = y0 - 1 - torch.arange(curr_bh, device=image.device).flip(0).clamp(max=ring - 1)
                else:
                    v_rows = y1 + torch.arange(curr_bh, device=image.device).clamp(max=ring - 1)
                if x0 > 0:
                    h_cols = x0 - 1 - torch.arange(curr_bw, device=image.device).flip(0).clamp(max=ring - 1)
                else:
                    h_cols = x1 + torch.arange(curr_bw, device=image.device).clamp(max=ring - 1)
                ref_v = image[:, v_rows, x0:x1, :3]  # vertical edge continuation [B, curr_bh, curr_bw, 3]
                ref_h = image[:, y0:y1, h_cols, :3]  # horizontal edge continuation [B, curr_bh, curr_bw, 3]
                clean_fill = 0.5 * (ref_v + ref_h)

                def _blur4(t):
                    k = max(3, (int(feather_radius) // 2) * 2 + 1)
                    return F.avg_pool2d(t.permute(0, -1, 1, 2), kernel_size=k, stride=1, padding=k // 2).permute(0, 2, 3, 1)

                clean_fill = _blur4(clean_fill)

                if heal_mode == "Aggressive Corner Fill (Forced)":
                    alpha = spatial_env.unsqueeze(-1)
                    if int(feather_radius) > 1:
                        k = max(3, (int(feather_radius) // 4) * 2 + 1)
                        alpha = F.avg_pool2d(alpha.permute(0, -1, 1, 2), kernel_size=k, stride=1, padding=k // 2).permute(0, 2, 3, 1)
                    alpha = alpha.clamp(min=0.0, max=1.0)
                else:
                    alpha = corner_mask.unsqueeze(-1)
                    if not full_fill and float(corner_mask.max().item()) < 0.05:
                        alpha = torch.zeros_like(alpha)  # nothing detected: leave pixels untouched

                out_img[:, y0:y1, x0:x1, :3] = patch * (1.0 - alpha) + clean_fill * alpha
            elif heal_mode == "Soft Corner Blur":
                blur_k = max(5, (int(feather_radius) // 2) * 2 + 1)
                blurred = F.avg_pool2d(patch_chw, kernel_size=blur_k, stride=1, padding=blur_k // 2).permute(0, 2, 3, 1)
                alpha = corner_mask.unsqueeze(-1)
                out_img[:, y0:y1, x0:x1, :3] = patch * (1.0 - alpha) + blurred * alpha

        det_str = ", ".join(triggered_corners) if triggered_corners else "Clean (Low HF Energy)"
        expanded_note = f" | Auto-Expanded (large mark): {', '.join(actually_expanded)}" if actually_expanded else ""
        info = (
            f"HaDeS Corner Scribble Shield | Box: {bw}x{bh}px ({corner_box_percent*100:.0f}% base){expanded_note} | "
            f"Mode: {heal_mode} | Active: {active_corners} | Detected: {det_str}"
        )
        return (out_img, mask_out, info)
