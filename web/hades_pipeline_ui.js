import {
    HADES_THEME,
    drawRoundedRect,
    setWidgetValue,
    hideBackingWidget,
    applyNodeFrameStyle,
    createSpectralBannerWidget,
    createStyledComboWidget,
    createSpectralSliderWidget,
    createStyledTextRowWidget,
    createStyledToggleWidget,
} from "./hades_widgets.js";

// =============================================================================
// UI INSTALLERS FOR THE 5 HADES ANIMA PIPELINE & GUARD NODES
// =============================================================================

function migratePromptStudioInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return;

    const migrateSlot = (oldName, newName) => {
        const oldIdx = node.inputs.findIndex((i) => i && i.name === oldName);
        if (oldIdx === -1) return;

        const newIdx = node.inputs.findIndex((i) => i && i.name === newName);
        const oldInput = node.inputs[oldIdx];

        if (newIdx !== -1 && oldInput) {
            // Transfer wire/link if old slot has one and new slot doesn't
            if (oldInput.link !== null && oldInput.link !== undefined && node.graph?.links) {
                const link = node.graph.links[oldInput.link];
                if (link && (node.inputs[newIdx].link === null || node.inputs[newIdx].link === undefined)) {
                    link.target_slot = newIdx;
                    node.inputs[newIdx].link = oldInput.link;
                    oldInput.link = null; // detach so removeInput doesn't destroy the transferred link
                }
            }
            if (typeof node.removeInput === "function") {
                node.removeInput(oldIdx);
            } else {
                node.inputs.splice(oldIdx, 1);
            }
        } else if (newIdx === -1 && oldInput) {
            oldInput.name = newName;
        }
    };

    migrateSlot("artist_in", "artist");
    migrateSlot("character_in", "character");
}

export function setupPromptStudioUI(node) {
    applyNodeFrameStyle(node);
    migratePromptStudioInputs(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wVariant = findW("model_variant");
    const wStyle = findW("style_mode");
    const wSanitize = findW("booru_sanitizer");
    const wT5Mult = findW("t5_weight_multiplier");
    const wChar = findW("character_tag");
    const wArtist = findW("artist_tag");
    const wPos = findW("positive_prompt");
    const wNeg = findW("negative_prompt");
    const wHeaderMode = findW("header_mode");
    const wHeaderTrig = findW("header_trigger");

    [wVariant, wStyle, wSanitize, wT5Mult, wChar, wArtist].forEach(hideBackingWidget);

    const TAG_W = 66;

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.emerald,
            customHeight: 52,
            actions: [
                {
                    label: "Aesthetic",
                    tooltip: `<strong>Anima Aesthetic Profile</strong><br>Enables Slop Guard to strip score_* tags and low-quality tokens for pure artistic prompt interpretation.`,
                    onClick: () =>
                        setWidgetValue(wVariant, "Anima Aesthetic (Strip score_* & Slop Guard)", node),
                },
                {
                    label: "2.9B/3.8B",
                    tooltip: `<strong>Anima 2.9B / 3.8B Expanded Profile</strong><br>Preserves quality tags and enables expanded vocabulary for deep DiT architectures.`,
                    onClick: () =>
                        setWidgetValue(wVariant, "Anima 2.9B / 3.8B Expanded (Keep Quality Tags)", node),
                },
                {
                    label: "Base 2B",
                    tooltip: `<strong>Anima Base 2B Profile</strong><br>Standard training baseline with full quality tag retention.`,
                    onClick: () => setWidgetValue(wVariant, "Anima Base 2B (Keep Quality Tags)", node),
                },
                {
                    label: "Vector",
                    tooltip: `<strong>Flat Vector / Graphic Style</strong><br>Applies official CircleStone 3-line header format (ye-pop trigger + optional title).`,
                    onClick: () =>
                        setWidgetValue(wStyle, "Flat Vector / Graphic (Non-Anime Dataset Header)", node),
                },
            ],
            drawCustomHUD(ctx, rect) {
                const isAesthetic = String(wVariant?.value || "").includes("Aesthetic");
                const isVector = String(wStyle?.value || "").includes("Flat Vector");
                const t5Scale = Number(wT5Mult?.value ?? 1.0);
                const posText = String(wPos?.value || "");
                const hasSlop = /score_\d|masterpiece|best quality/i.test(posText);

                ctx.font = "bold 9px monospace";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";

                const badgeY = rect.y + 7;
                ctx.fillStyle = isAesthetic ? HADES_THEME.emerald : HADES_THEME.blue;
                ctx.fillText(
                    isAesthetic ? "SLOP GUARD: STRIP score_*" : "BASE 2B: KEEP QUALITY TAGS",
                    rect.x + 8,
                    badgeY
                );

                ctx.textAlign = "right";
                if (isAesthetic && hasSlop) {
                    ctx.fillStyle = HADES_THEME.amber;
                    ctx.fillText("AUTO-STRIPPING SLOP TAGS", rect.x + rect.w - 8, badgeY);
                } else if (isVector) {
                    const trig = String(wHeaderTrig?.value || "ye-pop").toLowerCase();
                    const mode = String(wHeaderMode?.value || "Canonical");
                    const label = mode.includes("Dual") ? "DUAL: ye-pop + deviantart" : `HEADER: ${trig}`;
                    ctx.fillStyle = HADES_THEME.purple;
                    ctx.fillText(label, rect.x + rect.w - 8, badgeY);
                } else {
                    ctx.fillStyle = HADES_THEME.cyan;
                    ctx.fillText(`T5 EMPHASIS: ${t5Scale.toFixed(2)}x`, rect.x + rect.w - 8, badgeY);
                }

                // T5 Query Weight Gauge Bar
                const barX = rect.x + 8;
                const barY = rect.y + 16;
                const barW = rect.w - 16;
                const barH = 6;
                drawRoundedRect(ctx, barX, barY, barW, barH, 2);
                ctx.fillStyle = "#181c24";
                ctx.fill();

                const fillNorm = Math.max(0, Math.min(1, t5Scale / 2.5));
                if (fillNorm > 0) {
                    drawRoundedRect(ctx, barX, barY, barW * fillNorm, barH, 2);
                    ctx.fillStyle = isAesthetic ? HADES_THEME.emerald : HADES_THEME.cyan;
                    ctx.fill();
                }
            },
        })
    );

    const variantWidget = node.addCustomWidget(
        createStyledComboWidget(node, wVariant, {
            tag: "VARIANT",
            tagColor: HADES_THEME.emerald,
            tagWidth: TAG_W,
            label: "",
            tooltip: `<strong>Model Conditioning Variant</strong><br>• <strong>Aesthetic:</strong> Strips quality artifacts (score_9, masterpiece) for clean prompt execution.<br>• <strong>2.9B / 3.8B:</strong> Tailored for expanded deep DiT architectures.<br>• <strong>Base 2B:</strong> Preserves original Danbooru quality tags.<br>• <strong>Turbo / Lightning:</strong> High-adherence distilled mode.`,
        })
    );

    const styleWidget = node.addCustomWidget(
        createStyledComboWidget(node, wStyle, {
            tag: "STYLE",
            tagColor: HADES_THEME.purple,
            tagWidth: TAG_W,
            label: "",
            tooltip: `<strong>Artwork Styling Mode</strong><br>• <strong>Standard Anime:</strong> Traditional anime/illustration conditioning.<br>• <strong>Flat Vector / Graphic:</strong> Employs CircleStone dataset header for clean flat art.<br>• <strong>Cinematic:</strong> Enhances dynamic lighting and detailed textures.<br>• <strong>Raw Pass-Through:</strong> No style injections.`,
        })
    );

    const sanitizeWidget = node.addCustomWidget(
        createStyledToggleWidget(node, wSanitize, {
            tag: "BOORU",
            color: HADES_THEME.cyan,
            tagWidth: TAG_W,
            label: "",
            onText: "SANITIZE",
            offText: "RAW",
            tooltip: `<strong>Booru Grammar & Kaomoji Guard</strong><br>Normalizes Danbooru tags, converts underscores to spaces (protecting kaomojis and score tags), escapes literal parentheses, and arranges tags into canonical buckets.`,
        })
    );

    const t5Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wT5Mult, {
            tag: "T5 WGT",
            rangeTag: "T5 MULT",
            tagWidth: TAG_W,
            label: "",
            color: HADES_THEME.amber,
            min: 0.0,
            max: 2.5,
            visMin: 0.0,
            visMax: 2.5,
            step: 0.05,
            defaultVal: 1.0,
            tooltip: `<strong>T5 Query Emphasis Multiplier</strong><br>Scales token-level attention weight scaling exiting the llm_adapter bridge.<br><span style="color:#fbbf24">• 1.00:</span> Standard balanced emphasis.<br><span style="color:#fbbf24">• > 1.00:</span> Amplifies tag weights for stronger visual feature compliance.`,
        })
    );

    const charWidget = wChar
        ? node.addCustomWidget(
              createStyledTextRowWidget(node, wChar, {
                  tag: "CHAR",
                  color: HADES_THEME.cyan,
                  tagWidth: TAG_W,
                  promptTitle: "Enter Character Tag(s) (e.g. frieren, sousou no frieren)",
                  emptyPlaceholder: "— Optional character & series injection —",
                  allowClearOnRightClick: true,
                  tooltip: `<strong>Character & Series Tags</strong><br>Enter character names and franchises (e.g. frieren, sousou no frieren). Automatically formatted into canonical character position.`,
              })
          )
        : null;

    const artistWidget = wArtist
        ? node.addCustomWidget(
              createStyledTextRowWidget(node, wArtist, {
                  tag: "ARTIST",
                  color: HADES_THEME.blue,
                  tagWidth: TAG_W,
                  promptTitle: "Enter Artist Tag(s) (e.g. @zocehuy, (@zocehuy:1.2), wlop)",
                  emptyPlaceholder: "— Optional @artist / style injection —",
                  allowClearOnRightClick: true,
                  tooltip: `<strong>Artist & Style Tags</strong><br>Enter artist handles with optional weights (e.g. @zocehuy, (@zocehuy:1.2)). Automatically prefixed with '@' and placed in canonical artist position.`,
              })
          )
        : null;

    // Keep multiline prompt textboxes at the bottom of the node
    if (Array.isArray(node.widgets) && (wPos || wNeg)) {
        const tail = [wPos, wNeg].filter(Boolean);
        node.widgets = node.widgets.filter((w) => !tail.includes(w)).concat(tail);
    }

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        migratePromptStudioInputs(this);
        if (info?.size && Array.isArray(info.size) && info.size[0] > 0 && info.size[1] > 0) {
            this.size = [info.size[0], info.size[1]];
        }
        this.setDirtyCanvas?.(true, true);
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        migratePromptStudioInputs(this);
        this.setDirtyCanvas?.(true, true);
    };

    const minW = 440;
    const sz = node.computeSize();
    const initW = (Array.isArray(node.size) && node.size[0] > 0) ? Math.max(minW, node.size[0]) : Math.max(minW, sz[0]);
    const initH = (Array.isArray(node.size) && node.size[1] > 0) ? Math.max(node.size[1], sz[1] + 12) : Math.max(418, sz[1] + 12);
    node.setSize([initW, initH]);
}

export function setupAttentionGuidanceUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wPreset = findW("guidance_preset");
    const wScale = findW("nag_scale");
    const wAlpha = findW("nag_alpha");
    const wTau = findW("nag_tau");
    const wStart = findW("start_percent");
    const wEnd = findW("end_percent");

    [wPreset, wScale, wAlpha, wTau, wStart, wEnd].forEach(hideBackingWidget);

    const NAG_PRESET_MAP = {
        "Turbo / Lightning Negative Enabler": [3.5, 0.45, 2.5, 0.0, 0.70],
        "Aesthetic Crispness (Anti-Burn)": [2.2, 0.30, 2.0, 0.0, 0.60],
        "High Adherence Extrapolation": [4.5, 0.50, 3.0, 0.0, 0.75],
    };

    let isApplyingNagPreset = false;
    const applyPresetValues = (presetName, scale, alpha, tau, startP, endP) => {
        isApplyingNagPreset = true;
        try {
            setWidgetValue(wPreset, presetName, node);
            setWidgetValue(wScale, scale, node);
            setWidgetValue(wAlpha, alpha, node);
            setWidgetValue(wTau, tau, node);
            setWidgetValue(wStart, startP, node);
            setWidgetValue(wEnd, endP, node);
        } finally {
            isApplyingNagPreset = false;
        }
    };

    if (wPreset) {
        const origPresetCb = wPreset.callback;
        wPreset.callback = function (val) {
            if (typeof origPresetCb === "function") origPresetCb.call(this, val);
            if (!isApplyingNagPreset && NAG_PRESET_MAP[val]) {
                const [s, a, t, st, en] = NAG_PRESET_MAP[val];
                applyPresetValues(val, s, a, t, st, en);
            }
        };
    }

    [wScale, wAlpha, wTau, wStart, wEnd].forEach((sliderW) => {
        if (!sliderW) return;
        const origCb = sliderW.callback;
        sliderW.callback = function (val) {
            if (typeof origCb === "function") origCb.call(this, val);
            if (!isApplyingNagPreset && wPreset && wPreset.value !== "Custom") {
                wPreset.value = "Custom";
            }
        };
    });

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.amber,
            customHeight: 66,
            tooltip: `<strong>NAG Attention Guidance & L1 Guard</strong><br>Normalized Attention Guidance providing CFG=1.0 negative prompting on Turbo/Lightning and anti-burn stabilization on Base.<br><span style="color:#fbbf24">• Guidance Window:</span> Active percentage of the diffusion trajectory.<br><span style="color:#38bdf8">• L1 Ceiling (τ):</span> Bounds guided attention to prevent oversaturation and edge frying.`,
            drawCustomHUD(ctx, rect) {
                const startP = Math.max(0, Math.min(1, Number(wStart?.value ?? 0.0)));
                const endP = Math.max(0, Math.min(1, Number(wEnd?.value ?? 0.65)));
                const scale = Number(wScale?.value ?? 3.0);
                const tau = Number(wTau?.value ?? 2.5);

                const padX = 8;
                const trackX = rect.x + padX;
                const trackW = rect.w - padX * 2;
                const trackY = rect.y + 18;
                const trackH = 12;

                drawRoundedRect(ctx, trackX, trackY, trackW, trackH, 4);
                ctx.fillStyle = "#181b22";
                ctx.fill();

                if (endP > startP && scale > 1.0) {
                    const winX = trackX + startP * trackW;
                    const winW = Math.max(4, (endP - startP) * trackW);
                    drawRoundedRect(ctx, winX, trackY, winW, trackH, 4);
                    ctx.fillStyle = `${HADES_THEME.amber}66`;
                    ctx.fill();
                    ctx.strokeStyle = HADES_THEME.amber;
                    ctx.stroke();
                }

                ctx.font = "bold 9px monospace";
                ctx.textBaseline = "middle";
                ctx.textAlign = "left";
                ctx.fillStyle = HADES_THEME.amber;
                const leftTxt = `WINDOW: ${Math.round(startP * 100)}% → ${Math.round(endP * 100)}%`;
                const leftW = ctx.measureText(leftTxt).width;
                ctx.fillText(leftTxt, trackX, rect.y + 9);

                let rightTxt = `SCALE ${scale.toFixed(1)}x | L1 TAU ≤ ${tau.toFixed(1)}`;
                if (trackW - leftW - 12 < ctx.measureText(rightTxt).width) {
                    rightTxt = `S ${scale.toFixed(1)}x | τ ≤ ${tau.toFixed(1)}`;
                }
                if (trackW - leftW - 12 >= ctx.measureText(rightTxt).width) {
                    ctx.textAlign = "right";
                    ctx.fillStyle = HADES_THEME.cyan;
                    ctx.fillText(rightTxt, trackX + trackW, rect.y + 9);
                }
            },
            actions: [
                {
                    label: "Turbo CFG=1",
                    tooltip: `<strong>Turbo / Lightning Preset</strong><br>Sets Scale 3.5, Alpha 0.45, Tau 2.5, Window 0% - 70%. Enables negative prompt steering on CFG=1 models.`,
                    onClick: () =>
                        applyPresetValues("Turbo / Lightning Negative Enabler", 3.5, 0.45, 2.5, 0.0, 0.7),
                },
                {
                    label: "Anti-Burn",
                    tooltip: `<strong>Aesthetic Crispness (Anti-Burn)</strong><br>Sets Scale 2.2, Alpha 0.30, Tau 2.0, Window 0% - 60%. Smooth high-contrast control without oversaturation.`,
                    onClick: () => applyPresetValues("Aesthetic Crispness (Anti-Burn)", 2.2, 0.3, 2.0, 0.0, 0.6),
                },
                {
                    label: "High",
                    tooltip: `<strong>High Adherence Preset</strong><br>Sets Scale 4.5, Alpha 0.50, Tau 3.0, Window 0% - 75%. Strongest prompt adherence guidance.`,
                    onClick: () => applyPresetValues("High Adherence Extrapolation", 4.5, 0.5, 3.0, 0.0, 0.75),
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "PRESET",
            tagColor: HADES_THEME.amber,
            tooltip: `<strong>Attention Guidance Preset</strong><br>Pre-tuned guidance parameters for Turbo/Lightning, Base, or High Adherence.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wScale, {
            tag: "SCALE",
            rangeTag: "NAG s",
            label: "Guidance Extrapolation Scale",
            color: HADES_THEME.amber,
            min: 1.0,
            max: 15.0,
            visMin: 1.0,
            visMax: 8.0,
            step: 0.1,
            defaultVal: 3.0,
            tooltip: `<strong>Guidance Extrapolation Scale (s)</strong><br>Strength of normalized attention guidance vector extrapolation.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wAlpha, {
            tag: "ALPHA",
            rangeTag: "BLEND",
            label: "Guided Attention Blend (α)",
            color: HADES_THEME.emerald,
            min: 0.0,
            max: 1.0,
            visMin: 0.0,
            visMax: 1.0,
            step: 0.05,
            defaultVal: 0.35,
            tooltip: `<strong>Guided Attention Blend (α)</strong><br>Interpolation ratio blending the guidance delta into the base attention maps.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wTau, {
            tag: "TAU",
            rangeTag: "L1 MAX",
            label: "L1-Norm Ceiling (Anti-Burn τ)",
            color: HADES_THEME.cyan,
            min: 1.0,
            max: 10.0,
            visMin: 1.0,
            visMax: 5.0,
            step: 0.1,
            defaultVal: 2.5,
            tooltip: `<strong>L1-Norm Ceiling (τ)</strong><br>Clamps the magnitude of attention deltas to completely prevent pixel burning and artifacts.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wStart, {
            tag: "START",
            rangeTag: "STEP %",
            label: "Start Denoise Percent",
            color: HADES_THEME.blue,
            min: 0.0,
            max: 1.0,
            visMin: 0.0,
            visMax: 1.0,
            step: 0.05,
            defaultVal: 0.0,
            tooltip: `<strong>Guidance Start Step (%)</strong><br>Normalized diffusion timestep where NAG attention guidance begins.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wEnd, {
            tag: "END",
            rangeTag: "STEP %",
            label: "End Denoise Percent",
            color: HADES_THEME.purple,
            min: 0.0,
            max: 1.0,
            visMin: 0.0,
            visMax: 1.0,
            step: 0.05,
            defaultVal: 0.65,
            tooltip: `<strong>Guidance End Step (%)</strong><br>Normalized diffusion timestep where NAG guidance disengages (settling stage).`,
        })
    );

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

const PRESET_DIMS = {
    "1024x1024 (1:1 Square - 1.05 MP)": [1024, 1024],
    "896x1152 (7:9 Portrait - 1.03 MP)": [896, 1152],
    "832x1216 (13:19 Portrait - 1.01 MP)": [832, 1216],
    "1152x896 (9:7 Landscape - 1.03 MP)": [1152, 896],
    "1216x832 (19:13 Landscape - 1.01 MP)": [1216, 832],
    "1344x768 (7:4 Widescreen - 1.03 MP)": [1344, 768],
    "1536x640 (12:5 Ultrawide - 0.98 MP)": [1536, 640],
    "704x1408 (1:2 Tall Banner - 0.99 MP)": [704, 1408],
    "1408x704 (2:1 Wide Banner - 0.99 MP)": [1408, 704],
    "512x2048 (1:4 Extreme Vertical - 1.05 MP)": [512, 2048],
    "2048x512 (4:1 Extreme Panorama - 1.05 MP)": [2048, 512],
};

export function setupLatentPipelineGuardUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wPreset = findW("resolution_preset");
    const wWidth = findW("custom_width");
    const wHeight = findW("custom_height");
    const wBatch = findW("batch_size");
    const wGuard29b = findW("prevent_29b_white_bg_bias");

    [wPreset, wWidth, wHeight, wBatch, wGuard29b].forEach(hideBackingWidget);

    let isApplyingResPreset = false;
    const applyResPreset = (presetName) => {
        isApplyingResPreset = true;
        try {
            setWidgetValue(wPreset, presetName, node);
            const dims = PRESET_DIMS[presetName];
            if (dims) {
                setWidgetValue(wWidth, dims[0], node);
                setWidgetValue(wHeight, dims[1], node);
            }
        } finally {
            isApplyingResPreset = false;
        }
    };

    if (wPreset) {
        const origPresetCb = wPreset.callback;
        wPreset.callback = function (val) {
            if (typeof origPresetCb === "function") origPresetCb.call(this, val);
            if (!isApplyingResPreset && PRESET_DIMS[val]) {
                applyResPreset(val);
            }
        };
    }

    [wWidth, wHeight].forEach((dimW) => {
        if (!dimW) return;
        const origCb = dimW.callback;
        dimW.callback = function (val) {
            if (typeof origCb === "function") origCb.call(this, val);
            if (!isApplyingResPreset && wPreset && wPreset.value !== "Custom (Auto-Snap to 32px Grid)") {
                wPreset.value = "Custom (Auto-Snap to 32px Grid)";
            }
        };
    });

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.cyan,
            customHeight: 64,
            drawCustomHUD(ctx, rect) {
                const pVal = String(wPreset?.value || "");
                let [w, h] = PRESET_DIMS[pVal] || [
                    Math.max(256, Math.round(Number(wWidth?.value ?? 1024) / 32) * 32),
                    Math.max(256, Math.round(Number(wHeight?.value ?? 1024) / 32) * 32),
                ];
                const guardOn = wGuard29b ? Boolean(wGuard29b.value) : true;
                if (guardOn && w === 1280 && h === 1280) {
                    w = 1216;
                    h = 1216;
                }
                const mp = (w * h) / 1e6;
                const isOptimal = mp >= 0.5 && mp <= 1.55;
                const isNarrow = rect.w < 270;

                // Draw aspect-ratio preview box on the left
                const boxAreaW = isNarrow ? 36 : 44;
                const maxBoxH = rect.h - 8;
                const maxBoxW = isNarrow ? 28 : 36;
                const scale = Math.min(maxBoxW / w, maxBoxH / h);
                const pw = Math.max(8, w * scale);
                const ph = Math.max(8, h * scale);
                const px = rect.x + 8 + (maxBoxW - pw) * 0.5;
                const py = rect.y + (rect.h - ph) * 0.5;

                drawRoundedRect(ctx, px, py, pw, ph, 2);
                ctx.fillStyle = `${HADES_THEME.cyan}33`;
                ctx.fill();
                ctx.strokeStyle = HADES_THEME.cyan;
                ctx.stroke();

                ctx.font = isNarrow ? "9.5px monospace" : "bold 10px monospace";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(`${w} × ${h} px  (${mp.toFixed(2)} MP)`, rect.x + boxAreaW + 8, rect.y + 11);

                ctx.font = isNarrow ? "8.5px monospace" : "9px monospace";
                ctx.fillStyle = isOptimal ? HADES_THEME.emerald : HADES_THEME.amber;
                const line2 = isOptimal
                    ? (isNarrow ? "16CH WAN21 · 32PX · OPTIMAL" : "16CH WAN21 · 32PX ALIGNED · OPTIMAL MP")
                    : (isNarrow ? "OUTSIDE 0.5–1.5 MP" : "OUTSIDE 0.5–1.5 MP OPTIMAL WINDOW");
                ctx.fillText(line2, rect.x + boxAreaW + 8, rect.y + 25);
            },
            actions: [
                {
                    label: "1:1",
                    onClick: () => applyResPreset("1024x1024 (1:1 Square - 1.05 MP)"),
                },
                {
                    label: "Portrait",
                    onClick: () => applyResPreset("896x1152 (7:9 Portrait - 1.03 MP)"),
                },
                {
                    label: "Landsc",
                    onClick: () => applyResPreset("1152x896 (9:7 Landscape - 1.03 MP)"),
                },
                {
                    label: "Wide",
                    onClick: () => applyResPreset("1344x768 (7:4 Widescreen - 1.03 MP)"),
                },
            ],
        })
    );

    const TAG_W = 66;

    const presetWidget = node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "ASPECT",
            tagColor: HADES_THEME.cyan,
            tagWidth: TAG_W,
            label: "",
        })
    );

    const widthWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wWidth, {
            tag: "WIDTH",
            rangeTag: "WIDTH",
            tagWidth: TAG_W,
            label: "",
            color: HADES_THEME.blue,
            min: 256,
            max: 4096,
            visMin: 512,
            visMax: 2048,
            step: 32,
            defaultVal: 1024,
            isInt: true,
        })
    );

    const heightWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wHeight, {
            tag: "HEIGHT",
            rangeTag: "HEIGHT",
            tagWidth: TAG_W,
            label: "",
            color: HADES_THEME.blue,
            min: 256,
            max: 4096,
            visMin: 512,
            visMax: 2048,
            step: 32,
            defaultVal: 1024,
            isInt: true,
        })
    );

    const batchWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wBatch, {
            tag: "BATCH",
            rangeTag: "BATCH",
            tagWidth: TAG_W,
            label: "",
            color: HADES_THEME.purple,
            min: 1,
            max: 64,
            visMin: 1,
            visMax: 8,
            step: 1,
            defaultVal: 1,
            isInt: true,
        })
    );

    const guardWidget = wGuard29b
        ? node.addCustomWidget(
              createStyledToggleWidget(node, wGuard29b, {
                  tag: "GUARD",
                  tagWidth: TAG_W,
                  color: HADES_THEME.emerald,
                  label: "",
                  onText: "GUARD ON",
                  offText: "BYPASSED",
                  tooltip: `<strong>Anima 2.9B White-BG Guard</strong><br>Prevents the known Anima-2.9B (Preview 3) dataset artifact where 1280×1280 causes a blank white background or loss of contrast.<br><span style="color:#6ee7b7">When active, 1280×1280 automatically shifts to 1216×1216 (32px-grid aligned).</span>`,
              })
          )
        : null;

    node.widgets = [
        bannerWidget,
        presetWidget,
        widthWidget,
        heightWidget,
        batchWidget,
        guardWidget,
        wPreset,
        wWidth,
        wHeight,
        wBatch,
        wGuard29b,
    ].filter(Boolean);

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

export function setupUniversalLoraLoaderUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wLora = findW("lora_name");
    const wStrM = findW("strength_model");
    const wStrC = findW("strength_clip");
    const wStrip = findW("strip_llm_adapter");
    const wClamp = findW("clamp_spectral_outliers");
    const wSave = findW("save_converted_copy");
    const wOutFile = findW("output_filename");

    [wLora, wStrM, wStrC, wStrip, wClamp, wSave, wOutFile].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.purple,
            customHeight: 58,
            tooltip: `<strong>Universal 24-Rule LoRA Remapper</strong><br>Translates external LoRA architectures into native ComfyUI MiniTrainDIT format on-the-fly.<br><span style="color:#c084fc">• Cross-Format Compatibility:</span> Diffusers, Kohya (lora_unet_*), Musubi, and OneTrainer.<br><span style="color:#34d399">• Strip llm_adapter:</span> Strips dirty text-conditioner keys (lr=0 golden rule) to protect prompt adherence.<br><span style="color:#fbbf24">• Outlier Clamp:</span> Cleans NaN/Inf and clamps extreme >10σ weights.`,
            drawCustomHUD(ctx, rect) {
                const stripOn = wStrip ? Boolean(wStrip.value) : true;
                const clampOn = wClamp ? Boolean(wClamp.value) : false;
                const saveOn = wSave ? Boolean(wSave.value) : false;

                ctx.font = "bold 9px monospace";
                ctx.textBaseline = "middle";
                ctx.textAlign = "left";
                ctx.fillStyle = HADES_THEME.purple;
                ctx.fillText("DIFFUSERS / KOHYA / MUSUBI  ──►  MINITRAINDIT", rect.x + 8, rect.y + 10);

                ctx.fillStyle = stripOn ? HADES_THEME.emerald : HADES_THEME.amber;
                ctx.fillText(
                    `BRIDGE: ${stripOn ? "STRIPPED (SAFE)" : "UNFILTERED"}  |  OUTLIERS: ${clampOn ? "CLAMP 10σ" : "RAW"}  |  EXPORT: ${saveOn ? "ON" : "OFF"}`,
                    rect.x + 8,
                    rect.y + 23
                );
            },
            actions: [
                {
                    label: "1.0x",
                    tooltip: `<strong>Default Strengths</strong><br>Sets Model LoRA to 1.00 and Qwen CLIP LoRA to 0.50.`,
                    onClick: () => {
                        setWidgetValue(wStrM, 1.0, node);
                        if (wStrC) setWidgetValue(wStrC, 0.5, node);
                    },
                },
                {
                    label: "Safe Clean",
                    tooltip: `<strong>Recommended Safe Cleaning</strong><br>Enables llm_adapter stripping and 10σ outlier clamping.`,
                    onClick: () => {
                        setWidgetValue(wStrip, true, node);
                        setWidgetValue(wClamp, true, node);
                    },
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wLora, {
            tag: "LORA",
            tagColor: HADES_THEME.purple,
            cleanFileNames: true,
            tooltip: `<strong>Target LoRA File</strong><br>External (Diffusers, Kohya, Musubi) or native LoRA to load and adapt for Anima.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wStrM, {
            tag: "MODEL",
            rangeTag: "DIT STR",
            label: "Model LoRA Strength",
            color: HADES_THEME.blue,
            min: -4.0,
            max: 4.0,
            visMin: 0.0,
            visMax: 1.5,
            step: 0.05,
            defaultVal: 1.0,
            tooltip: `<strong>Diffusion Model Strength</strong><br>Strength applied to the 28 core Anima DiT transformer blocks.`,
        })
    );

    if (wStrC) {
        node.addCustomWidget(
            createSpectralSliderWidget(node, wStrC, {
                tag: "CLIP",
                rangeTag: "QWEN STR",
                label: "CLIP LoRA Strength",
                color: HADES_THEME.cyan,
                min: -4.0,
                max: 4.0,
                visMin: 0.0,
                visMax: 1.5,
                step: 0.05,
                defaultVal: 0.5,
                tooltip: `<strong>Qwen CLIP Strength</strong><br>Strength applied to the Qwen3 text conditioner layers (recommended 0.50x).`,
            })
        );
    }

    node.addCustomWidget(
        createStyledToggleWidget(node, wStrip, {
            tag: "BRIDGE",
            color: HADES_THEME.emerald,
            label: "Strip llm_adapter Weights (lr=0 Rule)",
            onText: "STRIP",
            offText: "KEEP",
            tooltip: `<strong>Strip llm_adapter Weights</strong><br>Golden rule enforcement: strips text conditioner weights trained at lr=0.`,
        })
    );

    node.addCustomWidget(
        createStyledToggleWidget(node, wClamp, {
            tag: "SCRUB",
            color: HADES_THEME.amber,
            label: "Scrub NaN/Inf & Clamp >10σ Outliers",
            onText: "ACTIVE",
            offText: "OFF",
            tooltip: `<strong>Weight Sanitization</strong><br>Detects and cleans NaN/Inf values and clamps runaway weights exceeding 10 standard deviations.`,
        })
    );

    node.addCustomWidget(
        createStyledToggleWidget(node, wSave, {
            tag: "EXPORT",
            color: HADES_THEME.cyan,
            label: "Save Converted Copy to models/loras/",
            onText: "SAVE COPY",
            offText: "IN-MEMORY",
            tooltip: `<strong>Export Converted SafeTensors</strong><br>Persists the converted native format LoRA to disk for instant loading next time.`,
        })
    );

    node.addCustomWidget(
        createStyledTextRowWidget(node, wOutFile, {
            tag: "FILE",
            color: HADES_THEME.emerald,
            promptTitle: "Output Filename for Converted LoRA",
            suffix: ".safetensors",
            emptyPlaceholder: "Anima_ComfyNative_LoRA",
            tooltip: `<strong>Converted Filename</strong><br>File name used when saving the converted LoRA to disk.`,
        })
    );

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

function migrateOutputSlot(node, oldName, newName) {
    if (!node || !Array.isArray(node.outputs)) return;
    const slot = node.outputs.find((o) => o && o.name === oldName);
    if (slot) {
        slot.name = newName;
        slot.localized_name = newName;
        slot.label = newName;
    }
}

export function setupCornerScribbleShieldUI(node) {
    applyNodeFrameStyle(node);
    migrateOutputSlot(node, "scribble_mask", "mask");

    const origConfigure = node.configure;
    node.configure = function (info) {
        const ret = origConfigure ? origConfigure.apply(this, arguments) : undefined;
        migrateOutputSlot(this, "scribble_mask", "mask");
        return ret;
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        const ret = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
        migrateOutputSlot(this, "scribble_mask", "mask");
        return ret;
    };

    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wBox = findW("corner_box_percent");
    const wSens = findW("detection_sensitivity");
    const wCorners = findW("active_corners");
    const wHeal = findW("heal_mode");
    const wFeather = findW("feather_radius");

    [wBox, wSens, wCorners, wHeal, wFeather].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.rose,
            customHeight: 66,
            tooltip: `<strong>Corner Scribble Shield & Watermark Healer</strong><br>Detects and heals Anima's ~20% corner signature scribbles and watermarks.<br><span style="color:#f43f5e">• Perimeter Box:</span> Inspects corner regions for high-frequency stroke energy.<br><span style="color:#34d399">• Smart Auto-Heal:</span> Inward edge continuation avoiding central pixel corruption.`,
            drawCustomHUD(ctx, rect) {
                const act = String(wCorners?.value || "All 4 Corners");
                const mode = String(wHeal?.value || "");
                const boxPct = Math.round(Number(wBox?.value ?? 0.10) * 100);

                // Mini 4-corner frame monitor
                const fw = 44;
                const fh = rect.h - 8;
                const fx = rect.x + 8;
                const fy = rect.y + 4;

                drawRoundedRect(ctx, fx, fy, fw, fh, 3);
                ctx.strokeStyle = "#333947";
                ctx.stroke();

                const cw = Math.max(6, fw * Math.min(0.32, Number(wBox?.value ?? 0.10) * 2.2));
                const ch = Math.max(5, fh * Math.min(0.32, Number(wBox?.value ?? 0.10) * 2.2));

                const hasTL = act === "All 4 Corners";
                const hasTR = act === "All 4 Corners";
                const hasBL = act === "All 4 Corners" || act.includes("Bottom Corners") || act.includes("Bottom-Left");
                const hasBR = act === "All 4 Corners" || act.includes("Bottom Corners") || act.includes("Bottom-Right");

                ctx.fillStyle = HADES_THEME.rose;
                if (hasTL) ctx.fillRect(fx + 2, fy + 2, cw, ch);
                if (hasTR) ctx.fillRect(fx + fw - cw - 2, fy + 2, cw, ch);
                if (hasBL) ctx.fillRect(fx + 2, fy + fh - ch - 2, cw, ch);
                if (hasBR) ctx.fillRect(fx + fw - cw - 2, fy + fh - ch - 2, cw, ch);

                ctx.font = "bold 9px monospace";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillStyle = HADES_THEME.textPrimary;
                let boxTxt = `INSPECTION BOX: ${boxPct}% PERIMETER`;
                if (ctx.measureText(boxTxt).width > rect.w - fw - 24) {
                    boxTxt = `BOX: ${boxPct}% PERIMETER`;
                }
                ctx.fillText(boxTxt, fx + fw + 10, rect.y + 11);

                ctx.fillStyle = mode.includes("Mask Only") ? HADES_THEME.cyan : HADES_THEME.emerald;
                let actTxt = `ACTION: ${mode.toUpperCase()}`;
                if (ctx.measureText(actTxt).width > rect.w - fw - 24) {
                    actTxt = mode.includes("Mask Only") ? "ACTION: MASK ONLY" : (mode.includes("Auto-Heal") ? "ACTION: AUTO-HEAL" : "ACTION: FORCED");
                }
                ctx.fillText(actTxt, fx + fw + 10, rect.y + 24);
            },
            actions: [
                {
                    label: "All 4",
                    tooltip: `<strong>All 4 Corners</strong><br>Inspects and processes all four corner margins of the image canvas.`,
                    onClick: () => setWidgetValue(wCorners, "All 4 Corners", node),
                },
                {
                    label: "Bottom",
                    tooltip: `<strong>Bottom Corners Only</strong><br>Restricts scanning to bottom-left and bottom-right corners (most common location for Anima signatures).`,
                    onClick: () => setWidgetValue(wCorners, "Bottom Corners Only (Left & Right)", node),
                },
                {
                    label: "Auto-Heal",
                    tooltip: `<strong>Smart Edge-Inward Blend</strong><br>Intelligently blends edge pixels inward across detected stroke areas to remove scribbles seamlessly.`,
                    onClick: () => setWidgetValue(wHeal, "Smart Edge-Inward Blend (Auto-Heal)", node),
                },
                {
                    label: "Forced",
                    tooltip: `<strong>Aggressive Corner Fill</strong><br>Forces inward edge fill across corner boxes regardless of stroke detection threshold.`,
                    onClick: () => setWidgetValue(wHeal, "Aggressive Corner Fill (Forced)", node),
                },
                {
                    label: "Mask Only",
                    tooltip: `<strong>Mask Only (No Pixel Modification)</strong><br>Outputs a feathered soft mask of affected corners without altering the RGB image pixels.`,
                    onClick: () => setWidgetValue(wHeal, "Mask Only (No Pixel Modification)", node),
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wCorners, {
            tag: "CORNERS",
            tagColor: HADES_THEME.rose,
            tooltip: `<strong>Target Corner Regions</strong><br>Select which image perimeter corners to inspect for signature scribbles.`,
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wHeal, {
            tag: "MODE",
            tagColor: HADES_THEME.emerald,
            tooltip: `<strong>Remediation Strategy</strong><br>Auto-Heal (inward edge continuation), Aggressive Fill (forced replacement), or Mask Only.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wBox, {
            tag: "BOX %",
            rangeTag: "MARGIN",
            label: "Corner Box Size Ratio",
            color: HADES_THEME.rose,
            min: 0.03,
            max: 0.50,
            visMin: 0.03,
            visMax: 0.50,
            step: 0.01,
            defaultVal: 0.10,
            tooltip: `<strong>Corner Box Perimeter Ratio</strong><br>Percentage of image width and height defining the corner inspection box (e.g., 0.10 = 10%).`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wSens, {
            tag: "SENS",
            rangeTag: "DETECT",
            label: "Scribble Detection Sensitivity",
            color: HADES_THEME.amber,
            min: 0.0,
            max: 1.0,
            visMin: 0.0,
            visMax: 1.0,
            step: 0.05,
            defaultVal: 0.75,
            tooltip: `<strong>Scribble Detection Sensitivity</strong><br>Laplacian high-frequency threshold for identifying artificial strokes and watermarks.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wFeather, {
            tag: "FTHR",
            rangeTag: "RADIUS",
            label: "Mask Feather Radius (px)",
            color: HADES_THEME.cyan,
            min: 1,
            max: 64,
            visMin: 1,
            visMax: 32,
            step: 1,
            defaultVal: 12,
            isInt: true,
            tooltip: `<strong>Mask Feather Radius (px)</strong><br>Gaussian blur kernel size for smooth alpha transitions at corner borders.`,
        })
    );

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

export function setupSuiteTelemetryUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wFormat = findW("report_format");
    const wTitle = findW("report_title");

    [wFormat, wTitle].forEach(hideBackingWidget);

    node._latestReport = "";
    node._latestJson = "";
    node._latestDiag = "";

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.cyan,
            customHeight: 58,
            tooltip: `<strong>Anima Suite Telemetry & Flight Auditor</strong><br>Aggregates, semantically classifies, and performs automated diagnostic health checks across all HaDeS Suite nodes in the active workflow.<br><span style="color:#38bdf8">• Health Checks:</span> Flow Shift sync, 32px grid alignment, 16ch Wan2.1 VAE, and llm_adapter locks.<br><span style="color:#34d399">• Clipboard Export:</span> Quick-copy full audit reports or structured JSON payload directly to clipboard.`,
            drawCustomHUD(ctx, rect) {
                const diag = node._latestDiag || "STATUS: READY / STANDBY (CONNECT INFO SLOTS)";
                const isWarning = diag.includes("WARNING") || diag.includes("DEGRADED");
                const isError = diag.includes("ERROR") || diag.includes("CRITICAL");
                const statusColor = isError ? HADES_THEME.rose : (isWarning ? HADES_THEME.amber : HADES_THEME.cyan);

                ctx.font = "bold 9px monospace";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";

                // Top line
                ctx.fillStyle = statusColor;
                ctx.fillText("HA-DES FLIGHT RECORDER // TELEMETRY AUDITOR", rect.x + 8, rect.y + 10);

                // Bottom status
                ctx.fillStyle = HADES_THEME.textPrimary;
                let statusTxt = diag;
                if (ctx.measureText(statusTxt).width > rect.w - 16) {
                    statusTxt = statusTxt.slice(0, 48) + "...";
                }
                ctx.fillText(statusTxt, rect.x + 8, rect.y + 24);
            },
            actions: [
                {
                    label: "Copy Report",
                    tooltip: `<strong>Copy Audit Report</strong><br>Copies the latest formatted audit report to system clipboard.`,
                    onClick: () => {
                        const txt = node._latestReport || "No report generated yet. Run the workflow to produce telemetry.";
                        if (typeof navigator !== "undefined" && navigator.clipboard) {
                            navigator.clipboard.writeText(txt);
                        }
                    },
                },
                {
                    label: "Copy JSON",
                    tooltip: `<strong>Copy JSON Raw Telemetry</strong><br>Copies the structured JSON telemetry payload to system clipboard.`,
                    onClick: () => {
                        const txt = node._latestJson || "{}";
                        if (typeof navigator !== "undefined" && navigator.clipboard) {
                            navigator.clipboard.writeText(txt);
                        }
                    },
                },
                {
                    label: "Full / Short",
                    tooltip: `<strong>Toggle Report Detail</strong><br>Switches between Full Audit Report (Markdown) and Compact Flight Summary.`,
                    onClick: () => {
                        if (!wFormat) return;
                        const curr = String(wFormat.value || "");
                        const next = curr.includes("Compact") ? "Full Audit Report (Markdown)" : "Compact Flight Summary";
                        setWidgetValue(wFormat, next, node);
                    },
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wFormat, {
            tag: "FORMAT",
            tagColor: HADES_THEME.cyan,
            tooltip: `<strong>Report Presentation Mode</strong><br>Selects the output style: Full Markdown, Compact Terminal Summary, Diagnostics Only, or Raw JSON.`,
        })
    );

    node.addCustomWidget(
        createStyledTextRowWidget(node, wTitle, {
            tag: "TITLE",
            color: HADES_THEME.emerald,
            promptTitle: "Enter Telemetry Report Title",
            emptyPlaceholder: "HaDeS Anima Suite Telemetry & Flight Audit",
            tooltip: `<strong>Report Document Title</strong><br>Header title embedded at the top of the generated flight telemetry report.`,
        })
    );

    const origOnExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        if (typeof origOnExecuted === "function") origOnExecuted.apply(this, arguments);
        if (output) {
            if (Array.isArray(output.report) && output.report[0]) {
                this._latestReport = output.report[0];
            } else if (typeof output.report === "string") {
                this._latestReport = output.report;
            }
            if (Array.isArray(output.json_telemetry) && output.json_telemetry[0]) {
                this._latestJson = output.json_telemetry[0];
            }
            if (Array.isArray(output.diagnostics) && output.diagnostics[0]) {
                this._latestDiag = output.diagnostics[0];
            }
        }
    };

    const sz = node.computeSize();
    node.setSize([Math.max(440, sz[0]), sz[1] + 8]);
}

export const PIPELINE_NODE_DISPATCH = {
    HaDeSAnimaPromptStudio: setupPromptStudioUI,
    HaDeSAnimaAttentionGuidance: setupAttentionGuidanceUI,
    HaDeSAnimaLatentPipelineGuard: setupLatentPipelineGuardUI,
    HaDeSAnimaUniversalLoraLoader: setupUniversalLoraLoaderUI,
    HaDeSAnimaCornerScribbleShield: setupCornerScribbleShieldUI,
    HaDeSAnimaSuiteTelemetry: setupSuiteTelemetryUI,
};
