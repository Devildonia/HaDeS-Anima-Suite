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

export function setupPromptStudioUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wVariant = findW("model_variant");
    const wStyle = findW("style_mode");
    const wSanitize = findW("booru_sanitizer");
    const wT5Mult = findW("t5_weight_multiplier");
    const wChar = findW("character_tag");
    const wArtist = findW("artist_tag");
    const wPos = findW("positive_prompt");
    const wNeg = findW("negative_prompt");

    [wVariant, wStyle, wSanitize, wT5Mult, wChar, wArtist].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "ANIMA PROMPT & AESTHETIC GUARD",
            accentColor: HADES_THEME.emerald,
            customHeight: 64,
            drawCustomHUD(ctx, rect) {
                const isAesthetic = String(wVariant?.value || "").includes("Aesthetic");
                const isVector = String(wStyle?.value || "").includes("Flat Vector");
                const t5Scale = Number(wT5Mult?.value ?? 1.0);
                const posText = String(wPos?.value || "");
                const hasSlop = /score_\d|masterpiece|best quality/i.test(posText);

                ctx.font = "bold 9px monospace";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";

                const badgeY = rect.y + 10;
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
                    ctx.fillStyle = HADES_THEME.purple;
                    ctx.fillText("HEADER: ye-pop + deviantart", rect.x + rect.w - 8, badgeY);
                } else {
                    ctx.fillStyle = HADES_THEME.cyan;
                    ctx.fillText(`T5 QUERY EMPHASIS: ${t5Scale.toFixed(2)}x`, rect.x + rect.w - 8, badgeY);
                }

                // T5 Query Weight Gauge Bar
                const barX = rect.x + 8;
                const barY = rect.y + 21;
                const barW = rect.w - 16;
                const barH = 8;
                drawRoundedRect(ctx, barX, barY, barW, barH, 3);
                ctx.fillStyle = "#181c24";
                ctx.fill();

                const fillNorm = Math.max(0, Math.min(1, t5Scale / 2.5));
                if (fillNorm > 0) {
                    drawRoundedRect(ctx, barX, barY, barW * fillNorm, barH, 3);
                    ctx.fillStyle = isAesthetic ? HADES_THEME.emerald : HADES_THEME.cyan;
                    ctx.fill();
                }
            },
            actions: [
                {
                    label: "Aesthetic",
                    onClick: () =>
                        setWidgetValue(wVariant, "Anima Aesthetic (Strip score_* & Slop Guard)", node),
                },
                {
                    label: "2.9B/3.8B",
                    onClick: () =>
                        setWidgetValue(wVariant, "Anima 2.9B / 3.8B Expanded (Keep Quality Tags)", node),
                },
                {
                    label: "Base 2B",
                    onClick: () => setWidgetValue(wVariant, "Anima Base 2B (Keep Quality Tags)", node),
                },
                {
                    label: "Vector",
                    onClick: () =>
                        setWidgetValue(wStyle, "Flat Vector / Graphic (Non-Anime Dataset Header)", node),
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wVariant, {
            tag: "VARIANT",
            tagColor: HADES_THEME.emerald,
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wStyle, {
            tag: "STYLE",
            tagColor: HADES_THEME.purple,
        })
    );

    node.addCustomWidget(
        createStyledToggleWidget(node, wSanitize, {
            tag: "BOORU",
            color: HADES_THEME.cyan,
            label: "Booru Grammar & Kaomoji Guard",
            onText: "SANITIZE",
            offText: "RAW",
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wT5Mult, {
            tag: "T5 WGT",
            rangeTag: "T5 MULT",
            label: "T5 Query Emphasis Scale",
            color: HADES_THEME.amber,
            min: 0.0,
            max: 2.5,
            visMin: 0.0,
            visMax: 2.5,
            step: 0.05,
            defaultVal: 1.0,
        })
    );

    if (wChar) {
        node.addCustomWidget(
            createStyledTextRowWidget(node, wChar, {
                tag: "CHAR",
                color: HADES_THEME.cyan,
                promptTitle: "Enter Character Tag(s) (e.g. frieren, sousou no frieren)",
                emptyPlaceholder: "— Optional character & series injection —",
                allowClearOnRightClick: true,
            })
        );
    }

    node.addCustomWidget(
        createStyledTextRowWidget(node, wArtist, {
            tag: "ARTIST",
            color: HADES_THEME.blue,
            promptTitle: "Enter Artist Tag(s) (e.g. @zocehuy, (@zocehuy:1.2), wlop)",
            emptyPlaceholder: "— Optional @artist / style injection —",
            allowClearOnRightClick: true,
        })
    );

    // Keep multiline prompt textboxes at the bottom of the node
    if (Array.isArray(node.widgets) && (wPos || wNeg)) {
        const tail = [wPos, wNeg].filter(Boolean);
        node.widgets = node.widgets.filter((w) => !tail.includes(w)).concat(tail);
    }

    const sz = node.computeSize();
    node.setSize([Math.max(440, sz[0]), Math.max(418, sz[1] + 12)]);
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
            title: "NAG ATTENTION WINDOW & L1 GUARD",
            accentColor: HADES_THEME.amber,
            customHeight: 66,
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
                ctx.fillText(`WINDOW: ${Math.round(startP * 100)}% → ${Math.round(endP * 100)}%`, trackX, rect.y + 9);

                ctx.textAlign = "right";
                ctx.fillStyle = HADES_THEME.cyan;
                ctx.fillText(`SCALE ${scale.toFixed(1)}x | L1 TAU ≤ ${tau.toFixed(1)}`, trackX + trackW, rect.y + 9);
            },
            actions: [
                {
                    label: "Turbo CFG=1",
                    onClick: () =>
                        applyPresetValues("Turbo / Lightning Negative Enabler", 3.5, 0.45, 2.5, 0.0, 0.7),
                },
                {
                    label: "Anti-Burn",
                    onClick: () => applyPresetValues("Aesthetic Crispness (Anti-Burn)", 2.2, 0.3, 2.0, 0.0, 0.6),
                },
                {
                    label: "High",
                    onClick: () => applyPresetValues("High Adherence Extrapolation", 4.5, 0.5, 3.0, 0.0, 0.75),
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "PRESET",
            tagColor: HADES_THEME.amber,
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

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "16CH WAN21 LATENT & 32PX GRID",
            accentColor: HADES_THEME.cyan,
            customHeight: 68,
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

                // Draw aspect-ratio preview box on the left
                const boxAreaW = 48;
                const maxBoxH = rect.h - 8;
                const maxBoxW = 40;
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

                ctx.font = "bold 10px monospace";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(`${w} × ${h} px  (${mp.toFixed(2)} MP)`, rect.x + boxAreaW + 8, rect.y + 11);

                ctx.font = "9px monospace";
                ctx.fillStyle = isOptimal ? HADES_THEME.emerald : HADES_THEME.amber;
                ctx.fillText(
                    isOptimal ? "16CH WAN21 | 32PX ALIGNED | OPTIMAL MP" : "OUTSIDE 0.5–1.5 MP OPTIMAL WINDOW",
                    rect.x + boxAreaW + 8,
                    rect.y + 25
                );
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

    node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "ASPECT",
            tagColor: HADES_THEME.cyan,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wWidth, {
            tag: "WIDTH",
            rangeTag: "CUSTOM W",
            label: "Custom Width (32px Grid)",
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

    node.addCustomWidget(
        createSpectralSliderWidget(node, wHeight, {
            tag: "HEIGHT",
            rangeTag: "CUSTOM H",
            label: "Custom Height (32px Grid)",
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

    node.addCustomWidget(
        createSpectralSliderWidget(node, wBatch, {
            tag: "BATCH",
            rangeTag: "COUNT",
            label: "Latent Batch Size",
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

    if (wGuard29b) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wGuard29b, {
                tag: "2.9B",
                color: HADES_THEME.emerald,
                label: "Prevent 1280x1280 White-BG Bias",
                onText: "GUARD ON",
                offText: "ALLOW 1280",
            })
        );
    }

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
            title: "24-RULE DIFFUSERS -> COMFY REMAPPER",
            accentColor: HADES_THEME.purple,
            customHeight: 58,
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
                    onClick: () => {
                        setWidgetValue(wStrM, 1.0, node);
                        if (wStrC) setWidgetValue(wStrC, 0.5, node);
                    },
                },
                {
                    label: "Safe Clean",
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
        })
    );

    node.addCustomWidget(
        createStyledToggleWidget(node, wClamp, {
            tag: "SCRUB",
            color: HADES_THEME.amber,
            label: "Scrub NaN/Inf & Clamp >10σ Outliers",
            onText: "ACTIVE",
            offText: "OFF",
        })
    );

    node.addCustomWidget(
        createStyledToggleWidget(node, wSave, {
            tag: "EXPORT",
            color: HADES_THEME.cyan,
            label: "Save Converted Copy to models/loras/",
            onText: "SAVE COPY",
            offText: "IN-MEMORY",
        })
    );

    node.addCustomWidget(
        createStyledTextRowWidget(node, wOutFile, {
            tag: "FILE",
            color: HADES_THEME.emerald,
            promptTitle: "Output Filename for Converted LoRA",
            suffix: ".safetensors",
            emptyPlaceholder: "Anima_ComfyNative_LoRA",
        })
    );

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

export function setupCornerScribbleShieldUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wBox = findW("corner_box_percent");
    const wSens = findW("detection_sensitivity");
    const wCorners = findW("active_corners");
    const wHeal = findW("heal_mode");
    const wFeather = findW("feather_radius");

    [wBox, wSens, wCorners, wHeal, wFeather].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "CORNER SIGNATURE SCRIBBLE SHIELD",
            accentColor: HADES_THEME.rose,
            customHeight: 66,
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
                ctx.fillText(`INSPECTION BOX: ${boxPct}% PERIMETER`, fx + fw + 10, rect.y + 11);

                ctx.fillStyle = mode.includes("Mask Only") ? HADES_THEME.cyan : HADES_THEME.emerald;
                ctx.fillText(`ACTION: ${mode.toUpperCase()}`, fx + fw + 10, rect.y + 24);
            },
            actions: [
                {
                    label: "All 4",
                    onClick: () => setWidgetValue(wCorners, "All 4 Corners", node),
                },
                {
                    label: "Bottom",
                    onClick: () => setWidgetValue(wCorners, "Bottom Corners Only (Left & Right)", node),
                },
                {
                    label: "Auto-Heal",
                    onClick: () => setWidgetValue(wHeal, "Smart Edge-Inward Blend (Auto-Heal)", node),
                },
                {
                    label: "Forced",
                    onClick: () => setWidgetValue(wHeal, "Aggressive Corner Fill (Forced)", node),
                },
                {
                    label: "Mask Only",
                    onClick: () => setWidgetValue(wHeal, "Mask Only (No Pixel Modification)", node),
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wCorners, {
            tag: "CORNERS",
            tagColor: HADES_THEME.rose,
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wHeal, {
            tag: "MODE",
            tagColor: HADES_THEME.emerald,
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
        })
    );

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

export const PIPELINE_NODE_DISPATCH = {
    HaDeSAnimaPromptStudio: setupPromptStudioUI,
    HaDeSAnimaAttentionGuidance: setupAttentionGuidanceUI,
    HaDeSAnimaLatentPipelineGuard: setupLatentPipelineGuardUI,
    HaDeSAnimaUniversalLoraLoader: setupUniversalLoraLoaderUI,
    HaDeSAnimaCornerScribbleShield: setupCornerScribbleShieldUI,
};
