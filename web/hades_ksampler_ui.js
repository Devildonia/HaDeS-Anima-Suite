import {
    HADES_THEME,
    setWidgetValue,
    drawRoundedRect,
    hideBackingWidget,
    applyNodeFrameStyle,
    createSpectralBannerWidget,
    createSpectralSliderWidget,
    createStyledComboWidget,
} from "./hades_widgets.js";

// =============================================================================
// HADES ANIMA KSAMPLER TRILOGY UI (SINGLE, DUAL-PASS & TRIPLE-PASS)
// =============================================================================

const SINGLE_PROFILES = {
    "Anima Base / Raehoshi (er_sde | 32 steps | CFG 4.5 | Shift 3.0)": [32, 4.5, "er_sde", "normal", 1.0, 3.0],
    "Anima Aesthetic Crisp (er_sde | 30 steps | CFG 4.0 | beta | Shift 3.0)": [30, 4.0, "er_sde", "beta", 1.0, 3.0],
    "Anima Aesthetic Painterly (euler_a | 35 steps | CFG 4.2 | anima_beta57 | Shift 2.8)": [
        35, 4.2, "euler_ancestral", "anima_beta57", 1.0, 2.8,
    ],
    "Anima Turbo (euler | 10 steps | CFG 1.0 | Shift 3.0)": [10, 1.0, "euler", "normal", 1.0, 3.0],
    "Anima Lightning TDM (euler | 4 steps | CFG 1.0 | anima_lightning_tdm)": [
        4, 1.0, "euler", "anima_lightning_tdm", 1.0, 3.0,
    ],
    "Anima 2.9B / 3.8B Deep DiT (er_sde | 36 steps | CFG 4.5 | Shift 3.0)": [36, 4.5, "er_sde", "normal", 1.0, 3.0],
};

const DUAL_PROFILES = {
    "HiRes Standard 1.5x (er_sde 30s -> er_sde + beta57 16s @ 0.42 denoise)": [
        30, 4.5, "er_sde", "normal", 1.0, 3.0,
        "Latent 16ch Variance-Preserving Bicubic", 1.5,
        16, 4.0, "er_sde", "anima_beta57", 0.42, 3.6,
    ],
    "Aesthetic Painterly 1.5x (er_sde 32s -> euler_a + beta57 18s @ 0.45 denoise)": [
        32, 4.2, "er_sde", "beta", 1.0, 3.0,
        "Latent 16ch Variance-Preserving Bicubic", 1.5,
        18, 3.8, "euler_ancestral", "anima_beta57", 0.45, 3.5,
    ],
    "Base-to-Turbo Fast HiRes 1.5x (er_sde 30s CFG 4.5 -> euler 8s CFG 1.0 @ 0.38 denoise)": [
        30, 4.5, "er_sde", "normal", 1.0, 3.0,
        "Latent 16ch Variance-Preserving Bicubic", 1.5,
        8, 1.0, "euler", "normal", 0.38, 3.5,
    ],
    "Same-Res Texture & Lineart Refiner 1.0x (30s -> 12s @ 0.28 denoise)": [
        30, 4.5, "er_sde", "normal", 1.0, 3.0,
        "Direct Latent Pass-Through (1.0x)", 1.0,
        12, 3.6, "euler", "anima_beta57", 0.28, 3.2,
    ],
};

const TRIPLE_PROFILES = {
    "Progressive Staircase 1.0x -> 1.35x -> 1.75x (Ultra-Res Poster Pipeline)": [
        "Latent 16ch Variance-Preserving Bicubic",
        28, 4.5, "er_sde", "normal", 1.0, 3.0,
        1.35, 16, 4.0, "er_sde", "beta", 0.48, 3.5,
        1.75, 12, 3.4, "euler", "anima_beta57", 0.26, 4.0,
    ],
    "Progressive 2x Master 1.0x -> 1.5x -> 2.0x (4 MP Studio Illustration)": [
        "Latent 16ch Variance-Preserving Bicubic",
        30, 4.5, "er_sde", "normal", 1.0, 3.0,
        1.5, 18, 4.0, "er_sde", "beta", 0.46, 3.6,
        2.0, 14, 3.5, "euler", "anima_beta57", 0.28, 4.2,
    ],
    "Concept -> Aesthetic Bridge 1.5x -> Micro-Polish 1.5x (Lock Res at Stage 2)": [
        "Latent 16ch Variance-Preserving Bicubic",
        28, 4.5, "er_sde", "normal", 1.0, 3.0,
        1.5, 16, 4.0, "er_sde", "beta", 0.45, 3.6,
        1.5, 10, 3.2, "euler", "anima_beta57", 0.22, 3.6,
    ],
    "Hybrid Base -> Aesthetic -> Turbo Polish (Fast 3-Stage Studio)": [
        "Latent 16ch Variance-Preserving Bicubic",
        28, 4.5, "er_sde", "normal", 1.0, 3.0,
        1.4, 14, 4.0, "er_sde", "beta", 0.46, 3.5,
        1.75, 6, 1.0, "euler", "normal", 0.28, 3.9,
    ],
};

function bindPresetSync(node, wPreset, presetTable, orderedWidgets) {
    let isApplying = false;
    const applyProfile = (profileName) => {
        const vals = presetTable[profileName];
        if (!vals) return;
        isApplying = true;
        try {
            setWidgetValue(wPreset, profileName, node);
            orderedWidgets.forEach((w, idx) => {
                if (w && vals[idx] !== undefined) {
                    setWidgetValue(w, vals[idx], node);
                }
            });
        } finally {
            isApplying = false;
        }
    };

    if (wPreset) {
        const origCb = wPreset.callback;
        wPreset.callback = function (val) {
            if (typeof origCb === "function") origCb.call(this, val);
            if (!isApplying && presetTable[val]) {
                applyProfile(val);
            }
        };
    }

    orderedWidgets.forEach((w) => {
        if (!w) return;
        const origCb = w.callback;
        w.callback = function (val) {
            if (typeof origCb === "function") origCb.call(this, val);
            if (!isApplying && wPreset && wPreset.value !== "Manual (Use Widget Settings)") {
                wPreset.value = "Manual (Use Widget Settings)";
            }
        };
    });

    return applyProfile;
}

export function setupAnimaKSamplerUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wProfile = findW("variant_profile");
    const wSteps = findW("steps");
    const wCfg = findW("cfg");
    const wSampler = findW("sampler_name");
    const wSched = findW("scheduler");
    const wDenoise = findW("denoise");
    const wShift = findW("flow_shift");
    const wSeed = findW("seed");
    const wControl = findW("control_after_generate");

    const managed = [wProfile, wSteps, wCfg, wSampler, wSched, wDenoise, wShift];
    managed.forEach(hideBackingWidget);

    const applyProfile = bindPresetSync(node, wProfile, SINGLE_PROFILES, [
        wSteps, wCfg, wSampler, wSched, wDenoise, wShift,
    ]);

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.blue,
            customHeight: 72,
            drawCustomHUD(ctx, rect) {
                const shift = Number(wShift?.value ?? 3.0);
                const denoise = Math.max(0.05, Math.min(1.0, Number(wDenoise?.value ?? 1.0)));
                const steps = Number(wSteps?.value ?? 32);
                const cfg = Number(wCfg?.value ?? 4.5);
                const sampler = String(wSampler?.value ?? "er_sde");
                const sched = String(wSched?.value ?? "normal");

                const isNarrow = rect.w < 260;
                const plotW = isNarrow ? Math.max(48, Math.round(rect.w * 0.28)) : 76;
                const plotX = rect.x + 6;
                const plotY = rect.y + 4;
                const plotH = rect.h - 8;

                drawRoundedRect(ctx, plotX, plotY, plotW, plotH, 4);
                ctx.fillStyle = "#161920";
                ctx.fill();
                ctx.strokeStyle = "#262b36";
                ctx.stroke();

                // Draw shifted sigma(t) curve
                ctx.beginPath();
                const nPts = 20;
                for (let i = 0; i <= nPts; i++) {
                    const u = i / nPts;
                    const t = (1.0 - u) * denoise;
                    const sig = (shift * t) / (1.0 + (shift - 1.0) * t);
                    const px = plotX + 4 + u * (plotW - 8);
                    const py = plotY + plotH - 4 - sig * (plotH - 8);
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.strokeStyle = HADES_THEME.cyan;
                ctx.lineWidth = 1.6;
                ctx.stroke();

                const infoX = plotX + plotW + 8;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";

                ctx.font = isNarrow ? "9px monospace" : "bold 10px monospace";
                ctx.fillStyle = HADES_THEME.textPrimary;
                const line1 = isNarrow ? `${sampler.toUpperCase()} · ${sched}` : `${sampler.toUpperCase()}  |  ${sched}`;
                ctx.fillText(line1, infoX, rect.y + 11);

                ctx.font = isNarrow ? "8.5px monospace" : "9px monospace";
                ctx.fillStyle = HADES_THEME.emerald;
                const line2 = isNarrow
                    ? `${steps}s · C:${cfg.toFixed(1)} · S:${shift.toFixed(2)}`
                    : `STEPS: ${steps}  |  CFG: ${cfg.toFixed(1)}  |  SHIFT: ${shift.toFixed(2)}`;
                ctx.fillText(line2, infoX, rect.y + 25);
            },
            actions: [
                {
                    label: "Base/Rae",
                    onClick: () =>
                        applyProfile("Anima Base / Raehoshi (er_sde | 32 steps | CFG 4.5 | Shift 3.0)"),
                },
                {
                    label: "Aesthetic",
                    onClick: () =>
                        applyProfile("Anima Aesthetic Crisp (er_sde | 30 steps | CFG 4.0 | beta | Shift 3.0)"),
                },
                {
                    label: "Beta57",
                    onClick: () =>
                        applyProfile(
                            "Anima Aesthetic Painterly (euler_a | 35 steps | CFG 4.2 | anima_beta57 | Shift 2.8)"
                        ),
                },
                {
                    label: "Turbo",
                    onClick: () => applyProfile("Anima Turbo (euler | 10 steps | CFG 1.0 | Shift 3.0)"),
                },
                {
                    label: "TDM 4s",
                    onClick: () =>
                        applyProfile("Anima Lightning TDM (euler | 4 steps | CFG 1.0 | anima_lightning_tdm)"),
                },
            ],
        })
    );

    const K_TAG_WIDTH = 66;

    const profileWidget = node.addCustomWidget(
        createStyledComboWidget(node, wProfile, { tag: "PROFILE", tagColor: HADES_THEME.blue, tagWidth: K_TAG_WIDTH })
    );
    const samplerWidget = node.addCustomWidget(
        createStyledComboWidget(node, wSampler, { tag: "SAMPLER", tagColor: HADES_THEME.cyan, tagWidth: K_TAG_WIDTH })
    );
    const schedWidget = node.addCustomWidget(
        createStyledComboWidget(node, wSched, { tag: "SCHEDULER", tagColor: HADES_THEME.purple, tagWidth: K_TAG_WIDTH })
    );

    const stepsWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps, {
            tag: "STEPS",
            rangeTag: "STEPS",
            tagWidth: K_TAG_WIDTH,
            label: "",
            color: HADES_THEME.blue,
            min: 1,
            max: 200,
            visMin: 4,
            visMax: 60,
            step: 1,
            defaultVal: 32,
            isInt: true,
        })
    );
    const cfgWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg, {
            tag: "CFG",
            rangeTag: "CFG",
            tagWidth: K_TAG_WIDTH,
            label: "",
            color: HADES_THEME.emerald,
            min: 1.0,
            max: 20.0,
            visMin: 1.0,
            visMax: 8.0,
            step: 0.1,
            defaultVal: 4.5,
        })
    );
    const denoiseWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDenoise, {
            tag: "DENOISE",
            rangeTag: "DENOISE",
            tagWidth: K_TAG_WIDTH,
            label: "",
            color: HADES_THEME.cyan,
            min: 0.0,
            max: 1.0,
            visMin: 0.0,
            visMax: 1.0,
            step: 0.01,
            defaultVal: 1.0,
        })
    );
    const shiftWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift, {
            tag: "SHIFT",
            rangeTag: "SHIFT",
            tagWidth: K_TAG_WIDTH,
            label: "",
            color: HADES_THEME.amber,
            min: 0.5,
            max: 10.0,
            visMin: 1.0,
            visMax: 6.0,
            step: 0.05,
            defaultVal: 3.0,
        })
    );

    const seedWidgets = [];
    if (wSeed) seedWidgets.push(wSeed);
    if (wControl) seedWidgets.push(wControl);

    node.widgets = [
        bannerWidget,
        profileWidget,
        wProfile,
        ...seedWidgets,
        samplerWidget,
        schedWidget,
        stepsWidget,
        cfgWidget,
        denoiseWidget,
        shiftWidget,
        wSteps,
        wCfg,
        wSampler,
        wSched,
        wDenoise,
        wShift,
    ];

    function sanitizeSinglePassValues() {
        if (wProfile) {
            const val = String(wProfile.value ?? "");
            if (!SINGLE_PROFILES[val] && val !== "Manual (Use Widget Settings)") {
                wProfile.value = "Manual (Use Widget Settings)";
            }
        }
        if (wSeed) {
            const num = Number(wSeed.value);
            if (isNaN(num) || num < 0 || typeof wSeed.value !== "number") {
                wSeed.value = Math.floor(Math.random() * 1000000000000);
            }
        }
        if (wControl) {
            if (!["fixed", "increment", "decrement", "randomize"].includes(wControl.value)) {
                wControl.value = "randomize";
            }
        }
    }

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        sanitizeSinglePassValues();
        this.setDirtyCanvas(true, true);
    };
    sanitizeSinglePassValues();

    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (slotType, slot, isChangeConnect, link_info, input_info) {
        if (origOnConnectionsChange) {
            origOnConnectionsChange.apply(this, arguments);
        }
        const sz = this.computeSize();
        this.setSize([Math.max(440, sz[0]), sz[1] + 8]);
        this.setDirtyCanvas(true, true);
    };

    const sz = node.computeSize();
    node.setSize([Math.max(440, sz[0]), sz[1] + 8]);
}

export function setupAnimaDualPassKSamplerUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wPreset = findW("dual_preset");
    const wSteps1 = findW("steps_pass1");
    const wCfg1 = findW("cfg_pass1");
    const wSamp1 = findW("sampler_pass1");
    const wSched1 = findW("scheduler_pass1");
    const wDen1 = findW("denoise_pass1");
    const wShift1 = findW("shift_pass1");
    const wMethod = findW("upscale_method");
    const wScale = findW("upscale_factor");
    const wSteps2 = findW("steps_pass2");
    const wCfg2 = findW("cfg_pass2");
    const wSamp2 = findW("sampler_pass2");
    const wSched2 = findW("scheduler_pass2");
    const wDen2 = findW("denoise_pass2");
    const wShift2 = findW("shift_pass2");
    const wSeed = findW("seed");
    const wControl = findW("control_after_generate");

    const ordered = [
        wSteps1, wCfg1, wSamp1, wSched1, wDen1, wShift1,
        wMethod, wScale,
        wSteps2, wCfg2, wSamp2, wSched2, wDen2, wShift2,
    ];
    [wPreset, ...ordered].forEach(hideBackingWidget);

    const applyPreset = bindPresetSync(node, wPreset, DUAL_PROFILES, ordered);

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "2-PASS HIRES & HYBRID REFINER (16CH 32PX)",
            accentColor: HADES_THEME.emerald,
            customHeight: 70,
            drawCustomHUD(ctx, rect) {
                const s1 = Number(wSteps1?.value ?? 30);
                const c1 = Number(wCfg1?.value ?? 4.5);
                const scale = Number(wScale?.value ?? 1.5);
                const s2 = Number(wSteps2?.value ?? 16);
                const c2 = Number(wCfg2?.value ?? 4.0);
                const d2 = Number(wDen2?.value ?? 0.42);
                const sh2 = Number(wShift2?.value ?? 3.6);

                const padX = 8;
                const boxY = rect.y + 4;
                const boxH = 26;
                const totalW = rect.w - padX * 2;
                const boxW = (totalW - 56) * 0.5;

                // Pass 1 pill
                drawRoundedRect(ctx, rect.x + padX, boxY, boxW, boxH, 4);
                ctx.fillStyle = `${HADES_THEME.blue}22`;
                ctx.fill();
                ctx.strokeStyle = HADES_THEME.blue;
                ctx.stroke();

                ctx.font = "bold 9px monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(`PASS 1: 1.0x (${s1}s | CFG ${c1.toFixed(1)})`, rect.x + padX + boxW * 0.5, boxY + boxH * 0.5);

                // Bridge arrow badge
                const midX = rect.x + padX + boxW + 28;
                ctx.fillStyle = HADES_THEME.amber;
                ctx.fillText(`×${scale.toFixed(2)}`, midX, boxY + boxH * 0.5);

                // Pass 2 pill
                const p2X = rect.x + padX + boxW + 56;
                drawRoundedRect(ctx, p2X, boxY, boxW, boxH, 4);
                ctx.fillStyle = `${HADES_THEME.emerald}22`;
                ctx.fill();
                ctx.strokeStyle = HADES_THEME.emerald;
                ctx.stroke();

                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(
                    `PASS 2: ${s2}s | DEN ${d2.toFixed(2)} | S ${sh2.toFixed(1)}`,
                    p2X + boxW * 0.5,
                    boxY + boxH * 0.5
                );
            },
            actions: [
                {
                    label: "HiRes 1.5x",
                    onClick: () =>
                        applyPreset("HiRes Standard 1.5x (er_sde 30s -> er_sde + beta57 16s @ 0.42 denoise)"),
                },
                {
                    label: "Painterly",
                    onClick: () =>
                        applyPreset(
                            "Aesthetic Painterly 1.5x (er_sde 32s -> euler_a + beta57 18s @ 0.45 denoise)"
                        ),
                },
                {
                    label: "Base→Turbo",
                    onClick: () =>
                        applyPreset(
                            "Base-to-Turbo Fast HiRes 1.5x (er_sde 30s CFG 4.5 -> euler 8s CFG 1.0 @ 0.38 denoise)"
                        ),
                },
                {
                    label: "1.0x Polish",
                    onClick: () =>
                        applyPreset("Same-Res Texture & Lineart Refiner 1.0x (30s -> 12s @ 0.28 denoise)"),
                },
            ],
        })
    );

    const presetWidget = node.addCustomWidget(createStyledComboWidget(node, wPreset, { tag: "PRESET", tagColor: HADES_THEME.emerald }));

    // Pass 1 Controls
    const samp1Widget = node.addCustomWidget(createStyledComboWidget(node, wSamp1, { tag: "P1 SAMP", tagColor: HADES_THEME.blue }));
    const sched1Widget = node.addCustomWidget(createStyledComboWidget(node, wSched1, { tag: "P1 SCHD", tagColor: HADES_THEME.blue }));
    const steps1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps1, {
            tag: "P1 STEP", rangeTag: "PASS 1", label: "Pass 1 Steps",
            color: HADES_THEME.blue, min: 1, max: 200, visMin: 4, visMax: 50, step: 1, defaultVal: 30, isInt: true,
        })
    );
    const cfg1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg1, {
            tag: "P1 CFG", rangeTag: "PASS 1", label: "Pass 1 CFG",
            color: HADES_THEME.blue, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 4.5,
        })
    );
    const den1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen1, {
            tag: "P1 DEN", rangeTag: "PASS 1", label: "Pass 1 Denoise",
            color: HADES_THEME.blue, min: 0.0, max: 1.0, visMin: 0.0, visMax: 1.0, step: 0.01, defaultVal: 1.0,
        })
    );
    const shift1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift1, {
            tag: "P1 SHFT", rangeTag: "M=1.0", label: "Pass 1 Flow Shift",
            color: HADES_THEME.blue, min: 0.5, max: 10.0, visMin: 1.0, visMax: 6.0, step: 0.05, defaultVal: 3.0,
        })
    );

    // Bridge Controls
    const methodWidget = node.addCustomWidget(createStyledComboWidget(node, wMethod, { tag: "BRIDGE", tagColor: HADES_THEME.amber }));
    const scaleWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wScale, {
            tag: "SCALE", rangeTag: "32PX", label: "HiRes Scale Factor (32px Grid)",
            color: HADES_THEME.amber, min: 1.0, max: 2.5, visMin: 1.0, visMax: 2.0, step: 0.05, defaultVal: 1.5,
        })
    );

    // Pass 2 Controls
    const samp2Widget = node.addCustomWidget(createStyledComboWidget(node, wSamp2, { tag: "P2 SAMP", tagColor: HADES_THEME.emerald }));
    const sched2Widget = node.addCustomWidget(createStyledComboWidget(node, wSched2, { tag: "P2 SCHD", tagColor: HADES_THEME.emerald }));
    const steps2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps2, {
            tag: "P2 STEP", rangeTag: "PASS 2", label: "Pass 2 Refiner Steps",
            color: HADES_THEME.emerald, min: 1, max: 200, visMin: 4, visMax: 40, step: 1, defaultVal: 16, isInt: true,
        })
    );
    const cfg2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg2, {
            tag: "P2 CFG", rangeTag: "PASS 2", label: "Pass 2 Refiner CFG",
            color: HADES_THEME.emerald, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 4.0,
        })
    );
    const den2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen2, {
            tag: "P2 DEN", rangeTag: "PASS 2", label: "Pass 2 Refiner Denoise",
            color: HADES_THEME.emerald, min: 0.05, max: 1.0, visMin: 0.1, visMax: 0.8, step: 0.01, defaultVal: 0.42,
        })
    );
    const shift2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift2, {
            tag: "P2 SHFT", rangeTag: "M=1.0", label: "Pass 2 High-Res Flow Shift",
            color: HADES_THEME.emerald, min: 0.5, max: 10.0, visMin: 1.0, visMax: 6.0, step: 0.05, defaultVal: 3.6,
        })
    );

    const seedWidgets = [];
    if (wSeed) seedWidgets.push(wSeed);
    if (wControl) seedWidgets.push(wControl);

    node.widgets = [
        bannerWidget,
        presetWidget,
        wPreset,
        ...seedWidgets,
        samp1Widget,
        sched1Widget,
        steps1Widget,
        cfg1Widget,
        den1Widget,
        shift1Widget,
        methodWidget,
        scaleWidget,
        samp2Widget,
        sched2Widget,
        steps2Widget,
        cfg2Widget,
        den2Widget,
        shift2Widget,
        ...ordered,
    ];

    function sanitizeDualPassValues() {
        if (wPreset) {
            const val = String(wPreset.value ?? "");
            if (!DUAL_PROFILES[val] && val !== "Manual (Use Widget Settings)") {
                wPreset.value = "Manual (Use Widget Settings)";
            }
        }
        if (wSeed) {
            const num = Number(wSeed.value);
            if (isNaN(num) || num < 0 || typeof wSeed.value !== "number") {
                wSeed.value = Math.floor(Math.random() * 1000000000000);
            }
        }
        if (wControl) {
            if (!["fixed", "increment", "decrement", "randomize"].includes(wControl.value)) {
                wControl.value = "randomize";
            }
        }
    }

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        sanitizeDualPassValues();
        this.setDirtyCanvas(true, true);
    };
    sanitizeDualPassValues();

    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (slotType, slot, isChangeConnect, link_info, input_info) {
        if (origOnConnectionsChange) {
            origOnConnectionsChange.apply(this, arguments);
        }
        const sz = this.computeSize();
        this.setSize([Math.max(460, sz[0]), sz[1] + 8]);
        this.setDirtyCanvas(true, true);
    };

    const sz = node.computeSize();
    node.setSize([Math.max(460, sz[0]), sz[1] + 8]);
}

export function setupAnimaTriplePassKSamplerUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wPreset = findW("triple_preset");
    const wMethod = findW("upscale_method");

    const wSteps1 = findW("steps_stage1");
    const wCfg1 = findW("cfg_stage1");
    const wSamp1 = findW("sampler_stage1");
    const wSched1 = findW("scheduler_stage1");
    const wDen1 = findW("denoise_stage1");
    const wShift1 = findW("shift_stage1");

    const wScale2 = findW("scale_stage2");
    const wSteps2 = findW("steps_stage2");
    const wCfg2 = findW("cfg_stage2");
    const wSamp2 = findW("sampler_stage2");
    const wSched2 = findW("scheduler_stage2");
    const wDen2 = findW("denoise_stage2");
    const wShift2 = findW("shift_stage2");

    const wScale3 = findW("scale_stage3");
    const wSteps3 = findW("steps_stage3");
    const wCfg3 = findW("cfg_stage3");
    const wSamp3 = findW("sampler_stage3");
    const wSched3 = findW("scheduler_stage3");
    const wDen3 = findW("denoise_stage3");
    const wShift3 = findW("shift_stage3");
    const wSeed = findW("seed");
    const wControl = findW("control_after_generate");

    const ordered = [
        wMethod,
        wSteps1, wCfg1, wSamp1, wSched1, wDen1, wShift1,
        wScale2, wSteps2, wCfg2, wSamp2, wSched2, wDen2, wShift2,
        wScale3, wSteps3, wCfg3, wSamp3, wSched3, wDen3, wShift3,
    ];
    [wPreset, ...ordered].forEach(hideBackingWidget);

    const applyPreset = bindPresetSync(node, wPreset, TRIPLE_PROFILES, ordered);

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "3-STAGE PROGRESSIVE STUDIO PIPELINE",
            accentColor: HADES_THEME.purple,
            customHeight: 72,
            drawCustomHUD(ctx, rect) {
                const sc2 = Number(wScale2?.value ?? 1.35);
                const sc3 = Number(wScale3?.value ?? 1.75);
                const d1 = Number(wDen1?.value ?? 1.0);
                const d2 = Number(wDen2?.value ?? 0.48);
                const d3 = Number(wDen3?.value ?? 0.26);
                const s1 = Number(wSteps1?.value ?? 28);
                const s2 = Number(wSteps2?.value ?? 16);
                const s3 = Number(wSteps3?.value ?? 12);

                const padX = 8;
                const gap = 6;
                const cardW = (rect.w - padX * 2 - gap * 2) / 3;
                const cardY = rect.y + 4;
                const cardH = 28;

                const stages = [
                    { label: `S1 1.00x (${s1}s)`, sub: `DEN ${d1.toFixed(2)}`, col: HADES_THEME.blue },
                    { label: `S2 ${sc2.toFixed(2)}x (${s2}s)`, sub: `DEN ${d2.toFixed(2)}`, col: HADES_THEME.cyan },
                    { label: `S3 ${sc3.toFixed(2)}x (${s3}s)`, sub: `DEN ${d3.toFixed(2)}`, col: HADES_THEME.purple },
                ];

                stages.forEach((st, i) => {
                    const cx = rect.x + padX + i * (cardW + gap);
                    drawRoundedRect(ctx, cx, cardY, cardW, cardH, 4);
                    ctx.fillStyle = `${st.col}22`;
                    ctx.fill();
                    ctx.strokeStyle = st.col;
                    ctx.stroke();

                    ctx.font = "bold 9px monospace";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillStyle = HADES_THEME.textPrimary;
                    ctx.fillText(`${st.label} | ${st.sub}`, cx + cardW * 0.5, cardY + cardH * 0.5);
                });
            },
            actions: [
                {
                    label: "1.35→1.75x",
                    onClick: () =>
                        applyPreset("Progressive Staircase 1.0x -> 1.35x -> 1.75x (Ultra-Res Poster Pipeline)"),
                },
                {
                    label: "1.5→2.0x",
                    onClick: () =>
                        applyPreset("Progressive 2x Master 1.0x -> 1.5x -> 2.0x (4 MP Studio Illustration)"),
                },
                {
                    label: "1.5x Lock",
                    onClick: () =>
                        applyPreset("Concept -> Aesthetic Bridge 1.5x -> Micro-Polish 1.5x (Lock Res at Stage 2)"),
                },
                {
                    label: "Turbo S3",
                    onClick: () =>
                        applyPreset("Hybrid Base -> Aesthetic -> Turbo Polish (Fast 3-Stage Studio)"),
                },
            ],
        })
    );

    const presetWidget = node.addCustomWidget(createStyledComboWidget(node, wPreset, { tag: "PRESET", tagColor: HADES_THEME.purple }));
    const methodWidget = node.addCustomWidget(createStyledComboWidget(node, wMethod, { tag: "BRIDGE", tagColor: HADES_THEME.amber }));

    // Stage 1
    const samp1Widget = node.addCustomWidget(createStyledComboWidget(node, wSamp1, { tag: "S1 SAMP", tagColor: HADES_THEME.blue }));
    const sched1Widget = node.addCustomWidget(createStyledComboWidget(node, wSched1, { tag: "S1 SCHD", tagColor: HADES_THEME.blue }));
    const steps1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps1, {
            tag: "S1 STEP", rangeTag: "STAGE 1", label: "Stage 1 Draft Steps",
            color: HADES_THEME.blue, min: 1, max: 200, visMin: 4, visMax: 50, step: 1, defaultVal: 28, isInt: true,
        })
    );
    const cfg1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg1, {
            tag: "S1 CFG", rangeTag: "STAGE 1", label: "Stage 1 CFG",
            color: HADES_THEME.blue, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 4.5,
        })
    );
    const den1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen1, {
            tag: "S1 DEN", rangeTag: "STAGE 1", label: "Stage 1 Denoise",
            color: HADES_THEME.blue, min: 0.0, max: 1.0, visMin: 0.0, visMax: 1.0, step: 0.01, defaultVal: 1.0,
        })
    );
    const shift1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift1, {
            tag: "S1 SHFT", rangeTag: "M=1.0", label: "Stage 1 Flow Shift",
            color: HADES_THEME.blue, min: 0.5, max: 10.0, visMin: 1.0, visMax: 6.0, step: 0.05, defaultVal: 3.0,
        })
    );

    // Stage 2
    const scale2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wScale2, {
            tag: "S2 SCAL", rangeTag: "32PX", label: "Stage 2 Scale (vs Base)",
            color: HADES_THEME.cyan, min: 1.0, max: 2.5, visMin: 1.0, visMax: 2.0, step: 0.05, defaultVal: 1.35,
        })
    );
    const samp2Widget = node.addCustomWidget(createStyledComboWidget(node, wSamp2, { tag: "S2 SAMP", tagColor: HADES_THEME.cyan }));
    const sched2Widget = node.addCustomWidget(createStyledComboWidget(node, wSched2, { tag: "S2 SCHD", tagColor: HADES_THEME.cyan }));
    const steps2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps2, {
            tag: "S2 STEP", rangeTag: "STAGE 2", label: "Stage 2 Bridge Steps",
            color: HADES_THEME.cyan, min: 1, max: 200, visMin: 4, visMax: 40, step: 1, defaultVal: 16, isInt: true,
        })
    );
    const cfg2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg2, {
            tag: "S2 CFG", rangeTag: "STAGE 2", label: "Stage 2 Bridge CFG",
            color: HADES_THEME.cyan, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 4.0,
        })
    );
    const den2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen2, {
            tag: "S2 DEN", rangeTag: "STAGE 2", label: "Stage 2 Bridge Denoise",
            color: HADES_THEME.cyan, min: 0.05, max: 1.0, visMin: 0.1, visMax: 0.8, step: 0.01, defaultVal: 0.48,
        })
    );
    const shift2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift2, {
            tag: "S2 SHFT", rangeTag: "M=1.0", label: "Stage 2 Flow Shift",
            color: HADES_THEME.cyan, min: 0.5, max: 10.0, visMin: 1.0, visMax: 6.0, step: 0.05, defaultVal: 3.5,
        })
    );

    // Stage 3
    const scale3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wScale3, {
            tag: "S3 SCAL", rangeTag: "32PX", label: "Stage 3 Final Scale (vs Base)",
            color: HADES_THEME.purple, min: 1.0, max: 3.0, visMin: 1.0, visMax: 2.5, step: 0.05, defaultVal: 1.75,
        })
    );
    const samp3Widget = node.addCustomWidget(createStyledComboWidget(node, wSamp3, { tag: "S3 SAMP", tagColor: HADES_THEME.purple }));
    const sched3Widget = node.addCustomWidget(createStyledComboWidget(node, wSched3, { tag: "S3 SCHD", tagColor: HADES_THEME.purple }));
    const steps3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps3, {
            tag: "S3 STEP", rangeTag: "STAGE 3", label: "Stage 3 Polish Steps",
            color: HADES_THEME.purple, min: 1, max: 200, visMin: 4, visMax: 30, step: 1, defaultVal: 12, isInt: true,
        })
    );
    const cfg3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg3, {
            tag: "S3 CFG", rangeTag: "STAGE 3", label: "Stage 3 Polish CFG",
            color: HADES_THEME.purple, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 3.4,
        })
    );
    const den3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen3, {
            tag: "S3 DEN", rangeTag: "STAGE 3", label: "Stage 3 Polish Denoise",
            color: HADES_THEME.purple, min: 0.05, max: 1.0, visMin: 0.1, visMax: 0.6, step: 0.01, defaultVal: 0.26,
        })
    );
    const shift3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift3, {
            tag: "S3 SHFT", rangeTag: "M=1.0", label: "Stage 3 Flow Shift",
            color: HADES_THEME.purple, min: 0.5, max: 10.0, visMin: 1.0, visMax: 6.0, step: 0.05, defaultVal: 4.0,
        })
    );

    const seedWidgets = [];
    if (wSeed) seedWidgets.push(wSeed);
    if (wControl) seedWidgets.push(wControl);

    node.widgets = [
        bannerWidget,
        presetWidget,
        wPreset,
        ...seedWidgets,
        methodWidget,
        samp1Widget,
        sched1Widget,
        steps1Widget,
        cfg1Widget,
        den1Widget,
        shift1Widget,
        scale2Widget,
        samp2Widget,
        sched2Widget,
        steps2Widget,
        cfg2Widget,
        den2Widget,
        shift2Widget,
        scale3Widget,
        samp3Widget,
        sched3Widget,
        steps3Widget,
        cfg3Widget,
        den3Widget,
        shift3Widget,
        ...ordered,
    ];

    function sanitizeTriplePassValues() {
        if (wPreset) {
            const val = String(wPreset.value ?? "");
            if (!TRIPLE_PROFILES[val] && val !== "Manual (Use Widget Settings)") {
                wPreset.value = "Manual (Use Widget Settings)";
            }
        }
        if (wSeed) {
            const num = Number(wSeed.value);
            if (isNaN(num) || num < 0 || typeof wSeed.value !== "number") {
                wSeed.value = Math.floor(Math.random() * 1000000000000);
            }
        }
        if (wControl) {
            if (!["fixed", "increment", "decrement", "randomize"].includes(wControl.value)) {
                wControl.value = "randomize";
            }
        }
    }

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        sanitizeTriplePassValues();
        this.setDirtyCanvas(true, true);
    };
    sanitizeTriplePassValues();

    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (slotType, slot, isChangeConnect, link_info, input_info) {
        if (origOnConnectionsChange) {
            origOnConnectionsChange.apply(this, arguments);
        }
        const sz = this.computeSize();
        this.setSize([Math.max(470, sz[0]), sz[1] + 8]);
        this.setDirtyCanvas(true, true);
    };

    const sz = node.computeSize();
    node.setSize([Math.max(470, sz[0]), sz[1] + 8]);
}

export const KSAMPLER_NODE_DISPATCH = {
    HaDeSAnimaKSampler: setupAnimaKSamplerUI,
    HaDeSAnimaDualPassKSampler: setupAnimaDualPassKSamplerUI,
    HaDeSAnimaTriplePassKSampler: setupAnimaTriplePassKSamplerUI,
};
