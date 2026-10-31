import {
    HADES_THEME,
    setWidgetValue,
    drawRoundedRect,
    hideBackingWidget,
    applyNodeFrameStyle,
    createSpectralBannerWidget,
    createSpectralSliderWidget,
    createStyledComboWidget,
    createStyledTextRowWidget,
    finalizeHadesNode,
} from "./hades_widgets.js";

// =============================================================================
// HADES ANIMA KSAMPLER TRILOGY UI (SINGLE, DUAL-PASS & TRIPLE-PASS)
// =============================================================================

const SINGLE_PROFILES = {
    "Anima Base / Raehoshi": [32, 4.5, "er_sde", "normal", 1.0, 0.0],
    "Anima Aesthetic Crisp": [30, 4.0, "er_sde", "beta", 1.0, 0.0],
    "Anima Aesthetic Painterly": [35, 4.2, "euler_ancestral", "anima_beta57", 1.0, 2.8],
    "Anima Turbo": [10, 1.0, "euler", "normal", 1.0, 0.0],
    "Anima Lightning TDM": [4, 1.0, "euler", "anima_lightning_tdm", 1.0, 0.0],
    "Anima 2.9B / 3.8B Deep DiT": [36, 4.5, "er_sde", "normal", 1.0, 0.0],
};

const LEGACY_SINGLE_PROFILE_MAP = {
    "Manual (Use Widget Settings)": "Manual",
    "Anima Base / Raehoshi (er_sde | 32 steps | CFG 4.5 | Shift 3.0)": "Anima Base / Raehoshi",
    "Anima Aesthetic Crisp (er_sde | 30 steps | CFG 4.0 | beta | Shift 3.0)": "Anima Aesthetic Crisp",
    "Anima Aesthetic Painterly (euler_a | 35 steps | CFG 4.2 | anima_beta57 | Shift 2.8)": "Anima Aesthetic Painterly",
    "Anima Turbo (euler | 10 steps | CFG 1.0 | Shift 3.0)": "Anima Turbo",
    "Anima Lightning TDM (euler | 4 steps | CFG 1.0 | anima_lightning_tdm)": "Anima Lightning TDM",
    "Anima 2.9B / 3.8B Deep DiT (er_sde | 36 steps | CFG 4.5 | Shift 3.0)": "Anima 2.9B / 3.8B Deep DiT",
};

const DUAL_PROFILES = {
    "HiRes Standard 1.5x (er_sde 30s -> er_sde + beta57 16s @ 0.42 denoise)": [
        30, 4.5, "er_sde", "normal", 1.0, 0.0,
        "Latent 16ch Variance-Preserving Bicubic", 1.5,
        16, 4.0, "er_sde", "anima_beta57", 0.42, 3.6,
    ],
    "Aesthetic Painterly 1.5x (er_sde 32s -> euler_a + beta57 18s @ 0.45 denoise)": [
        32, 4.2, "er_sde", "beta", 1.0, 0.0,
        "Latent 16ch Variance-Preserving Bicubic", 1.5,
        18, 3.8, "euler_ancestral", "anima_beta57", 0.45, 3.5,
    ],
    "Base-to-Turbo Fast HiRes 1.5x (er_sde 30s CFG 4.5 -> euler 8s CFG 1.0 @ 0.38 denoise)": [
        30, 4.5, "er_sde", "normal", 1.0, 0.0,
        "Latent 16ch Variance-Preserving Bicubic", 1.5,
        8, 1.0, "euler", "normal", 0.38, 3.5,
    ],
    "Same-Res Texture & Lineart Refiner 1.0x (30s -> 12s @ 0.28 denoise)": [
        30, 4.5, "er_sde", "normal", 1.0, 0.0,
        "Direct Latent Pass-Through (1.0x)", 1.0,
        12, 3.6, "euler", "anima_beta57", 0.28, 3.2,
    ],
};

const TRIPLE_PROFILES = {
    "Progressive Staircase 1.0x -> 1.35x -> 1.75x (Ultra-Res Poster Pipeline)": [
        "Latent 16ch Variance-Preserving Bicubic",
        28, 4.5, "er_sde", "normal", 1.0, 0.0,
        1.35, 16, 4.0, "er_sde", "beta", 0.48, 3.5,
        1.75, 12, 3.4, "euler", "anima_beta57", 0.26, 4.0,
    ],
    "Progressive 2x Master 1.0x -> 1.5x -> 2.0x (4 MP Studio Illustration)": [
        "Latent 16ch Variance-Preserving Bicubic",
        30, 4.5, "er_sde", "normal", 1.0, 0.0,
        1.5, 18, 4.0, "er_sde", "beta", 0.46, 3.6,
        2.0, 14, 3.5, "euler", "anima_beta57", 0.28, 4.2,
    ],
    "Concept -> Aesthetic Bridge 1.5x -> Micro-Polish 1.5x (Lock Res at Stage 2)": [
        "Latent 16ch Variance-Preserving Bicubic",
        28, 4.5, "er_sde", "normal", 1.0, 0.0,
        1.5, 16, 4.0, "er_sde", "beta", 0.45, 3.6,
        1.5, 10, 3.2, "euler", "anima_beta57", 0.22, 3.6,
    ],
    "Hybrid Base -> Aesthetic -> Turbo Polish (Fast 3-Stage Studio)": [
        "Latent 16ch Variance-Preserving Bicubic",
        28, 4.5, "er_sde", "normal", 1.0, 0.0,
        1.4, 14, 4.0, "er_sde", "beta", 0.46, 3.5,
        1.75, 6, 1.0, "euler", "normal", 0.28, 3.9,
    ],
};

// Seed + control_after_generate rendered as regular HaDeS rows. The native widgets stay in the list (hidden) so
// ComfyUI's control_after_generate logic and serialization keep working unchanged.
export function createSeedRows(node, wSeed, wControl) {
    const rows = [wSeed, wControl].filter(Boolean);
    rows.forEach(hideBackingWidget);
    if (wSeed) {
        rows.push(node.addCustomWidget(createStyledTextRowWidget(node, wSeed, {
            tag: "SEED",
            numeric: true,
            promptTitle: "Seed",
            tooltip: "<strong>Seed</strong><br>Click to type a seed. Later passes use seed + 1 / seed + 2.",
        })));
    }
    if (wControl) {
        rows.push(node.addCustomWidget(createStyledComboWidget(node, wControl, {
            tag: "AFTER RUN",
            tooltip: "<strong>Control after generate</strong><br>fixed, increment, decrement or randomize the seed after each queued run.",
        })));
    }
    return rows;
}

function bindPresetSync(node, wPreset, presetTable, orderedWidgets, manualLabel = "Manual (Use Widget Settings)") {
    let isApplying = false;
    const applyProfile = (profileName) => {
        const canonical = LEGACY_SINGLE_PROFILE_MAP[profileName] || profileName;
        const vals = presetTable[canonical] || presetTable[profileName];
        if (!vals) return;
        isApplying = true;
        try {
            setWidgetValue(wPreset, canonical, node);
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
            const canonical = LEGACY_SINGLE_PROFILE_MAP[val] || val;
            if (!isApplying && (presetTable[canonical] || presetTable[val])) {
                applyProfile(canonical);
            }
        };
    }

    orderedWidgets.forEach((w) => {
        if (!w) return;
        const origCb = w.callback;
        w.callback = function (val) {
            if (typeof origCb === "function") origCb.call(this, val);
            if (!isApplying && wPreset && wPreset.value !== manualLabel) {
                wPreset.value = manualLabel;
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

    const applyProfile = bindPresetSync(
        node,
        wProfile,
        SINGLE_PROFILES,
        [wSteps, wCfg, wSampler, wSched, wDenoise, wShift],
        "Manual"
    );

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.blue,
            customHeight: 72,
            drawCustomHUD(ctx, rect) {
                const shiftRaw = Number(wShift?.value ?? 0.0);
                const shift = shiftRaw > 0 ? shiftRaw : 3.0;
                const shiftText = shiftRaw > 0 ? shift.toFixed(2) : "INHERIT";
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

                ctx.font = isNarrow ? "9px ui-monospace, Consolas, monospace" : "600 10px ui-monospace, Consolas, monospace";
                ctx.fillStyle = HADES_THEME.textPrimary;
                const line1 = isNarrow ? `${sampler.toUpperCase()} · ${sched}` : `${sampler.toUpperCase()}  |  ${sched}`;
                ctx.fillText(line1, infoX, rect.y + 11);

                ctx.font = isNarrow ? "8.5px ui-monospace, Consolas, monospace" : "9px ui-monospace, Consolas, monospace";
                ctx.fillStyle = HADES_THEME.emerald;
                const line2 = isNarrow
                    ? `${steps}s · C:${cfg.toFixed(1)} · S:${shiftText}`
                    : `STEPS: ${steps}  |  CFG: ${cfg.toFixed(1)}  |  SHIFT: ${shiftText}`;
                ctx.fillText(line2, infoX, rect.y + 25);
            },
            actions: [
                {
                    label: "Base/Rae",
                    onClick: () => applyProfile("Anima Base / Raehoshi"),
                    isActive: () => wProfile?.value === "Anima Base / Raehoshi",
                },
                {
                    label: "Aesthetic",
                    onClick: () => applyProfile("Anima Aesthetic Crisp"),
                    isActive: () => wProfile?.value === "Anima Aesthetic Crisp",
                },
                {
                    label: "Beta57",
                    onClick: () => applyProfile("Anima Aesthetic Painterly"),
                    isActive: () => wProfile?.value === "Anima Aesthetic Painterly",
                },
                {
                    label: "Turbo",
                    onClick: () => applyProfile("Anima Turbo"),
                    isActive: () => wProfile?.value === "Anima Turbo",
                },
                {
                    label: "TDM 4s",
                    onClick: () => applyProfile("Anima Lightning TDM"),
                    isActive: () => wProfile?.value === "Anima Lightning TDM",
                },
            ],
        })
    );

    const K_TAG_WIDTH = 66;

    const profileWidget = node.addCustomWidget(
        createStyledComboWidget(node, wProfile, {
            tag: "PROFILE",
            tagColor: HADES_THEME.blue,
            tagWidth: K_TAG_WIDTH,
            tooltip: `<strong>Variant Calibration Profile</strong><br>Instantly applies the officially recommended Sampler, Scheduler, Steps, CFG, and Flow Shift for each Anima architecture variant.`,
        })
    );
    const samplerWidget = node.addCustomWidget(
        createStyledComboWidget(node, wSampler, {
            tag: "SAMPLER",
            tagColor: HADES_THEME.cyan,
            tagWidth: K_TAG_WIDTH,
            tooltip: `<strong>ODE / SDE Solver</strong><br>Numerical solver for Anima's continuous-time Rectified Flow equations.<br><span style="color:#8fb4f0">• er_sde:</span> Recommended stochastic SDE solver for Anima.<br><span style="color:#8fb4f0">• euler / euler_ancestral:</span> Fast deterministic or ancestral solvers.`,
        })
    );
    const schedWidget = node.addCustomWidget(
        createStyledComboWidget(node, wSched, {
            tag: "SCHED",
            tagColor: HADES_THEME.purple,
            tagWidth: K_TAG_WIDTH,
            tooltip: `<strong>Sigma Noise Schedule</strong><br>Defines noise distribution curve across steps.<br><span style="color:#8fb4f0">• normal:</span> Linear Rectified Flow schedule.<br><span style="color:#8fb4f0">• anima_beta57:</span> Custom schedule (α=0.50, β=0.70) giving denser steps to fine texture & lineart.<br><span style="color:#8fb4f0">• anima_lightning_tdm:</span> Exact 4-step Trajectory Distribution Matching schedule.`,
        })
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
            tooltip: `<strong>Sampling Steps</strong><br>Total iterative steps along the trajectory.<br><span style="color:#8fb4f0">• Base / Aesthetic:</span> 30–36 steps.<br><span style="color:#8fb4f0">• Turbo:</span> 8–10 steps.<br><span style="color:#8fb4f0">• Lightning TDM:</span> Exactly 4 steps.`,
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
            tooltip: `<strong>Classifier-Free Guidance (CFG Scale)</strong><br>Adherence strength to positive prompt vs unconditional trajectory.<br><span style="color:#8fb4f0">• Standard Anima:</span> 4.0 – 4.5.<br><span style="color:#8fb4f0">• Distilled (Turbo / Lightning):</span> Fixed at 1.0.`,
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
            tooltip: `<strong>Denoising Strength (0.0 – 1.0)</strong><br>Fraction of total noise trajectory to denoise.<br><span style="color:#8fb4f0">• 1.0:</span> Full txt2img generation from pure random noise.<br><span style="color:#8fb4f0">• 0.25 – 0.60:</span> img2img / HiRes latent refinement.`,
        })
    );
    const shiftWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift, {
            tag: "SHIFT",
            rangeTag: "SHIFT",
            tagWidth: K_TAG_WIDTH,
            label: "",
            color: HADES_THEME.amber,
            min: 0.0,
            max: 10.0,
            visMin: 0.0,
            visMax: 6.0,
            step: 0.05,
            defaultVal: 0.0,
            zeroLabel: "INHERIT",
            tooltip: `<strong>Anima Flow Shift (multiplier=1.0)</strong><br><span style="color:#8fb4f0">• 0 (INHERIT):</span> Keeps the shift set upstream (Model Loader flow_shift / ModelSampling nodes).<br>Time-step density warping for Anima's Rectified Flow architecture.<br><span style="color:#8fb4f0">• Calibrated default (3.0):</span> Native balance for optimal contrast & anatomy.<br><span style="color:#8fb4f0">• Higher (3.5–5.0):</span> Allocates more steps to early high-noise structural formation (ideal for high-res).<br><span style="color:#8fb4f0">• Lower (1.5–2.5):</span> Allocates more steps to late-stage fine detail refinement.`,
        })
    );

    const seedWidgets = createSeedRows(node, wSeed, wControl);

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
            let val = String(wProfile.value ?? "");
            if (LEGACY_SINGLE_PROFILE_MAP[val]) {
                val = LEGACY_SINGLE_PROFILE_MAP[val];
                wProfile.value = val;
            }
            if (!SINGLE_PROFILES[val] && val !== "Manual") {
                wProfile.value = "Manual";
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

    finalizeHadesNode(node, 440);
}

function migrateDualPassSlots(node) {
    if (!node) return;
    // Input migration: model_pass2_optional -> model_pass2
    if (Array.isArray(node.inputs)) {
        const oldIdx = node.inputs.findIndex((i) => i && i.name === "model_pass2_optional");
        if (oldIdx !== -1) {
            const newIdx = node.inputs.findIndex((i) => i && i.name === "model_pass2");
            const oldInput = node.inputs[oldIdx];
            if (newIdx !== -1 && oldInput) {
                if (oldInput.link !== null && oldInput.link !== undefined && node.graph?.links) {
                    const link = node.graph.links[oldInput.link];
                    if (link && (node.inputs[newIdx].link === null || node.inputs[newIdx].link === undefined)) {
                        link.target_slot = newIdx;
                        node.inputs[newIdx].link = oldInput.link;
                        oldInput.link = null;
                    }
                }
                if (typeof node.removeInput === "function") node.removeInput(oldIdx);
                else node.inputs.splice(oldIdx, 1);
            } else if (newIdx === -1 && oldInput) {
                oldInput.name = "model_pass2";
                if (oldInput.label) oldInput.label = "model_pass2";
            }
        }
    }
    // Output migration: sampler_info -> info
    if (Array.isArray(node.outputs)) {
        const oldOutIdx = node.outputs.findIndex((o) => o && o.name === "sampler_info");
        if (oldOutIdx !== -1) {
            const newOutIdx = node.outputs.findIndex((o) => o && o.name === "info");
            const oldOut = node.outputs[oldOutIdx];
            if (newOutIdx !== -1 && oldOut) {
                if (Array.isArray(oldOut.links) && oldOut.links.length > 0 && node.graph?.links) {
                    for (const lId of oldOut.links) {
                        const link = node.graph.links[lId];
                        if (link) {
                            link.origin_slot = newOutIdx;
                            if (!Array.isArray(node.outputs[newOutIdx].links)) node.outputs[newOutIdx].links = [];
                            node.outputs[newOutIdx].links.push(lId);
                        }
                    }
                    oldOut.links = [];
                }
                if (typeof node.removeOutput === "function") node.removeOutput(oldOutIdx);
                else node.outputs.splice(oldOutIdx, 1);
            } else if (newOutIdx === -1 && oldOut) {
                oldOut.name = "info";
                if (oldOut.label) oldOut.label = "info";
            }
        }
    }
}

export function setupAnimaDualPassKSamplerUI(node) {
    applyNodeFrameStyle(node);
    migrateDualPassSlots(node);
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
            title: "",
            accentColor: HADES_THEME.emerald,
            customHeight: 52,
            tooltip: `<strong>Dual-Pass Pipeline Monitor</strong><br>Telemetry showing Pass 1 steps/CFG, 16ch bridge scale factor, and Pass 2 steps/denoise/shift.`,
            drawCustomHUD(ctx, rect) {
                const s1 = Number(wSteps1?.value ?? 30);
                const c1 = Number(wCfg1?.value ?? 4.5);
                const scale = Number(wScale?.value ?? 1.5);
                const s2 = Number(wSteps2?.value ?? 16);
                const c2 = Number(wCfg2?.value ?? 4.0);
                const d2 = Number(wDen2?.value ?? 0.42);
                const sh2 = Number(wShift2?.value ?? 3.6);

                const padX = 6;
                const boxY = rect.y + 3;
                const boxH = Math.max(18, rect.h - 6);
                const totalW = rect.w - padX * 2;
                
                // Allow dynamic space for bridge badge based on available width
                const badgeW = totalW < 220 ? 24 : (totalW < 300 ? 36 : 48);
                const availForBoxes = totalW - badgeW - 8;
                const boxW = Math.max(30, Math.floor(availForBoxes * 0.5));

                ctx.font = totalW < 240 ? "600 8px ui-monospace, Consolas, monospace" : "600 9px ui-monospace, Consolas, monospace";

                // Pass 1 pill
                const p1X = rect.x + padX;
                drawRoundedRect(ctx, p1X, boxY, boxW, boxH, 4);
                ctx.fillStyle = `${HADES_THEME.blue}22`;
                ctx.fill();
                ctx.strokeStyle = HADES_THEME.blue;
                ctx.stroke();

                ctx.save();
                ctx.beginPath();
                ctx.rect(p1X + 2, boxY, boxW - 4, boxH);
                ctx.clip();
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = HADES_THEME.textPrimary;

                let p1Text = `PASS 1: 1.0x (${s1}s | CFG ${c1.toFixed(1)})`;
                if (ctx.measureText(p1Text).width > boxW - 4) {
                    p1Text = `P1: ${s1}s · C${c1.toFixed(1)}`;
                }
                if (ctx.measureText(p1Text).width > boxW - 4) {
                    p1Text = `P1 (${s1}s)`;
                }
                if (ctx.measureText(p1Text).width > boxW - 4) {
                    p1Text = `${s1}s`;
                }
                ctx.fillText(p1Text, p1X + boxW * 0.5, boxY + boxH * 0.5);
                ctx.restore();

                // Bridge badge
                const midX = p1X + boxW + 4 + badgeW * 0.5;
                ctx.fillStyle = HADES_THEME.amber;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.font = totalW < 220 ? "600 8px ui-monospace, Consolas, monospace" : "600 9px ui-monospace, Consolas, monospace";
                const badgeText = badgeW < 30 ? "→" : `×${scale.toFixed(badgeW < 40 ? 1 : 2)}`;
                ctx.fillText(badgeText, midX, boxY + boxH * 0.5);

                // Pass 2 pill
                const p2X = p1X + boxW + 8 + badgeW;
                const p2W = Math.max(30, rect.x + rect.w - padX - p2X);
                drawRoundedRect(ctx, p2X, boxY, p2W, boxH, 4);
                ctx.fillStyle = `${HADES_THEME.emerald}22`;
                ctx.fill();
                ctx.strokeStyle = HADES_THEME.emerald;
                ctx.stroke();

                ctx.save();
                ctx.beginPath();
                ctx.rect(p2X + 2, boxY, p2W - 4, boxH);
                ctx.clip();
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.font = totalW < 240 ? "600 8px ui-monospace, Consolas, monospace" : "600 9px ui-monospace, Consolas, monospace";

                let p2Text = `PASS 2: ${s2}s | DEN ${d2.toFixed(2)} | S ${sh2 > 0 ? sh2.toFixed(1) : "INH"}`;
                if (ctx.measureText(p2Text).width > p2W - 4) {
                    p2Text = `P2: ${s2}s · D${d2.toFixed(2)}`;
                }
                if (ctx.measureText(p2Text).width > p2W - 4) {
                    p2Text = `P2 (${s2}s)`;
                }
                if (ctx.measureText(p2Text).width > p2W - 4) {
                    p2Text = `${s2}s`;
                }
                ctx.fillText(p2Text, p2X + p2W * 0.5, boxY + boxH * 0.5);
                ctx.restore();
            },
            actions: [
                {
                    label: "HiRes 1.5x",
                    tooltip: `<strong>HiRes Standard 1.5x</strong><br>30-step Pass 1 (er_sde) scaled 1.5x via 16ch bicubic bridge, refined in Pass 2 (16 steps, beta57, 0.42 denoise).`,
                    onClick: () =>
                        applyPreset("HiRes Standard 1.5x (er_sde 30s -> er_sde + beta57 16s @ 0.42 denoise)"),
                    isActive: () => wPreset?.value === "HiRes Standard 1.5x (er_sde 30s -> er_sde + beta57 16s @ 0.42 denoise)",
                },
                {
                    label: "Painterly",
                    tooltip: `<strong>Aesthetic Painterly 1.5x</strong><br>Painterly aesthetic profile using euler_ancestral in Pass 2 with beta57 scheduler at 0.45 denoise.`,
                    onClick: () =>
                        applyPreset(
                            "Aesthetic Painterly 1.5x (er_sde 32s -> euler_a + beta57 18s @ 0.45 denoise)"
                        ),
                    isActive: () => wPreset?.value === "Aesthetic Painterly 1.5x (er_sde 32s -> euler_a + beta57 18s @ 0.45 denoise)",
                },
                {
                    label: "Base→Turbo",
                    tooltip: `<strong>Base-to-Turbo Fast HiRes 1.5x</strong><br>Draft composition with Base model, then handoff to Turbo model in Pass 2 (8 steps, CFG 1.0, 0.38 denoise).`,
                    onClick: () =>
                        applyPreset(
                            "Base-to-Turbo Fast HiRes 1.5x (er_sde 30s CFG 4.5 -> euler 8s CFG 1.0 @ 0.38 denoise)"
                        ),
                    isActive: () => wPreset?.value === "Base-to-Turbo Fast HiRes 1.5x (er_sde 30s CFG 4.5 -> euler 8s CFG 1.0 @ 0.38 denoise)",
                },
                {
                    label: "1.0x Polish",
                    tooltip: `<strong>Same-Res Texture & Lineart Polish</strong><br>Direct 1.0x pass-through refiner to sharpen lineart and surface textures without resolution upscaling.`,
                    onClick: () =>
                        applyPreset("Same-Res Texture & Lineart Refiner 1.0x (30s -> 12s @ 0.28 denoise)"),
                    isActive: () => wPreset?.value === "Same-Res Texture & Lineart Refiner 1.0x (30s -> 12s @ 0.28 denoise)",
                },
            ],
        })
    );

    const presetWidget = node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "PRESET",
            tagColor: HADES_THEME.emerald,
            tooltip: `<strong>Dual-Pass Preset Profile</strong><br>Select curated generation workflows or switch to Manual for custom widget settings.`,
        })
    );

    // Pass 1 Controls
    const samp1Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSamp1, {
            tag: "PASS 1", label: "Sampler",
            tagColor: HADES_THEME.blue,
            tooltip: `<strong>Pass 1 Sampler</strong><br>Sampling algorithm for initial composition (er_sde recommended for Anima).`,
        })
    );
    const sched1Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSched1, {
            tag: "PASS 1", label: "Scheduler",
            tagColor: HADES_THEME.blue,
            tooltip: `<strong>Pass 1 Scheduler</strong><br>Noise schedule curve for Pass 1 (normal or beta).`,
        })
    );
    const steps1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps1, {
            tag: "PASS 1", label: "Pass 1 Steps",
            color: HADES_THEME.blue, min: 1, max: 200, visMin: 4, visMax: 50, step: 1, defaultVal: 30, isInt: true,
            tooltip: `<strong>Pass 1 Steps</strong><br>Number of denoising iterations for the base composition pass (28–32 steps).`,
        })
    );
    const cfg1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg1, {
            tag: "PASS 1", label: "Pass 1 CFG",
            color: HADES_THEME.blue, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 4.5,
            tooltip: `<strong>Pass 1 CFG Scale</strong><br>Classifier-Free Guidance scale for prompt adherence in Pass 1 (4.0–5.0 recommended).`,
        })
    );
    const den1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen1, {
            tag: "PASS 1", label: "Pass 1 Denoise",
            color: HADES_THEME.blue, min: 0.0, max: 1.0, visMin: 0.0, visMax: 1.0, step: 0.01, defaultVal: 1.0,
            tooltip: `<strong>Pass 1 Denoise Factor</strong><br>Initial denoising strength (1.0 for generating from pure latent noise).`,
        })
    );
    const shift1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift1, {
            tag: "PASS 1", label: "Pass 1 Flow Shift",
            color: HADES_THEME.blue, min: 0.0, max: 10.0, visMin: 0.0, visMax: 6.0, step: 0.05, defaultVal: 0.0, zeroLabel: "INHERIT",
            tooltip: `<strong>Pass 1 Flow Shift</strong><br>Time-step warping parameter for Anima's Rectified Flow (multiplier=1.0). 0 = inherit the model's shift (Model Loader / ModelSampling).`,
        })
    );

    // Bridge Controls
    const methodWidget = node.addCustomWidget(
        createStyledComboWidget(node, wMethod, {
            tag: "BRIDGE",
            tagColor: HADES_THEME.amber,
            tooltip: `<strong>Inter-Pass Latent Bridge</strong><br>Method used to transition Pass 1 latents into Pass 2. 16ch Variance-Preserving Bicubic preserves latent statistics and snaps strictly to 32px boundaries.`,
        })
    );
    const scaleWidget = node.addCustomWidget(
        createSpectralSliderWidget(node, wScale, {
            tag: "SCALE", label: "Upscale factor",
            color: HADES_THEME.amber, min: 1.0, max: 2.5, visMin: 1.0, visMax: 2.0, step: 0.05, defaultVal: 1.5,
            tooltip: `<strong>HiRes Magnification Scale</strong><br>Resolution multiplier relative to initial latent. Automatically snaps pixel dimensions to the nearest 32px grid required by Anima DiT.`,
        })
    );

    // Pass 2 Controls
    const samp2Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSamp2, {
            tag: "PASS 2", label: "Sampler",
            tagColor: HADES_THEME.emerald,
            tooltip: `<strong>Pass 2 Refiner Sampler</strong><br>Sampling algorithm for the high-resolution refinement pass.`,
        })
    );
    const sched2Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSched2, {
            tag: "PASS 2", label: "Scheduler",
            tagColor: HADES_THEME.emerald,
            tooltip: `<strong>Pass 2 Refiner Scheduler</strong><br>Noise schedule for Pass 2. anima_beta57 puts more steps at low noise; from full noise it converges slower than stock beta (tools/ABLATION_RESULTS.md).`,
        })
    );
    const steps2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps2, {
            tag: "PASS 2", label: "Pass 2 Refiner Steps",
            color: HADES_THEME.emerald, min: 1, max: 200, visMin: 4, visMax: 40, step: 1, defaultVal: 16, isInt: true,
            tooltip: `<strong>Pass 2 Refiner Steps</strong><br>Number of denoising steps in the refinement pass (12–18 steps typical).`,
        })
    );
    const cfg2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg2, {
            tag: "PASS 2", label: "Pass 2 Refiner CFG",
            color: HADES_THEME.emerald, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 4.0,
            tooltip: `<strong>Pass 2 Refiner CFG</strong><br>Guidance scale during refinement (lower than Pass 1 to prevent contrast burns).`,
        })
    );
    const den2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen2, {
            tag: "PASS 2", label: "Pass 2 Refiner Denoise",
            color: HADES_THEME.emerald, min: 0.05, max: 1.0, visMin: 0.1, visMax: 0.8, step: 0.01, defaultVal: 0.42,
            tooltip: `<strong>Pass 2 Refiner Denoise</strong><br>Fraction of noise added to the upscaled latent (0.35–0.48 preserves anatomy while enriching fine details).`,
        })
    );
    const shift2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift2, {
            tag: "PASS 2", label: "Pass 2 High-Res Flow Shift",
            color: HADES_THEME.emerald, min: 0.0, max: 10.0, visMin: 0.0, visMax: 6.0, step: 0.05, defaultVal: 3.6, zeroLabel: "INHERIT",
            tooltip: `<strong>Pass 2 High-Res Flow Shift</strong><br>Rectified Flow time-step shift for high-resolution refinement (3.5–4.0 focuses compute on fine details).`,
        })
    );

    const seedWidgets = createSeedRows(node, wSeed, wControl);

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
        migrateDualPassSlots(this);
        sanitizeDualPassValues();
        this.setDirtyCanvas(true, true);
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        migrateDualPassSlots(this);
        this.setDirtyCanvas?.(true, true);
    };
    sanitizeDualPassValues();

    finalizeHadesNode(node, 460);
}

function migrateTriplePassSlots(node) {
    if (!node) return;
    // Input migrations: model_stage2_optional -> model_pass2, model_stage3_optional -> model_pass3
    if (Array.isArray(node.inputs)) {
        const inputPairs = [
            ["model_stage2_optional", "model_pass2"],
            ["model_stage3_optional", "model_pass3"],
        ];
        for (const [oldName, newName] of inputPairs) {
            const oldIdx = node.inputs.findIndex((i) => i && i.name === oldName);
            if (oldIdx === -1) continue;
            const newIdx = node.inputs.findIndex((i) => i && i.name === newName);
            const oldInput = node.inputs[oldIdx];
            if (newIdx !== -1 && oldInput) {
                if (oldInput.link !== null && oldInput.link !== undefined && node.graph?.links) {
                    const link = node.graph.links[oldInput.link];
                    if (link && (node.inputs[newIdx].link === null || node.inputs[newIdx].link === undefined)) {
                        link.target_slot = newIdx;
                        node.inputs[newIdx].link = oldInput.link;
                        oldInput.link = null;
                    }
                }
                if (typeof node.removeInput === "function") node.removeInput(oldIdx);
                else node.inputs.splice(oldIdx, 1);
            } else if (newIdx === -1 && oldInput) {
                oldInput.name = newName;
                if (oldInput.label) oldInput.label = newName;
            }
        }
    }
    // Output migrations: stage1_latent -> pass1_latent, stage2_latent -> pass2_latent, sampler_info -> info
    if (Array.isArray(node.outputs)) {
        const outputPairs = [
            ["stage1_latent", "pass1_latent"],
            ["stage2_latent", "pass2_latent"],
            ["sampler_info", "info"],
        ];
        for (const [oldName, newName] of outputPairs) {
            const oldIdx = node.outputs.findIndex((o) => o && o.name === oldName);
            if (oldIdx === -1) continue;
            const newIdx = node.outputs.findIndex((o) => o && o.name === newName);
            const oldOut = node.outputs[oldIdx];
            if (newIdx !== -1 && oldOut) {
                if (Array.isArray(oldOut.links) && oldOut.links.length > 0 && node.graph?.links) {
                    for (const lId of oldOut.links) {
                        const link = node.graph.links[lId];
                        if (link) {
                            link.origin_slot = newIdx;
                            if (!Array.isArray(node.outputs[newIdx].links)) node.outputs[newIdx].links = [];
                            node.outputs[newIdx].links.push(lId);
                        }
                    }
                    oldOut.links = [];
                }
                if (typeof node.removeOutput === "function") node.removeOutput(oldIdx);
                else node.outputs.splice(oldIdx, 1);
            } else if (newIdx === -1 && oldOut) {
                oldOut.name = newName;
                if (oldOut.label) oldOut.label = newName;
            }
        }
    }
}

export function setupAnimaTriplePassKSamplerUI(node) {
    applyNodeFrameStyle(node);
    migrateTriplePassSlots(node);
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
            title: "",
            accentColor: HADES_THEME.purple,
            customHeight: 54,
            tooltip: `<strong>Triple-Pass Studio Monitor</strong><br>Real-time telemetry showing Stage 1 steps/denoise, Stage 2 scale/steps, and Stage 3 final scale/polish.`,
            drawCustomHUD(ctx, rect) {
                const sc2 = Number(wScale2?.value ?? 1.35);
                const sc3 = Number(wScale3?.value ?? 1.75);
                const d1 = Number(wDen1?.value ?? 1.0);
                const d2 = Number(wDen2?.value ?? 0.48);
                const d3 = Number(wDen3?.value ?? 0.26);
                const s1 = Number(wSteps1?.value ?? 28);
                const s2 = Number(wSteps2?.value ?? 16);
                const s3 = Number(wSteps3?.value ?? 12);

                const padX = 6;
                const gap = 4;
                const totalW = rect.w - padX * 2;
                const cardW = Math.max(20, (totalW - gap * 2) / 3);
                const cardY = rect.y + 3;
                const cardH = Math.max(18, rect.h - 6);

                const stages = [
                    { tag: "S1", scale: "1.00", steps: s1, den: d1, col: HADES_THEME.blue },
                    { tag: "S2", scale: sc2.toFixed(2), steps: s2, den: d2, col: HADES_THEME.cyan },
                    { tag: "S3", scale: sc3.toFixed(2), steps: s3, den: d3, col: HADES_THEME.purple },
                ];

                ctx.font = totalW < 260 ? "600 8px ui-monospace, Consolas, monospace" : "600 9px ui-monospace, Consolas, monospace";

                stages.forEach((st, i) => {
                    const cx = rect.x + padX + i * (cardW + gap);
                    drawRoundedRect(ctx, cx, cardY, cardW, cardH, 4);
                    ctx.fillStyle = `${st.col}22`;
                    ctx.fill();
                    ctx.strokeStyle = st.col;
                    ctx.stroke();

                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(cx + 2, cardY, cardW - 4, cardH);
                    ctx.clip();

                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillStyle = HADES_THEME.textPrimary;

                    let text = `${st.tag} ${st.scale}x (${st.steps}s) | D${st.den.toFixed(2)}`;
                    if (ctx.measureText(text).width > cardW - 4) {
                        text = `${st.tag} ${st.scale}x (${st.steps}s)`;
                    }
                    if (ctx.measureText(text).width > cardW - 4) {
                        text = `${st.tag} (${st.steps}s)`;
                    }
                    if (ctx.measureText(text).width > cardW - 4) {
                        text = `${st.tag}`;
                    }
                    ctx.fillText(text, cx + cardW * 0.5, cardY + cardH * 0.5);
                    ctx.restore();
                });
            },
            actions: [
                {
                    label: "1.35→1.75x",
                    tooltip: `<strong>Progressive Staircase (1.0x -> 1.35x -> 1.75x)</strong><br>Curated 3-stage poster pipeline: 28-step draft, 1.35x bridge at 0.48 denoise, and 1.75x final surface polish.`,
                    onClick: () =>
                        applyPreset("Progressive Staircase 1.0x -> 1.35x -> 1.75x (Ultra-Res Poster Pipeline)"),
                    isActive: () => wPreset?.value === "Progressive Staircase 1.0x -> 1.35x -> 1.75x (Ultra-Res Poster Pipeline)",
                },
                {
                    label: "1.5→2.0x",
                    tooltip: `<strong>Progressive 2x Master (1.0x -> 1.5x -> 2.0x)</strong><br>Studio illustration pipeline reaching 4 Megapixels with zero anatomy drift or limb duplication.`,
                    onClick: () =>
                        applyPreset("Progressive 2x Master 1.0x -> 1.5x -> 2.0x (4 MP Studio Illustration)"),
                    isActive: () => wPreset?.value === "Progressive 2x Master 1.0x -> 1.5x -> 2.0x (4 MP Studio Illustration)",
                },
                {
                    label: "1.5x Lock",
                    tooltip: `<strong>Aesthetic Bridge & Micro-Polish (1.5x Lock)</strong><br>Upscales to 1.5x at Stage 2 and locks resolution, using Stage 3 for delicate micro-lineart sharpening.`,
                    onClick: () =>
                        applyPreset("Concept -> Aesthetic Bridge 1.5x -> Micro-Polish 1.5x (Lock Res at Stage 2)"),
                    isActive: () => wPreset?.value === "Concept -> Aesthetic Bridge 1.5x -> Micro-Polish 1.5x (Lock Res at Stage 2)",
                },
                {
                    label: "Turbo S3",
                    tooltip: `<strong>Hybrid Base -> Aesthetic -> Turbo Polish</strong><br>Drafts with Base, styles with Aesthetic, and rapidly polishes final high-res surface with Turbo in 6 steps.`,
                    onClick: () =>
                        applyPreset("Hybrid Base -> Aesthetic -> Turbo Polish (Fast 3-Stage Studio)"),
                    isActive: () => wPreset?.value === "Hybrid Base -> Aesthetic -> Turbo Polish (Fast 3-Stage Studio)",
                },
            ],
        })
    );

    const presetWidget = node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "PRESET",
            tagColor: HADES_THEME.purple,
            tooltip: `<strong>Triple-Pass Preset Profile</strong><br>Curated multi-stage studio workflows or Manual for full custom control.`,
        })
    );
    const methodWidget = node.addCustomWidget(
        createStyledComboWidget(node, wMethod, {
            tag: "BRIDGE",
            tagColor: HADES_THEME.amber,
            tooltip: `<strong>Progressive Latent Bridge</strong><br>16-channel variance-preserving bicubic interpolation snapped to the 32px grid across all progressive stages.`,
        })
    );

    // Stage 1
    const samp1Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSamp1, {
            tag: "STAGE 1", label: "Sampler",
            tagColor: HADES_THEME.blue,
            tooltip: `<strong>Stage 1 Draft Sampler</strong><br>Sampling algorithm for draft layout and character pose lock.`,
        })
    );
    const sched1Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSched1, {
            tag: "STAGE 1", label: "Scheduler",
            tagColor: HADES_THEME.blue,
            tooltip: `<strong>Stage 1 Draft Scheduler</strong><br>Noise schedule for initial base stage (normal standard).`,
        })
    );
    const steps1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps1, {
            tag: "STAGE 1", label: "Stage 1 Draft Steps",
            color: HADES_THEME.blue, min: 1, max: 200, visMin: 4, visMax: 50, step: 1, defaultVal: 28, isInt: true,
            tooltip: `<strong>Stage 1 Draft Steps</strong><br>Denoising iterations for base composition (24–32 steps).`,
        })
    );
    const cfg1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg1, {
            tag: "STAGE 1", label: "Stage 1 CFG",
            color: HADES_THEME.blue, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 4.5,
            tooltip: `<strong>Stage 1 CFG Scale</strong><br>Prompt guidance for the draft stage (4.0–4.8).`,
        })
    );
    const den1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen1, {
            tag: "STAGE 1", label: "Stage 1 Denoise",
            color: HADES_THEME.blue, min: 0.0, max: 1.0, visMin: 0.0, visMax: 1.0, step: 0.01, defaultVal: 1.0,
            tooltip: `<strong>Stage 1 Denoise</strong><br>Full denoise (1.0) from raw latent noise.`,
        })
    );
    const shift1Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift1, {
            tag: "STAGE 1", label: "Stage 1 Flow Shift",
            color: HADES_THEME.blue, min: 0.0, max: 10.0, visMin: 0.0, visMax: 6.0, step: 0.05, defaultVal: 0.0, zeroLabel: "INHERIT",
            tooltip: `<strong>Stage 1 Flow Shift</strong><br>Rectified Flow shift for base resolution. 0 = inherit the model's shift (Model Loader / ModelSampling).`,
        })
    );

    // Stage 2
    const scale2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wScale2, {
            tag: "STAGE 2", label: "Stage 2 Scale (vs Base)",
            color: HADES_THEME.cyan, min: 1.0, max: 2.5, visMin: 1.0, visMax: 2.0, step: 0.05, defaultVal: 1.35,
            tooltip: `<strong>Stage 2 Scale Factor</strong><br>Intermediate resolution magnification relative to Stage 1 (e.g. 1.35x–1.50x).`,
        })
    );
    const samp2Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSamp2, {
            tag: "STAGE 2", label: "Sampler",
            tagColor: HADES_THEME.cyan,
            tooltip: `<strong>Stage 2 Bridge Sampler</strong><br>Sampler for structural refinement and color harmony.`,
        })
    );
    const sched2Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSched2, {
            tag: "STAGE 2", label: "Scheduler",
            tagColor: HADES_THEME.cyan,
            tooltip: `<strong>Stage 2 Bridge Scheduler</strong><br>Scheduler for intermediate bridge pass (beta recommended).`,
        })
    );
    const steps2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps2, {
            tag: "STAGE 2", label: "Stage 2 Bridge Steps",
            color: HADES_THEME.cyan, min: 1, max: 200, visMin: 4, visMax: 40, step: 1, defaultVal: 16, isInt: true,
            tooltip: `<strong>Stage 2 Bridge Steps</strong><br>Iterations for the intermediate structural bridge (14–18 steps).`,
        })
    );
    const cfg2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg2, {
            tag: "STAGE 2", label: "Stage 2 Bridge CFG",
            color: HADES_THEME.cyan, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 4.0,
            tooltip: `<strong>Stage 2 Bridge CFG</strong><br>Guidance scale during intermediate stage.`,
        })
    );
    const den2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen2, {
            tag: "STAGE 2", label: "Stage 2 Bridge Denoise",
            color: HADES_THEME.cyan, min: 0.05, max: 1.0, visMin: 0.1, visMax: 0.8, step: 0.01, defaultVal: 0.48,
            tooltip: `<strong>Stage 2 Bridge Denoise</strong><br>Denoise strength for Stage 2 (0.42–0.50).`,
        })
    );
    const shift2Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift2, {
            tag: "STAGE 2", label: "Stage 2 Flow Shift",
            color: HADES_THEME.cyan, min: 0.0, max: 10.0, visMin: 0.0, visMax: 6.0, step: 0.05, defaultVal: 3.5, zeroLabel: "INHERIT",
            tooltip: `<strong>Stage 2 Flow Shift</strong><br>Rectified Flow shift for intermediate resolution (3.40–3.60).`,
        })
    );

    // Stage 3
    const scale3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wScale3, {
            tag: "STAGE 3", label: "Stage 3 Final Scale (vs Base)",
            color: HADES_THEME.purple, min: 1.0, max: 3.0, visMin: 1.0, visMax: 2.5, step: 0.05, defaultVal: 1.75,
            tooltip: `<strong>Stage 3 Final Scale Factor</strong><br>Final output scale calculated directly against Stage 1 base dimensions to prevent cumulative rounding errors.`,
        })
    );
    const samp3Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSamp3, {
            tag: "STAGE 3", label: "Sampler",
            tagColor: HADES_THEME.purple,
            tooltip: `<strong>Stage 3 Polish Sampler</strong><br>Fast high-frequency sampler for final polish (euler or er_sde).`,
        })
    );
    const sched3Widget = node.addCustomWidget(
        createStyledComboWidget(node, wSched3, {
            tag: "STAGE 3", label: "Scheduler",
            tagColor: HADES_THEME.purple,
            tooltip: `<strong>Stage 3 Polish Scheduler</strong><br>Noise schedule for Stage 3. anima_beta57 puts more steps at low noise; from full noise it converges slower than stock beta (tools/ABLATION_RESULTS.md).`,
        })
    );
    const steps3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wSteps3, {
            tag: "STAGE 3", label: "Stage 3 Polish Steps",
            color: HADES_THEME.purple, min: 1, max: 200, visMin: 4, visMax: 30, step: 1, defaultVal: 12, isInt: true,
            tooltip: `<strong>Stage 3 Polish Steps</strong><br>Final micro-lineart and texture polish steps (10–14 steps).`,
        })
    );
    const cfg3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wCfg3, {
            tag: "STAGE 3", label: "Stage 3 Polish CFG",
            color: HADES_THEME.purple, min: 1.0, max: 20.0, visMin: 1.0, visMax: 8.0, step: 0.1, defaultVal: 3.4,
            tooltip: `<strong>Stage 3 Polish CFG</strong><br>Final polish guidance scale (3.0–3.5 to avoid over-saturation).`,
        })
    );
    const den3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wDen3, {
            tag: "STAGE 3", label: "Stage 3 Polish Denoise",
            color: HADES_THEME.purple, min: 0.05, max: 1.0, visMin: 0.1, visMax: 0.6, step: 0.01, defaultVal: 0.26,
            tooltip: `<strong>Stage 3 Polish Denoise</strong><br>Gentle denoise (0.22–0.28) preserving structure while adding micro-acuity.`,
        })
    );
    const shift3Widget = node.addCustomWidget(
        createSpectralSliderWidget(node, wShift3, {
            tag: "STAGE 3", label: "Stage 3 Flow Shift",
            color: HADES_THEME.purple, min: 0.0, max: 10.0, visMin: 0.0, visMax: 6.0, step: 0.05, defaultVal: 4.0, zeroLabel: "INHERIT",
            tooltip: `<strong>Stage 3 Flow Shift</strong><br>Flow shift for high resolution (3.80–4.20).`,
        })
    );

    const seedWidgets = createSeedRows(node, wSeed, wControl);

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
        migrateTriplePassSlots(this);
        sanitizeTriplePassValues();
        this.setDirtyCanvas(true, true);
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        migrateTriplePassSlots(this);
        this.setDirtyCanvas?.(true, true);
    };
    sanitizeTriplePassValues();

    finalizeHadesNode(node, 470);
}

export function setupSamplerMetadataUI(node) {
    applyNodeFrameStyle(node);
    node._lastSettings = null;

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.blue,
            customHeight: 58,
            tooltip: `<strong>Sampler Metadata</strong><br>Settings the connected KSampler actually used (profile / preset applied). Dual / Triple-Pass report pass 1.`,
            drawCustomHUD(ctx, rect) {
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                const s = node._lastSettings;
                if (!s) {
                    ctx.font = "9px ui-monospace, Consolas, monospace";
                    ctx.fillStyle = HADES_THEME.textMuted;
                    ctx.fillText("Connect a KSampler info output and queue", rect.x + 8, rect.y + rect.h / 2);
                    return;
                }
                const [seed, steps, cfg, sampler, sched, denoise] = s;
                ctx.font = "600 10px ui-monospace, Consolas, monospace";
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(`${String(sampler).toUpperCase()}  |  ${sched}`, rect.x + 8, rect.y + 11);
                ctx.font = "9px ui-monospace, Consolas, monospace";
                ctx.fillStyle = HADES_THEME.emerald;
                ctx.fillText(`STEPS: ${steps}  |  CFG: ${Number(cfg).toFixed(1)}  |  DENOISE: ${Number(denoise).toFixed(2)}`, rect.x + 8, rect.y + 25);
                ctx.fillStyle = HADES_THEME.textMuted;
                ctx.fillText(`SEED: ${seed}`, rect.x + 8, rect.y + 39);
            },
        })
    );

    const origOnExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        origOnExecuted?.apply(this, arguments);
        const settings = (output?.ui || output)?.settings?.[0];
        if (settings) {
            this._lastSettings = settings;
            this.setDirtyCanvas(true, true);
        }
    };

    finalizeHadesNode(node, 300);
}

export const KSAMPLER_NODE_DISPATCH = {
    HaDeSAnimaKSampler: setupAnimaKSamplerUI,
    HaDeSAnimaDualPassKSampler: setupAnimaDualPassKSamplerUI,
    HaDeSAnimaTriplePassKSampler: setupAnimaTriplePassKSamplerUI,
    HaDeSAnimaSamplerMetadata: setupSamplerMetadataUI,
};
