import {
    HADES_THEME,
    GROUP_META,
    CLIP_GROUP_META,
    setWidgetValue,
    hideBackingWidget,
    applyNodeFrameStyle,
    parseClientRecipePreview,
    estimatePredictiveCurve,
    createSpectralBannerWidget,
    createStyledComboWidget,
    createSpectralSliderWidget,
    createStyledTextRowWidget,
    createStyledToggleWidget,
    createSlotRowWidget,
} from "./hades_widgets.js";

// =============================================================================
// UI INSTALLERS FOR THE 5 HADES ANIMA MERGER & LORA LOADER NODES
// =============================================================================

export function setupManualBlockMergerUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const groupKeys = [
        "base_ratio",
        "BLOCKS_Layout_Geometry",
        "BLOCKS_Global_Composition",
        "BLOCKS_Subject_Identity",
        "BLOCKS_Material_Substance",
        "BLOCKS_Art_Style_Medium",
        "BLOCKS_Lighting_Sharpness",
        "ADAPTER_Text_Alignment",
    ];

    const backingMap = {};
    for (const k of groupKeys) {
        const w = findW(k);
        hideBackingWidget(w);
        backingMap[k] = w;
    }
    const wRecipe = findW("recipe_override");
    hideBackingWidget(wRecipe);

    const setAllGroups = (ratioObj) => {
        if (wRecipe) setWidgetValue(wRecipe, "", node);
        for (const k of groupKeys) {
            if (backingMap[k] && ratioObj[k] !== undefined) {
                setWidgetValue(backingMap[k], ratioObj[k], node);
            }
        }
    };

    const blockColors = new Array(34).fill(HADES_THEME.blue);
    for (const meta of Object.values(GROUP_META)) {
        for (const idx of meta.indices) blockColors[idx] = meta.color;
    }

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.blue,
            getBars() {
                const overrideArr = parseClientRecipePreview(wRecipe?.value);
                if (overrideArr) {
                    return overrideArr.map((r, idx) => ({ ratio: r, color: blockColors[idx] }));
                }
                const bars = new Array(34);
                const fallback = Number(backingMap.base_ratio?.value ?? 0.5);
                for (let i = 0; i < 34; i++) {
                    bars[i] = { ratio: fallback, color: blockColors[i] };
                }
                for (const [gName, meta] of Object.entries(GROUP_META)) {
                    if (!meta.indices.length) continue;
                    const defVal = gName === "ADAPTER_Text_Alignment" ? 0.0 : fallback;
                    const r = Number(backingMap[gName]?.value ?? defVal);
                    for (const idx of meta.indices) {
                        bars[idx] = { ratio: r, color: meta.color };
                    }
                }
                return bars;
            },
            actions: [
                {
                    label: "M1",
                    tooltip: `<strong>100% Model 1</strong><br>Sets all block groups to 0.0 (pure Model 1).`,
                    onClick: () => setAllGroups(Object.fromEntries(groupKeys.map((k) => [k, 0.0]))),
                },
                {
                    label: "50/50 (Lock Brg)",
                    tooltip: `<strong>50/50 Balanced Merge</strong><br>Even 0.50 blend while strictly preserving Model 1's llm_adapter bridge (0.0).`,
                    onClick: () =>
                        setAllGroups({
                            base_ratio: 0.5,
                            BLOCKS_Layout_Geometry: 0.5,
                            BLOCKS_Global_Composition: 0.5,
                            BLOCKS_Subject_Identity: 0.5,
                            BLOCKS_Material_Substance: 0.5,
                            BLOCKS_Art_Style_Medium: 0.5,
                            BLOCKS_Lighting_Sharpness: 0.5,
                            ADAPTER_Text_Alignment: 0.0,
                        }),
                },
                {
                    label: "Style B",
                    tooltip: `<strong>Model 1 Structure / Model 2 Style</strong><br>Keeps M1 geometry/identity while transferring M2 textures, lighting, and style.`,
                    onClick: () =>
                        setAllGroups({
                            base_ratio: 0.5,
                            BLOCKS_Layout_Geometry: 0.1,
                            BLOCKS_Global_Composition: 0.15,
                            BLOCKS_Subject_Identity: 0.1,
                            BLOCKS_Material_Substance: 0.65,
                            BLOCKS_Art_Style_Medium: 0.9,
                            BLOCKS_Lighting_Sharpness: 0.85,
                            ADAPTER_Text_Alignment: 0.0,
                        }),
                },
                {
                    label: "M2",
                    tooltip: `<strong>100% Model 2</strong><br>Sets all block groups to 1.0 (pure Model 2).`,
                    onClick: () => setAllGroups(Object.fromEntries(groupKeys.map((k) => [k, 1.0]))),
                },
            ],
        })
    );

    const friendlyLabels = {
        base_ratio: "Base Fallback Ratio",
        BLOCKS_Layout_Geometry: "Layout & Geometry",
        BLOCKS_Global_Composition: "Global Composition",
        BLOCKS_Subject_Identity: "Subject Identity",
        BLOCKS_Material_Substance: "Material & Substance",
        BLOCKS_Art_Style_Medium: "Art Style & Medium",
        BLOCKS_Lighting_Sharpness: "Lighting & Sharpness",
        ADAPTER_Text_Alignment: "LLM Adapter Bridge (0=Lock)",
    };

    const groupTooltips = {
        base_ratio: `<strong>Global Base Fallback Ratio</strong><br>Ratio applied to root embeddings, timesteps, and non-block parameters.`,
        BLOCKS_Layout_Geometry: `<strong>Layout & Geometry (Blocks 0–4)</strong><br>Core spatial composition, framing, perspective, and global anatomy.`,
        BLOCKS_Global_Composition: `<strong>Global Composition (Blocks 5–9)</strong><br>Scene staging, major structural forms, and background elements.`,
        BLOCKS_Subject_Identity: `<strong>Subject Identity (Blocks 10–14)</strong><br>Character features, facial contours, clothing, and primary subjects.`,
        BLOCKS_Material_Substance: `<strong>Material & Substance (Blocks 15–19)</strong><br>Surface textures, materials, reflections, and skin rendering.`,
        BLOCKS_Art_Style_Medium: `<strong>Art Style & Medium (Blocks 20–23)</strong><br>Brushwork, line weight, painterly vs flat vector aesthetics.`,
        BLOCKS_Lighting_Sharpness: `<strong>Lighting & Sharpness (Blocks 24–27)</strong><br>Micro-contrast, highlight bloom, rim lighting, and final acuity.`,
        ADAPTER_Text_Alignment: `<strong>LLM Adapter Bridge (Blocks 28–33)</strong><br>Anima text conditioning adapter. Keep locked at 0.0 per official guidelines.`,
    };

    for (const k of groupKeys) {
        const meta = GROUP_META[k];
        const defVal = k === "ADAPTER_Text_Alignment" ? 0.0 : 0.5;
        node.addCustomWidget(
            createSpectralSliderWidget(node, backingMap[k], {
                tag: meta.tag,
                rangeTag: meta.range,
                label: friendlyLabels[k],
                color: meta.color,
                min: -10.0,
                max: 10.0,
                visMin: 0.0,
                visMax: 1.0,
                step: 0.05,
                defaultVal: defVal,
                tooltip: groupTooltips[k],
            })
        );
    }

    if (wRecipe) {
        node.addCustomWidget(
            createStyledTextRowWidget(node, wRecipe, {
                tag: "RECIPE",
                color: HADES_THEME.cyan,
                promptTitle: "Paste 34-Block CSV Vector or Predictive JSON Recipe (Right-click to clear)",
                emptyPlaceholder: "— Manual Sliders Active (Click to paste CSV/JSON) —",
                allowClearOnRightClick: true,
                tooltip: `<strong>CSV / JSON Recipe Vector</strong><br>Paste a custom 34-element vector or predictive recipe to override sliders. Right-click to clear.`,
            })
        );
    }

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        if (info?.size && Array.isArray(info.size) && info.size[0] > 0 && info.size[1] > 0) {
            this.size = [info.size[0], info.size[1]];
        }
        this.setDirtyCanvas?.(true, true);
    };

    const minW = 430;
    const sz = node.computeSize();
    const initW = (Array.isArray(node.size) && node.size[0] > 0) ? Math.max(minW, node.size[0]) : Math.max(minW, sz[0]);
    const initH = (Array.isArray(node.size) && node.size[1] > 0) ? Math.max(node.size[1], sz[1] + 8) : sz[1] + 8;
    node.setSize([initW, initH]);
}

export function setupPredictiveModelMergerUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wPreset = findW("preset");
    const wInf = findW("target_influence");
    const wAgg = findW("aggressiveness");
    const wProtect = findW("protect_llm_adapter");

    [wPreset, wInf, wAgg, wProtect].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.purple,
            getBars() {
                return estimatePredictiveCurve(
                    wPreset?.value,
                    Number(wInf?.value ?? 0.5),
                    Number(wAgg?.value ?? 1.0),
                    wProtect ? Boolean(wProtect.value) : true
                );
            },
            actions: [
                {
                    label: "Soft",
                    tooltip: `<strong>Soft Influence (0.35 / 0.80)</strong><br>Subtle integration of Model 2 with restrained divergence response.`,
                    onClick: () => {
                        setWidgetValue(wInf, 0.35, node);
                        setWidgetValue(wAgg, 0.8, node);
                    },
                },
                {
                    label: "Balanced",
                    tooltip: `<strong>Balanced Influence (0.50 / 1.00)</strong><br>Standard calibrated predictive spectral fusion.`,
                    onClick: () => {
                        setWidgetValue(wInf, 0.5, node);
                        setWidgetValue(wAgg, 1.0, node);
                    },
                },
                {
                    label: "Strong",
                    tooltip: `<strong>Strong Influence (0.75 / 1.35)</strong><br>High Model 2 dominance with pronounced divergence adaptation.`,
                    onClick: () => {
                        setWidgetValue(wInf, 0.75, node);
                        setWidgetValue(wAgg, 1.35, node);
                    },
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "PRESET",
            tagColor: HADES_THEME.purple,
            tooltip: `<strong>Predictive Fusion Heuristic</strong><br>• <strong>Style Transfer:</strong> Retains Model 1 geometry while blending Model 2 style bands.<br>• <strong>Interference Guard:</strong> Dampens high-divergence blocks to prevent burning/artifacts.<br>• <strong>Complementary Richness:</strong> Fills capacity gaps of Model 1 using Model 2.<br>• <strong>Balanced Geometric Fusion:</strong> Harmonic block-by-block spectral integration.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wInf, {
            tag: "INFL",
            rangeTag: "MODEL 2",
            label: "Target Influence",
            color: HADES_THEME.emerald,
            min: 0.0,
            max: 1.0,
            visMin: 0.0,
            visMax: 1.0,
            step: 0.05,
            defaultVal: 0.5,
            tooltip: `<strong>Global Target Influence (Model 2)</strong><br>Baseline weight allocation for Model 2 across the 34 Anima blocks.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wAgg, {
            tag: "AGGR",
            rangeTag: "REACT",
            label: "Spectral Aggressiveness",
            color: HADES_THEME.amber,
            min: 0.1,
            max: 2.5,
            visMin: 0.1,
            visMax: 2.5,
            step: 0.05,
            defaultVal: 1.0,
            tooltip: `<strong>Spectral Aggressiveness Multiplier</strong><br>Sensitivity scaling factor for block-level task-vector divergence and cosine angle warping.`,
        })
    );

    if (wProtect) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wProtect, {
                tag: "BRIDGE",
                color: HADES_THEME.cyan,
                label: "Lock llm_adapter (Model 1 Bridge)",
                onText: "LOCKED (0.0)",
                offText: "ADAPTIVE",
                tooltip: `<strong>Lock llm_adapter Bridge (Blocks 28–33)</strong><br>Ensures Anima's text adapter layers remain locked to Model 1 (0.0) per official guidelines to prevent prompt drift.`,
            })
        );
    }

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        if (info?.size && Array.isArray(info.size) && info.size[0] > 0 && info.size[1] > 0) {
            this.size = [info.size[0], info.size[1]];
        }
        this.setDirtyCanvas?.(true, true);
    };

    const minW = 430;
    const sz = node.computeSize();
    const initW = (Array.isArray(node.size) && node.size[0] > 0) ? Math.max(minW, node.size[0]) : Math.max(minW, sz[0]);
    const initH = (Array.isArray(node.size) && node.size[1] > 0) ? Math.max(node.size[1], sz[1] + 8) : sz[1] + 8;
    node.setSize([initW, initH]);
}

export function setupMultiLoraLoaderUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wEnableAll = findW("enable_all");
    const wPreset = findW("preset");
    const wAgg = findW("aggressiveness");
    const wProtect = findW("protect_llm_adapter");

    [wEnableAll, wPreset, wAgg, wProtect].forEach(hideBackingWidget);

    const slotRefs = [];
    for (let i = 1; i <= 4; i++) {
        const wLora = findW(`lora_${i}`);
        const wOn = findW(`on_${i}`);
        const wStrength = findW(`strength_${i}`);

        hideBackingWidget(wLora);
        hideBackingWidget(wOn);
        hideBackingWidget(wStrength);

        slotRefs.push({ index: i, wLora, wOn, wStrength });
    }

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.emerald,
            tooltip: `<strong>Predictive Subspace Multi-LoRA</strong><br>Concurrent multi-LoRA stacking with dynamic subspace task-vector projection.<br><span style="color:#34d399">• Primary Slot (PRI):</span> Full DiT influence baseline.<br><span style="color:#60a5fa">• Secondary Slots (S2..S4):</span> Dynamically projected onto non-interfering Grassmann angle subspaces.<br><span style="color:#38bdf8">• llm_adapter Lock:</span> Golden rule protection for Anima's 6-layer text conditioner bridge.`,
            actions: [
                {
                    label: "Master",
                    tooltip: `<strong>Master Multi-LoRA Switch</strong><br>Enables or bypasses all 4 LoRA slots globally.`,
                    onClick: () => {
                        if (wEnableAll) setWidgetValue(wEnableAll, !wEnableAll.value, node);
                    },
                },
                {
                    label: "All On",
                    tooltip: `<strong>Enable All Slots</strong><br>Activates all configured LoRA slots simultaneously.`,
                    onClick: () => {
                        if (wEnableAll) setWidgetValue(wEnableAll, true, node);
                        slotRefs.forEach((s) => {
                            if (s.wOn) setWidgetValue(s.wOn, true, node);
                        });
                    },
                },
                {
                    label: "Mute",
                    tooltip: `<strong>Mute All Slots</strong><br>Turns off individual slot toggles while keeping LoRA assignments.`,
                    onClick: () => {
                        slotRefs.forEach((s) => {
                            if (s.wOn) setWidgetValue(s.wOn, false, node);
                        });
                    },
                },
                {
                    label: "1.0x",
                    tooltip: `<strong>Reset Strengths to 1.0x</strong><br>Restores standard 1.00 strength across all slots.`,
                    onClick: () => {
                        slotRefs.forEach((s) => {
                            if (s.wStrength) setWidgetValue(s.wStrength, 1.0, node);
                        });
                    },
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "PRESET",
            tagColor: HADES_THEME.purple,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wAgg, {
            tag: "AGGR",
            rangeTag: "REACT",
            label: "Spectral Aggressiveness",
            color: HADES_THEME.amber,
            min: 0.1,
            max: 2.5,
            visMin: 0.1,
            visMax: 2.5,
            step: 0.05,
            defaultVal: 1.0,
        })
    );

    if (wProtect) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wProtect, {
                tag: "BRIDGE",
                color: HADES_THEME.cyan,
                label: "Strip Dirty llm_adapter LoRA Keys",
                onText: "PROTECTED",
                offText: "ALLOW ALL",
            })
        );
    }

    for (const slot of slotRefs) {
        node.addCustomWidget(
            createSlotRowWidget(node, slot.index, slot.wOn, slot.wLora, slot.wStrength, wEnableAll)
        );
    }

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

export function setupStandaloneLoraMergerUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wLoraA = findW("lora_a");
    const wLoraB = findW("lora_b");
    const wPreset = findW("preset");
    const wMode = findW("merge_mode");
    const wTopo = findW("topology_override");
    const wInfB = findW("target_influence_b");
    const wAgg = findW("aggressiveness");
    const wRank = findW("target_rank");
    const wFile = findW("export_filename");
    const wProtect = findW("protect_llm_adapter");
    const wOverwrite = findW("overwrite_existing");

    [wLoraA, wLoraB, wPreset, wMode, wTopo, wInfB, wAgg, wRank, wFile, wProtect, wOverwrite].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.purple,
            tooltip: `<strong>QR-SVD LoRA Subspace Exporter</strong><br>High-speed mathematical fusion of Donor A and Donor B into a standalone LoRA.<br><span style="color:#c084fc">• QR-Reduced SVD:</span> Orthogonal Grassmann projection maintaining rank fidelity under 6ms.<br><span style="color:#34d399">• Spectrum Topology:</span> Canonical 34-band allocation (0..27 DiT blocks + 28..33 Bridge blocks).<br><span style="color:#38bdf8">• llm_adapter Lock:</span> Locks text conditioner modifications to prevent prompt drift.`,
            getBars() {
                return estimatePredictiveCurve(
                    wPreset?.value,
                    Number(wInfB?.value ?? 0.5),
                    Number(wAgg?.value ?? 1.0),
                    wProtect ? Boolean(wProtect.value) : true
                );
            },
            actions: [
                {
                    label: "R16",
                    tooltip: `<strong>Target Rank 16</strong><br>Compact rank export (~150MB). Ideal for lightweight style/character LoRAs.`,
                    onClick: () => setWidgetValue(wRank, 16, node),
                },
                {
                    label: "R32",
                    tooltip: `<strong>Target Rank 32 (Balanced)</strong><br>Standard high-fidelity export (~300MB). Optimal balance of quality and size.`,
                    onClick: () => setWidgetValue(wRank, 32, node),
                },
                {
                    label: "R64",
                    tooltip: `<strong>Target Rank 64 (Full Depth)</strong><br>Maximum capacity export (~600MB). Captures subtle micro-textures and intricate styles.`,
                    onClick: () => setWidgetValue(wRank, 64, node),
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wLoraA, {
            tag: "DONOR A",
            tagColor: HADES_THEME.blue,
            cleanFileNames: true,
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wLoraB, {
            tag: "DONOR B",
            tagColor: HADES_THEME.purple,
            cleanFileNames: true,
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "PRESET",
            tagColor: HADES_THEME.cyan,
        })
    );

    if (wMode) {
        node.addCustomWidget(
            createStyledComboWidget(node, wMode, {
                tag: "MODE",
                tagColor: HADES_THEME.emerald,
            })
        );
    }

    if (wTopo) {
        node.addCustomWidget(
            createStyledComboWidget(node, wTopo, {
                tag: "TOPO",
                tagColor: HADES_THEME.blue,
            })
        );
    }

    node.addCustomWidget(
        createSpectralSliderWidget(node, wInfB, {
            tag: "INFL B",
            rangeTag: "DONOR B",
            label: "Target Influence B",
            color: HADES_THEME.purple,
            min: 0.0,
            max: 1.0,
            visMin: 0.0,
            visMax: 1.0,
            step: 0.05,
            defaultVal: 0.5,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wAgg, {
            tag: "AGGR",
            rangeTag: "REACT",
            label: "Spectral Aggressiveness",
            color: HADES_THEME.amber,
            min: 0.1,
            max: 2.5,
            visMin: 0.1,
            visMax: 2.5,
            step: 0.05,
            defaultVal: 1.0,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wRank, {
            tag: "RANK",
            rangeTag: "SVD k",
            label: "Target SVD Rank",
            color: HADES_THEME.cyan,
            min: 4,
            max: 128,
            visMin: 4,
            visMax: 128,
            step: 4,
            defaultVal: 32,
            isInt: true,
        })
    );

    if (wProtect) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wProtect, {
                tag: "BRIDGE",
                color: HADES_THEME.cyan,
                label: "Strip Dirty llm_adapter Keys",
                onText: "STRIP",
                offText: "KEEP",
            })
        );
    }

    if (wOverwrite) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wOverwrite, {
                tag: "WRITE",
                color: HADES_THEME.amber,
                label: "Overwrite Existing Output File",
                onText: "OVERWRITE",
                offText: "AUTO-INC",
            })
        );
    }

    node.addCustomWidget(
        createStyledTextRowWidget(node, wFile, {
            tag: "FILE",
            color: HADES_THEME.emerald,
            promptTitle: "Export LoRA Filename (saved in models/loras/)",
            suffix: ".safetensors",
            emptyPlaceholder: "HaDeS_Merged_LoRA",
        })
    );

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

export function setupCLIPBlockMergerUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const clipKeys = [
        "base_ratio",
        "vocab_embeddings",
        "syntax_rigidity",
        "semantic_focus",
        "style_abstraction",
        "final_projections",
    ];

    const backingMap = {};
    for (const k of clipKeys) {
        const w = findW(k);
        hideBackingWidget(w);
        backingMap[k] = w;
    }

    const setAll = (obj) => {
        for (const k of clipKeys) {
            if (backingMap[k] && obj[k] !== undefined) {
                setWidgetValue(backingMap[k], obj[k], node);
            }
        }
    };

    const clipTooltips = {
        base_ratio: `<strong>Global Base Fallback Ratio</strong><br>Interpolation ratio applied to non-block and root text encoder tensors.`,
        vocab_embeddings: `<strong>Vocab & Token Embeddings</strong><br>Blend ratio for word embedding lookups and initial token projections.`,
        syntax_rigidity: `<strong>Syntax & Tag Rigidity</strong><br>Low-level grammar, token positioning, and punctuation sensitivity layers (0–35% depth).`,
        semantic_focus: `<strong>Semantic & Concept Focus</strong><br>Mid-depth layers governing conceptual understanding and subject semantics (35–72% depth).`,
        style_abstraction: `<strong>Style & Mood Abstraction</strong><br>High-depth layers governing artistic atmosphere and aesthetic interpretation (72%+ depth).`,
        final_projections: `<strong>Final Norm Projections</strong><br>Output layernorm and final conditioning projection layers.`,
    };

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.cyan,
            getBars() {
                const bandKeys = [
                    "vocab_embeddings",
                    "syntax_rigidity",
                    "semantic_focus",
                    "style_abstraction",
                    "final_projections",
                ];
                return bandKeys.map((k) => ({
                    ratio: Number(backingMap[k]?.value ?? 0.5),
                    color: CLIP_GROUP_META[k].color,
                }));
            },
            actions: [
                {
                    label: "CLIP 1",
                    tooltip: `<strong>100% CLIP 1</strong><br>Sets all linguistic bands to 0.0 (pure CLIP 1).`,
                    onClick: () => setAll(Object.fromEntries(clipKeys.map((k) => [k, 0.0]))),
                },
                {
                    label: "50/50",
                    tooltip: `<strong>50/50 Balanced Interpolation</strong><br>Even blend across all linguistic bands.`,
                    onClick: () => setAll(Object.fromEntries(clipKeys.map((k) => [k, 0.5]))),
                },
                {
                    label: "A-Synt/B-Styl",
                    tooltip: `<strong>Syntax A / Style B</strong><br>Prioritizes CLIP 1 for syntax/structure and CLIP 2 for aesthetic/style abstraction.`,
                    onClick: () =>
                        setAll({
                            base_ratio: 0.5,
                            vocab_embeddings: 0.15,
                            syntax_rigidity: 0.15,
                            semantic_focus: 0.5,
                            style_abstraction: 0.85,
                            final_projections: 0.75,
                        }),
                },
                {
                    label: "CLIP 2",
                    tooltip: `<strong>100% CLIP 2</strong><br>Sets all linguistic bands to 1.0 (pure CLIP 2).`,
                    onClick: () => setAll(Object.fromEntries(clipKeys.map((k) => [k, 1.0]))),
                },
            ],
        })
    );

    for (const k of clipKeys) {
        const meta = CLIP_GROUP_META[k];
        node.addCustomWidget(
            createSpectralSliderWidget(node, backingMap[k], {
                tag: meta.tag,
                rangeTag: meta.range,
                label: meta.label,
                color: meta.color,
                min: -10.0,
                max: 10.0,
                visMin: 0.0,
                visMax: 1.0,
                step: 0.05,
                defaultVal: 0.5,
                tooltip: clipTooltips[k],
            })
        );
    }

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        if (info?.size && Array.isArray(info.size) && info.size[0] > 0 && info.size[1] > 0) {
            this.size = [info.size[0], info.size[1]];
        }
        this.setDirtyCanvas?.(true, true);
    };

    const minW = 430;
    const sz = node.computeSize();
    const initW = (Array.isArray(node.size) && node.size[0] > 0) ? Math.max(minW, node.size[0]) : Math.max(minW, sz[0]);
    const initH = (Array.isArray(node.size) && node.size[1] > 0) ? Math.max(node.size[1], sz[1] + 8) : sz[1] + 8;
    node.setSize([initW, initH]);
}

export const MERGER_NODE_DISPATCH = {
    HaDeSAnimaModelBlockMerger: setupManualBlockMergerUI,
    HaDeSAnimaPredictiveModelMerger: setupPredictiveModelMergerUI,
    HaDeSAnimaPredictiveMultiLoraLoader: setupMultiLoraLoaderUI,
    HaDeSAnimaStandaloneLoraMerger: setupStandaloneLoraMergerUI,
    HaDeSAnimaCLIPBlockMerger: setupCLIPBlockMergerUI,
};
