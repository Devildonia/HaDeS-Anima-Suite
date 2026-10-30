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
    finalizeHadesNode,
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
                marker: k === "base_ratio" ? null : meta.color,
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
                emptyPlaceholder: "None (sliders active). Click to paste a CSV / JSON recipe",
                allowClearOnRightClick: true,
                tooltip: `<strong>CSV / JSON Recipe Vector</strong><br>Paste a custom 34-element vector or predictive recipe to override sliders. Right-click to clear.`,
            })
        );
    }

    finalizeHadesNode(node, 430);
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
            label: "Aggressiveness",
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
                label: "Keep llm_adapter from Model 1",
                onText: "LOCKED (0.0)",
                offText: "ADAPTIVE",
                tooltip: `<strong>Lock llm_adapter Bridge (Blocks 28–33)</strong><br>Ensures Anima's text adapter layers remain locked to Model 1 (0.0) per official guidelines to prevent prompt drift.`,
            })
        );
    }

    finalizeHadesNode(node, 430);
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
            tooltip: `<strong>Predictive Subspace Multi-LoRA</strong><br>Concurrent multi-LoRA stacking with dynamic subspace task-vector projection.<br><span style="color:#8fb4f0">• Primary Slot (PRI):</span> Full DiT influence baseline.<br><span style="color:#8fb4f0">• Secondary Slots (S2..S4):</span> Dynamically projected onto non-interfering Grassmann angle subspaces.<br><span style="color:#8fb4f0">• llm_adapter Lock:</span> Golden rule protection for Anima's 6-layer text conditioner bridge.`,
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
            tooltip: `<strong>Predictive Multi-LoRA Preset</strong><br>Grassmann subspace allocation strategy (Character + Style Fusion, Style Harmony, or Balanced Fusion).`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wAgg, {
            tag: "AGGR",
            label: "Aggressiveness",
            color: HADES_THEME.amber,
            min: 0.1,
            max: 2.5,
            visMin: 0.1,
            visMax: 2.5,
            step: 0.05,
            defaultVal: 1.0,
            tooltip: `<strong>Subspace Aggressiveness</strong><br>Sensitivity multiplier for singular value truncation and angle steering across secondary slots.`,
        })
    );

    if (wProtect) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wProtect, {
                tag: "BRIDGE",
                color: HADES_THEME.cyan,
                label: "Strip llm_adapter keys from LoRAs",
                onText: "PROTECTED",
                offText: "ALLOW ALL",
                tooltip: `<strong>Protect llm_adapter Bridge</strong><br>Strips secondary text adapter weights from LoRAs to keep Anima's Qwen3 conditioner stable.`,
            })
        );
    }

    for (const slot of slotRefs) {
        node.addCustomWidget(
            createSlotRowWidget(node, slot.index, slot.wOn, slot.wLora, slot.wStrength, wEnableAll)
        );
    }

    finalizeHadesNode(node, 430);
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
            tooltip: `<strong>QR-SVD LoRA Subspace Exporter</strong><br>High-speed mathematical fusion of Donor A and Donor B into a standalone LoRA.<br><span style="color:#8fb4f0">• QR-Reduced SVD:</span> Orthogonal Grassmann projection maintaining rank fidelity under 6ms.<br><span style="color:#8fb4f0">• Spectrum Topology:</span> Canonical 34-band allocation (0..27 DiT blocks + 28..33 Bridge blocks).<br><span style="color:#8fb4f0">• llm_adapter Lock:</span> Locks text conditioner modifications to prevent prompt drift.`,
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
            tooltip: `<strong>Donor LoRA A</strong><br>Primary baseline LoRA model for subspace fusion.`,
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wLoraB, {
            tag: "DONOR B",
            tagColor: HADES_THEME.purple,
            cleanFileNames: true,
            tooltip: `<strong>Donor LoRA B</strong><br>Secondary donor LoRA contributing style or supplementary features.`,
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wPreset, {
            tag: "PRESET",
            tagColor: HADES_THEME.cyan,
            tooltip: `<strong>Merge Preset</strong><br>Grassmann subspace allocation curve across the 34 Anima model layers.`,
        })
    );

    if (wMode) {
        node.addCustomWidget(
            createStyledComboWidget(node, wMode, {
                tag: "MODE",
                tagColor: HADES_THEME.emerald,
                tooltip: `<strong>Merge Mathematical Mode</strong><br>Additive, Orthogonal Task-Vector, or Normalized Subspace projection.`,
            })
        );
    }

    if (wTopo) {
        node.addCustomWidget(
            createStyledComboWidget(node, wTopo, {
                tag: "TOPO",
                tagColor: HADES_THEME.blue,
                tooltip: `<strong>Topology Architecture</strong><br>Target model topology (Auto-Detect, Anima-2.9B 40L, Anima-3.8B 52L).`,
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
            tooltip: `<strong>Target Influence B</strong><br>Weight ratio applied to Donor B in the subspace projection.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wAgg, {
            tag: "AGGR",
            label: "Aggressiveness",
            color: HADES_THEME.amber,
            min: 0.1,
            max: 2.5,
            visMin: 0.1,
            visMax: 2.5,
            step: 0.05,
            defaultVal: 1.0,
            tooltip: `<strong>Spectral Aggressiveness</strong><br>Sensitivity multiplier for singular value truncation and angle steering.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wRank, {
            tag: "RANK",
            label: "Target SVD Rank",
            color: HADES_THEME.cyan,
            min: 4,
            max: 128,
            visMin: 4,
            visMax: 128,
            step: 4,
            defaultVal: 32,
            isInt: true,
            tooltip: `<strong>Target SVD Rank (k)</strong><br>Low-rank truncation dimension for the fused safetensors weights (R16, R32, R64).`,
        })
    );

    if (wProtect) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wProtect, {
                tag: "BRIDGE",
                color: HADES_THEME.cyan,
                label: "Strip llm_adapter keys",
                onText: "STRIP",
                offText: "KEEP",
                tooltip: `<strong>llm_adapter Protection</strong><br>Prevents dirty text conditioner weights from corrupting prompt understanding.`,
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
                tooltip: `<strong>File Overwrite Policy</strong><br>When OFF, automatically appends sequential numbers (e.g. _001) to protect existing files.`,
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
            tooltip: `<strong>Export Filename</strong><br>Name of the merged LoRA file saved to ComfyUI models/loras/.`,
        })
    );

    finalizeHadesNode(node, 430);
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

    finalizeHadesNode(node, 430);
}

export function setupModelSaverUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wFile = findW("filename_prefix");
    const wFormat = findW("save_format");
    const wPrecision = findW("precision");
    const wAdapter = findW("llm_adapter_mode");
    const wMetadata = findW("include_metadata");
    const wAuthor = findW("custom_author");
    const wTitle = findW("custom_model_title");

    [wFile, wFormat, wPrecision, wAdapter, wMetadata, wAuthor, wTitle].forEach(hideBackingWidget);

    const TAG_W = 76;

    // Spectral Banner Widget with HUD and Quick Action Presets
    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.emerald,
            customHeight: 68,
            actions: [
                {
                    label: "BF16 Master",
                    isActive: () => wPrecision?.value === "bfloat16 (Original Master)",
                    tooltip: `<strong>Master Precision (BF16)</strong><br>Saves in full unquantized bfloat16 master format with complete numeric fidelity (~4.8 GB).`,
                    onClick: () => {
                        setWidgetValue(wPrecision, "bfloat16 (Original Master)", node);
                    },
                },
                {
                    label: "ConvRot INT8",
                    isActive: () => wPrecision?.value === "convrot_int8 (Hadamard Rotated INT8)",
                    tooltip: `<strong>ConvRot INT8 (ComfyUI native)</strong><br>Hadamard-rotated per-channel INT8 on the DiT block linears via comfy.quant_ops (~2.3 GB). Loads in the stock Load Diffusion Model node; fast matmul needs an int8-capable GPU.`,
                    onClick: () => {
                        setWidgetValue(wPrecision, "convrot_int8 (Hadamard Rotated INT8)", node);
                    },
                },
                {
                    label: "ConvRot INT4",
                    isActive: () => wPrecision?.value === "convrot_int4 (Hadamard Rotated INT4)",
                    tooltip: `<strong>ConvRot W4A4 (ComfyUI native)</strong><br>Hadamard-rotated 4-bit weights on the DiT block linears via comfy.quant_ops (~1.4 GB). Noticeably lossier than INT8.`,
                    onClick: () => {
                        setWidgetValue(wPrecision, "convrot_int4 (Hadamard Rotated INT4)", node);
                    },
                },
                {
                    label: "Diffusers HF",
                    isActive: () => wFormat?.value === "Diffusers (HuggingFace / Trainer)",
                    tooltip: `<strong>Hugging Face Diffusers Format</strong><br>Applies 24-rule inverse conversion to transformer_blocks.*, attn1.*, and text_conditioner.* for training scripts and Hugging Face distribution.`,
                    onClick: () => {
                        setWidgetValue(wFormat, "Diffusers (HuggingFace / Trainer)", node);
                    },
                },
            ],
            drawCustomHUD(ctx, rect) {
                const isDiffusers = String(wFormat?.value || "").includes("Diffusers");
                const prec = String(wPrecision?.value || "bfloat16");
                const isFp8 = prec.includes("fp8");
                const hasMeta = wMetadata ? Boolean(wMetadata.value) : true;
                const adapterMode = String(wAdapter?.value || "Embed");

                const isNarrow = rect.w < 350;
                const maxRowW = Math.max(60, rect.w - 16);

                ctx.font = isNarrow ? "600 8.5px ui-monospace, Consolas, monospace" : "600 9px ui-monospace, Consolas, monospace";
                ctx.textBaseline = "middle";

                const badgeY = rect.y + 11;

                // --- Row 1: Format & Precision ---
                let fmtText = isDiffusers
                    ? (isNarrow ? "DIFFUSERS (HF)" : "FORMAT: DIFFUSERS (HF)")
                    : (isNarrow ? "COMFY NATIVE" : "FORMAT: COMFY NATIVE (DIT)");

                let precLabel = "BF16 (~4.8GB)";
                let precColor = HADES_THEME.cyan;
                if (prec.includes("convrot_int8")) {
                    precLabel = isNarrow ? "CR-INT8 (~2.3GB)" : "CONVROT INT8 (~2.3GB)";
                    precColor = HADES_THEME.emerald;
                } else if (prec.includes("convrot_int4")) {
                    precLabel = isNarrow ? "CR-INT4 (~1.4GB)" : "CONVROT INT4 (~1.4GB)";
                    precColor = HADES_THEME.amber;
                } else if (prec.includes("float32")) {
                    precLabel = "FP32 (~9.6GB)";
                    precColor = HADES_THEME.amber;
                } else if (isFp8) {
                    precLabel = isNarrow ? "FP8 (~2.6GB)" : "FP8 E4M3 (~2.6GB)";
                    precColor = HADES_THEME.blue;
                } else if (isNarrow) {
                    precLabel = "BF16 (~4.8GB)";
                }

                // Prevent Row 1 horizontal overlap on very small widths
                if (ctx.measureText(fmtText).width + ctx.measureText(precLabel).width > maxRowW) {
                    fmtText = isDiffusers ? "DIFFUSERS" : "COMFY DIT";
                    precLabel = prec.includes("convrot_int8") ? "INT8" : (prec.includes("convrot_int4") ? "INT4" : (prec.includes("float32") ? "FP32" : (isFp8 ? "FP8" : "BF16")));
                }
                if (ctx.measureText(fmtText).width + ctx.measureText(precLabel).width > maxRowW) {
                    fmtText = isDiffusers ? "DIFF" : "DIT";
                }

                ctx.textAlign = "left";
                ctx.fillStyle = isDiffusers ? HADES_THEME.purple : HADES_THEME.emerald;
                ctx.fillText(fmtText, rect.x + 8, badgeY);

                ctx.textAlign = "right";
                ctx.fillStyle = precColor;
                ctx.fillText(precLabel, rect.x + rect.w - 8, badgeY);

                // --- Row 2: Bridge Status & Metadata ---
                const subY = rect.y + 25;

                let bridgeStr = "BRIDGE: EMBEDDED";
                if (adapterMode.includes("Strip")) {
                    bridgeStr = isNarrow ? "BRIDGE: STRIP" : "BRIDGE: STRIPPED (PURE DIT)";
                } else if (adapterMode.includes("Separate")) {
                    bridgeStr = isNarrow ? "BRIDGE: EXPORT" : "BRIDGE: EXPORT SEPARATE";
                }

                let metaStr = hasMeta
                    ? (isNarrow ? "META: ACTIVE" : "METADATA: MODELSPEC ACTIVE")
                    : (isNarrow ? "META: OFF" : "METADATA: BARE TENSORS (OFF)");

                // Prevent Row 2 horizontal overlap on very small widths
                if (ctx.measureText(bridgeStr).width + ctx.measureText(metaStr).width > maxRowW) {
                    bridgeStr = adapterMode.includes("Strip") ? "STRIP" : (adapterMode.includes("Separate") ? "EXPORT" : "EMBED");
                    metaStr = hasMeta ? "META: ON" : "META: OFF";
                }

                ctx.textAlign = "left";
                ctx.fillStyle = HADES_THEME.textMuted;
                ctx.fillText(bridgeStr, rect.x + 8, subY);

                ctx.textAlign = "right";
                ctx.fillStyle = hasMeta ? HADES_THEME.cyan : HADES_THEME.textMuted;
                ctx.fillText(metaStr, rect.x + rect.w - 8, subY);
            },
        })
    );

    // Styled Widgets
    if (wFile) {
        node.addCustomWidget(
            createStyledTextRowWidget(node, wFile, {
                tag: "OUTPUT",
                color: HADES_THEME.cyan,
                tagWidth: TAG_W,
                suffix: ".safetensors",
                promptTitle: "Enter Model Output Subfolder and Filename Prefix",
                tooltip: `<strong>Export Checkpoint Path</strong><br>Saved under ComfyUI/output/ (e.g. output/diffusion_models/...) with automatic numeric incrementing. Move it to models/diffusion_models/ to load it.`,
            })
        );
    }

    if (wFormat) {
        node.addCustomWidget(
            createStyledComboWidget(node, wFormat, {
                tag: "FORMAT",
                tagColor: HADES_THEME.purple,
                tagWidth: TAG_W,
                label: "",
                tooltip: `<strong>Output Checkpoint Format</strong><br>• <strong>Native ComfyUI (MiniTrainDIT):</strong> Native blocks naming convention.<br>• <strong>Diffusers (HuggingFace / Trainer):</strong> Converts keys back to transformer_blocks.* and text_conditioner.* for OneTrainer, Musubi, Kohya, or Hugging Face.`,
            })
        );
    }

    if (wPrecision) {
        node.addCustomWidget(
            createStyledComboWidget(node, wPrecision, {
                tag: "PRECISION",
                tagColor: HADES_THEME.emerald,
                tagWidth: TAG_W,
                label: "",
                tooltip: `<strong>Precision & Quantization Engine</strong><br>• <strong>bfloat16:</strong> Full master fidelity (~4.8 GB).<br>• <strong>fp8_e4m3fn:</strong> ComfyUI native float8 on linear weights (~2.6 GB).<br>• <strong>convrot_int8:</strong> Native ComfyUI ConvRot INT8 on DiT block linears (~2.3 GB).<br>• <strong>convrot_int4:</strong> Native ComfyUI ConvRot W4A4 (~1.4 GB, lossier).<br>• <strong>float32:</strong> Uncompressed.`,
            })
        );
    }

    if (wAdapter) {
        node.addCustomWidget(
            createStyledComboWidget(node, wAdapter, {
                tag: "BRIDGE",
                tagColor: HADES_THEME.blue,
                tagWidth: TAG_W,
                label: "",
                tooltip: `<strong>AnimaTextConditioner (llm_adapter) Handling</strong><br>• <strong>Embed:</strong> Keeps full 6-layer bridge in the checkpoint.<br>• <strong>Strip:</strong> Drops bridge weights for pure DiT.<br>• <strong>Export Separately:</strong> Saves companion _adapter.safetensors.`,
            })
        );
    }

    if (wMetadata) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wMetadata, {
                tag: "METADATA",
                color: HADES_THEME.cyan,
                tagWidth: TAG_W,
                label: "SAI ModelSpec Header",
                onText: "ACTIVE",
                offText: "DISABLED",
                tooltip: `<strong>Safetensors Metadata Injection</strong><br>When active, injects modelspec.architecture ("anima/mini_train_dit"), topology (28L/40L/52L), and author details into the JSON header. Turn OFF for bare unlabelled tensors.`,
            })
        );
    }

    if (wAuthor) {
        node.addCustomWidget(
            createStyledTextRowWidget(node, wAuthor, {
                tag: "AUTHOR",
                color: HADES_THEME.purple,
                tagWidth: TAG_W,
                promptTitle: "Enter Author / Creator Name",
                emptyPlaceholder: "HaDeS",
                tooltip: `<strong>Model Author / Creator</strong><br>Embedded into modelspec.author in the safetensors header.`,
            })
        );
    }

    if (wTitle) {
        node.addCustomWidget(
            createStyledTextRowWidget(node, wTitle, {
                tag: "TITLE",
                color: HADES_THEME.amber,
                tagWidth: TAG_W,
                promptTitle: "Enter Custom Model Display Title",
                emptyPlaceholder: "Leave empty to use filename",
                tooltip: `<strong>Model Display Title</strong><br>Embedded into modelspec.title in the safetensors header.`,
            })
        );
    }

    finalizeHadesNode(node, 390);
}

export const MERGER_NODE_DISPATCH = {
    HaDeSAnimaModelBlockMerger: setupManualBlockMergerUI,
    HaDeSAnimaPredictiveModelMerger: setupPredictiveModelMergerUI,
    HaDeSAnimaPredictiveMultiLoraLoader: setupMultiLoraLoaderUI,
    HaDeSAnimaStandaloneLoraMerger: setupStandaloneLoraMergerUI,
    HaDeSAnimaCLIPBlockMerger: setupCLIPBlockMergerUI,
    HaDeSAnimaModelSaver: setupModelSaverUI,
};
