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
                    label: "Late B",
                    tooltip: `<strong>Model 2 on late blocks</strong><br>0.10–0.15 on blocks 0–14, 0.65–0.90 on blocks 15–27. Mostly keeps Model 1: measured, it does not isolate style from structure.`,
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
        BLOCKS_Layout_Geometry: "Early Blocks",
        BLOCKS_Global_Composition: "Early-Mid Blocks",
        BLOCKS_Subject_Identity: "Core Blocks (strongest)",
        BLOCKS_Material_Substance: "Mid-Late Blocks",
        BLOCKS_Art_Style_Medium: "Late Blocks",
        BLOCKS_Lighting_Sharpness: "Final Blocks (weakest)",
        ADAPTER_Text_Alignment: "LLM Adapter Bridge (0=Lock)",
    };

    const groupTooltips = {
        base_ratio: `<strong>Global Base Fallback Ratio</strong><br>Ratio applied to root embeddings, timesteps, and non-block parameters.`,
        BLOCKS_Layout_Geometry: `<strong>Early Blocks (0–4)</strong><br>Moves composition and texture together: 0.22–0.84 of the change. Measured share of an A→B change, see tools/ABLATION_RESULTS.md.`,
        BLOCKS_Global_Composition: `<strong>Early-Mid Blocks (5–9)</strong><br>Moves composition and texture together: 0.18–0.57 of the change. Measured share of an A→B change, see tools/ABLATION_RESULTS.md.`,
        BLOCKS_Subject_Identity: `<strong>Core Blocks (10–14)</strong><br>Strongest group for composition, texture and color alike: 0.44–0.75 of the change. Measured share of an A→B change, see tools/ABLATION_RESULTS.md.`,
        BLOCKS_Material_Substance: `<strong>Mid-Late Blocks (15–19)</strong><br>Second strongest group: 0.38–0.62 of the change. Measured share of an A→B change, see tools/ABLATION_RESULTS.md.`,
        BLOCKS_Art_Style_Medium: `<strong>Late Blocks (20–23)</strong><br>Low influence, including on texture and color: 0.19–0.38 of the change. Measured share of an A→B change, see tools/ABLATION_RESULTS.md.`,
        BLOCKS_Lighting_Sharpness: `<strong>Final Blocks (24–27)</strong><br>Weakest DiT group: 0.12–0.32 of the change. Measured share of an A→B change, see tools/ABLATION_RESULTS.md.`,
        ADAPTER_Text_Alignment: `<strong>LLM Adapter Bridge (Blocks 28–33)</strong><br>Anima text conditioning adapter (0.0 = Model 1). Identical in most finetunes, but anima-aesthetic-v1.1 retrained it.`,
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
                promptTitle: "Paste 34-Block CSV Vector or Predictive JSON Recipe (empty clears)",
                emptyPlaceholder: "None (sliders active). Click to paste a CSV / JSON recipe",
                tooltip: `<strong>CSV / JSON Recipe Vector</strong><br>Paste a custom 34-element vector or predictive recipe to override sliders. Submit an empty value to clear.`,
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
            displayNames: { "Style Transfer (Keep A identity, B style)": "Conservative (mostly Model 1)" },
            tooltip: `<strong>Predictive Fusion Heuristic</strong><br>Per-block weights from weight divergence. Measured, the presets act mainly as overall Model 2 strength, weakest to strongest:<br>• <strong>Conservative:</strong> mostly Model 1 (saved as "Style Transfer"). A merge cannot keep Model 1's composition with Model 2's style in Anima; for Model 2 detail on Model 1's composition use the Dual-Pass KSampler with model_pass2 at low denoise.<br>• <strong>Interference Guard / Balanced Geometric Fusion:</strong> a little below a plain blend at the same influence.<br>• <strong>Complementary Richness:</strong> above a plain blend.<br>See tools/ABLATION_RESULTS.md.`,
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
            displayNames: { "Character + Style Fusion (A identity, B style)": "Favor Slot 1 (PRI)" },
            tooltip: `<strong>Predictive Multi-LoRA Preset</strong><br>Per-block slot weights from LoRA overlap. Measured with two style LoRAs:<br>• <strong>Favor Slot 1:</strong> moves the result slightly towards the PRI LoRA in composition and style alike (saved as "Character + Style Fusion"); it does not separate identity from style.<br>• <strong>Anti-Overcook / Orthogonal Synthesis / Balanced 50/50:</strong> within ±0.03 of stacking the LoRAs at 1.0.<br>See tools/ABLATION_RESULTS.md.`,
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
            displayNames: { "Character + Style Fusion (A identity, B style)": "Favor Donor A" },
            tooltip: `<strong>Merge Preset</strong><br>Per-block donor weights from LoRA overlap, the same formula family as the Multi-LoRA presets. Measured there, "Character + Style Fusion" (shown as Favor Donor A) leans towards the first LoRA without separating identity from style, and the others behave close to plain stacking. See tools/ABLATION_RESULTS.md.`,
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

export function setupLoraExtractorUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wRank = findW("rank");
    const wFile = findW("export_filename");
    const wAdapter = findW("include_llm_adapter");
    const wOverwrite = findW("overwrite_existing");

    [wRank, wFile, wAdapter, wOverwrite].forEach(hideBackingWidget);

    const rankAction = (r, hint) => ({
        label: `R${r}`,
        isActive: () => Number(wRank?.value) === r,
        tooltip: `<strong>Rank ${r}</strong><br>${hint} The info output reports how much of the difference the rank keeps.`,
        onClick: () => setWidgetValue(wRank, r, node),
    });

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.purple,
            tooltip: `<strong>LoRA Extractor</strong><br>Saves (tuned − base) as a low-rank LoRA in models/loras/: SVD per linear layer, full diffs for norms and biases, identical layers skipped. Merge and LoRA patches on either input are included; quantized inputs are rejected.`,
            actions: [
                rankAction(16, "Smallest file; keeps the broad change."),
                rankAction(32, "Default."),
                rankAction(64, "Closer to the tuned model."),
                rankAction(128, "Largest file; for merges far from the base."),
            ],
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wRank, {
            tag: "RANK",
            label: "LoRA Rank",
            color: HADES_THEME.purple,
            min: 4,
            max: 256,
            visMin: 4,
            visMax: 256,
            step: 4,
            defaultVal: 32,
            isInt: true,
            tooltip: `<strong>LoRA Rank</strong><br>Higher keeps more of the difference and makes a larger file.`,
        })
    );

    if (wAdapter) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wAdapter, {
                tag: "BRIDGE",
                color: HADES_THEME.cyan,
                label: "Extract llm_adapter differences",
                onText: "EXTRACT",
                offText: "BASE",
                tooltip: `<strong>llm_adapter</strong><br>Off keeps the base model's adapter. Turn on for releases that retrained it (e.g. anima-aesthetic-v1.1).`,
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
                tooltip: `<strong>File Overwrite Policy</strong><br>When OFF, appends _001, _002… instead of replacing an existing file.`,
            })
        );
    }

    node.addCustomWidget(
        createStyledTextRowWidget(node, wFile, {
            tag: "FILE",
            color: HADES_THEME.emerald,
            promptTitle: "Export LoRA Filename (saved in models/loras/)",
            suffix: ".safetensors",
            emptyPlaceholder: "HaDeS_Extracted_LoRA",
            tooltip: `<strong>Export Filename</strong><br>Name of the extracted LoRA file in ComfyUI models/loras/.`,
        })
    );

    finalizeHadesNode(node, 400);
}

const LORA_BLOCK_GROUPS = [
    { key: "blocks_00_04", meta: "BLOCKS_Layout_Geometry", label: "Early Blocks" },
    { key: "blocks_05_09", meta: "BLOCKS_Global_Composition", label: "Early-Mid Blocks" },
    { key: "blocks_10_14", meta: "BLOCKS_Subject_Identity", label: "Core Blocks" },
    { key: "blocks_15_19", meta: "BLOCKS_Material_Substance", label: "Mid-Late Blocks" },
    { key: "blocks_20_23", meta: "BLOCKS_Art_Style_Medium", label: "Late Blocks" },
    { key: "blocks_24_27", meta: "BLOCKS_Lighting_Sharpness", label: "Final Blocks" },
];

export function setupLoraBlockWeightUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wLora = findW("lora_name");
    const wStrength = findW("strength_model");
    const wClip = findW("strength_clip");
    const wOther = findW("other_layers");
    const wProtect = findW("protect_llm_adapter");
    const groupWidgets = LORA_BLOCK_GROUPS.map((g) => findW(g.key));

    [wLora, wStrength, wClip, wOther, wProtect, ...groupWidgets].forEach(hideBackingWidget);

    const setWeights = (values) => {
        groupWidgets.forEach((w, i) => setWidgetValue(w, values[i], node));
        setWidgetValue(wOther, values[6], node);
    };

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.purple,
            tooltip: `<strong>LoRA Block Weight</strong><br>Applies one LoRA with a separate multiplier per DiT depth group (canonical 28-block index; 40L / 52L models are mapped proportionally).<br>Where a LoRA acts differs per LoRA (measured: one style LoRA mostly in blocks 05–14, another in 00–04 and 20–23), so there are no fixed "style only" presets. Measure a LoRA with <code>tools/anima_ablation.py --mode lora</code>. Blocks 24–27 carried the least change for every model and LoRA tested.`,
            getBars() {
                const bars = [];
                LORA_BLOCK_GROUPS.forEach((g, i) => {
                    const meta = GROUP_META[g.meta];
                    for (let k = 0; k < meta.indices.length; k++) {
                        bars.push({ ratio: Math.max(0.03, Math.min(1.0, Number(groupWidgets[i]?.value ?? 1) / 2)), color: meta.color });
                    }
                });
                return bars;
            },
            actions: [
                {
                    label: "1.0",
                    isActive: () => groupWidgets.every((w) => Number(w?.value) === 1) && Number(wOther?.value) === 1,
                    tooltip: `<strong>Uniform</strong><br>Every group at 1.0 (same as a plain LoRA loader).`,
                    onClick: () => setWeights([1, 1, 1, 1, 1, 1, 1]),
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wLora, {
            tag: "LORA",
            tagColor: HADES_THEME.purple,
            cleanFileNames: true,
            tooltip: `<strong>LoRA</strong><br>Diffusers, Kohya, OneTrainer and Musubi keys are remapped to native Anima keys.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wStrength, {
            tag: "MODEL",
            label: "Model Strength",
            color: HADES_THEME.purple,
            min: -10.0,
            max: 10.0,
            visMin: 0.0,
            visMax: 2.0,
            step: 0.05,
            defaultVal: 1.0,
            tooltip: `<strong>Model Strength</strong><br>Overall LoRA strength; each group multiplies it.`,
        })
    );

    LORA_BLOCK_GROUPS.forEach((g, i) => {
        const meta = GROUP_META[g.meta];
        node.addCustomWidget(
            createSpectralSliderWidget(node, groupWidgets[i], {
                tag: meta.tag,
                rangeTag: meta.range,
                label: g.label,
                color: meta.color,
                marker: meta.color,
                min: -2.0,
                max: 3.0,
                visMin: 0.0,
                visMax: 2.0,
                step: 0.05,
                defaultVal: 1.0,
                tooltip: `<strong>${g.label} (${meta.range})</strong><br>Multiplier for the LoRA on these blocks.`,
            })
        );
    });

    node.addCustomWidget(
        createSpectralSliderWidget(node, wOther, {
            tag: "OTHER",
            label: "Other Layers",
            color: HADES_THEME.textSecondary,
            min: -2.0,
            max: 3.0,
            visMin: 0.0,
            visMax: 2.0,
            step: 0.05,
            defaultVal: 1.0,
            tooltip: `<strong>Other Layers</strong><br>Embedders, final layer and llm_adapter (when not protected).`,
        })
    );

    if (wClip) {
        node.addCustomWidget(
            createSpectralSliderWidget(node, wClip, {
                tag: "CLIP",
                label: "CLIP Strength",
                color: HADES_THEME.cyan,
                min: -10.0,
                max: 10.0,
                visMin: 0.0,
                visMax: 2.0,
                step: 0.05,
                defaultVal: 1.0,
                tooltip: `<strong>CLIP Strength</strong><br>Strength for text-encoder LoRA layers, when the LoRA has them and a CLIP is connected.`,
            })
        );
    }

    if (wProtect) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wProtect, {
                tag: "BRIDGE",
                color: HADES_THEME.cyan,
                label: "Strip llm_adapter keys",
                onText: "STRIP",
                offText: "KEEP",
                tooltip: `<strong>llm_adapter</strong><br>Removes llm_adapter weights from the LoRA (llm_adapter_lr = 0 rule).`,
            })
        );
    }

    finalizeHadesNode(node, 430);
}

export function setupLoraResizeUI(node) {
    applyNodeFrameStyle(node);
    const findW = (name) => node.widgets?.find((w) => w.name === name);

    const wLora = findW("lora_name");
    const wRank = findW("target_rank");
    const wFile = findW("export_filename");
    const wDynamic = findW("dynamic_energy");
    const wOverwrite = findW("overwrite_existing");

    [wLora, wRank, wFile, wDynamic, wOverwrite].forEach(hideBackingWidget);

    const rankAction = (r) => ({
        label: `R${r}`,
        isActive: () => Number(wRank?.value) === r && Number(wDynamic?.value ?? 0) === 0,
        tooltip: `<strong>Fixed rank ${r}</strong><br>Every layer above rank ${r} is reduced to ${r}.`,
        onClick: () => {
            setWidgetValue(wRank, r, node);
            setWidgetValue(wDynamic, 0.0, node);
        },
    });

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.purple,
            tooltip: `<strong>LoRA Resize</strong><br>Lowers a LoRA's rank with an exact SVD of its low-rank factors and saves a new file in models/loras/ (the source is never overwritten). Norm / bias diffs and unsupported formats (LoHa, LoKr, DoRA) are copied unchanged. The info output reports the energy kept and the size before / after.`,
            actions: [
                rankAction(8),
                rankAction(16),
                rankAction(32),
                {
                    label: "95% Dyn",
                    isActive: () => Number(wDynamic?.value) === 0.95,
                    tooltip: `<strong>Dynamic 95%</strong><br>Each layer keeps the smallest rank that holds 95% of its energy, up to the target rank.`,
                    onClick: () => setWidgetValue(wDynamic, 0.95, node),
                },
            ],
        })
    );

    node.addCustomWidget(
        createStyledComboWidget(node, wLora, {
            tag: "LORA",
            tagColor: HADES_THEME.purple,
            cleanFileNames: true,
            tooltip: `<strong>Source LoRA</strong><br>Diffusers, Kohya, OneTrainer and Musubi keys are remapped to native Anima keys in the output.`,
        })
    );

    node.addCustomWidget(
        createSpectralSliderWidget(node, wRank, {
            tag: "RANK",
            label: "Target Rank (max)",
            color: HADES_THEME.purple,
            min: 1,
            max: 256,
            visMin: 1,
            visMax: 128,
            step: 1,
            defaultVal: 32,
            isInt: true,
            tooltip: `<strong>Target Rank</strong><br>Maximum rank per layer; layers already at or below it are copied unchanged.`,
        })
    );

    if (wDynamic) {
        node.addCustomWidget(
            createSpectralSliderWidget(node, wDynamic, {
                tag: "DYN",
                label: "Dynamic Energy (0 = off)",
                color: HADES_THEME.cyan,
                min: 0.0,
                max: 1.0,
                visMin: 0.0,
                visMax: 1.0,
                step: 0.005,
                defaultVal: 0.0,
                decimals: 3,
                zeroLabel: "OFF",
                tooltip: `<strong>Dynamic Energy</strong><br>0 = fixed rank. Otherwise each layer gets the smallest rank that keeps this share of its energy, capped at the target rank.`,
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
                tooltip: `<strong>File Overwrite Policy</strong><br>When OFF, appends _001, _002… instead of replacing an existing file. The source LoRA is never overwritten.`,
            })
        );
    }

    node.addCustomWidget(
        createStyledTextRowWidget(node, wFile, {
            tag: "FILE",
            color: HADES_THEME.emerald,
            promptTitle: "Export LoRA Filename (saved in models/loras/)",
            suffix: ".safetensors",
            emptyPlaceholder: "HaDeS_Resized_LoRA",
            tooltip: `<strong>Export Filename</strong><br>Name of the resized LoRA file in ComfyUI models/loras/.`,
        })
    );

    finalizeHadesNode(node, 400);
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
    HaDeSAnimaLoraExtractor: setupLoraExtractorUI,
    HaDeSAnimaLoraBlockWeight: setupLoraBlockWeightUI,
    HaDeSAnimaLoraResize: setupLoraResizeUI,
};
