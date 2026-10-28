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
            title: "34-BLOCK SPECTRUM (28/40/52L + BRIDGE)",
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
                    onClick: () => setAllGroups(Object.fromEntries(groupKeys.map((k) => [k, 0.0]))),
                },
                {
                    label: "50/50 (Lock Brg)",
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
            })
        );
    }

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
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
            title: "PREDICTIVE PROFILE (28/40/52L)",
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
                    onClick: () => {
                        setWidgetValue(wInf, 0.35, node);
                        setWidgetValue(wAgg, 0.8, node);
                    },
                },
                {
                    label: "Balanced",
                    onClick: () => {
                        setWidgetValue(wInf, 0.5, node);
                        setWidgetValue(wAgg, 1.0, node);
                    },
                },
                {
                    label: "Strong",
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
                label: "Lock llm_adapter (Model 1 Bridge)",
                onText: "LOCKED (0.0)",
                offText: "ADAPTIVE",
            })
        );
    }

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
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
            title: "SPECTRAL MULTI-LORA",
            accentColor: HADES_THEME.emerald,
            getStatusColor() {
                const masterOn = wEnableAll ? Boolean(wEnableAll.value) : true;
                if (!masterOn) return HADES_THEME.rose;
                const activeCount = slotRefs.filter(
                    (s) => s.wOn?.value && s.wLora?.value && s.wLora.value !== "None" && Number(s.wStrength?.value) !== 0
                ).length;
                return activeCount > 0 ? HADES_THEME.emerald : HADES_THEME.textMuted;
            },
            getStatusText() {
                const masterOn = wEnableAll ? Boolean(wEnableAll.value) : true;
                if (!masterOn) return "SPECTRAL MULTI-LORA (BYPASSED)";
                const activeCount = slotRefs.filter(
                    (s) => s.wOn?.value && s.wLora?.value && s.wLora.value !== "None" && Number(s.wStrength?.value) !== 0
                ).length;
                return `SPECTRAL SLOTS (${activeCount}/4 ACTIVE)`;
            },
            actions: [
                {
                    label: "Master",
                    onClick: () => {
                        if (wEnableAll) setWidgetValue(wEnableAll, !wEnableAll.value, node);
                    },
                },
                {
                    label: "All On",
                    onClick: () => {
                        if (wEnableAll) setWidgetValue(wEnableAll, true, node);
                        slotRefs.forEach((s) => {
                            if (s.wOn) setWidgetValue(s.wOn, true, node);
                        });
                    },
                },
                {
                    label: "Mute",
                    onClick: () => {
                        slotRefs.forEach((s) => {
                            if (s.wOn) setWidgetValue(s.wOn, false, node);
                        });
                    },
                },
                {
                    label: "1.0x",
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
            title: "QR-SVD LORA EXPORTER",
            accentColor: HADES_THEME.purple,
            getBars() {
                return estimatePredictiveCurve(
                    wPreset?.value,
                    Number(wInfB?.value ?? 0.5),
                    Number(wAgg?.value ?? 1.0),
                    wProtect ? Boolean(wProtect.value) : true
                );
            },
            actions: [
                { label: "R16", onClick: () => setWidgetValue(wRank, 16, node) },
                { label: "R32", onClick: () => setWidgetValue(wRank, 32, node) },
                { label: "R64", onClick: () => setWidgetValue(wRank, 64, node) },
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

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "QWEN3 LINGUISTIC BANDS (28L / 36L)",
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
                    onClick: () => setAll(Object.fromEntries(clipKeys.map((k) => [k, 0.0]))),
                },
                {
                    label: "50/50",
                    onClick: () => setAll(Object.fromEntries(clipKeys.map((k) => [k, 0.5]))),
                },
                {
                    label: "A-Synt/B-Styl",
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
            })
        );
    }

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

export const MERGER_NODE_DISPATCH = {
    HaDeSAnimaModelBlockMerger: setupManualBlockMergerUI,
    HaDeSAnimaPredictiveModelMerger: setupPredictiveModelMergerUI,
    HaDeSAnimaPredictiveMultiLoraLoader: setupMultiLoraLoaderUI,
    HaDeSAnimaStandaloneLoraMerger: setupStandaloneLoraMergerUI,
    HaDeSAnimaCLIPBlockMerger: setupCLIPBlockMergerUI,
};
