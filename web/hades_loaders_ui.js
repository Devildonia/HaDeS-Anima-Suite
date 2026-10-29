import {
    HADES_THEME,
    formatCleanLabel,
    setWidgetValue,
    drawRoundedRect,
    hideBackingWidget,
    applyNodeFrameStyle,
    createSpectralBannerWidget,
    createSpectralSliderWidget,
    createStyledComboWidget,
    createStyledToggleWidget,
} from "./hades_widgets.js";

// =============================================================================
// HADES ANIMA LOADERS & SMART VAE DECODE UI (NODES 14 - 17)
// =============================================================================

function drawPill(ctx, x, y, text, color) {
    ctx.font = "bold 9px monospace";
    const tw = ctx.measureText(text).width + 12;
    drawRoundedRect(ctx, x, y, tw, 16, 4);
    ctx.fillStyle = `${color}22`;
    ctx.fill();
    ctx.strokeStyle = `${color}77`;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + tw * 0.5, y + 8.5);
    return tw + 5;
}

export function setupAnimaModelLoaderUI(node) {
    applyNodeFrameStyle(node);
    const wUnet = node.widgets?.find((w) => w.name === "unet_name");
    const wDtype = node.widgets?.find((w) => w.name === "weight_dtype");
    const wShift = node.widgets?.find((w) => w.name === "flow_shift");
    const wRemap = node.widgets?.find((w) => w.name === "auto_remap_diffusers");
    const wAdapter = node.widgets?.find((w) => w.name === "external_llm_adapter");

    [wUnet, wDtype, wShift, wRemap, wAdapter].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "ANIMA DIFFUSION MODEL LOADER",
            accentColor: HADES_THEME.blue,
            customHeight: 58,
            getStatusText: () => `ANIMA DIT LOADER • ${formatCleanLabel(wUnet?.value, "Select Checkpoint")}`,
            getStatusColor: () => (wRemap?.value ? HADES_THEME.emerald : HADES_THEME.blue),
            actions: [
                {
                    label: "Reset 3.0",
                    onClick: (n) => {
                        setWidgetValue(wShift, 3.0, n);
                        setWidgetValue(wDtype, "default", n);
                        setWidgetValue(wRemap, true, n);
                    },
                },
            ],
            drawCustomHUD: (ctx, rect) => {
                let cx = rect.x + 6;
                const cy = rect.y + (rect.h - 16) * 0.5;
                cx += drawPill(ctx, cx, cy, `DTYPE: ${String(wDtype?.value || "default").toUpperCase()}`, HADES_THEME.cyan);
                cx += drawPill(ctx, cx, cy, `SHIFT: ${Number(wShift?.value ?? 3.0).toFixed(2)} (mult=1.0)`, HADES_THEME.blue);
                cx += drawPill(
                    ctx,
                    cx,
                    cy,
                    wRemap?.value ? "24-RULE DIFFUSERS AUTO-REMAP" : "RAW KEYS",
                    wRemap?.value ? HADES_THEME.emerald : HADES_THEME.amber
                );
                if (wAdapter?.value && wAdapter.value !== "None") {
                    drawPill(ctx, cx, cy, "EXT ADAPTER INJECT", HADES_THEME.purple);
                }
            },
        })
    );

    if (wUnet) {
        node.addCustomWidget(
            createStyledComboWidget(node, wUnet, {
                tag: "MODEL",
                tagColor: HADES_THEME.blue,
                label: "Anima Checkpoint",
                cleanFileNames: true,
            })
        );
    }
    if (wDtype) {
        node.addCustomWidget(
            createStyledComboWidget(node, wDtype, {
                tag: "DTYPE",
                tagColor: HADES_THEME.cyan,
                label: "Weight Precision",
            })
        );
    }
    if (wShift) {
        node.addCustomWidget(
            createSpectralSliderWidget(node, wShift, {
                tag: "SHFT",
                rangeTag: "M=1.0",
                label: "Rectified Flow Shift (multiplier=1.0)",
                color: HADES_THEME.blue,
                min: 0.5,
                max: 10.0,
                visMin: 1.0,
                visMax: 6.0,
                step: 0.05,
                defaultVal: 3.0,
            })
        );
    }
    if (wRemap) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wRemap, {
                tag: "CONV",
                color: HADES_THEME.emerald,
                label: "Auto-Convert Diffusers Keys (24 Rules)",
                onText: "AUTO REMAP",
                offText: "PASS-THROUGH",
            })
        );
    }
    if (wAdapter) {
        node.addCustomWidget(
            createStyledComboWidget(node, wAdapter, {
                tag: "ADPT",
                tagColor: HADES_THEME.purple,
                label: "External llm_adapter (Optional / 3.8B)",
                cleanFileNames: true,
            })
        );
    }

    const sz = node.computeSize();
    node.setSize([Math.max(430, sz[0]), sz[1] + 8]);
}

export function setupAnimaCLIPLoaderUI(node) {
    applyNodeFrameStyle(node);
    const wClip = node.widgets?.find((w) => w.name === "clip_name");
    const wDevice = node.widgets?.find((w) => w.name === "device_mode");
    const wPrec = node.widgets?.find((w) => w.name === "precision");
    const wGuard = node.widgets?.find((w) => w.name === "strict_qwen3_guard");

    [wClip, wDevice, wPrec, wGuard].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "ANIMA QWEN3 TEXT ENCODER LOADER",
            accentColor: HADES_THEME.cyan,
            customHeight: 56,
            getStatusText: () => `ANIMA QWEN3 TE • ${formatCleanLabel(wClip?.value, "qwen_3_06b_base")}`,
            getStatusColor: () => (wGuard?.value ? HADES_THEME.emerald : HADES_THEME.amber),
            drawCustomHUD: (ctx, rect) => {
                let cx = rect.x + 6;
                const cy = rect.y + (rect.h - 16) * 0.5;
                const isCpu = String(wDevice?.value || "").startsWith("cpu");
                cx += drawPill(
                    ctx,
                    cx,
                    cy,
                    wGuard?.value ? "QWEN3-0.6B (1024d) GUARD" : "GUARD BYPASSED",
                    wGuard?.value ? HADES_THEME.emerald : HADES_THEME.amber
                );
                cx += drawPill(ctx, cx, cy, isCpu ? "CPU LOW-VRAM SAFE" : "GPU DYNAMIC", isCpu ? HADES_THEME.purple : HADES_THEME.blue);
                drawPill(ctx, cx, cy, `PREC: ${String(wPrec?.value || "auto").toUpperCase()}`, HADES_THEME.cyan);
            },
        })
    );

    if (wClip) {
        node.addCustomWidget(
            createStyledComboWidget(node, wClip, {
                tag: "QWEN",
                tagColor: HADES_THEME.cyan,
                label: "Text Encoder File",
                cleanFileNames: true,
            })
        );
    }
    if (wDevice) {
        node.addCustomWidget(
            createStyledComboWidget(node, wDevice, {
                tag: "DEV",
                tagColor: HADES_THEME.blue,
                label: "Execution Device",
            })
        );
    }
    if (wPrec) {
        node.addCustomWidget(
            createStyledComboWidget(node, wPrec, {
                tag: "PREC",
                tagColor: HADES_THEME.purple,
                label: "Compute Precision",
            })
        );
    }
    if (wGuard) {
        node.addCustomWidget(
            createStyledToggleWidget(node, wGuard, {
                tag: "SHLD",
                color: HADES_THEME.emerald,
                label: "Block Incompatible CLIP / T5 / 7B",
                onText: "1024d LOCKED",
                offText: "UNCHECKED",
            })
        );
    }

    const sz = node.computeSize();
    node.setSize([Math.max(420, sz[0]), sz[1] + 8]);
}

export function setupAnimaVAELoaderUI(node) {
    applyNodeFrameStyle(node);
    const wVae = node.widgets?.find((w) => w.name === "vae_name");
    const wPrec = node.widgets?.find((w) => w.name === "vae_precision");

    [wVae, wPrec].forEach(hideBackingWidget);

    node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "ANIMA 16CH WAN21 VAE LOADER",
            accentColor: HADES_THEME.emerald,
            customHeight: 56,
            getStatusText: () => `ANIMA 16CH VAE • ${formatCleanLabel(wVae?.value, "qwen_image_vae")}`,
            getStatusColor: () => HADES_THEME.emerald,
            drawCustomHUD: (ctx, rect) => {
                let cx = rect.x + 6;
                const cy = rect.y + (rect.h - 16) * 0.5;
                cx += drawPill(ctx, cx, cy, "16CH WAN2.1 ONLY (BLOCKS 64CH TRAP)", HADES_THEME.emerald);
                cx += drawPill(ctx, cx, cy, "8x SPATIAL", HADES_THEME.cyan);
                const precShort = String(wPrec?.value || "bfloat16").split(" ")[0].toUpperCase();
                drawPill(ctx, cx, cy, precShort, HADES_THEME.blue);
            },
        })
    );

    if (wVae) {
        node.addCustomWidget(
            createStyledComboWidget(node, wVae, {
                tag: "VAE",
                tagColor: HADES_THEME.emerald,
                label: "16ch Wan21 VAE File",
                cleanFileNames: true,
            })
        );
    }
    if (wPrec) {
        node.addCustomWidget(
            createStyledComboWidget(node, wPrec, {
                tag: "PREC",
                tagColor: HADES_THEME.blue,
                label: "Wan21 RMSNorm Precision",
            })
        );
    }

    const sz = node.computeSize();
    node.setSize([Math.max(420, sz[0]), sz[1] + 8]);
}

export function setupAnimaVAEDecodeUI(node) {
    applyNodeFrameStyle(node);
    const wMode = node.widgets?.find((w) => w.name === "decode_mode");
    const wTile = node.widgets?.find((w) => w.name === "tile_size_px");
    const wOverlap = node.widgets?.find((w) => w.name === "tile_overlap_px");
    const wThresh = node.widgets?.find((w) => w.name === "auto_tile_mp_threshold");

    [wMode, wTile, wOverlap, wThresh].forEach(hideBackingWidget);

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.emerald,
            customHeight: 52,
            actions: [
                {
                    label: "Auto",
                    onClick: (n) => {
                        setWidgetValue(wMode, "Auto Smart-Tile (VRAM & HiRes Aware)", n);
                    },
                },
                {
                    label: "6GB Safe",
                    onClick: (n) => {
                        setWidgetValue(wMode, "Force Seamless Tiled (Low-VRAM 6GB Safe)", n);
                        setWidgetValue(wTile, 512, n);
                        setWidgetValue(wOverlap, 64, n);
                    },
                },
                {
                    label: "Direct",
                    onClick: (n) => {
                        setWidgetValue(wMode, "Direct Full Decode (Max Speed)", n);
                    },
                },
            ],
            drawCustomHUD: (ctx, rect) => {
                const isNarrow = rect.w < 260;
                let cx = rect.x + 6;
                const cy = rect.y + (rect.h - 16) * 0.5;
                const tileVal = wTile?.value ?? 512;
                const ovlVal = wOverlap?.value ?? 64;
                const mpVal = Number(wThresh?.value ?? 1.35).toFixed(2);
                cx += drawPill(ctx, cx, cy, isNarrow ? `${tileVal}/${ovlVal}px` : `TILE: ${tileVal}px / ${ovlVal}px`, HADES_THEME.cyan);
                drawPill(ctx, cx, cy, isNarrow ? `>${mpVal} MP` : `AUTO > ${mpVal} MP`, HADES_THEME.blue);
            },
        })
    );

    const TAG_W = 66;

    const modeWidget = wMode
        ? node.addCustomWidget(
              createStyledComboWidget(node, wMode, {
                  tag: "MODE",
                  tagColor: HADES_THEME.emerald,
                  tagWidth: TAG_W,
                  label: "",
              })
          )
        : null;

    const tileWidget = wTile
        ? node.addCustomWidget(
              createSpectralSliderWidget(node, wTile, {
                  tag: "TILE",
                  rangeTag: "TILE",
                  tagWidth: TAG_W,
                  label: "",
                  color: HADES_THEME.cyan,
                  min: 256,
                  max: 2048,
                  visMin: 256,
                  visMax: 2048,
                  step: 64,
                  defaultVal: 512,
                  decimals: 0,
                  isInt: true,
              })
          )
        : null;

    const overlapWidget = wOverlap
        ? node.addCustomWidget(
              createSpectralSliderWidget(node, wOverlap, {
                  tag: "OVERLAP",
                  rangeTag: "OVERLAP",
                  tagWidth: TAG_W,
                  label: "",
                  color: HADES_THEME.blue,
                  min: 32,
                  max: 512,
                  visMin: 32,
                  visMax: 512,
                  step: 32,
                  defaultVal: 64,
                  decimals: 0,
                  isInt: true,
              })
          )
        : null;

    const threshWidget = wThresh
        ? node.addCustomWidget(
              createSpectralSliderWidget(node, wThresh, {
                  tag: "AUTO-MP",
                  rangeTag: "AUTO-MP",
                  tagWidth: TAG_W,
                  label: "",
                  color: HADES_THEME.emerald,
                  min: 0.5,
                  max: 4.0,
                  visMin: 0.5,
                  visMax: 4.0,
                  step: 0.05,
                  defaultVal: 1.35,
              })
          )
        : null;

    node.widgets = [
        bannerWidget,
        modeWidget,
        tileWidget,
        overlapWidget,
        threshWidget,
        wMode,
        wTile,
        wOverlap,
        wThresh,
    ].filter(Boolean);

    const sz = node.computeSize();
    node.setSize([Math.max(420, sz[0]), sz[1] + 8]);
}

export const LOADER_NODE_DISPATCH = {
    HaDeSAnimaModelLoader: setupAnimaModelLoaderUI,
    HaDeSAnimaCLIPLoader: setupAnimaCLIPLoaderUI,
    HaDeSAnimaVAELoader: setupAnimaVAELoaderUI,
    HaDeSAnimaVAEDecode: setupAnimaVAEDecodeUI,
};
