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

const TAG_W = 66;

export function setupAnimaModelLoaderUI(node) {
    applyNodeFrameStyle(node);
    const wUnet = node.widgets?.find((w) => w.name === "unet_name");
    const wDtype = node.widgets?.find((w) => w.name === "weight_dtype");
    const wShift = node.widgets?.find((w) => w.name === "flow_shift");
    const wRemap = node.widgets?.find((w) => w.name === "auto_remap_diffusers");
    const wAdapter = node.widgets?.find((w) => w.name === "external_llm_adapter");

    [wUnet, wDtype, wShift, wRemap, wAdapter].forEach(hideBackingWidget);

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.blue,
            customHeight: 52,
            actions: [
                {
                    label: "Reset 3.0",
                    onClick: (n) => {
                        setWidgetValue(wShift, 3.0, n);
                        setWidgetValue(wDtype, "default", n);
                        setWidgetValue(wRemap, true, n);
                    },
                },
                {
                    label: "Default",
                    onClick: (n) => setWidgetValue(wDtype, "default", n),
                },
                {
                    label: "BF16",
                    onClick: (n) => setWidgetValue(wDtype, "bfloat16", n),
                },
                {
                    label: "FP8",
                    onClick: (n) => setWidgetValue(wDtype, "fp8_e4m3fn", n),
                },
            ],
            drawCustomHUD: (ctx, rect) => {
                const isNarrow = rect.w < 270;
                let cx = rect.x + 6;
                const cy = rect.y + (rect.h - 16) * 0.5;
                const dtypeVal = String(wDtype?.value || "default").toUpperCase();
                const shiftVal = Number(wShift?.value ?? 3.0).toFixed(2);
                cx += drawPill(ctx, cx, cy, isNarrow ? dtypeVal : `DTYPE: ${dtypeVal}`, HADES_THEME.cyan);
                cx += drawPill(ctx, cx, cy, isNarrow ? `S: ${shiftVal}` : `SHIFT: ${shiftVal} (m=1.0)`, HADES_THEME.blue);
                cx += drawPill(
                    ctx,
                    cx,
                    cy,
                    wRemap?.value ? (isNarrow ? "24-REMAP" : "24-RULE REMAP") : "RAW KEYS",
                    wRemap?.value ? HADES_THEME.emerald : HADES_THEME.amber
                );
                if (wAdapter?.value && wAdapter.value !== "None") {
                    drawPill(ctx, cx, cy, isNarrow ? "ADPT" : "EXT ADAPTER", HADES_THEME.purple);
                }
            },
        })
    );

    const unetWidget = wUnet
        ? node.addCustomWidget(
              createStyledComboWidget(node, wUnet, {
                  tag: "MODEL",
                  tagColor: HADES_THEME.blue,
                  tagWidth: TAG_W,
                  label: "",
                  cleanFileNames: true,
                  tooltip: `<strong>Anima Diffusion Transformer</strong><br>Select an Anima DiT model checkpoint (Base 2B, Aesthetic, Turbo, Lightning, 2.9B, or 3.8B).`,
              })
          )
        : null;

    const dtypeWidget = wDtype
        ? node.addCustomWidget(
              createStyledComboWidget(node, wDtype, {
                  tag: "DTYPE",
                  tagColor: HADES_THEME.cyan,
                  tagWidth: TAG_W,
                  label: "",
                  tooltip: `<strong>Weight Precision</strong><br>Precision for loading the DiT weights into VRAM.<br><span style="color:#38bdf8">• default:</span> Preserves checkpoint native precision.<br><span style="color:#38bdf8">• bfloat16:</span> Native 16-bit brain float.<br><span style="color:#38bdf8">• fp8_e4m3fn:</span> Low-VRAM 8-bit quantization.`,
              })
          )
        : null;

    const shiftWidget = wShift
        ? node.addCustomWidget(
              createSpectralSliderWidget(node, wShift, {
                  tag: "SHIFT",
                  color: HADES_THEME.amber,
                  tagWidth: TAG_W,
                  label: "",
                  min: 0.5,
                  max: 10.0,
                  visMin: 1.0,
                  visMax: 6.0,
                  step: 0.05,
                  defaultVal: 3.0,
                  tooltip: `<strong>Rectified Flow Shift (multiplier=1.0)</strong><br>Time-step density warping for Anima's continuous flow formulation.<br><span style="color:#fbbf24">• Default: 3.00</span> calibrated baseline for optimal contrast and anatomy.`,
              })
          )
        : null;

    const remapWidget = wRemap
        ? node.addCustomWidget(
              createStyledToggleWidget(node, wRemap, {
                  tag: "CONV",
                  color: HADES_THEME.emerald,
                  tagWidth: TAG_W,
                  label: "",
                  onText: "AUTO REMAP",
                  offText: "BYPASS",
                  tooltip: `<strong>24-Rule Diffusers Auto-Remap</strong><br>Automatically converts official Diffusers DiT key format to ComfyUI native format on-the-fly, preventing missing block errors.`,
              })
          )
        : null;

    const adapterWidget = wAdapter
        ? node.addCustomWidget(
              createStyledComboWidget(node, wAdapter, {
                  tag: "ADPT",
                  tagColor: HADES_THEME.purple,
                  tagWidth: TAG_W,
                  label: "",
                  cleanFileNames: true,
                  tooltip: `<strong>External llm_adapter Injection</strong><br>Optional llm_adapter layer injection for Anima 3.8B or hybrid text conditioning architectures.`,
              })
          )
        : null;

    node.widgets = [
        bannerWidget,
        unetWidget,
        dtypeWidget,
        shiftWidget,
        remapWidget,
        adapterWidget,
        wUnet,
        wDtype,
        wShift,
        wRemap,
        wAdapter,
    ].filter(Boolean);

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        if (info?.size && Array.isArray(info.size) && info.size[0] > 0 && info.size[1] > 0) {
            this.size = [info.size[0], info.size[1]];
        }
        this.setDirtyCanvas?.(true, true);
    };

    const minW = 420;
    const sz = node.computeSize();
    const initW = (Array.isArray(node.size) && node.size[0] > 0) ? Math.max(minW, node.size[0]) : Math.max(minW, sz[0]);
    const initH = (Array.isArray(node.size) && node.size[1] > 0) ? Math.max(node.size[1], sz[1] + 8) : sz[1] + 8;
    node.setSize([initW, initH]);
}

export function setupAnimaCLIPLoaderUI(node) {
    applyNodeFrameStyle(node);
    const wClip = node.widgets?.find((w) => w.name === "clip_name");
    const wDevice = node.widgets?.find((w) => w.name === "device_mode");
    const wPrec = node.widgets?.find((w) => w.name === "precision");
    const wGuard = node.widgets?.find((w) => w.name === "strict_qwen3_guard");

    [wClip, wDevice, wPrec, wGuard].forEach(hideBackingWidget);

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.cyan,
            customHeight: 52,
            actions: [
                {
                    label: "GPU Dyn",
                    onClick: (n) => setWidgetValue(wDevice, "default (GPU Dynamic)", n),
                },
                {
                    label: "CPU Safe",
                    onClick: (n) => setWidgetValue(wDevice, "cpu (Low-VRAM Safe)", n),
                },
                {
                    label: "Auto Prec",
                    onClick: (n) => setWidgetValue(wPrec, "auto", n),
                },
                {
                    label: "BF16",
                    onClick: (n) => setWidgetValue(wPrec, "bfloat16", n),
                },
            ],
            drawCustomHUD: (ctx, rect) => {
                const isNarrow = rect.w < 270;
                let cx = rect.x + 6;
                const cy = rect.y + (rect.h - 16) * 0.5;
                const isCpu = String(wDevice?.value || "").startsWith("cpu");
                const precVal = String(wPrec?.value || "auto").toUpperCase();
                cx += drawPill(
                    ctx,
                    cx,
                    cy,
                    wGuard?.value ? (isNarrow ? "1024d GUARD" : "QWEN3 (1024d) GUARD") : "GUARD OFF",
                    wGuard?.value ? HADES_THEME.emerald : HADES_THEME.amber
                );
                cx += drawPill(ctx, cx, cy, isCpu ? "CPU SAFE" : (isNarrow ? "GPU DYN" : "GPU DYNAMIC"), isCpu ? HADES_THEME.purple : HADES_THEME.blue);
                drawPill(ctx, cx, cy, isNarrow ? precVal : `PREC: ${precVal}`, HADES_THEME.cyan);
            },
        })
    );

    const clipWidget = wClip
        ? node.addCustomWidget(
              createStyledComboWidget(node, wClip, {
                  tag: "QWEN",
                  tagColor: HADES_THEME.cyan,
                  tagWidth: TAG_W,
                  label: "",
                  cleanFileNames: true,
                  tooltip: `<strong>Qwen3-0.6B Text Encoder</strong><br>Anima's native 1024-dim text encoder model file.`,
              })
          )
        : null;

    const deviceWidget = wDevice
        ? node.addCustomWidget(
              createStyledComboWidget(node, wDevice, {
                  tag: "DEV",
                  tagColor: HADES_THEME.blue,
                  tagWidth: TAG_W,
                  label: "",
                  tooltip: `<strong>Execution Device Mode</strong><br><span style="color:#38bdf8">• default (GPU Dynamic):</span> Moves text encoder to GPU during prompt conditioning, then offloads when idle.<br><span style="color:#a855f7">• cpu:</span> Keeps text encoder in system RAM to preserve maximum GPU VRAM.`,
              })
          )
        : null;

    const precWidget = wPrec
        ? node.addCustomWidget(
              createStyledComboWidget(node, wPrec, {
                  tag: "PREC",
                  tagColor: HADES_THEME.purple,
                  tagWidth: TAG_W,
                  label: "",
                  tooltip: `<strong>Compute Precision</strong><br>Calculation precision for prompt conditioning embeddings (auto, bfloat16, fp16, fp32).`,
              })
          )
        : null;

    const guardWidget = wGuard
        ? node.addCustomWidget(
              createStyledToggleWidget(node, wGuard, {
                  tag: "SHLD",
                  color: HADES_THEME.emerald,
                  tagWidth: TAG_W,
                  label: "",
                  onText: "1024d LOCKED",
                  offText: "UNCHECKED",
                  tooltip: `<strong>Strict Qwen3-0.6B (1024d) Guard</strong><br>Blocks incompatible CLIP, T5, or Qwen2.5-7B models that would cause shape mismatch crashes in Anima.`,
              })
          )
        : null;

    node.widgets = [
        bannerWidget,
        clipWidget,
        deviceWidget,
        precWidget,
        guardWidget,
        wClip,
        wDevice,
        wPrec,
        wGuard,
    ].filter(Boolean);

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        if (info?.size && Array.isArray(info.size) && info.size[0] > 0 && info.size[1] > 0) {
            this.size = [info.size[0], info.size[1]];
        }
        this.setDirtyCanvas?.(true, true);
    };

    const minW = 420;
    const sz = node.computeSize();
    const initW = (Array.isArray(node.size) && node.size[0] > 0) ? Math.max(minW, node.size[0]) : Math.max(minW, sz[0]);
    const initH = (Array.isArray(node.size) && node.size[1] > 0) ? Math.max(node.size[1], sz[1] + 8) : sz[1] + 8;
    node.setSize([initW, initH]);
}

export function setupAnimaVAELoaderUI(node) {
    applyNodeFrameStyle(node);
    const wVae = node.widgets?.find((w) => w.name === "vae_name");
    const wPrec = node.widgets?.find((w) => w.name === "vae_precision");

    [wVae, wPrec].forEach(hideBackingWidget);

    const bannerWidget = node.addCustomWidget(
        createSpectralBannerWidget(node, {
            title: "",
            accentColor: HADES_THEME.emerald,
            customHeight: 52,
            actions: [
                {
                    label: "BF16 (Rec)",
                    onClick: (n) => setWidgetValue(wPrec, "bfloat16 (Recommended Wan21)", n),
                },
                {
                    label: "FP32 HiPrec",
                    onClick: (n) => setWidgetValue(wPrec, "float32 (High Precision)", n),
                },
                {
                    label: "FP16",
                    onClick: (n) => setWidgetValue(wPrec, "float16", n),
                },
            ],
            drawCustomHUD: (ctx, rect) => {
                const isNarrow = rect.w < 270;
                let cx = rect.x + 6;
                const cy = rect.y + (rect.h - 16) * 0.5;
                cx += drawPill(ctx, cx, cy, isNarrow ? "16CH WAN21" : "16CH WAN2.1 ONLY", HADES_THEME.emerald);
                cx += drawPill(ctx, cx, cy, "8x SPATIAL", HADES_THEME.cyan);
                const precShort = String(wPrec?.value || "bfloat16").split(" ")[0].toUpperCase();
                drawPill(ctx, cx, cy, precShort, HADES_THEME.blue);
            },
        })
    );

    const vaeWidget = wVae
        ? node.addCustomWidget(
              createStyledComboWidget(node, wVae, {
                  tag: "VAE",
                  tagColor: HADES_THEME.emerald,
                  tagWidth: TAG_W,
                  label: "",
                  cleanFileNames: true,
                  tooltip: `<strong>16-Channel Wan2.1 VAE Checkpoint</strong><br>Anima requires the Wan2.1 16-channel VAE. Incompatible 64-channel Qwen2.1 VAEs are blocked automatically.`,
              })
          )
        : null;

    const precWidget = wPrec
        ? node.addCustomWidget(
              createStyledComboWidget(node, wPrec, {
                  tag: "PREC",
                  tagColor: HADES_THEME.blue,
                  tagWidth: TAG_W,
                  label: "",
                  tooltip: `<strong>Wan2.1 RMSNorm Precision</strong><br>Precision for VAE RMSNorm and convolutional layers.<br><span style="color:#38bdf8">• bfloat16:</span> Officially recommended native precision for Wan2.1 VAE.`,
              })
          )
        : null;

    node.widgets = [
        bannerWidget,
        vaeWidget,
        precWidget,
        wVae,
        wPrec,
    ].filter(Boolean);

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        if (info?.size && Array.isArray(info.size) && info.size[0] > 0 && info.size[1] > 0) {
            this.size = [info.size[0], info.size[1]];
        }
        this.setDirtyCanvas?.(true, true);
    };

    const minW = 420;
    const sz = node.computeSize();
    const initW = (Array.isArray(node.size) && node.size[0] > 0) ? Math.max(minW, node.size[0]) : Math.max(minW, sz[0]);
    const initH = (Array.isArray(node.size) && node.size[1] > 0) ? Math.max(node.size[1], sz[1] + 8) : sz[1] + 8;
    node.setSize([initW, initH]);
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
                  tooltip: `<strong>Decode Mode (Strategy)</strong><br>• <strong>Auto Smart-Tile:</strong> Automatically enables seamless tiling only when image exceeds AUTO-MP threshold.<br>• <strong>6GB Safe:</strong> Forces tiling on all resolutions to prevent VRAM spikes.<br>• <strong>Direct Full Decode:</strong> Decodes in a single un-tiled pass for maximum speed.`,
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
                  tooltip: `<strong>Spatial Tile Size (px)</strong><br>Pixel dimension of each square patch decoded independently.<br><span style="color:#38bdf8">• Smaller (256–512px):</span> Drastically reduces VRAM usage (safe for 6GB–8GB GPUs).<br><span style="color:#38bdf8">• Larger (768–1024px):</span> Faster decoding and fewer tile boundaries if VRAM permits.`,
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
                  tooltip: `<strong>Tile Blend Overlap (px)</strong><br>Margin of overlapping pixels between adjacent tiles.<br><span style="color:#60a5fa">• Edge Feathering:</span> Blends boundaries with smooth cosine/linear weighting.<br>Prevents visible seams, grid artifacts, or luminance lines across tile borders.`,
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
                  tooltip: `<strong>Auto-Tile Resolution Threshold (Megapixels)</strong><br>Resolution cutoff (W × H / 1M) triggering tiled decoding in Auto mode.<br><span style="color:#34d399">• Below threshold (e.g. 1024×1024 = 1.05 MP):</span> Direct fast decode.<br><span style="color:#34d399">• Above threshold (e.g. 1536×1536 = 2.36 MP):</span> Activates seamless tiling to prevent VRAM OOM.`,
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

    const origConfigure = node.configure;
    node.configure = function (info) {
        if (origConfigure) origConfigure.apply(this, arguments);
        if (info?.size && Array.isArray(info.size) && info.size[0] > 0 && info.size[1] > 0) {
            this.size = [info.size[0], info.size[1]];
        }
        this.setDirtyCanvas?.(true, true);
    };

    const minW = 420;
    const sz = node.computeSize();
    const initW = (Array.isArray(node.size) && node.size[0] > 0) ? Math.max(minW, node.size[0]) : Math.max(minW, sz[0]);
    const initH = (Array.isArray(node.size) && node.size[1] > 0) ? Math.max(node.size[1], sz[1] + 8) : sz[1] + 8;
    node.setSize([initW, initH]);
}

export const LOADER_NODE_DISPATCH = {
    HaDeSAnimaModelLoader: setupAnimaModelLoaderUI,
    HaDeSAnimaCLIPLoader: setupAnimaCLIPLoaderUI,
    HaDeSAnimaVAELoader: setupAnimaVAELoaderUI,
    HaDeSAnimaVAEDecode: setupAnimaVAEDecodeUI,
};
