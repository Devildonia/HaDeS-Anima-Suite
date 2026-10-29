import { app } from "../../scripts/app.js";

// =============================================================================
// HADES SPECTRAL UI - DESIGN TOKENS & CORE CANVAS HELPERS
// =============================================================================
export const HADES_THEME = {
    bgNode: "#131519",
    bgPanel: "#181b22",
    bgControl: "#12141a",
    bgControlHover: "#1e232d",
    borderSubtle: "#282d37",
    borderActive: "#3b82f688",
    textPrimary: "#e5e7eb",
    textSecondary: "#9da5b4",
    textMuted: "#5c6370",
    blue: "#3b82f6",
    cyan: "#06b6d4",
    emerald: "#10b981",
    purple: "#a855f7",
    amber: "#f59e0b",
    rose: "#f43f5e",
};

export const GROUP_META = {
    base_ratio:                { tag: "BASE", range: "GLOBAL", color: "#94a3b8", indices: [] },
    BLOCKS_Layout_Geometry:    { tag: "GEOM", range: "B00–04", color: "#3b82f6", indices: [0, 1, 2, 3, 4] },
    BLOCKS_Global_Composition: { tag: "COMP", range: "B05–09", color: "#06b6d4", indices: [5, 6, 7, 8, 9] },
    BLOCKS_Subject_Identity:   { tag: "IDEN", range: "B10–14", color: "#6366f1", indices: [10, 11, 12, 13, 14] },
    BLOCKS_Material_Substance: { tag: "MATS", range: "B15–19", color: "#10b981", indices: [15, 16, 17, 18, 19] },
    BLOCKS_Art_Style_Medium:   { tag: "STYL", range: "B20–23", color: "#a855f7", indices: [20, 21, 22, 23] },
    BLOCKS_Lighting_Sharpness: { tag: "LIGH", range: "B24–27", color: "#f59e0b", indices: [24, 25, 26, 27] },
    ADAPTER_Text_Alignment:    { tag: "ADPT", range: "B28–33", color: "#ec4899", indices: [28, 29, 30, 31, 32, 33] },
};

export const CLIP_GROUP_META = {
    base_ratio:        { tag: "BASE", range: "GLOBAL", label: "Base Fallback Ratio", color: "#94a3b8" },
    vocab_embeddings:  { tag: "TOKS", range: "EMBED",  label: "Vocab & Token Embeddings", color: "#3b82f6" },
    syntax_rigidity:   { tag: "SYNT", range: "0–35%",  label: "Syntax & Tag Rigidity", color: "#06b6d4" },
    semantic_focus:    { tag: "SEMA", range: "35–72%", label: "Semantic & Concept Focus", color: "#10b981" },
    style_abstraction: { tag: "STYL", range: "72%+",   label: "Style & Mood Abstraction", color: "#a855f7" },
    final_projections: { tag: "NORM", range: "OUT",    label: "Final Norm Projections", color: "#f59e0b" },
};

export function formatCleanLabel(rawName, emptyText = "— None —") {
    if (!rawName || rawName === "None") return emptyText;
    const base = String(rawName).split(/[\\/]/).pop();
    return base.replace(/\.(safetensors|ckpt|pt|bin)$/i, "");
}

export function clampNum(val, min, max, decimals = 2) {
    const num = Number(val);
    if (Number.isNaN(num)) return min;
    const clamped = Math.max(min, Math.min(max, num));
    const factor = 10 ** decimals;
    return Math.round(clamped * factor) / factor;
}

export function setWidgetValue(widget, value, node) {
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value, app.canvas, node, [0, 0], null);
    if (node?.graph) {
        node.graph._version = (node.graph._version || 0) + 1;
    }
    node?.setDirtyCanvas?.(true, true);
}

export function drawRoundedRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
}

export function hideBackingWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
}

const HADES_ICON_IMG = new Image();
HADES_ICON_IMG.src = new URL("./icon.png", import.meta.url).href;
HADES_ICON_IMG.onload = () => {
    app?.canvas?.setDirty?.(true, true);
};

export function enableNodeHeaderIcon(node, options = {}) {
    if (!node) return;
    if (!node._default_hades_title) {
        node._default_hades_title = node.title || node.constructor?.title || "";
    }
    const stripPrefix = options.stripPrefix ?? true;

    // 1. Right-click context menu option to show/hide icon on demand
    const origGetExtraMenuOptions = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, menuOptions) {
        if (typeof origGetExtraMenuOptions === "function") {
            origGetExtraMenuOptions.call(this, canvas, menuOptions);
        }
        const currentFlags = this.flags || {};
        const defTitle = this._default_hades_title || this.constructor?.title || "";
        const curTitle = this.getTitle?.() ?? this.title ?? "";
        const isDefault = (
            curTitle.replace(/^HaDeS\s+/i, "").trim() === defTitle.replace(/^HaDeS\s+/i, "").trim()
        );
        const isVisible = currentFlags.show_hades_icon !== undefined
            ? Boolean(currentFlags.show_hades_icon)
            : isDefault;

        if (Array.isArray(menuOptions)) {
            menuOptions.push({
                content: isVisible ? "Hide HaDeS Icon" : "Show HaDeS Icon",
                callback: () => {
                    this.flags = this.flags || {};
                    this.flags.show_hades_icon = !isVisible;
                    this.setDirtyCanvas?.(true, true);
                },
            });
        }
    };

    // 2. Custom header text & icon rendering
    const origOnDrawTitleText = node.onDrawTitleText;
    node.onDrawTitleText = function (ctx, title_height, size, scale, titleFontStyle, selected) {
        if (typeof origOnDrawTitleText === "function") {
            origOnDrawTitleText.call(this, ctx, title_height, size, scale, titleFontStyle, selected);
            return;
        }

        ctx.font = titleFontStyle || this.titleFontStyle || "bold 12px sans-serif";
        let rawTitle = this.getTitle?.() ?? this.title ?? `❌ ${this.type}`;
        if (stripPrefix) {
            rawTitle = String(rawTitle).replace(/^HaDeS\s+/i, "");
        }
        if (this.pinned) rawTitle += " 📌";

        const defTitle = this._default_hades_title || this.constructor?.title || "";
        const isDefaultName = (
            rawTitle.trim() === defTitle.replace(/^HaDeS\s+/i, "").trim()
            || rawTitle.trim() === defTitle.trim()
        );

        const currentFlags = this.flags || {};
        const showIcon = currentFlags.show_hades_icon !== undefined
            ? Boolean(currentFlags.show_hades_icon)
            : isDefaultName;

        const h = title_height || 30;
        const iconSize = Math.max(14, Math.min(20, Math.round(h * 0.6)));
        const iconY = Math.round(-h + (h - iconSize) * 0.5);
        const iconX = Math.round(h * 0.75);

        let textX = iconX;
        if (showIcon) {
            if (HADES_ICON_IMG.complete && HADES_ICON_IMG.naturalWidth > 0) {
                ctx.drawImage(HADES_ICON_IMG, iconX, iconY, iconSize, iconSize);
            }
            textX = iconX + iconSize + 6;
        }

        const maxTextW = size[0] - textX - 10;
        let displayTitle = rawTitle;

        if (this.collapsed) {
            displayTitle = rawTitle.substring(0, 18);
        } else if (maxTextW > 20 && ctx.measureText(displayTitle).width > maxTextW) {
            while (displayTitle.length > 2 && ctx.measureText(displayTitle + "…").width > maxTextW) {
                displayTitle = displayTitle.slice(0, -1);
            }
            displayTitle += "…";
        }

        ctx.fillStyle = selected
            ? "#ffffff"
            : (this.constructor.title_text_color || "#e5e7eb");
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(displayTitle, textX, Math.round(-h * 0.5));
    };
}

export function applyNodeFrameStyle(node) {
    node.color = "#1b2028";
    node.bgcolor = "#12151a";
    enableNodeHeaderIcon(node);
}

export function parseClientRecipePreview(recipeStr) {
    if (!recipeStr || !String(recipeStr).trim()) return null;
    const text = String(recipeStr).trim();
    try {
        if (text.startsWith("{")) {
            const obj = JSON.parse(text);
            if (Array.isArray(obj.predicted_ratios_34) && obj.predicted_ratios_34.length === 34) {
                return obj.predicted_ratios_34.map(Number);
            }
        }
        const parts = text.split(",").map((s) => s.trim()).filter(Boolean);
        if (parts.length === 34) {
            const nums = parts.map(Number);
            if (nums.every((n) => !Number.isNaN(n))) return nums;
        }
    } catch {
        return null;
    }
    return null;
}

export function estimatePredictiveCurve(preset, targetInfluence, aggressiveness, protectAdapter = true) {
    const bars = [];
    const blockColors = new Array(34).fill(HADES_THEME.blue);
    for (const meta of Object.values(GROUP_META)) {
        for (const idx of meta.indices) blockColors[idx] = meta.color;
    }

    const p = String(preset || "");
    const sampleDiv = 0.35;
    for (let b = 0; b < 34; b++) {
        if (protectAdapter && b >= 28) {
            bars.push({ ratio: 0.02, color: "#4b5563" });
            continue;
        }
        let r = targetInfluence;
        if (p.includes("Style Transfer") || p.includes("Character + Style")) {
            if (b < 15) r = targetInfluence * 0.15 * Math.max(0.05, 1.0 - sampleDiv * 0.7 * aggressiveness);
            else if (b < 20) r = targetInfluence * 0.65;
            else if (b < 28) r = Math.min(1.2, targetInfluence * (1.0 + sampleDiv * 0.65 * aggressiveness));
            else r = targetInfluence * 0.70;
        } else if (p.includes("Interference") || p.includes("Anti-Overcook")) {
            r = targetInfluence * Math.max(0.15, 1.0 - sampleDiv * 0.55 * aggressiveness);
        } else if (p.includes("Complementary") || p.includes("Orthogonal")) {
            r = Math.min(1.2, targetInfluence * (0.75 + sampleDiv * 0.65 * aggressiveness));
        } else {
            r = targetInfluence * Math.max(0.2, 1.0 - sampleDiv * 0.35 * aggressiveness);
        }
        bars.push({ ratio: Math.max(0.03, Math.min(1.0, r)), color: blockColors[b] });
    }
    return bars;
}

// =============================================================================
// REUSABLE HADES CANVAS WIDGET COMPONENTS
// =============================================================================

export function createSpectralBannerWidget(node, config) {
    const {
        title,
        accentColor = HADES_THEME.blue,
        getStatusText = null,
        getStatusColor = null,
        getBars = null,
        drawCustomHUD = null,
        customHeight = null,
        actions = [],
    } = config;

    const hasGraph = typeof getBars === "function" || typeof drawCustomHUD === "function";
    const widgetHeight = customHeight || (hasGraph ? 62 : 28);

    return {
        name: `hades_banner_${title}`,
        type: "custom",
        value: "",
        options: { serialize: false },
        serialize: false,
        _btnBounds: [],

        computeSize(width) {
            return [width, widgetHeight];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = widgetHeight - 4;
            const topY = y + 2;

            ctx.save();

            drawRoundedRect(ctx, x, topY, w, h, 6);
            ctx.fillStyle = HADES_THEME.bgPanel;
            ctx.fill();
            ctx.strokeStyle = HADES_THEME.borderSubtle;
            ctx.lineWidth = 1;
            ctx.stroke();

            const ledColor = getStatusColor ? getStatusColor() : accentColor;
            ctx.beginPath();
            ctx.arc(x + 11, topY + 12, 4, 0, Math.PI * 2);
            ctx.fillStyle = ledColor;
            ctx.fill();

            ctx.font = "bold 10px sans-serif";
            ctx.fillStyle = "#abb2bf";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            const labelText = getStatusText ? getStatusText() : title;
            ctx.fillText(labelText, x + 20, topY + 12);

            const btnH = 16;
            const btnGap = 4;
            let cursorRight = x + w - 5;
            this._btnBounds = [];

            ctx.font = "10px sans-serif";
            ctx.textAlign = "center";

            for (let i = actions.length - 1; i >= 0; i--) {
                const act = actions[i];
                const btnW = Math.max(32, ctx.measureText(act.label).width + 10);
                const bx = cursorRight - btnW;
                const by = topY + 4;

                drawRoundedRect(ctx, bx, by, btnW, btnH, 3);
                ctx.fillStyle = "#232731";
                ctx.fill();
                ctx.strokeStyle = "#363c49";
                ctx.stroke();
                ctx.fillStyle = "#9da5b4";
                ctx.fillText(act.label, bx + btnW * 0.5, by + btnH * 0.5 + 0.5);

                this._btnBounds.push({ x: bx, y: by, w: btnW, h: btnH, onClick: act.onClick });
                cursorRight = bx - btnGap;
            }

            if (hasGraph) {
                const graphX = x + 6;
                const graphY = topY + 24;
                const graphW = w - 12;
                const graphH = h - 28;

                drawRoundedRect(ctx, graphX, graphY, graphW, graphH, 4);
                ctx.fillStyle = "#101217";
                ctx.fill();

                if (typeof drawCustomHUD === "function") {
                    drawCustomHUD(ctx, { x: graphX, y: graphY, w: graphW, h: graphH }, n);
                } else if (typeof getBars === "function") {
                    const bars = getBars() || [];
                    const midY = graphY + graphH * 0.5;
                    ctx.strokeStyle = "#222630";
                    ctx.beginPath();
                    ctx.moveTo(graphX + 2, midY);
                    ctx.lineTo(graphX + graphW - 2, midY);
                    ctx.stroke();

                    if (bars.length > 0) {
                        const pad = 4;
                        const availW = graphW - pad * 2;
                        const stepW = availW / bars.length;
                        const barW = Math.max(2, stepW - 1.5);

                        for (let i = 0; i < bars.length; i++) {
                            const b = bars[i];
                            const norm = Math.max(0, Math.min(1, Number(b.ratio) || 0));
                            const bh = Math.max(2, norm * (graphH - 6));
                            const bx = graphX + pad + i * stepW;
                            const by = graphY + graphH - 3 - bh;

                            ctx.fillStyle = b.color || accentColor;
                            ctx.fillRect(bx, by, barW, bh);
                        }
                    }
                }
            }

            ctx.restore();
        },

        mouse(event, pos, n) {
            if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
            const [mx, my] = pos;
            for (const btn of this._btnBounds) {
                if (mx >= btn.x && mx <= btn.x + btn.w && my >= btn.y && my <= btn.y + btn.h) {
                    btn.onClick?.(n);
                    n.setDirtyCanvas(true, true);
                    return true;
                }
            }
            return false;
        },
    };
}

export function createStyledComboWidget(node, backingWidget, config = {}) {
    const {
        tag = "MODE",
        tagColor = config.color || HADES_THEME.purple,
        label = "",
        cleanFileNames = false,
    } = config;

    return {
        name: `hades_combo_${backingWidget?.name || label}`,
        type: "custom",
        value: "",
        options: { serialize: false },
        serialize: false,
        _bounds: null,

        computeSize(width) {
            return [width, 28];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 24;
            const rowY = y + 2;

            ctx.save();
            drawRoundedRect(ctx, x, rowY, w, h, 5);
            ctx.fillStyle = HADES_THEME.bgPanel;
            ctx.fill();
            ctx.strokeStyle = HADES_THEME.borderSubtle;
            ctx.lineWidth = 1;
            ctx.stroke();

            const tagW = 52;
            drawRoundedRect(ctx, x + 3, rowY + 3, tagW, h - 6, 3);
            ctx.fillStyle = `${tagColor}22`;
            ctx.fill();
            ctx.strokeStyle = `${tagColor}66`;
            ctx.stroke();

            ctx.font = "bold 9px sans-serif";
            ctx.fillStyle = tagColor;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(tag, x + 3 + tagW * 0.5, rowY + h * 0.5 + 0.5);

            const rawVal = backingWidget?.value ?? "";
            const valText = cleanFileNames ? formatCleanLabel(rawVal) : String(rawVal);

            const textStartX = x + tagW + 10;
            const textMaxW = w - tagW - 26;

            ctx.save();
            ctx.beginPath();
            ctx.rect(textStartX, rowY, textMaxW, h);
            ctx.clip();

            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            if (label) {
                ctx.fillStyle = HADES_THEME.textSecondary;
                ctx.fillText(`${label}: `, textStartX, rowY + h * 0.5 + 0.5);
                const lw = ctx.measureText(`${label}: `).width;
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(valText, textStartX + lw, rowY + h * 0.5 + 0.5);
            } else {
                ctx.fillStyle = HADES_THEME.textPrimary;
                ctx.fillText(valText, textStartX, rowY + h * 0.5 + 0.5);
            }
            ctx.restore();

            ctx.fillStyle = HADES_THEME.textSecondary;
            ctx.font = "10px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("▾", x + w - 10, rowY + h * 0.5 + 0.5);

            this._bounds = { x, y: rowY, w, h };
            ctx.restore();
        },

        mouse(event, pos, n) {
            if ((event.type !== "pointerdown" && event.type !== "mousedown") || !this._bounds || !backingWidget) {
                return false;
            }
            const [mx, my] = pos;
            const b = this._bounds;
            if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                const values = backingWidget.options?.values || [];
                new LiteGraph.ContextMenu(values, {
                    event,
                    title: label || backingWidget.name,
                    scale: Math.max(1, app.canvas?.ds?.scale || 1),
                    callback: (selected) => {
                        if (selected !== undefined) {
                            setWidgetValue(backingWidget, selected, n);
                        }
                    },
                });
                return true;
            }
            return false;
        },
    };
}

export function createSpectralSliderWidget(node, backingWidget, config = {}) {
    const {
        tag = "VAL",
        rangeTag = config.range || "",
        label = backingWidget?.name || "",
        color = HADES_THEME.blue,
        min = -10.0,
        max = 10.0,
        visMin = config.min ?? 0.0,
        visMax = config.max ?? 1.0,
        step = 0.05,
        defaultVal = config.defVal ?? 0.5,
        decimals = 2,
        isInt = false,
    } = config;

    return {
        name: `hades_slider_${backingWidget?.name || label}`,
        type: "custom",
        value: "",
        options: { serialize: false },
        serialize: false,
        _zones: null,
        _dragStartX: null,
        _dragStartVal: null,
        _didDrag: false,

        computeSize(width) {
            return [width, 28];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 24;
            const rowY = y + 2;

            const curVal = clampNum(backingWidget?.value ?? defaultVal, min, max, isInt ? 0 : decimals);

            ctx.save();
            drawRoundedRect(ctx, x, rowY, w, h, 5);
            ctx.fillStyle = HADES_THEME.bgPanel;
            ctx.fill();
            ctx.strokeStyle = HADES_THEME.borderSubtle;
            ctx.lineWidth = 1;
            ctx.stroke();

            const tagW = rangeTag ? 52 : 46;
            drawRoundedRect(ctx, x + 3, rowY + 3, tagW, h - 6, 3);
            ctx.fillStyle = `${color}22`;
            ctx.fill();
            ctx.strokeStyle = `${color}66`;
            ctx.stroke();

            ctx.font = "bold 9px monospace";
            ctx.fillStyle = color;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(rangeTag || tag, x + 3 + tagW * 0.5, rowY + h * 0.5 + 0.5);

            const ctrlW = 114;
            const ctrlX = x + w - ctrlW - 3;
            const ctrlY = rowY + 3;
            const ctrlH = h - 6;
            const btnW = 18;

            drawRoundedRect(ctx, ctrlX, ctrlY, ctrlW, ctrlH, 4);
            ctx.fillStyle = HADES_THEME.bgControl;
            ctx.fill();
            ctx.strokeStyle = "#2b303c";
            ctx.stroke();

            const trackX = ctrlX + btnW;
            const trackW = ctrlW - btnW * 2;
            const norm = Math.max(0, Math.min(1, (curVal - visMin) / (visMax - visMin || 1)));
            const fillW = norm * trackW;
            if (fillW > 0.5) {
                ctx.fillStyle = `${color}44`;
                ctx.fillRect(trackX, ctrlY + 1, fillW, ctrlH - 2);
            }

            ctx.font = "bold 11px monospace";
            ctx.textAlign = "center";
            ctx.fillStyle = HADES_THEME.textSecondary;
            ctx.fillText("−", ctrlX + btnW * 0.5, ctrlY + ctrlH * 0.5 + 0.5);
            ctx.fillText("+", ctrlX + ctrlW - btnW * 0.5, ctrlY + ctrlH * 0.5 + 0.5);

            ctx.font = "11px monospace";
            ctx.fillStyle = HADES_THEME.textPrimary;
            const valStr = isInt ? String(Math.round(curVal)) : curVal.toFixed(decimals);
            ctx.fillText(valStr, trackX + trackW * 0.5, ctrlY + ctrlH * 0.5 + 0.5);

            const labelX = x + tagW + 9;
            const labelMaxW = ctrlX - labelX - 6;
            ctx.save();
            ctx.beginPath();
            ctx.rect(labelX, rowY, Math.max(20, labelMaxW), h);
            ctx.clip();
            ctx.font = "11px sans-serif";
            ctx.fillStyle = "#d1d5db";
            ctx.textAlign = "left";
            ctx.fillText(label, labelX, rowY + h * 0.5 + 0.5);
            ctx.restore();

            this._zones = {
                rowY,
                h,
                dec: { x: ctrlX, w: btnW },
                track: { x: x + tagW + 6, w: ctrlX + ctrlW - btnW - (x + tagW + 6) },
                inc: { x: ctrlX + ctrlW - btnW, w: btnW },
            };

            ctx.restore();
        },

        mouse(event, pos, n) {
            if (!this._zones || !backingWidget) return false;
            const [mx, my] = pos;
            const z = this._zones;

            if (event.type === "pointerdown" || event.type === "mousedown") {
                if (my < z.rowY || my > z.rowY + z.h) return false;

                if (event.button === 2) {
                    setWidgetValue(backingWidget, defaultVal, n);
                    return true;
                }

                if (mx >= z.dec.x && mx <= z.dec.x + z.dec.w) {
                    const next = clampNum(Number(backingWidget.value) - step, min, max, isInt ? 0 : decimals);
                    setWidgetValue(backingWidget, next, n);
                    return true;
                }

                if (mx >= z.inc.x && mx <= z.inc.x + z.inc.w) {
                    const next = clampNum(Number(backingWidget.value) + step, min, max, isInt ? 0 : decimals);
                    setWidgetValue(backingWidget, next, n);
                    return true;
                }

                if (mx >= z.track.x && mx <= z.track.x + z.track.w) {
                    this._dragStartX = mx;
                    this._dragStartVal = Number(backingWidget.value) ?? defaultVal;
                    this._didDrag = false;
                    return true;
                }
            } else if (event.type === "pointermove" || event.type === "mousemove") {
                if (this._dragStartX !== null) {
                    const dx = mx - this._dragStartX;
                    if (Math.abs(dx) > 2) {
                        this._didDrag = true;
                        const sensitivity = isInt ? (visMax - visMin) / 160 : (visMax - visMin) / 140;
                        const rawNext = this._dragStartVal + dx * sensitivity;
                        const stepped = isInt ? Math.round(rawNext / step) * step : rawNext;
                        const next = clampNum(stepped, min, max, isInt ? 0 : decimals);
                        setWidgetValue(backingWidget, next, n);
                    }
                    return true;
                }
            } else if (event.type === "pointerup" || event.type === "mouseup") {
                if (this._dragStartX !== null) {
                    const wasDrag = this._didDrag;
                    this._dragStartX = null;
                    this._dragStartVal = null;
                    this._didDrag = false;

                    if (!wasDrag && app.canvas) {
                        app.canvas.prompt(
                            `${label} (${min} to ${max})`,
                            String(backingWidget.value),
                            (val) => {
                                const parsed = isInt ? parseInt(val, 10) : parseFloat(val);
                                if (!Number.isNaN(parsed)) {
                                    const next = clampNum(parsed, min, max, isInt ? 0 : decimals);
                                    setWidgetValue(backingWidget, next, n);
                                }
                            },
                            event
                        );
                    }
                    return true;
                }
            }
            return false;
        },
    };
}

export function createStyledTextRowWidget(node, backingWidget, config = {}) {
    const {
        tag = "FILE",
        color = HADES_THEME.emerald,
        promptTitle = "Enter Value",
        suffix = "",
        emptyPlaceholder = "— None —",
        allowClearOnRightClick = false,
    } = config;

    return {
        name: `hades_text_${backingWidget?.name || tag}`,
        type: "custom",
        value: "",
        options: { serialize: false },
        serialize: false,
        _bounds: null,

        computeSize(width) {
            return [width, 28];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 24;
            const rowY = y + 2;

            ctx.save();
            drawRoundedRect(ctx, x, rowY, w, h, 5);
            ctx.fillStyle = HADES_THEME.bgPanel;
            ctx.fill();
            ctx.strokeStyle = HADES_THEME.borderSubtle;
            ctx.stroke();

            const tagW = 52;
            drawRoundedRect(ctx, x + 3, rowY + 3, tagW, h - 6, 3);
            ctx.fillStyle = `${color}22`;
            ctx.fill();
            ctx.strokeStyle = `${color}66`;
            ctx.stroke();

            ctx.font = "bold 9px monospace";
            ctx.fillStyle = color;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(tag, x + 3 + tagW * 0.5, rowY + h * 0.5 + 0.5);

            let sfxW = 0;
            if (suffix) {
                ctx.font = "10px monospace";
                sfxW = ctx.measureText(suffix).width + 10;
                ctx.fillStyle = HADES_THEME.textMuted;
                ctx.textAlign = "right";
                ctx.fillText(suffix, x + w - 6, rowY + h * 0.5 + 0.5);
            }

            const rawVal = String(backingWidget?.value ?? "").trim();
            const displayVal = suffix ? rawVal.replace(new RegExp(`${suffix}$`, "i"), "") : rawVal;

            ctx.save();
            ctx.beginPath();
            ctx.rect(x + tagW + 8, rowY, w - tagW - sfxW - 14, h);
            ctx.clip();
            ctx.font = displayVal ? "11px monospace" : "italic 11px sans-serif";
            ctx.fillStyle = displayVal ? HADES_THEME.textPrimary : HADES_THEME.textMuted;
            ctx.textAlign = "left";
            ctx.fillText(displayVal || emptyPlaceholder, x + tagW + 10, rowY + h * 0.5 + 0.5);
            ctx.restore();

            this._bounds = { x, y: rowY, w, h };
            ctx.restore();
        },

        mouse(event, pos, n) {
            if ((event.type !== "pointerdown" && event.type !== "mousedown") || !this._bounds || !backingWidget) {
                return false;
            }
            const [mx, my] = pos;
            const b = this._bounds;
            if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                if (event.button === 2 && allowClearOnRightClick) {
                    setWidgetValue(backingWidget, "", n);
                    return true;
                }
                if (app.canvas) {
                    app.canvas.prompt(
                        promptTitle,
                        String(backingWidget.value || ""),
                        (val) => {
                            setWidgetValue(backingWidget, String(val ?? "").trim(), n);
                        },
                        event
                    );
                }
                return true;
            }
            return false;
        },
    };
}

export function createStyledToggleWidget(node, backingWidget, config = {}) {
    const {
        tag = "FLAG",
        color = HADES_THEME.emerald,
        label = backingWidget?.name || "",
        onText = "ENABLED",
        offText = "DISABLED",
    } = config;

    return {
        name: `hades_toggle_${backingWidget?.name || label}`,
        type: "custom",
        value: "",
        options: { serialize: false },
        serialize: false,
        _bounds: null,

        computeSize(width) {
            return [width, 28];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 24;
            const rowY = y + 2;
            const isOn = Boolean(backingWidget?.value);

            ctx.save();
            drawRoundedRect(ctx, x, rowY, w, h, 5);
            ctx.fillStyle = HADES_THEME.bgPanel;
            ctx.fill();
            ctx.strokeStyle = isOn ? `${color}66` : HADES_THEME.borderSubtle;
            ctx.lineWidth = 1;
            ctx.stroke();

            const tagW = 52;
            drawRoundedRect(ctx, x + 3, rowY + 3, tagW, h - 6, 3);
            ctx.fillStyle = `${color}22`;
            ctx.fill();
            ctx.strokeStyle = `${color}66`;
            ctx.stroke();

            ctx.font = "bold 9px monospace";
            ctx.fillStyle = color;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(tag, x + 3 + tagW * 0.5, rowY + h * 0.5 + 0.5);

            ctx.font = "11px sans-serif";
            ctx.fillStyle = "#d1d5db";
            ctx.textAlign = "left";
            ctx.fillText(label, x + tagW + 10, rowY + h * 0.5 + 0.5);

            const pillW = 84;
            const pillX = x + w - pillW - 4;
            const pillY = rowY + 3;
            const pillH = h - 6;

            drawRoundedRect(ctx, pillX, pillY, pillW, pillH, 9);
            ctx.fillStyle = isOn ? `${color}33` : "#1e222b";
            ctx.fill();
            ctx.strokeStyle = isOn ? color : "#333947";
            ctx.stroke();

            ctx.font = "bold 9px monospace";
            ctx.fillStyle = isOn ? color : HADES_THEME.textMuted;
            ctx.textAlign = "center";
            ctx.fillText(isOn ? onText : offText, pillX + pillW * 0.5, pillY + pillH * 0.5 + 0.5);

            this._bounds = { x, y: rowY, w, h };
            ctx.restore();
        },

        mouse(event, pos, n) {
            if ((event.type !== "pointerdown" && event.type !== "mousedown") || !this._bounds || !backingWidget) {
                return false;
            }
            const [mx, my] = pos;
            const b = this._bounds;
            if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                setWidgetValue(backingWidget, !backingWidget.value, n);
                return true;
            }
            return false;
        },
    };
}

export function createSlotRowWidget(node, slotIndex, wOn, wLora, wStrength, wEnableAll) {
    return {
        name: `hades_lora_slot_ui_${slotIndex}`,
        type: "custom",
        value: "",
        options: { serialize: false },
        serialize: false,
        _hitZones: null,
        _dragStartX: null,
        _dragStartVal: null,
        _didDrag: false,

        computeSize(width) {
            return [width, 30];
        },

        draw(ctx, n, widgetWidth, y) {
            const margin = 12;
            const x = margin;
            const w = widgetWidth - margin * 2;
            const h = 26;
            const rowY = y + 2;

            const masterOn = wEnableAll ? Boolean(wEnableAll.value) : true;
            const isOn = Boolean(wOn?.value);
            const loraVal = wLora?.value ?? "None";
            const hasLora = loraVal !== "None";
            const strength = clampNum(wStrength?.value ?? 1.0, -10.0, 10.0, 2);
            const effectiveActive = masterOn && isOn && hasLora && strength !== 0;

            ctx.save();

            drawRoundedRect(ctx, x, rowY, w, h, 6);
            ctx.fillStyle = effectiveActive ? "#1e242d" : "#15181e";
            ctx.fill();
            ctx.strokeStyle = effectiveActive ? (slotIndex === 1 ? "#3b82f688" : "#10b98177") : "#262a34";
            ctx.lineWidth = 1;
            ctx.stroke();

            const toggleX = x + 6;
            const toggleY = rowY + 5;
            const toggleW = 30;
            const toggleH = 16;

            drawRoundedRect(ctx, toggleX, toggleY, toggleW, toggleH, 8);
            ctx.fillStyle = isOn && masterOn ? (slotIndex === 1 ? HADES_THEME.blue : HADES_THEME.emerald) : "#2d323e";
            ctx.fill();

            const knobR = 6;
            const knobX = isOn ? toggleX + toggleW - knobR - 2 : toggleX + knobR + 2;
            const knobY = toggleY + toggleH * 0.5;
            ctx.beginPath();
            ctx.arc(knobX, knobY, knobR, 0, Math.PI * 2);
            ctx.fillStyle = isOn && masterOn ? "#ffffff" : "#8b92a5";
            ctx.fill();

            ctx.font = "bold 9px sans-serif";
            ctx.fillStyle = effectiveActive ? (slotIndex === 1 ? "#60a5fa" : "#34d399") : HADES_THEME.textMuted;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(slotIndex === 1 ? "PRI" : `S${slotIndex}`, toggleX + toggleW + 5, rowY + h * 0.5);

            const strZoneW = 100;
            const strZoneX = x + w - strZoneW - 4;
            const strZoneY = rowY + 3;
            const strZoneH = h - 6;
            const btnW = 18;

            drawRoundedRect(ctx, strZoneX, strZoneY, strZoneW, strZoneH, 4);
            ctx.fillStyle = HADES_THEME.bgControl;
            ctx.fill();
            ctx.strokeStyle = "#2c313a";
            ctx.stroke();

            const centerX = strZoneX + btnW;
            const centerW = strZoneW - btnW * 2;
            const norm = Math.max(-1, Math.min(1, strength / 2.0));
            const midX = centerX + centerW * 0.5;
            const barW = Math.abs(norm) * (centerW * 0.5);
            if (barW > 0.5 && isOn && hasLora) {
                ctx.fillStyle = strength >= 0 ? "#2563eb44" : "#ef444444";
                ctx.fillRect(strength >= 0 ? midX : midX - barW, strZoneY + 1, barW, strZoneH - 2);
            }

            ctx.font = "bold 11px monospace";
            ctx.textAlign = "center";
            ctx.fillStyle = isOn ? HADES_THEME.textSecondary : "#4b5263";
            ctx.fillText("−", strZoneX + btnW * 0.5, strZoneY + strZoneH * 0.5 + 0.5);
            ctx.fillText("+", strZoneX + strZoneW - btnW * 0.5, strZoneY + strZoneH * 0.5 + 0.5);

            ctx.font = "11px monospace";
            ctx.fillStyle = effectiveActive ? HADES_THEME.textPrimary : "#6b7280";
            const strText = (strength >= 0 ? "+" : "") + strength.toFixed(2);
            ctx.fillText(strText, centerX + centerW * 0.5, strZoneY + strZoneH * 0.5 + 0.5);

            const comboX = x + 62;
            const comboW = Math.max(40, strZoneX - comboX - 6);
            const comboY = rowY + 3;
            const comboH = h - 6;

            drawRoundedRect(ctx, comboX, comboY, comboW, comboH, 4);
            ctx.fillStyle = "#14161b";
            ctx.fill();
            ctx.strokeStyle = hasLora && isOn ? "#374151" : "#21252b";
            ctx.stroke();

            ctx.save();
            ctx.beginPath();
            ctx.rect(comboX + 6, comboY, comboW - 20, comboH);
            ctx.clip();

            ctx.font = hasLora ? "11px sans-serif" : "italic 11px sans-serif";
            ctx.fillStyle = !isOn || !masterOn ? HADES_THEME.textMuted : hasLora ? "#d1d5db" : "#6b7280";
            ctx.textAlign = "left";
            ctx.fillText(formatCleanLabel(loraVal), comboX + 7, comboY + comboH * 0.5 + 0.5);
            ctx.restore();

            ctx.fillStyle = "#6b7280";
            ctx.font = "9px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("▾", comboX + comboW - 9, comboY + comboH * 0.5 + 0.5);

            this._hitZones = {
                rowY,
                h,
                toggle: { x: toggleX, w: 54 },
                combo: { x: comboX, w: comboW },
                decBtn: { x: strZoneX, w: btnW },
                strCenter: { x: centerX, w: centerW },
                incBtn: { x: strZoneX + strZoneW - btnW, w: btnW },
            };

            ctx.restore();
        },

        mouse(event, pos, n) {
            const zones = this._hitZones;
            if (!zones) return false;
            const [mx, my] = pos;

            if (event.type === "pointerdown" || event.type === "mousedown") {
                if (my < zones.rowY || my > zones.rowY + zones.h) return false;

                if (mx >= zones.toggle.x && mx <= zones.toggle.x + zones.toggle.w) {
                    if (wOn) setWidgetValue(wOn, !wOn.value, n);
                    return true;
                }

                if (mx >= zones.combo.x && mx <= zones.combo.x + zones.combo.w) {
                    const values = wLora?.options?.values || ["None"];
                    new LiteGraph.ContextMenu(values, {
                        event,
                        title: `Slot ${slotIndex} LoRA`,
                        scale: Math.max(1, app.canvas?.ds?.scale || 1),
                        callback: (selected) => {
                            if (wLora && selected !== undefined) {
                                setWidgetValue(wLora, selected, n);
                                if (selected !== "None" && wOn && !wOn.value) {
                                    setWidgetValue(wOn, true, n);
                                }
                            }
                        },
                    });
                    return true;
                }

                if (mx >= zones.decBtn.x && mx <= zones.decBtn.x + zones.decBtn.w) {
                    if (wStrength) {
                        setWidgetValue(wStrength, clampNum(Number(wStrength.value) - 0.05, -10.0, 10.0, 2), n);
                    }
                    return true;
                }

                if (mx >= zones.incBtn.x && mx <= zones.incBtn.x + zones.incBtn.w) {
                    if (wStrength) {
                        setWidgetValue(wStrength, clampNum(Number(wStrength.value) + 0.05, -10.0, 10.0, 2), n);
                    }
                    return true;
                }

                if (mx >= zones.strCenter.x && mx <= zones.strCenter.x + zones.strCenter.w) {
                    if (event.button === 2) {
                        if (wStrength) setWidgetValue(wStrength, 1.0, n);
                        return true;
                    }
                    this._dragStartX = mx;
                    this._dragStartVal = clampNum(wStrength?.value ?? 1.0, -10.0, 10.0, 2);
                    this._didDrag = false;
                    return true;
                }
            } else if (event.type === "pointermove" || event.type === "mousemove") {
                if (this._dragStartX !== null && wStrength) {
                    const dx = mx - this._dragStartX;
                    if (Math.abs(dx) > 2) {
                        this._didDrag = true;
                        setWidgetValue(wStrength, clampNum(this._dragStartVal + dx * 0.01, -10.0, 10.0, 2), n);
                    }
                    return true;
                }
            } else if (event.type === "pointerup" || event.type === "mouseup") {
                if (this._dragStartX !== null) {
                    const wasDrag = this._didDrag;
                    this._dragStartX = null;
                    this._dragStartVal = null;
                    this._didDrag = false;

                    if (!wasDrag && wStrength && app.canvas) {
                        app.canvas.prompt(
                            `Slot ${slotIndex} Strength (-10.0 to 10.0)`,
                            String(wStrength.value),
                            (val) => {
                                const parsed = parseFloat(val);
                                if (!Number.isNaN(parsed)) {
                                    setWidgetValue(wStrength, clampNum(parsed, -10.0, 10.0, 2), n);
                                }
                            },
                            event
                        );
                    }
                    return true;
                }
            }
            return false;
        },
    };
}
