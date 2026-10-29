import { app } from "../../scripts/app.js";
import { setGlobalClassicUI } from "./hades_widgets.js";
import { MERGER_NODE_DISPATCH } from "./hades_mergers_ui.js";
import { PIPELINE_NODE_DISPATCH } from "./hades_pipeline_ui.js";
import { KSAMPLER_NODE_DISPATCH } from "./hades_ksampler_ui.js";
import { LOADER_NODE_DISPATCH } from "./hades_loaders_ui.js";

// =============================================================================
// HADES ANIMA SPECTRAL SUITE UI - EXTENSION ORCHESTRATOR (17 NODES)
// =============================================================================
const NODE_SETUP_DISPATCH = {
    ...MERGER_NODE_DISPATCH,
    ...PIPELINE_NODE_DISPATCH,
    ...KSAMPLER_NODE_DISPATCH,
    ...LOADER_NODE_DISPATCH,
};

app.registerExtension({
    name: "HaDeS.AnimaSpectralSuiteUI",
    async setup() {
        if (app?.ui?.settings?.addSetting) {
            app.ui.settings.addSetting({
                id: "HaDeS.ClassicUI",
                name: "HaDeS Suite: Force Classic ComfyUI Theme (Vanilla)",
                type: "boolean",
                defaultValue: false,
                tooltip: "When enabled, hides HaDeS spectral neon widgets and shows standard ComfyUI controls across all 17 nodes",
                onChange(value) {
                    setGlobalClassicUI(Boolean(value));
                },
            });
            const savedVal = app.ui.settings.getSettingValue?.("HaDeS.ClassicUI");
            if (savedVal !== undefined && savedVal !== null) {
                setGlobalClassicUI(Boolean(savedVal));
            }
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const installer = NODE_SETUP_DISPATCH[nodeData.name];
        if (!installer) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            this.serialize_widgets = true;
            installer(this);
        };
    },
});
