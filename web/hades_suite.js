import { app } from "../../scripts/app.js";
import { setGlobalClassicUI } from "./hades_widgets.js";
import { MERGER_NODE_DISPATCH } from "./hades_mergers_ui.js";
import { PIPELINE_NODE_DISPATCH } from "./hades_pipeline_ui.js";
import { KSAMPLER_NODE_DISPATCH } from "./hades_ksampler_ui.js";
import { LOADER_NODE_DISPATCH } from "./hades_loaders_ui.js";

// =============================================================================
// HADES ANIMA SUITE UI - EXTENSION ORCHESTRATOR (24 NODES)
// =============================================================================
const NODE_SETUP_DISPATCH = {
    ...MERGER_NODE_DISPATCH,
    ...PIPELINE_NODE_DISPATCH,
    ...KSAMPLER_NODE_DISPATCH,
    ...LOADER_NODE_DISPATCH,
};

app.registerExtension({
    name: "HaDeS.AnimaSpectralSuiteUI",
    settings: [
        {
            id: "HaDeS.ClassicUI",
            category: ["HaDeS Anima Suite", "Appearance", "Classic widgets"],
            name: "Use standard ComfyUI widgets on HaDeS nodes",
            type: "boolean",
            defaultValue: false,
            tooltip: "Hides the HaDeS canvas widgets and shows the plain ComfyUI inputs on every HaDeS node.",
            onChange(value) {
                setGlobalClassicUI(Boolean(value));
            },
        },
    ],
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
