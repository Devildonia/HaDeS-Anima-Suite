// In-page checks for tests/frontend_test.py. They pin the LiteGraph internals the HaDeS widgets rely on, so a
// ComfyUI frontend update that changes them fails CI instead of reaching users. Returns a list of failure strings.
async (mode) => {
    const app = window.comfyAPI.app.app;
    const failures = [];
    const fail = (msg) => failures.push(`[${mode}] ${msg}`);
    const tick = (ms) => new Promise((r) => setTimeout(r, ms));

    await app.extensionManager.setting.set("Comfy.VueNodes.Enabled", mode === "vue");
    await tick(500);
    app.graph.clear();
    LGraphCanvas.active_canvas = app.canvas;

    const types = Object.keys(LiteGraph.registered_node_types).filter((t) => t.startsWith("HaDeSAnima")).sort();
    if (types.length !== 25) fail(`expected 25 HaDeS node types, got ${types.length}`);

    const nodes = [];
    let x = 0;
    for (const type of types) {
        try {
            const node = LiteGraph.createNode(type);
            app.graph.add(node);
            node.pos = [x, 0];
            x += 600;
            nodes.push(node);
        } catch (e) {
            fail(`${type}: createNode threw ${e}`);
        }
    }
    app.canvas.ds.offset = [0, 0];
    app.canvas.setDirty(true, true);
    try {
        app.canvas.draw(true, true);
    } catch (e) {
        fail(`canvas draw threw ${e}`);
    }
    await tick(mode === "vue" ? 1500 : 200);

    const widgetsModule = await import((await (await fetch("/extensions")).json()).find((u) => u.endsWith("/hades_widgets.js")));

    for (const node of nodes) {
        const t = node.type;
        const styled = (node.widgets || []).filter((w) => w._hadesBackingWidget);
        const backing = (node.widgets || []).filter((w) => w._isHadesBacking);
        if (!node._hasHadesFrameStyle || !node._hadesCustomWidgets?.length) {
            fail(`${t}: HaDeS UI was not installed`);
            continue;
        }

        // Native widgets behind the styled rows stay hidden and take no space
        for (const w of backing) {
            if (!w.hidden || w.computeSize?.(400)?.[1] >= 0) fail(`${t}: backing widget '${w.name}' is visible`);
        }

        for (const row of styled) {
            const input = node.inputs?.find((i) => i.widget?.name === row._hadesBackingWidget.name);
            if (!input) continue;
            // A link dropped on a row goes to that row's input, not to the first input of its type
            if (node.getSlotFromWidget(row) !== input) fail(`${t}: dropping on row '${row.name}' does not target '${input.name}'`);
            // The input socket sits on its row (hidden widgets are skipped by the layout)
            if (mode === "classic" && row.y != null && Math.abs(input.pos?.[1] - (row.y + 14)) > 2) {
                fail(`${t}: socket of '${input.name}' at y=${input.pos?.[1]}, its row is at y=${row.y}`);
            }
        }

        // Vanilla view: the native widgets come back with a real height
        widgetsModule.setNodeClassicUI(node, true, true);
        for (const w of backing) {
            const h = w.computeSize ? w.computeSize(400)[1] : LiteGraph.NODE_WIDGET_HEIGHT;
            if (w.hidden || h <= 0) fail(`${t}: '${w.name}' is missing in the vanilla view`);
        }
        widgetsModule.setNodeClassicUI(node, false, true);

        if (mode === "vue") {
            const el = document.querySelector(`[data-node-id="${node.id}"]`);
            if (!el) fail(`${t}: no Vue node element`);
            else if (el.querySelectorAll("canvas").length < styled.length) fail(`${t}: Vue node renders ${el.querySelectorAll("canvas").length} widget canvases for ${styled.length} styled rows`);
        }
    }

    // Dropdowns open ComfyUI's searchable menu ("Filter list"). ComfyUI adds the box only to lists of more than 4
    // entries, so this uses the sampler list (a fresh install has no model files to fill the loader lists).
    // The row needs to be drawn once to know its bounds.
    app.graph.clear();
    const sampler = LiteGraph.createNode("HaDeSAnimaKSampler");
    app.graph.add(sampler);
    sampler.pos = [50, 50];
    app.canvas.ds.offset = [0, 0];
    app.canvas.ds.scale = 1;
    app.canvas.setDirty(true, true);
    app.canvas.draw(true, true);
    await tick(mode === "vue" ? 1500 : 100);
    const combo = sampler.widgets.find((w) => w.name === "hades_combo_sampler_name");
    if (combo?._bounds) {
        app.canvas.current_node = sampler;
        const b = combo._bounds;
        combo.mouse(new PointerEvent("pointerdown", { clientX: 200, clientY: 200 }), [b.x + 40, b.y + 5], sampler);
        await tick(300);
        const menu = document.querySelector(".litecontextmenu");
        if (!menu?.querySelector(".comfy-context-menu-filter")) fail("sampler dropdown has no filter box");
        menu?.remove();
    } else {
        fail("KSampler sampler dropdown row was not drawn");
    }

    app.graph.clear();
    return failures;
}
