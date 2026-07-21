import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// "+ add textbox" button for RedNode Prompt Combine (Power-Lora-Loader style growth).
app.registerExtension({
  name: "rednode.promptcombine",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "RedNodePromptCombine") return;

    function addBox(node, value = "") {
      const n = node.widgets.filter((w) => /^text_\d+$/.test(w.name || "")).length + 1;
      const w = ComfyWidgets.STRING(node, "text_" + n, ["STRING", { multiline: true, default: "" }], app).widget;
      w.value = value;
      const bi = node.widgets.findIndex((x) => x.type === "button");
      if (bi !== -1) node.widgets.push(node.widgets.splice(bi, 1)[0]);
      node.setSize(node.computeSize());
      return w;
    }

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const btn = this.addWidget("button", "+ add textbox", null, () => addBox(this));
      btn.serialize = false;
      btn.serializeValue = () => undefined;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      // recreate saved extra boxes: values beyond [text_1, text_2, separator, order]
      const vals = (info.widgets_values || []).filter((v) => v !== null && v !== undefined);
      const extras = Math.max(0, vals.length - 4);
      const have = this.widgets.filter((w) => /^text_\d+$/.test(w.name || "")).length - 2;
      for (let i = have; i < extras; i++) addBox(this);
      onConfigure?.apply(this, arguments);
    };
  },
});
