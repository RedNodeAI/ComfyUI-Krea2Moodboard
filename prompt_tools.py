"""RedNode Prompt Combine — prompt combiner with a JS-powered "+ add textbox" button
(web/rednode_prompt.js), Power-Lora-Loader style: one node, unlimited boxes.

Two textboxes by default; the button adds text_3, text_4, ... dynamically. `order`
rearranges parts without rewiring ("2,1", "3,1" — unlisted parts follow in natural
order). Empty parts are skipped. Separator kept (\n escape supported). No help output.
"""


class _FlexText(dict):
    """Accepts any dynamically-added text_N widget as a valid optional STRING input."""

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return ("STRING", {"multiline": True, "default": ""})

    def get(self, key, default=None):
        return self[key]


class RedNodePromptCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "text_2": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "separator": ("STRING", {"default": ", ", "tooltip": "placed between parts; type \\n for a newline"}),
                "order": ("STRING", {"default": "", "tooltip": "rearrange parts without rewiring: 1-based indices, e.g. '2,1' or '3,1'. Unlisted parts follow in natural order. Empty = natural order."}),
            },
            "optional": _FlexText(),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "combine"
    CATEGORY = "conditioning/krea2"
    DESCRIPTION = "Combine prompt parts with a separator. Two boxes by default, '+ add textbox' for more, reorderable via the order field."

    def combine(self, text_1, text_2, separator=", ", order="", **kwargs):
        extras = sorted(
            ((int(k.split("_")[1]), v) for k, v in kwargs.items()
             if isinstance(v, str) and k.startswith("text_") and k.split("_")[1].isdigit()),
            key=lambda t: t[0])
        parts = [text_1, text_2] + [v for _, v in extras]

        if order.strip():
            try:
                idx = [int(p) for p in order.replace(";", ",").split(",") if p.strip()]
            except ValueError:
                raise ValueError(f"RedNode Prompt Combine: 'order' must be comma-separated numbers, got {order!r}")
            bad = [i for i in idx if not 1 <= i <= len(parts)]
            if bad:
                raise ValueError(
                    f"RedNode Prompt Combine: 'order' references part {bad[0]} but only {len(parts)} part(s) exist")
            parts = [parts[i - 1] for i in idx] + [p for n, p in enumerate(parts, 1) if n not in idx]

        sep = separator.replace("\\n", "\n")
        return (sep.join(p.strip() for p in parts if p and p.strip()),)


NODE_CLASS_MAPPINGS = {"RedNodePromptCombine": RedNodePromptCombine}
NODE_DISPLAY_NAME_MAPPINGS = {"RedNodePromptCombine": "RedNode Prompt Combine"}
