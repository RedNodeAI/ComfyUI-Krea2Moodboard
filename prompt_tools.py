"""RedNode prompt utilities — a cleaner take on the classic prompt-combine node.

Two textboxes by default; add more by chaining "Prompt Textbox (add)" nodes into
`more_parts` (pure-Python packs can't grow widgets dynamically, so extra boxes are
extra nodes — unlimited). `order` rearranges parts without rewiring: 1-based indices
over [text_1, text_2, chained...], e.g. "2,1" swaps, "3,1" puts part 3 first and
appends the rest in natural order. Empty parts are skipped. No help output.
"""

PARTS_TYPE = "REDNODE_PROMPT_PARTS"


class RedNodePromptTextbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"text": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True})},
            "optional": {"chain": (PARTS_TYPE, {"tooltip": "previous Prompt Textbox (add) node — chain as many as you need"})},
        }

    RETURN_TYPES = (PARTS_TYPE,)
    RETURN_NAMES = ("parts",)
    FUNCTION = "add"
    CATEGORY = "conditioning/krea2"
    DESCRIPTION = "One extra textbox for RedNode Prompt Combine; chain several for more."

    def add(self, text, chain=None):
        return ((list(chain) if chain else []) + [text],)


class RedNodePromptCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "text_2": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "separator": ("STRING", {"default": ", ", "tooltip": "placed between parts; type \\n for a newline"}),
                "order": ("STRING", {"default": "", "tooltip": "rearrange parts without rewiring: 1-based indices over [text_1, text_2, chained...], e.g. '2,1' or '3,1'. Unlisted parts follow in natural order. Empty = natural order."}),
            },
            "optional": {
                "more_parts": (PARTS_TYPE, {"tooltip": "chain of Prompt Textbox (add) nodes — parts 3, 4, 5..."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "combine"
    CATEGORY = "conditioning/krea2"
    DESCRIPTION = "Combine prompt parts with a separator. Two boxes by default, unlimited via chained textboxes, reorderable via the order field."

    def combine(self, text_1, text_2, separator=", ", order="", more_parts=None):
        parts = [text_1, text_2] + (list(more_parts) if more_parts else [])

        if order.strip():
            try:
                idx = [int(p) for p in order.replace(";", ",").split(",") if p.strip()]
            except ValueError:
                raise ValueError(f"RedNode Prompt Combine: 'order' must be comma-separated numbers, got {order!r}")
            bad = [i for i in idx if not 1 <= i <= len(parts)]
            if bad:
                raise ValueError(
                    f"RedNode Prompt Combine: 'order' references part {bad[0]} but only {len(parts)} part(s) exist")
            picked = [parts[i - 1] for i in idx]
            picked += [p for n, p in enumerate(parts, 1) if n not in idx]
            parts = picked

        sep = separator.replace("\\n", "\n")
        return (sep.join(p.strip() for p in parts if p and p.strip()),)


NODE_CLASS_MAPPINGS = {
    "RedNodePromptCombine": RedNodePromptCombine,
    "RedNodePromptTextbox": RedNodePromptTextbox,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RedNodePromptCombine": "RedNode Prompt Combine",
    "RedNodePromptTextbox": "RedNode Prompt Textbox (add)",
}
