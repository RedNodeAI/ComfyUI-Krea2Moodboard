"""ComfyUI-Krea2Moodboard — moodboard / vibe transfer + identity editing for the open Krea 2 model.

Nodes:
- Krea 2 Moodboard (moodboard.py): one-node vibe transfer — prompt + reference images in,
  conditioning out. Effects applied post-encode (strength, style/subject extract, crops, indirect,
  directives). Multiple refs/crops form separate vision spans (use indirect if outputs grid).
- Krea2 Moodboard Encode (this file): the packed-span variant — multiple references are packed into
  ONE vision span (structurally grid-safe, references blend jointly). Best as the `fuse_with` feeder.
- Krea 2 Identity Edit (identity.py): instruction-based identity-preserving editing for krea2_edit
  LoRAs (e.g. krea2_identity_edit_v1) — dual conditioning (in-context ref latents at RoPE frames
  1..N + image-grounded instruction), with a `fuse_with` input to fuse a moodboard conditioning in
  front (style from the moodboard, identity from the edit source).

Strength is an information knob, not a magnitude knob (per-token RMSNorms erase scaling): "style"
extract collapses spans toward a mean/±std statistics signature, "subject" whitens the statistics
away and keeps the token structure. Ported from the authors' Forge Neo implementation.
"""

import math

import torch

import comfy.text_encoders.krea2
import comfy.text_encoders.qwen3vl
from comfy.text_encoders.krea2 import KREA2_TEMPLATE, Krea2TEModel

from .identity import Krea2IdentityEdit
from .moodboard import Krea2Moodboard

STYLE_DIRECTIVE = (
    "The image uses only the art style, color palette, lighting, texture, rendering technique and "
    "overall mood of the reference images. The subjects, objects and composition of the image come "
    "from the following text description alone. "
)
SUBJECT_DIRECTIVE = (
    "The image depicts the same subjects, objects and composition as the reference images, "
    "rendered in the art style described by the following text. "
)
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# Sizes of spliced vision spans, recorded in splice order during the current encode.
_SPAN_SIZES = []


# ---------------------------------------------------------------------------
# Patch 1: packed multi-image spans — Qwen3VL.preprocess_embed accepts a LIST of images and
# concatenates their tokens into one contiguous span (single image reference to the model).
# ---------------------------------------------------------------------------
_orig_preprocess = comfy.text_encoders.qwen3vl.Qwen3VL.preprocess_embed


def _packed_preprocess_embed(self, embed, device):
    data = embed.get("data", None)
    if isinstance(data, (list, tuple)):
        merged_all, deepstack_all, grids = [], [], []
        for img in data:
            merged, extra = _orig_preprocess(self, {"type": "image", "data": img, "original_type": "image"}, device)
            merged_all.append(merged)
            deepstack_all.append(extra["deepstack"] if isinstance(extra, dict) else None)
            grids.append(extra["grid"] if isinstance(extra, dict) else extra)
        merged = torch.cat(merged_all, dim=0)
        deepstack = None
        if deepstack_all and deepstack_all[0] is not None:
            deepstack = [torch.cat([ds[i] for ds in deepstack_all], dim=0) for i in range(len(deepstack_all[0]))]
        _SPAN_SIZES.append(merged.shape[0])
        return merged, {"grid": grids[0], "deepstack": deepstack, "packed": True}

    merged, extra = _orig_preprocess(self, embed, device)
    if merged is not None:
        _SPAN_SIZES.append(merged.shape[0])
    return merged, extra


comfy.text_encoders.qwen3vl.Qwen3VL.preprocess_embed = _packed_preprocess_embed


# ---------------------------------------------------------------------------
# Patch 2: moodboard effects inside Krea2's encode (pre-template-strip, spans known exactly).
# Controlled by attributes set on clip.cond_stage_model by the nodes; no-ops otherwise.
# ---------------------------------------------------------------------------
_orig_encode_token_weights = Krea2TEModel.encode_token_weights


def _apply_moodboard_effects(model, out, spans, extra):
    """out: (B, 12, seq, 2560) pre-strip. spans: [(start, end)] post-splice indices."""
    limit = getattr(model, "moodboard_span_limit", None)
    if limit is not None:
        spans = spans[:limit]
    if not spans:
        return out, extra

    if getattr(model, "moodboard_hide_refs", False):
        keep = torch.ones(out.shape[2], dtype=torch.bool, device=out.device)
        for start, end in spans:
            keep[start:end] = False
        out = out[:, :, keep]
        if "attention_mask" in extra:
            extra["attention_mask"] = extra["attention_mask"][:, keep]
        return out, extra

    strength = float(getattr(model, "moodboard_strength", 1.0))
    if strength >= 1.0:
        return out, extra
    strength = max(0.0, strength)

    joint = torch.cat([out[:, :, start:end] for start, end in spans], dim=2)
    mu = joint.mean(dim=2, keepdim=True)
    sigma = joint.std(dim=2, keepdim=True)

    if getattr(model, "moodboard_extract", "style") == "subject":
        safe_sigma = sigma.clamp_min(1e-4)
        for start, end in spans:
            span = out[:, :, start:end]
            whitened = (span - mu) / safe_sigma
            out[:, :, start:end] = whitened + strength * (span - whitened)
    else:
        for start, end in spans:
            span = out[:, :, start:end]
            n = span.shape[2]
            coef = torch.tensor([0.0, 1.0, -1.0], device=span.device, dtype=span.dtype).repeat((n + 2) // 3)[:n].view(1, 1, n, 1)
            target = mu + coef * sigma
            out[:, :, start:end] = target + strength * (span - target)
    return out, extra


def _moodboard_encode_token_weights(self, token_weight_pairs, template_end=-1):
    _SPAN_SIZES.clear()
    out, pooled, extra = super(Krea2TEModel, self).encode_token_weights(token_weight_pairs)
    tok_pairs = token_weight_pairs["qwen3vl_4b"][0]

    import numbers

    # Strip index (original logic) — always resolves before the first image span.
    count_im_start = 0
    if template_end == -1:
        for i, v in enumerate(tok_pairs):
            elem = v[0]
            if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral):
                if elem == 151644 and count_im_start < 2:
                    template_end = i
                    count_im_start += 1
        if out.shape[2] > (template_end + 3):
            if tok_pairs[template_end + 1][0] == 872:
                if tok_pairs[template_end + 2][0] == 198:
                    template_end += 3

    # Post-splice span positions: walk the (pre-splice) token pairs, expanding image entries by the
    # sizes recorded during preprocessing (same order).
    spans = []
    offset = 0
    sizes = iter(list(_SPAN_SIZES))
    for i, v in enumerate(tok_pairs):
        elem = v[0]
        if isinstance(elem, dict):
            n = next(sizes, 0)
            start = i + offset
            spans.append((start, start + n))
            offset += n - 1

    out, extra = _apply_moodboard_effects(self, out, spans, extra)

    out = out[:, :, template_end:]
    b, n, seq, h = out.shape
    out = out.permute(0, 2, 1, 3).reshape(b, seq, n * h)

    if "attention_mask" in extra:
        extra["attention_mask"] = extra["attention_mask"][:, template_end:]
        if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
            extra.pop("attention_mask")

    return out, pooled, extra


Krea2TEModel.encode_token_weights = _moodboard_encode_token_weights


# ---------------------------------------------------------------------------
# Packed-span moodboard node
# ---------------------------------------------------------------------------
def expand_style_crops(images, n=2):
    orders = {
        2: (2, 0, 3, 1),
        4: (10, 3, 12, 5, 0, 15, 6, 9, 2, 13, 4, 11, 8, 1, 14, 7),
    }
    order = orders.get(n) or tuple(range(n * n))
    crops = []
    for image in images:
        h, w = image.shape[1] // n, image.shape[2] // n
        grid = [image[:, r * h:(r + 1) * h, c * w:(c + 1) * w] for r in range(n) for c in range(n)]
        crops.extend(grid[i] for i in order)
    return crops


def resize_area(image, total_px, never_upscale=False):
    samples = image.movedim(-1, 1)
    scale_by = math.sqrt(total_px / (samples.shape[3] * samples.shape[2]))
    if never_upscale:
        scale_by = min(1.0, scale_by)
    height = max(32, round(samples.shape[2] * scale_by))
    width = max(32, round(samples.shape[3] * scale_by))
    samples = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
    return samples.movedim(1, -1)[:, :, :, :3]


def set_flags(clip, strength=1.0, hide=False, extract="style", span_limit=None):
    model = clip.cond_stage_model
    model.moodboard_strength = strength
    model.moodboard_hide_refs = hide
    model.moodboard_extract = extract
    model.moodboard_span_limit = span_limit


REF_MODES = ["full image", "quadrant crops (2x2)", "fine tiles (4x4)"]


class Krea2MoodboardEncode:
    """Packed-span moodboard: all references share ONE vision span (grid-safe, joint blending).
    Leave the prompt empty when feeding Krea 2 Identity Edit's `fuse_with` input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "images": ("IMAGE", {"tooltip": "reference images (batch them for multiple)"}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "1.0 = raw reference detail; lower = purer extract of the selected aspect"}),
                "extract": (["style / vibe", "subject / concept"],),
                "reference_processing": (REF_MODES,),
                "style_directive": ("BOOLEAN", {"default": True}),
                "indirect": ("BOOLEAN", {"default": False, "tooltip": "hide reference tokens from the DiT: style arrives only via prompt re-contextualization; cannot copy pose/subject"}),
                "position": (["before prompt", "after prompt"],),
                "budget_px": ("INT", {"default": 384, "min": 128, "max": 1536, "step": 64,
                                      "tooltip": "area budget per reference fed to the vision encoder"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/krea2"
    DESCRIPTION = "Moodboard conditioning with all references packed into one vision span. Use standalone (with prompt) or as the fuse_with feeder for Krea 2 Identity Edit (empty prompt)."

    def encode(self, clip, prompt, images, strength, extract, reference_processing, style_directive, indirect, position, budget_px):
        refs = [images[i:i + 1] for i in range(images.shape[0])]
        crops_n = 4 if "4x4" in reference_processing else 2 if "2x2" in reference_processing else 0
        total = int(budget_px) * int(budget_px)
        if crops_n:
            refs = expand_style_crops(refs, n=crops_n)
            total = total // (3 if crops_n == 2 else 12)
        refs = [resize_area(r, total, never_upscale=(budget_px >= 1024)) for r in refs]

        extract_key = "subject" if extract.startswith("subject") else "style"
        directive = ""
        if style_directive:
            directive = SUBJECT_DIRECTIVE if extract_key == "subject" else STYLE_DIRECTIVE

        if position == "after prompt":
            text = prompt + (" " + directive if directive else "") + VISION_BLOCK
        else:
            text = VISION_BLOCK + directive + prompt

        set_flags(clip, strength=float(strength), hide=bool(indirect), extract=extract_key, span_limit=None)
        try:
            tokens = clip.tokenize(text, images=[refs], llama_template=KREA2_TEMPLATE)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        finally:
            set_flags(clip)
        return (conditioning,)


class Krea2MoodboardIdentityFusion:
    """Single-encode fusion: moodboard style + identity-edit source in ONE LLM pass, exactly like
    the Forge Neo implementation. Because the instruction and the edit grounding attend the
    moodboard span inside the encoder, `indirect` genuinely works here (unlike feeding a separate
    moodboard encode into `fuse_with`, where deleting rows deletes all image influence).

    Use for the KSampler POSITIVE; keep the negative a Krea 2 Identity Edit with an EMPTY prompt
    and the same source image. Requires a krea2_edit LoRA at strength 1.0 on the model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "instruction": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "the edit instruction, e.g. 'create a photo of this person at a night market'"}),
                "edit_source": ("IMAGE", {"tooltip": "identity source image"}),
                "moodboard_images": ("IMAGE", {"tooltip": "style references (batch for several)"}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "extract": (["style / vibe", "subject / concept"],),
                "reference_processing": (REF_MODES,),
                "style_directive": ("BOOLEAN", {"default": True}),
                "indirect": ("BOOLEAN", {"default": True, "tooltip": "delete moodboard rows after encoding; style survives via in-encoder attention. Safest when style refs contain people."}),
                "budget_px": ("INT", {"default": 384, "min": 128, "max": 1536, "step": 64}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 2048, "step": 32,
                                         "tooltip": "longest-side cap for the edit source fed to the encoder"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "connect to attach the in-context identity latents (required for actual editing)"}),
                "edit_source2": ("IMAGE", {"tooltip": "2nd reference for two-ref LoRAs (scene first, subject second)"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/krea2"
    DESCRIPTION = "Moodboard style + identity edit fused in a single encode (Neo-parity). Positive only; negative = Krea 2 Identity Edit with empty prompt + same image."

    def encode(self, clip, instruction, edit_source, moodboard_images, strength, extract, reference_processing, style_directive, indirect, budget_px, grounding_px, vae=None, edit_source2=None):
        import comfy.utils
        import node_helpers

        # moodboard side: crops + area budget, packed into one span
        refs = [moodboard_images[i:i + 1] for i in range(moodboard_images.shape[0])]
        crops_n = 4 if "4x4" in reference_processing else 2 if "2x2" in reference_processing else 0
        total = int(budget_px) * int(budget_px)
        if crops_n:
            refs = expand_style_crops(refs, n=crops_n)
            total = total // (3 if crops_n == 2 else 12)
        refs = [resize_area(r, total, never_upscale=(budget_px >= 1024)) for r in refs]

        extract_key = "subject" if extract.startswith("subject") else "style"
        directive = ""
        if style_directive:
            directive = SUBJECT_DIRECTIVE if extract_key == "subject" else STYLE_DIRECTIVE

        # edit side: grounding images + in-context ref latents (training-matched order: scene, subject)
        edit_images = []
        ref_latents = []
        edit_blocks = ""
        for img in (edit_source, edit_source2):
            if img is None:
                continue
            samples = img[:1].movedim(-1, 1)
            h, w = samples.shape[2], samples.shape[3]
            if grounding_px and max(h, w) > grounding_px:
                scale_by = grounding_px / max(h, w)
                samples = comfy.utils.common_upscale(samples, round(w * scale_by), round(h * scale_by), "area", "disabled")
            edit_images.append(samples.movedim(1, -1)[:, :, :, :3])
            if vae is not None:
                ref_latents.append(vae.encode(img[:1, :, :, :3]))
            edit_blocks += VISION_BLOCK

        text = VISION_BLOCK + directive + edit_blocks + instruction
        images = [refs] + edit_images  # span 1 = packed moodboard; spans 2.. = edit grounding

        set_flags(clip, strength=float(strength), hide=bool(indirect), extract=extract_key, span_limit=1)
        try:
            tokens = clip.tokenize(text, images=images, llama_template=KREA2_TEMPLATE)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        finally:
            set_flags(clip)

        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "Krea2Moodboard": Krea2Moodboard,
    "Krea2MoodboardEncode": Krea2MoodboardEncode,
    "Krea2IdentityEdit": Krea2IdentityEdit,
    "Krea2MoodboardIdentityFusion": Krea2MoodboardIdentityFusion,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2Moodboard": "Krea 2 Moodboard",
    "Krea2MoodboardEncode": "Krea 2 Moodboard Encode (packed)",
    "Krea2IdentityEdit": "Krea 2 Identity Edit",
    "Krea2MoodboardIdentityFusion": "Krea 2 Moodboard + Identity Fusion",
}
