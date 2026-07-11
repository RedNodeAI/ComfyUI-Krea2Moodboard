"""Krea 2 Identity Edit for ComfyUI.

In-context identity/edit conditioning for Krea 2 edit LoRAs (e.g. krea2_identity_edit_v1):
clean source latents are prepended as extra image frames, distinguished from the noisy target
only by the RoPE frame index (sources 1..N, target 0), and the instruction prompt is grounded
through Qwen3-VL with the source image(s). The negative should be grounded too (empty prompt +
same image = the training unconditional), which matters for CFG > 1 recipes.

Mechanics match the verified sd-forge-krea2-edit port (itself ported from
github.com/lbouaraba/comfyui-krea2edit), rebuilt against ComfyUI's stock Krea 2 code.
The core classes are extended at import time; with no reference latents attached the
patched paths are bit-identical to stock, so normal Krea 2 use is unaffected.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.conds
import comfy.ldm.common_dit
import comfy.model_base
import comfy.utils
import node_helpers
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.text_encoders.krea2 import KREA2_TEMPLATE

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


# --------------------------------------------------------------------------
# model_base.Krea2: forward "reference_latents" from the conditioning to the
# DiT (same contract as QwenImage/Flux edit models use).
# --------------------------------------------------------------------------

def _krea2_extra_conds(self, **kwargs):
    out = _orig_extra_conds(self, **kwargs)
    ref_latents = kwargs.get("reference_latents", None)
    if ref_latents is not None:
        out["ref_latents"] = comfy.conds.CONDList([self.process_latent_in(lat) for lat in ref_latents])
    return out


def _krea2_extra_conds_shapes(self, **kwargs):
    out = _orig_extra_conds_shapes(self, **kwargs)
    ref_latents = kwargs.get("reference_latents", None)
    if ref_latents is not None:
        out["ref_latents"] = list([1, 16, sum(map(lambda a: math.prod(a.size()[2:]), ref_latents))])
    return out


# --------------------------------------------------------------------------
# SingleStreamDiT._forward with the in-context source branch.
# Sequence becomes [text | source(s) | target]; positions: text at frame 0,
# source k at frame k (own h/w grid), target at frame 0. Only target tokens
# are returned. Sources are resized to the target latent size in latent space.
# --------------------------------------------------------------------------

def _krea2_forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
    ref_latents = kwargs.get("ref_latents", None) or []
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, c, H_orig, W_orig = x.shape
    patch = self.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    srcs = []
    for source in ref_latents:
        src = source.to(device=x.device, dtype=x.dtype)
        if src.ndim == 5:
            sb, sc, st, sh, sw = src.shape
            src = src.reshape(sb * st, sc, sh, sw)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        if src.shape[-2:] != (H, W):
            src = F.interpolate(src.float(), size=(H, W), mode="bilinear").to(x.dtype)
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch)))

    context = self._unpack_context(context)

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    img = self.first(img)
    src_imgs = [self.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)) for s_ in srcs]

    t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec = self.tproj(t)

    context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    srclen = sum(si.shape[1] for si in src_imgs)
    combined = torch.cat([context] + src_imgs + [img], dim=1)

    device = combined.device
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    srcpos = []
    for i in range(len(src_imgs)):
        sp = imgpos.clone()
        sp[..., 0] = i + 1
        srcpos.append(sp)
    pos = torch.cat([txtpos] + srcpos + [imgpos], dim=1)

    freqs = self.pe_embedder(pos)

    for block in self.blocks:
        combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    final = self.last(combined, t)
    out = final[:, txtlen + srclen:txtlen + srclen + imglen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
    return out


if not getattr(SingleStreamDiT, "_krea2_identity_patched", False):
    _orig_extra_conds = comfy.model_base.Krea2.extra_conds
    _orig_extra_conds_shapes = comfy.model_base.Krea2.extra_conds_shapes
    comfy.model_base.Krea2.extra_conds = _krea2_extra_conds
    comfy.model_base.Krea2.extra_conds_shapes = _krea2_extra_conds_shapes
    SingleStreamDiT._forward = _krea2_forward
    SingleStreamDiT._krea2_identity_patched = True


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class Krea2IdentityEdit:
    """Grounded Krea 2 edit conditioning.

    Use one for the positive (edit instruction) and one for the negative with an EMPTY
    prompt but the SAME image(s). With no image connected it behaves exactly like a plain
    Krea 2 CLIPTextEncode. Two-ref order: scene first, subject second.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "Edit instruction. Leave empty on the negative node."}),
            },
            "optional": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32,
                                         "tooltip": "Cap on the longest side fed to Qwen3-VL (the identity LoRA trained with 384-768px). 0 = never resize."}),
                "fuse_with": ("CONDITIONING", {"tooltip": "Optional conditioning to fuse in front of this one (e.g. Krea 2 Moodboard for scene/style vibe). Its token rows are prepended; this node's identity reference latents are kept. Matches the Neo moodboard+edit fusion layout."}),
                "sources": ("KREA2_SOURCES", {"tooltip": "chained sources (Krea2 Edit Source Chain) — appended after image/image2 as frames 3..N. 3+ refs is beyond the LoRA's training; identities may blend."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/krea2"

    def encode(self, clip, prompt, vae=None, image=None, image2=None, grounding_px=768, fuse_with=None, sources=None):
        all_sources = [image, image2] + (list(sources) if sources else [])
        n_refs = sum(1 for s in all_sources if s is not None)
        if n_refs > 2:
            print(f"[Krea2 Identity Edit] {n_refs} references - the edit LoRA trained on 1-2; expect identity blending beyond that")
        images_vl = []
        ref_latents = []
        vision_prompt = ""
        for img in all_sources:
            if img is None:
                continue
            samples = img.movedim(-1, 1)
            h, w = samples.shape[2], samples.shape[3]
            if grounding_px and max(h, w) > grounding_px:
                scale_by = grounding_px / max(h, w)
                vl = comfy.utils.common_upscale(samples, round(w * scale_by), round(h * scale_by), "area", "disabled")
            else:
                vl = samples
            images_vl.append(vl.movedim(1, -1)[:, :, :, :3])
            if vae is not None:
                # Appearance path: encode at source resolution; the DiT resizes in latent space.
                ref_latents.append(vae.encode(img[:, :, :, :3]))
            vision_prompt += VISION_BLOCK

        print(f"[Krea2 Identity Edit] encoding ({len(images_vl)} ref(s)): {(vision_prompt + prompt)[:120]!r}")
        tokens = clip.tokenize(vision_prompt + prompt, images=images_vl, llama_template=KREA2_TEMPLATE)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        if fuse_with:
            # Fusion layout matches Neo: [moodboard rows][edit grounding + instruction rows].
            # Krea2 text tokens all sit at RoPE position 0, so order only affects attention, not
            # positions. This node's extras (identity ref latents) are the ones that must survive.
            #
            # The fused conditioning ends with the standard template tail
            # ("<|im_end|>\n<|im_start|>assistant\n" = 5 rows) — keeping it would put a
            # description boundary mid-sequence, which K2 can read as a SECOND subject being
            # described (two-people outputs). Trim it so the fusion reads as one description.
            f_cond = fuse_with[0][0]
            if f_cond.shape[1] > 5:
                f_cond = f_cond[:, :-5]
            conditioning = [[torch.cat((f_cond.to(cond.device, cond.dtype), cond), dim=1), extras]
                            for cond, extras in conditioning]
        return (conditioning,)


