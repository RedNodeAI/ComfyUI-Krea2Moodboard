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

CROP_TOL = 0.08  # near-matched-AR tolerance for the "fit" geometry (v1.2 upstream value)


def fit_image_to_latent(image, lat_h, lat_w, fit_mode="fit"):
    """Pixel-space source prep (upstream v1.2 'fit' geometry — the blur/stretch fix): resample
    the IMAGE to the target grid BEFORE VAE-encoding, so latents are never resized (latent
    resizing softens results; plain interpolate stretches mixed-AR sources).

    fit: AR-preserving fit-inside at the target's grid density, /16 floor-snapped and capped at
    the target's /16 floor (training-matched — a different node grid produces a different
    centered offset and a visible margin seam). Near-matched AR fills the target grid exactly
    via a minimal center-crop, restoring fit == crop at matched AR.
    crop: center-crop to the target AR then resize (v1/v1.1 legacy geometry, for older weights).
    """
    px_h, px_w = lat_h * 8, lat_w * 8
    img = image[..., :3].movedim(-1, 1).float()
    ih, iw = img.shape[-2:]
    if fit_mode == "fit":
        sc = min(px_h / ih, px_w / iw)
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0:y0 + ch, x0:x0 + cw]
            nh, nw = px_h, px_w
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
        img = F.interpolate(img, size=(nh, nw), mode="bicubic", antialias=True)
    else:
        s = max(px_h / ih, px_w / iw)
        ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img[..., y0:y0 + ch, x0:x0 + cw]
        img = F.interpolate(img, size=(px_h, px_w), mode="bicubic", antialias=True)
    return img.movedim(1, -1).clamp(0, 1)


def _fit_latent(src, H, W):
    """Latent-space fallback fit: center-crop to the target AR, then resize. Equals the old
    plain-bilinear behavior at matched AR; at mismatched AR it crops instead of stretching."""
    sh, sw = src.shape[-2:]
    if (sh, sw) == (H, W):
        return src
    s = max(H / sh, W / sw)
    ch, cw = min(sh, int(round(H / s))), min(sw, int(round(W / s)))
    y0, x0 = (sh - ch) // 2, (sw - cw) // 2
    src = src[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(src.float(), size=(H, W), mode="bilinear")


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    """Stride-1 position grid for a (gh,gw) reference centered inside a (th,tw) target frame.
    The fit path resamples pixels to the target grid density, so positions stay stride-1 BY
    CONSTRUCTION — scaling them would only manufacture skip/collision artifacts. Equals the
    plain own-grid ids when gh==th and gw==tw (offset 0)."""
    off_h, off_w = max(0, (th - gh) // 2), max(0, (tw - gw) // 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


_warned = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(msg, flush=True)


def _ref_attn_bias(boosts, txtlen, slens, tgtlen, device, dtype):
    """ref_boost: additive attention-logit bias on the [text | refs... | target] sequence —
    target rows get log(boost) on reference-key columns, i.e. it multiplies target->reference
    attention weight before renormalization. Per-ref, aligned with the source blocks (last
    entry = last ref = the subject by workflow convention)."""
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        bias[:, :, rows0:, offs[i]:offs[i] + slens[i]] = math.log(max(b, 1e-4))
    return bias


# --------------------------------------------------------------------------
# model_base.Krea2: forward "reference_latents" from the conditioning to the
# DiT (same contract as QwenImage/Flux edit models use).
# --------------------------------------------------------------------------

def _krea2_extra_conds(self, **kwargs):
    out = _orig_extra_conds(self, **kwargs)
    ref_latents = kwargs.get("reference_latents", None)
    if ref_latents is not None:
        out["ref_latents"] = comfy.conds.CONDList([self.process_latent_in(lat) for lat in ref_latents])
        ref_boosts = kwargs.get("reference_boosts", None)
        if ref_boosts is not None:
            out["ref_boosts"] = comfy.conds.CONDConstant(list(ref_boosts))
        ref_fit = kwargs.get("reference_fit", None)
        if ref_fit is not None:
            out["ref_fit"] = comfy.conds.CONDConstant(list(ref_fit))
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

def _krea2_forward(self, x, timesteps, context, attention_mask=None, *_drift, transformer_options=None, **kwargs):
    # ComfyUI signature-drift guard: older cores call (..., attention_mask, transformer_options);
    # newer cores insert ref_latents positionally: (..., attention_mask, ref_latents,
    # transformer_options). Accept both so stock Krea 2 generation never breaks on update.
    native_ref = None
    drift = list(_drift)
    if drift and isinstance(drift[-1], dict) and transformer_options is None:
        transformer_options = drift.pop()
    if drift:
        native_ref = drift.pop(0)
    if transformer_options is None:
        transformer_options = {}
    ref_latents = kwargs.get("ref_latents", None) or native_ref or []
    ref_boosts = list(kwargs.get("ref_boosts", None) or [])
    ref_fit = list(kwargs.get("ref_fit", None) or [])
    # Defensive alignment: pad from the LEFT so attached values always map to the LAST refs
    # (last ref = the subject by workflow convention, matching the boost semantics).
    n = len(ref_latents)
    ref_boosts = [1.0] * max(0, n - len(ref_boosts)) + ref_boosts[-n:] if n else []
    ref_fit = [False] * max(0, n - len(ref_fit)) + ref_fit[-n:] if n else []
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
    for i, source in enumerate(ref_latents):
        src = source.to(device=x.device, dtype=x.dtype)
        if src.ndim == 5:
            sb, sc, st, sh, sw = src.shape
            src = src.reshape(sb * st, sc, sh, sw)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        # fit-prepared refs that fit inside the target keep their OWN grid (stride-1 offset
        # positions below); everything else is fitted to the target grid in latent space.
        native = ref_fit[i] and src.shape[-2] <= H and src.shape[-1] <= W
        if src.shape[-2:] != (H, W) and not native:
            src = _fit_latent(src, H, W).to(x.dtype)
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
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]
    # _imgids_offset == plain own-grid ids when a ref's grid matches the target's, so this is
    # bit-identical to the old positions for every non-fit ref.
    srcpos = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device) for i, (gh, gw) in enumerate(src_grids)]
    if any(f and (h_ - gh > 2 or w_ - gw > 2) for f, (gh, gw) in zip(ref_fit, src_grids)):
        _warn_once(("fit-margin", tuple(src_grids), (h_, w_)),
                   f"[Krea2 Identity Edit] fit margins >2 tokens (ref grids {src_grids} in target ({h_},{w_})): "
                   "the fit geometry is trained for matched/near-matched aspect ratios. For a big AR "
                   "change prefer 'crop (legacy)' or set the output AR closer to the source.")
    imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    pos = torch.cat([txtpos] + srcpos + [imgpos], dim=1)

    freqs = self.pe_embedder(pos)

    attn_bias = None
    if src_imgs and any(b != 1.0 for b in ref_boosts):
        attn_bias = _ref_attn_bias(ref_boosts, txtlen, [si.shape[1] for si in src_imgs], imglen,
                                   combined.device, combined.dtype)

    for block in self.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

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
                "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                        "tooltip": "reference-fidelity dial: multiplies target->reference attention. Applies to the LAST ref (the subject in two-ref workflows, the only ref in single-ref). 1.0 = off; >1 pulls harder toward the reference's appearance (the v1.2 edit-LoRA author suggests 2-6); <1 loosens. Set on the POSITIVE node; leave the negative at 1.0."}),
                "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                          "tooltip": "same dial for the earlier refs (the scene in two-ref workflows). No effect in single-ref workflows. 1.0 = off"}),
                "target_latent": ("LATENT", {"tooltip": "connect your (empty) sampling latent to enable the v1.2 'fit' geometry: refs are fitted in PIXEL space to the output resolution before VAE-encoding — fixes blur from resolution mismatch and removes the match-the-aspect-ratio requirement. With CFG > 1, connect it to the negative edit node too so both passes share one geometry."}),
                "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                             "tooltip": "how refs fit a mismatched output AR (needs target_latent + vae): fit = resample to the target grid at a centered offset, matching how the v1.2 edit LoRA was trained; crop (legacy) = center-crop to the target AR then resize (v1/v1.1 geometry, for older weights)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/krea2"

    def encode(self, clip, prompt, vae=None, image=None, image2=None, grounding_px=768, fuse_with=None, sources=None,
               ref_boost=1.0, ref_boost_a=1.0, target_latent=None, fit_mode="fit"):
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
                if target_latent is not None:
                    # v1.2 pixel-space path: fit the IMAGE to the output grid, then encode —
                    # the DiT never resizes these latents (blur-proof, AR-safe).
                    lh, lw = target_latent["samples"].shape[-2:]
                    mode = "crop" if fit_mode.startswith("crop") else "fit"
                    fitted = fit_image_to_latent(img[:1], lh, lw, mode)
                    ref_latents.append(vae.encode(fitted))
                else:
                    # legacy path: encode at source resolution; the DiT fits in latent space.
                    ref_latents.append(vae.encode(img[:, :, :, :3]))
            vision_prompt += VISION_BLOCK

        print(f"[Krea2 Identity Edit] encoding ({len(images_vl)} ref(s)): {(vision_prompt + prompt)[:120]!r}")
        tokens = clip.tokenize(vision_prompt + prompt, images=images_vl, llama_template=KREA2_TEMPLATE)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents:
            extra = {"reference_latents": ref_latents,
                     "reference_fit": [target_latent is not None] * len(ref_latents)}
            boosts = [ref_boost_a] * (len(ref_latents) - 1) + [ref_boost]
            if any(b != 1.0 for b in boosts):
                extra["reference_boosts"] = boosts
            conditioning = node_helpers.conditioning_set_values(conditioning, extra, append=True)
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


