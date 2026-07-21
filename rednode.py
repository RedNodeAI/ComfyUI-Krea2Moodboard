"""Krea 2 RedNode — the simple front door.

One node covers the 90% path: identity source(s) + style references + instruction in,
positive AND matching grounded negative out (same sources, same VAE, same fit geometry —
the mismatch class of bugs is structurally impossible). A preset dropdown replaces the
knob wall; the advanced settings node plugs into `settings` to take full manual control
(connected settings replace the preset entirely — all-or-nothing, no field mixing).

Internally this orchestrates the proven nodes (Fusion for the positive, Identity Edit for
the grounded negative), so every code path is the one already validated. The originals
remain available as advanced/legacy nodes.
"""

SETTINGS_TYPE = "KREA2_SETTINGS"

REF_MODES = ["full image", "quadrant crops (2x2)", "fine tiles (4x4)"]

# preset -> full internal config (documented in plain language in the settings node)
PRESETS = {
    "balanced": dict(transfer="style", reference_processing="full image", style_directive=True,
                     hide_style_refs=True, style_detail_px=384, likeness_vs_obedience=768,
                     reference_fidelity=2.5, scene_fidelity=1.0, fit_mode="fit"),
    "max identity": dict(transfer="style", reference_processing="full image", style_directive=True,
                         hide_style_refs=True, style_detail_px=384, likeness_vs_obedience=1024,
                         reference_fidelity=4.0, scene_fidelity=1.0, fit_mode="fit"),
    "style only": dict(transfer="style", reference_processing="full image", style_directive=True,
                       hide_style_refs=True, style_detail_px=768, likeness_vs_obedience=768,
                       reference_fidelity=1.0, scene_fidelity=1.0, fit_mode="fit"),
}


def _fusion_args(cfg):
    """Map plain-language settings onto the legacy fusion/identity node argument names."""
    return dict(
        strength=None,  # filled from the basic node's style_strength
        extract="subject / concept" if cfg["transfer"] == "subject" else "style / vibe",
        reference_processing=cfg["reference_processing"],
        style_directive=bool(cfg["style_directive"]),
        indirect=bool(cfg["hide_style_refs"]),
        budget_px=int(cfg["style_detail_px"]),
        grounding_px=int(cfg["likeness_vs_obedience"]),
        ref_boost=float(cfg["reference_fidelity"]),
        ref_boost_a=float(cfg["scene_fidelity"]),
        fit_mode=cfg["fit_mode"],
    )


class Krea2RedNodeSettings:
    """Advanced control surface: plug into the RedNode's `settings` input to replace the preset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "transfer": (["style", "subject"], {"default": "style",
                         "tooltip": "what the style references contribute: style = palette/lighting/texture/mood; subject = composition/content, the prompt controls the look. (technical: extract mode)"}),
            "reference_processing": (REF_MODES, {"default": "full image",
                         "tooltip": "crops/tiles scramble composition so only style survives from the references"}),
            "style_directive": ("BOOLEAN", {"default": True,
                         "tooltip": "adds a 'style from the refs, subjects from the text' sentence next to the references"}),
            "hide_style_refs": ("BOOLEAN", {"default": True,
                         "tooltip": "style references are deleted after encoding, so the image model never sees them — style transfers via the prompt, poses/people in the refs structurally cannot leak. (technical: indirect mode)"}),
            "style_detail_px": ("INT", {"default": 384, "min": 128, "max": 1536, "step": 64,
                         "tooltip": "resolution budget per style reference fed to the vision encoder. (technical: budget_px)"}),
            "likeness_vs_obedience": ("INT", {"default": 768, "min": 0, "max": 2048, "step": 64,
                         "tooltip": "cap on the identity source fed to the vision encoder: lower = follows the instruction more, higher = preserves likeness more (768 balanced, 1024+ for faces). (technical: grounding_px)"}),
            "reference_fidelity": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.05,
                         "tooltip": "pull toward the identity reference's appearance: 1.0 = off, 2-6 recommended with the v1.2 edit LoRA. (technical: ref_boost on the subject ref)"}),
            "scene_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                         "tooltip": "same dial for the scene reference in two-source setups. (technical: ref_boost_a)"}),
            "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                         "tooltip": "fit = v1.2 pixel-space geometry (blur-proof, any aspect ratio); crop = v1/v1.1 legacy behavior"}),
        }}

    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = "conditioning/krea2"
    DESCRIPTION = "Advanced settings for Krea 2 RedNode. Connecting this replaces the preset entirely."

    def build(self, **kwargs):
        return (dict(kwargs),)


class Krea2RedNode:
    """Simple front door: sources + instruction in, positive + grounded negative out."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "instruction": ("STRING", {"multiline": True, "dynamicPrompts": True,
                               "tooltip": "what to make, e.g. 'create a photo of this person at a night market'"}),
                "preset": (list(PRESETS), {"default": "balanced",
                           "tooltip": "balanced = identity + style fusion; max identity = strongest face lock; style only = pure vibe transfer. Connect a RedNode Settings node to take full manual control instead."}),
                "style_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "how much of the style references survives: 1.0 = raw reference detail, lower = purer style extract"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "required when a subject or scene image is connected"}),
                "subject_image": ("IMAGE", {"tooltip": "the person/subject to preserve — the face you want kept"}),
                "scene_image": ("IMAGE", {"tooltip": "optional environment/scene reference to place the subject into (two-ref edit LoRAs; ordering is handled for you)"}),
                "moodboard_style": ("IMAGE", {"tooltip": "style/vibe reference images — batch several for a joint moodboard"}),
                "extra_subjects": ("KREA2_SOURCES", {"tooltip": "chain more references (Krea2 Edit Source Chain); 3+ is beyond the edit LoRA's training"}),
                "output_latent": ("LATENT", {"tooltip": "connect the SAME empty latent you feed the sampler — enables the v1.2 blur-proof fit geometry on both outputs"}),
                "settings": (SETTINGS_TYPE, {"tooltip": "optional Krea 2 RedNode Settings node — replaces the preset entirely when connected"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "conditioning/krea2"
    DESCRIPTION = "Moodboard + identity edit in one node, with a matched grounded negative output. Presets for the common modes; plug in RedNode Settings for full control."

    def encode(self, clip, instruction, preset, style_strength, vae=None, subject_image=None,
               scene_image=None, moodboard_style=None, extra_subjects=None, output_latent=None,
               settings=None):
        import sys
        pkg = sys.modules[__package__]

        if settings is not None:
            cfg = dict(settings)
            print("[Krea2 RedNode] settings node connected - preset ignored")
        else:
            cfg = dict(PRESETS[preset])
            print(f"[Krea2 RedNode] preset '{preset}'")
        args = _fusion_args(cfg)
        args["strength"] = float(style_strength)

        # training order for two-ref edit LoRAs is scene first, subject second — handled here
        # so users never have to know it.
        if scene_image is not None:
            ref1, ref2 = scene_image, subject_image
        else:
            ref1, ref2 = subject_image, None

        (positive,) = pkg.Krea2MoodboardIdentityFusion().encode(
            clip=clip, instruction=instruction,
            edit_source=ref1, edit_source2=ref2,
            moodboard_images=moodboard_style, vae=vae,
            sources=extra_subjects, target_latent=output_latent, **args)

        from .identity import Krea2IdentityEdit
        (negative,) = Krea2IdentityEdit().encode(
            clip=clip, prompt="", vae=vae,
            image=ref1, image2=ref2, sources=extra_subjects,
            grounding_px=args["grounding_px"],
            target_latent=output_latent, fit_mode=args["fit_mode"])

        return (positive, negative)
