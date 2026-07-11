# ComfyUI-Krea2Moodboard

krea.ai-style **moodboard / vibe transfer** for the open **Krea 2** model, as ComfyUI nodes.
Companion to the [Forge Neo version](https://github.com/TdogCreations/forge-neo-krea2-toolkit) — same algorithms, same knobs.

## Nodes

### Krea2 Moodboard Encode
Reference images → style/vibe conditioning. Batch multiple images (Batch Images node) — they are
**packed into ONE vision span** and blend into a joint vibe (repeated spans or "Picture N:" labels
make K2 render N-panel grids; packing is the fix).

- **strength** — 1.0 = raw reference; lower = purer extract. This is an *information* knob, not a
  multiplier (per-token RMSNorms erase plain scaling).
- **extract** — `style / vibe`: spans collapse toward a mean/±std statistics signature (palette,
  texture contrast, mood; subjects fade). `subject / concept`: statistics are whitened away, token
  structure (subjects/composition) survives — your prompt controls the look.
- **reference_processing** — full image / quadrant crops / fine 4×4 tiles (tiles = subjects are
  largely never encoded; strongest style-only setting).
- **indirect** — reference tokens are deleted after the text encoder ran: the DiT never sees them,
  style arrives only through prompt re-contextualization. Structurally cannot copy poses.
- **style_directive** — declarative "style from the references, subjects from the text" sentence
  (auto-matches the extract mode).

Recipes: Krea vibe = style extract, strength 0.5, fine tiles, directive on. Subject transfer =
subject extract, strength 0.4–0.6, full image.

### Krea2 Moodboard + Edit Fusion
Style from the moodboard **plus** identity from an edit source, in one conditioning — designed to
compose with [ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit):

```
LoadImage(source) ─┬─ VAEEncode ─→ Krea2EditModelPatch(model, source_latent) ─→ KSampler
                   └────────────→ Krea2MoodboardEditFusion(edit_source) ─→ positive
LoadImages(style refs) ─────────→ Krea2MoodboardEditFusion(moodboard_images)
```

The moodboard span gets the style effects; the edit grounding span stays raw (span-limited).
Ground the **negative** too: an empty-instruction encode with the same source (Krea2Edit's grounded
encode node, or this node with empty instruction and strength 1.0). krea2_edit LoRA at 1.0.

## Requirements

- ComfyUI with native Krea 2 support; the **qwen3vl_4b** text encoder (with vision weights) loaded
  via CLIPLoader type `krea2`.
- For the fusion node: [ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit) +
  the [krea2_identity_edit LoRA](https://civitai.com/models/2761113).

## Install

Clone into `custom_nodes` (or install via Manager once indexed):

```
git clone https://github.com/TdogCreations/ComfyUI-Krea2Moodboard ComfyUI/custom_nodes/ComfyUI-Krea2Moodboard
```

## How it works / credits

Packs references into a single vision span via a small additive patch to the Qwen3-VL preprocessing,
and applies the moodboard effects inside Krea 2's `encode_token_weights` (pre-template-strip, exact
span indices). No behavior changes when the nodes aren't used.

Credits: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) · [lbouaraba/ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit)
(Apache-2.0) · ethanfel & ostris (K2 vision-conditioning recipes) · Krea.ai (Krea 2, Community License).
License: GPL-3.0 (ComfyUI-compatible). Not affiliated with Krea.ai.
