# ComfyUI-Krea2Moodboard

krea.ai-style **moodboard / vibe transfer** and **identity-preserving editing** for the open
**Krea 2** model, as ComfyUI nodes. Companion to the
[Forge Neo version](https://github.com/RedNodeAI/forge-neo-krea2-toolkit) — same algorithms, same knobs.

**On Civitai** (release zips + showcase):
[civitai.com/models/2794961](https://civitai.com/models/2794961/krea-2-moodboard-identity-edit-comfyui-nodes-forge-neo)

## Nodes

### Krea 2 Moodboard
One-node vibe transfer: prompt + reference image(s) in, conditioning out (replaces `CLIPTextEncode`
on the positive). The image inputs are optional — with nothing connected it behaves exactly like a
plain Krea 2 text encode, so you can leave it wired in and just unplug the references. Knobs:

- **strength** — 1.0 = raw reference detail (layout/pose can leak); lower = purer extract. This is an
  *information* knob, not a multiplier (per-token RMSNorms erase plain scaling).
- **extract** — `style`: palette/lighting/texture/mood survive, subjects fade (spans collapse toward a
  mean/±std statistics signature). `subject`: statistics are whitened away, subject/composition
  survives and your prompt controls the look.
- **reference_processing** — full / 2×2 crops / 4×4 tiles (tiles: subjects are largely never encoded —
  strongest style-only setting).
- **indirect** — reference tokens are deleted after the text encoder ran: the DiT never sees them,
  style arrives only through prompt re-contextualization. Cannot copy poses; also the safe mode for
  crops/multi-ref (deleted spans cannot grid).
- **style_directive** — declarative "style from the refs, subjects from the text" sentence
  (auto-matches the extract mode).

### Krea 2 Moodboard Encode (packed)
The multi-reference specialist: all references (or crops) are packed into **ONE vision span** —
structurally grid-safe, references blend into a joint vibe. Same knobs; `images` is optional
(unconnected = plain Krea 2 text encode). Use it standalone (with a
prompt) or as the **`fuse_with` feeder** for the identity node (empty prompt, `indirect` OFF —
fuse_with concatenates a separate encode, so deleted rows would carry no image influence).

### Krea 2 Moodboard + Identity Fusion
The recommended way to combine style refs with an identity edit: **one node, one encode** — the
instruction and the edit grounding attend the moodboard span inside the encoder, so `indirect`
(default ON) genuinely works: moodboard rows are deleted after encoding, people in your style refs
cannot appear in the output, and the style still transfers. Wire it as the KSampler positive
(connect `vae`!), keep the negative a Krea 2 Identity Edit with an empty prompt + the same source.
Both image inputs are **optional**: connect only `edit_source` for a plain identity edit, only
`moodboard_images` for plain vibe transfer, both for fusion — one node covers all three modes.
The identity-side dials from the edit node are here too: `ref_boost`/`ref_boost_a` (they boost the
identity refs only — the moodboard span is unaffected) and `target_latent` + `fit_mode` (v1.2 fit
geometry for the edit sources).

### Krea 2 Identity Edit
Instruction-based identity-preserving editing with community **krea2_edit LoRAs**
([krea2_identity_edit](https://civitai.com/models/2761113), weights also on
[HF conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit)):
*"create a photo of this person at a night market"* — same face, same outfit, relit. Dual
conditioning: clean source latents ride in-context at RoPE frames 1..N (the LoRA's preserve-this
signal) + the instruction is grounded on the source through Qwen3-VL. `grounding_px` =
likeness↔obedience dial (768 balanced, 1024+ for people).

- Use **two** of these: positive (instruction) + negative (**empty prompt, same image**) — the
  training unconditional, needed for CFG > 1 recipes.
- **`ref_boost`** — reference-fidelity dial: multiplies target→reference attention (additive
  logit bias). 1.0 = off; >1 pulls harder toward the reference's appearance (the v1.2 LoRA author
  suggests 2–6); <1 loosens. Applies to the LAST ref (= the subject); `ref_boost_a` is the same
  dial for the scene ref in two-ref workflows. Set on the positive node only.
- **`target_latent` + `fit_mode`** — connect your (empty) sampling latent to enable the v1.2
  **fit geometry**: refs are fitted in *pixel space* to the output resolution before VAE-encoding.
  Fixes blurry results from resolution mismatches (latents are never resized) and removes the old
  "match the source aspect ratio" requirement (AR-preserving fit at a centered stride-1 offset,
  matching v1.2 training). `crop (legacy)` keeps the v1/v1.1 geometry for older weights. With
  CFG > 1, connect the same latent to the negative node too so both passes share one geometry.
- **`fuse_with`** input: feed a Moodboard Encode conditioning to fuse style-from-moodboard with
  identity-from-source. Fuse the POSITIVE only (style in the negative cancels under CFG).
- Two-ref (experimental upstream): scene in `image`, subject in `image2`.

**v1.2 LoRA notes** (`krea2_identity_edit_v1_2.safetensors`): adds head/face swap, inpaint/outpaint
grounding, try-on, character sheets, and a 1024 high-res pass; on Turbo run 8–12 steps (8 favors
composition, 12 favors face detail). The `fit` default matches how v1.2 was trained; use
`crop (legacy)` with v1/v1.1 weights.

### Krea2 Edit Source Chain
Chainable multi-reference input: each node appends one image; connect chains into the `sources`
input on Identity Edit or the Fusion node (frames 3..N after `image`/`image2`). Unlimited by the
architecture — but the edit LoRA trained on 1–2 references, so 3+ tends to blend identities (the
LoRA author's multi-person recipe is chaining *edit passes* instead: place person A, then run a
second edit adding person B from their reference).

### Krea 2 Conditioning Rebalance
Per-layer reweighting of K2's conditioning: the model conditions on a **12-layer Qwen3-VL stack**
whose taps the DiT mixes linearly — shallow taps carry broad syntax/composition, deep taps carry
fine detail (identity, texture, precise attributes). Insert between any conditioning node and the
sampler. Presets: `balanced` (the classic community profile — deep taps 2.5/5.0/1.1/4.0),
`detail`, `subtle`, `uniform`, or `custom` weights. **`renormalize` (default ON)** holds the
overall magnitude so only the tap *ratios* change — the quality-preserving mode; global
amplification (multiplier > 1) mostly degrades likeness/color. Compatible drop-in for workflows
built around the community "Conditioning Krea2 Rebalance" node (its behavior = `renormalize` OFF,
multiplier 4.0).

## Example workflows (`workflows/`)

Both basic examples run the positive through **Krea 2 Conditioning Rebalance** (`balanced`,
renormalized — set preset to `uniform` to bypass), and the fusion example ships the v1.2 wiring
(`target_latent` connected, ref_boost dials exposed, v1_2 LoRA).

- `krea2_moodboard_t2i.json` — basic vibe transfer text-to-image
- `krea2_identity_edit_fusion.json` — identity edit + moodboard style fusion

### Full pipeline: `Krea_Workflow_Public.json` (advanced)

The author's complete daily-driver workflow: JoyCaption auto-captioning of scene/subject references,
wildcard prompting, identity edit + moodboard fusion, AR handling, optional upscale pass, group
bypass switches. Requires these custom node packs (all installable via ComfyUI-Manager):

| Pack | Used for |
|---|---|
| rgthree-comfy | switches, group bypassers, Power Lora Loader |
| ComfyUI-mxToolkit | sliders |
| ComfyUI-Impact-Pack | wildcard processor |
| ComfyUI-KJNodes | Set/Get nodes |
| ComfyUI_Comfyroll_CustomNodes | prompt combine, aspect ratio, text replace |
| ComfyUI-JoyCaption | image -> prompt captioning (downloads its captioner model on first run) |
| ComfyUI-Custom-Scripts (pythongosssss) | text display |
| comfyui-ollama-describer | Text Transformer (regex caption cleanup) — needs the `ollama` pip package to load (Manager installs it); NO Ollama server required |
| ComfyUI-Image-Saver | sampler selector |
| comfyui_layerstyle | seed node |
| ComfyUI-WhiteRabbit | batch Lanczos resize |
| Derfuu_ComfyUI_ModdedNodes | text box |

The two lean example workflows above need NONE of these — core nodes + this pack only.

Settings baked in: ModelSamplingAuraFlow shift 1.15 (= ComfyUI's stock Krea 2 default — the node
is there as a handle; raise it for Raw-checkpoint recipes), Euler/Simple, Turbo 8 steps CFG 1
(removals: Raw checkpoint, 20–40 steps, CFG 3). With the v1.2 LoRA, 8–12 steps (8 = composition, 12 = face
detail). Generate ≤2MP. Matching the output AR to the source is no longer required when
`target_latent` is connected (fit geometry) — but staying close still gives the best results.

## Requirements

- ComfyUI with native Krea 2 support; **qwen3vl_4b** text encoder (vision weights) via CLIPLoader
  type `krea2`; `qwen_image_vae`.
- For editing: a krea2_edit LoRA at strength 1.0 (LoraLoaderModelOnly).

## Install

```
git clone https://github.com/RedNodeAI/ComfyUI-Krea2Moodboard ComfyUI/custom_nodes/ComfyUI-Krea2Moodboard
```

## How it works / credits

Small additive patches at import: packed list-spans in Qwen3-VL preprocessing, moodboard effects
inside Krea 2's `encode_token_weights`, and the in-context ref-latents branch on the Krea 2 DiT
(`reference_latents` conditioning contract, like QwenImage/Flux edit models). All paths are
bit-identical to stock when the nodes aren't used.

Credits: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) ·
[lbouaraba/ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit) (Apache-2.0 — the
identity-edit dual-conditioning recipe this reimplements) ·
[nova452/ComfyUI-ConditioningKrea2Rebalance](https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance)
& [huwhitememes/comfyui-krea2-conditioning](https://github.com/huwhitememes/comfyui-krea2-conditioning)
(Apache-2.0 — the per-layer rebalance mechanic and its RMS-renormalized variant) · ethanfel & ostris
(K2 vision-conditioning recipes) · Krea.ai (Krea 2, Community License). License: GPL-3.0
(ComfyUI-compatible). Not affiliated with Krea.ai.
