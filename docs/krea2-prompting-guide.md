# Krea 2 Studio Prompting Guide

This guide is for the local Krea 2 Studio in this repository. It combines current public Krea 2 guidance with the behavior of this app: Turbo/RAW checkpoints, mood presets, style stacks, LoRAs, reference images, prompt expansion, and the sharing UI.

## Mental Model

Krea 2 is aesthetic-first. It is not only trying to obey a literal prompt; it is good at exploring visual directions, texture, lighting, medium, and mood. A short prompt can be useful when you want variety. A structured prompt is better when you already know the subject, composition, and style.

Use this loop:

1. Start with a clear seed idea.
2. Generate a few directions.
3. Keep the strongest visual language.
4. Add camera, lighting, material, mood, and composition details.
5. Use moodboards, style stacks, LoRAs, or reference images to lock the look.

## Best Default Formula

Use this order for reliable prompts:

```text
Subject + setting + composition + lighting + mood + medium/style + texture/camera detail
```

Example:

```text
A lone woman in a raincoat standing at the edge of a cornfield, medium close-up portrait, overcast dusk, practical low light from a farmhouse window, eerie folk horror mood, photorealistic 35mm film photograph, visible grain, damp fabric texture, shallow depth of field
```

For simple exploration, start smaller:

```text
A haunted roadside motel at night
```

Then refine after seeing what Krea gives you:

```text
A haunted roadside motel at night, seen from across an empty wet highway, flickering red neon sign, low-angle cinematic composition, fog, analog horror photography, grainy 35mm film, muted sodium-vapor lighting
```

## Turbo vs RAW

For this app, use Turbo for normal generation.

Recommended Turbo settings:

- Steps: `8`
- CFG: `0.0`
- Mu: `1.15`
- Quantization: `fp8` when using the installed turbo fp8 checkpoint

RAW is mainly useful for training, LoRA work, research, or slower exploratory generation. Public Krea 2 guidance recommends training LoRAs on RAW and running them on Turbo.

## Prompt Length

Krea 2 does not need a giant prompt. Too many style adjectives can muddy the output, especially when a moodboard or reference image already carries the look.

Good prompt length:

- Exploration: 5-20 words.
- Controlled image: 30-80 words.
- Complex scene: 80-140 words.

Avoid long comma soup. Prefer clear clauses:

```text
A weathered fisherman sits alone in a dim kitchen. Rain streaks the window behind him. The image is a natural-light documentary photograph with muted blue-gray tones, worn wood textures, and subtle film grain.
```

## Photography Prompts

Use real camera and physical cues:

- `photorealistic`
- `documentary photography`
- `35mm film`
- `50mm lens`
- `shallow depth of field`
- `available light`
- `practical lighting`
- `motion blur`
- `visible film grain`
- `natural skin texture`
- `low dynamic range`
- `slight underexposure`

Example:

```text
A nervous teenager in a dim convenience store aisle at 2 a.m., handheld documentary photograph, 35mm lens, harsh fluorescent ceiling lights, slight motion blur, visible film grain, realistic skin texture, muted colors, uncanny quiet mood
```

## Horror Photography

All Horror mood presets in this app are now grounded in photorealistic horror photography by default. They add language like documentary realism, grainy analog film, low-light practical lighting, and natural lens texture. They also avoid illustration, painting, cartoon, anime, CGI, plastic render, and overly clean digital art.

For best horror results:

- Use ordinary real-world settings.
- Keep the threat partly hidden.
- Name the light source.
- Add grain, imperfect exposure, and lens texture.
- Avoid overexplaining the monster.

Strong horror structure:

```text
Real place + human-scale composition + practical light + implied threat + photographic medium
```

Examples:

```text
A boarded-up farmhouse hallway at midnight, viewed from waist height, a single flashlight beam catching dust in the air, something just beyond the open basement door, photorealistic analog horror photograph, visible film grain, underexposed shadows
```

```text
A lone figure in a raincoat standing motionless at the far edge of a cornfield, overcast dusk, long-lens documentary photo, muted greens and browns, wet fabric, grainy 35mm film, unsettling folk horror realism
```

## Moodboards And Style Stacks

Mood presets add keywords to the prompt and avoid-terms to the negative prompt. In this app you can select multiple presets to build a Style Stack.

Good stacks:

- `Gothic Horror + Vintage 35mm + Film Noir`
- `Folk Horror + Editorial Portrait`
- `Liminal Dread + Brutalist Mono`
- `Cosmic Horror + Cinematic Film`
- `Slasher Night + Golden Hour` for a warmer but still threatening look

Use two or three presets at most. More than that can blur the direction.

Moodboard strength:

- `0.25`: light flavor
- `0.50`: balanced
- `0.70`: strong style direction
- `0.90+`: dominant style, less literal prompt adherence

## Reference Images

Use reference images when you want a specific look, not just a subject.

Best uses:

- A color palette
- A lighting style
- A lens/film texture
- A composition pattern
- A character or product look

Tips:

- Keep the written prompt about content.
- Let reference images carry visual style.
- If the image reference is strong, reduce style adjectives in the text.
- Use multiple references only when they agree visually.

## Magic Wand Backends

The prompt magic wand can expand a short prompt into a stronger image prompt.

Available backends:

- Local Qwen3-VL: default. Uses the bundled support model already used for local image understanding.
- OpenRouter: optional hosted backend. Uses the configured OpenRouter model and free-model fallback.
- Ideogram JSON: calls Ideogram's hosted Magic Prompt API, receives structured JSON, then flattens it into a Krea-friendly paragraph.

Local Qwen3-VL is the self-contained default. OpenRouter can be useful when you want hosted model quality, and Ideogram JSON can help when a short prompt needs richer scene structure. Krea itself does not consume Ideogram JSON directly, so this app converts that JSON into a normal paragraph.

If the wand appears to do nothing, check the snackbar:

- Local Qwen3-VL missing: use System > Krea Moodboard Conditioning / Local AI Assets to download or repair the model.
- OpenRouter rate-limited: wait or switch back to Local Qwen3-VL.
- Missing API key: add it in System > Magic Wand.
- No change returned: the model returned the original prompt.

## LoRA Prompting

LoRAs usually need trigger words or visual cues.

Use:

```text
main subject, LoRA trigger phrase, setting, composition, style/mood
```

Keep LoRA strength moderate at first. If the LoRA overpowers the scene, reduce strength or simplify the prompt.

## Common Problems

### Output Is Too Generic

Add physical detail:

- material
- lens
- light source
- time of day
- camera distance
- color palette
- texture

### Output Is Too Stylized

Add:

- `photorealistic`
- `documentary photography`
- `natural lens texture`
- `real-world detail`

And avoid:

- `illustration`
- `CGI`
- `digital painting`
- `cartoon`

### Output Ignores Prompt Details

Reduce competing style instructions. Use fewer mood presets. Lower moodboard strength. Put the subject first.

### Horror Looks Like Concept Art

Use a Horror preset and add:

```text
photorealistic analog horror photography, practical low light, visible film grain, documentary realism, natural lens imperfections
```

### Prompt Is Too Literal Or Boring

Use the magic wand, raise moodboard strength slightly, or add a style stack.

## Reliable Recipes

### Grainy Folk Horror Portrait

```text
A solitary farmer standing in a dead cornfield at dusk, centered medium portrait, overcast sky, distant farmhouse barely visible through fog, photorealistic folk horror photography, 35mm film grain, muted earth tones, practical natural light, unsettling stillness
```

Suggested stack: `Folk Horror + Vintage 35mm + Editorial Portrait`

### Cosmic Horror Realism

```text
A research team finds an impossible black monolith half-buried in Arctic ice, wide documentary photograph, headlamps and snow glare, tiny human figures for scale, photorealistic cosmic horror, visible film grain, cold desaturated palette, unsettling naturalism
```

Suggested stack: `Cosmic Horror + Cinematic Film`

### Liminal Flash Photo

```text
An empty hotel corridor at 3 a.m., direct on-camera flash, beige walls, patterned carpet, one door slightly open at the far end, photorealistic liminal horror photography, harsh shadows, low-resolution analog grain, uncanny silence
```

Suggested stack: `Liminal Dread + Vintage 35mm`

### Product Shot

```text
A matte black perfume bottle on wet obsidian stone, close studio product photograph, softbox reflection, subtle rim light, shallow depth of field, luxury editorial composition, crisp material texture
```

Suggested stack: `Product Studio + Minimalist`

### Character Concept Without Looking Like AI Art

```text
A weathered detective in a narrow alley after rain, three-quarter portrait, sodium streetlight behind him, damp wool coat, realistic face texture, cinematic noir photograph, 50mm lens, muted color grade, grounded urban detail
```

Suggested stack: `Film Noir + Cinematic Film`

## Source Notes

This guide was informed by public Krea 2 prompting material from Krea, fal.ai, the Krea 2 open-source repository/model cards, Hugging Face Diffusers Krea 2 docs, and local behavior verified in this app.
