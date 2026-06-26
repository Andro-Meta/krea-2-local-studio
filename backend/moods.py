"""Moodboard presets for Krea 2 — local equivalent of Krea's hand-curated moods.

Krea's cloud moodboards are stored server-side by UUID and can't be used with the
open-source weights. Each Krea preset is described as a taste profile with
"keywords" and "avoids". We reproduce that structure locally: a mood prepends
its keywords to the prompt and adds its avoids to the negative prompt, scaled by
moodboard_strength. Combine with a custom image board (reference images via the
Qwen3-VL multimodal encoder) for a full local moodboard.

"Retro Web" and "Expressive Marker" mirror Krea's two publicly-documented presets;
the rest are an original curated set covering Krea's aesthetic range plus
Utopia / Dystopia / Horror category packs.
"""
from __future__ import annotations

# id, name, emoji, category, keywords (added to prompt), avoids (added to negative)
MOODS: list[dict] = [
    # ----- Classic -----
    {"id": "retro_web", "name": "Retro Web", "emoji": "🕸️", "category": "Classic",
     "keywords": "late-90s web aesthetic, pixelated 3d-collage, rendered objects and stickers, early CGI, glossy chrome, slightly chaotic web 1.0 graphics",
     "avoids": "photorealistic, modern minimal, clean flat design"},
    {"id": "expressive_marker", "name": "Expressive Marker", "emoji": "🖍️", "category": "Classic",
     "keywords": "character-driven illustration, funny stylized marker drawing, bold expressive linework, personality over realism, playful cartoon energy",
     "avoids": "photoreal, literal rendering, stiff, corporate"},
    {"id": "cinematic_film", "name": "Cinematic Film", "emoji": "🎬", "category": "Classic",
     "keywords": "cinematic still, anamorphic lens, shallow depth of field, dramatic key light, color-graded teal and orange, filmic contrast, 35mm",
     "avoids": "flat lighting, snapshot, oversaturated, low detail"},
    {"id": "editorial_portrait", "name": "Editorial Portrait", "emoji": "📸", "category": "Classic",
     "keywords": "editorial portrait photography, soft window light, natural skin texture, 85mm, magazine cover quality, refined color",
     "avoids": "plastic skin, harsh flash, distorted face, cartoon"},
    {"id": "product_studio", "name": "Product Studio", "emoji": "💡", "category": "Classic",
     "keywords": "studio product photography, seamless backdrop, soft gradient lighting, crisp reflections, material focus, commercial render",
     "avoids": "cluttered background, harsh shadows, noise, amateur"},
    {"id": "fashion_editorial", "name": "Fashion Editorial", "emoji": "👗", "category": "Classic",
     "keywords": "high fashion editorial, lookbook aesthetic, dramatic styling, bold composition, glossy magazine grade, couture",
     "avoids": "casual snapshot, dull lighting, low fashion, plain"},
    {"id": "anime_cel", "name": "Anime Cel", "emoji": "🌸", "category": "Classic",
     "keywords": "anime cel shading, clean lineart, vibrant flat colors, expressive eyes, studio anime key frame, crisp 2d",
     "avoids": "photorealistic, 3d render, muddy colors, western cartoon"},
    {"id": "oil_painting", "name": "Oil Painting", "emoji": "🎨", "category": "Classic",
     "keywords": "classical oil painting, visible brushstrokes, rich impasto texture, old master lighting, painterly canvas, fine art",
     "avoids": "digital flat, photo, smooth airbrush, low detail"},
    {"id": "watercolor", "name": "Watercolor", "emoji": "💧", "category": "Classic",
     "keywords": "soft watercolor painting, bleeding pigments, paper texture, delicate washes, loose organic edges, light and airy",
     "avoids": "hard edges, digital, heavy contrast, photoreal"},
    {"id": "scifi_concept", "name": "Sci-Fi Concept", "emoji": "🚀", "category": "Classic",
     "keywords": "sci-fi concept art, dramatic scale, volumetric lighting, intricate hard-surface design, cinematic atmosphere, matte painting",
     "avoids": "flat, cartoon, low detail, mundane"},
    {"id": "vaporwave", "name": "Vaporwave", "emoji": "🌴", "category": "Classic",
     "keywords": "vaporwave aesthetic, neon pink and cyan, retro 80s grid, glitch art, chrome and palm, dreamy synthwave",
     "avoids": "natural color, muted, realistic, dull"},
    {"id": "film_noir", "name": "Film Noir", "emoji": "🕵️", "category": "Classic",
     "keywords": "film noir, high-contrast black and white, hard chiaroscuro shadows, venetian blind light, moody 1940s cinema, dramatic",
     "avoids": "colorful, bright even lighting, cheerful, flat"},
    {"id": "golden_hour", "name": "Golden Hour", "emoji": "🌅", "category": "Classic",
     "keywords": "golden hour light, warm sun flare, long soft shadows, glowing rim light, hazy atmosphere, dreamy warmth",
     "avoids": "cold light, midday flat, overcast, dull"},
    {"id": "cyberpunk_neon", "name": "Cyberpunk Neon", "emoji": "🌃", "category": "Classic",
     "keywords": "cyberpunk city, neon signage, rain-slick streets, volumetric fog, moody blue and magenta, blade-runner atmosphere, cinematic",
     "avoids": "daylight, rural, clean, pastel"},
    {"id": "pastel_dream", "name": "Pastel Dream", "emoji": "🍬", "category": "Classic",
     "keywords": "soft pastel palette, dreamy diffused light, gentle gradients, airy and delicate, kawaii soft aesthetic",
     "avoids": "high contrast, dark, gritty, saturated neon"},
    {"id": "brutalist_mono", "name": "Brutalist Mono", "emoji": "🏛️", "category": "Classic",
     "keywords": "brutalist concrete, monochrome, stark geometric forms, raw textured surfaces, dramatic hard shadow, minimal severe",
     "avoids": "ornate, colorful, soft, cluttered"},
    {"id": "vintage_35mm", "name": "Vintage 35mm", "emoji": "🎞️", "category": "Classic",
     "keywords": "vintage 35mm film photograph, kodak portra grain, faded colors, light leaks, nostalgic analog warmth, slight vignette",
     "avoids": "digital clarity, oversharp, neon, clean modern"},
    {"id": "storybook", "name": "Storybook", "emoji": "📖", "category": "Classic",
     "keywords": "storybook illustration, whimsical hand-drawn, warm gouache textures, charming character design, childrens book art",
     "avoids": "photoreal, dark, gritty, 3d render"},
    {"id": "dark_fantasy", "name": "Dark Fantasy", "emoji": "🗡️", "category": "Classic",
     "keywords": "dark fantasy art, moody atmospheric, intricate detail, dramatic low-key lighting, epic painterly, ominous mood",
     "avoids": "bright cheerful, flat, cartoon, simple"},
    {"id": "minimalist", "name": "Minimalist", "emoji": "⬜", "category": "Classic",
     "keywords": "minimalist composition, lots of negative space, limited palette, clean simple forms, calm balanced, refined",
     "avoids": "cluttered, busy, ornate, chaotic"},

    # ----- Utopia -----
    {"id": "solarpunk", "name": "Solarpunk", "emoji": "🌿", "category": "Utopia",
     "keywords": "solarpunk utopia, lush vertical gardens, renewable energy, sunlit eco-architecture, hopeful green future, art nouveau technology",
     "avoids": "dystopian, gray, polluted, ruined, dark"},
    {"id": "crystal_utopia", "name": "Crystal Utopia", "emoji": "💎", "category": "Utopia",
     "keywords": "gleaming crystalline city, translucent spires, prismatic light, pristine futuristic architecture, radiant utopia",
     "avoids": "decay, dark, gritty, ruined, polluted"},
    {"id": "floating_isles", "name": "Floating Isles", "emoji": "🏞️", "category": "Utopia",
     "keywords": "floating sky islands, verdant terraced gardens, waterfalls into clouds, serene aerial utopia, soft sunlight",
     "avoids": "ground-level decay, urban grime, dark, ruined"},
    {"id": "golden_age", "name": "Golden Age", "emoji": "🏛️", "category": "Utopia",
     "keywords": "idealized golden age, classical marble architecture, warm prosperous light, harmonious utopian grandeur, serene",
     "avoids": "modern, gritty, ruined, dark, industrial"},
    {"id": "bio_harmony", "name": "Bio-Harmony", "emoji": "🌱", "category": "Utopia",
     "keywords": "living architecture, bioluminescent flora fused with structures, organic eco-futurism, harmonious nature and technology",
     "avoids": "industrial, sterile, ruined, polluted, dark"},
    {"id": "celestial_paradise", "name": "Celestial Paradise", "emoji": "☁️", "category": "Utopia",
     "keywords": "heavenly paradise, ethereal golden light, soft luminous clouds, divine serene atmosphere, radiant utopia",
     "avoids": "dark, grim, earthly decay, gritty"},
    {"id": "pristine_future", "name": "Pristine Future", "emoji": "🤍", "category": "Utopia",
     "keywords": "clean white utopian future, minimalist gleaming surfaces, soft ambient light, optimistic sci-fi, spotless and bright",
     "avoids": "dirty, decayed, cluttered, dark, gritty"},
    {"id": "arcadian", "name": "Arcadian Idyll", "emoji": "🌾", "category": "Utopia",
     "keywords": "pastoral arcadian paradise, rolling green meadows, soft idealized nature, peaceful golden idyll, romantic landscape",
     "avoids": "urban, dystopian, dark, industrial, ruined"},
    {"id": "luminous_metropolis", "name": "Luminous Metropolis", "emoji": "🌇", "category": "Utopia",
     "keywords": "bright optimistic megacity, clean soaring towers, vibrant daylight, hopeful prosperous future, gleaming skyline",
     "avoids": "dark, oppressive, polluted, ruined, grimy"},
    {"id": "garden_tomorrow", "name": "Garden of Tomorrow", "emoji": "🌷", "category": "Utopia",
     "keywords": "abundant futuristic gardens, blooming flora everywhere, harmony of life and design, lush vibrant utopia, sunlit",
     "avoids": "barren, gray, dystopian, dark, ruined"},
    {"id": "aurora_haven", "name": "Aurora Haven", "emoji": "🌌", "category": "Utopia",
     "keywords": "serene aurora-lit haven, northern lights, tranquil glowing sky, peaceful utopian calm, ethereal colors",
     "avoids": "harsh, gritty, dark dystopia, ruined"},
    {"id": "oceanic_utopia", "name": "Oceanic Utopia", "emoji": "🐚", "category": "Utopia",
     "keywords": "radiant coastal paradise, turquoise waters, gleaming seaside utopia, sunlit harmony, pristine ocean architecture",
     "avoids": "polluted, dark, ruined, grimy, bleak"},
    {"id": "harmonic_spires", "name": "Harmonic Spires", "emoji": "🕊️", "category": "Utopia",
     "keywords": "elegant white utopian spires, clear blue sky, graceful flowing architecture, peaceful advanced civilization, serene light",
     "avoids": "brutalist, dark, decayed, oppressive"},
    {"id": "radiant_commune", "name": "Radiant Commune", "emoji": "🌻", "category": "Utopia",
     "keywords": "warm communal utopia, soft inviting light, people in harmony, cozy hopeful future, gentle abundance, golden glow",
     "avoids": "cold, isolated, dystopian, dark, bleak"},

    # ----- Dystopia -----
    {"id": "wasteland", "name": "Wasteland", "emoji": "🏜️", "category": "Dystopia",
     "keywords": "post-apocalyptic wasteland, cracked barren earth, rusted debris, harsh desolate horizon, lone survivor atmosphere",
     "avoids": "lush, clean, utopian, vibrant, thriving"},
    {"id": "industrial_decay", "name": "Industrial Decay", "emoji": "🏭", "category": "Dystopia",
     "keywords": "decaying industrial sprawl, rusted machinery, smog-choked sky, grimy abandoned factories, bleak gritty atmosphere",
     "avoids": "clean, natural, bright, utopian, pristine"},
    {"id": "surveillance_state", "name": "Surveillance State", "emoji": "📹", "category": "Dystopia",
     "keywords": "cold authoritarian dystopia, looming propaganda screens, gray monolithic architecture, oppressive order, bleak conformity",
     "avoids": "warm, free, lush, colorful, hopeful"},
    {"id": "toxic_megacity", "name": "Toxic Megacity", "emoji": "☢️", "category": "Dystopia",
     "keywords": "polluted dystopian megacity, acid-green haze, toxic smog, grimy oppressive neon, hazardous decay, choking atmosphere",
     "avoids": "clean, fresh, natural, bright, healthy"},
    {"id": "ruined_empire", "name": "Ruined Empire", "emoji": "🏚️", "category": "Dystopia",
     "keywords": "crumbling fallen empire, broken monuments, overgrown ruins, faded former glory, melancholic decay, somber light",
     "avoids": "pristine, new, thriving, bright, intact"},
    {"id": "brutalist_dystopia", "name": "Brutalist Dystopia", "emoji": "🧱", "category": "Dystopia",
     "keywords": "oppressive brutalist megastructures, endless gray concrete, cold dehumanizing scale, bleak authoritarian architecture",
     "avoids": "warm, ornate, natural, colorful, inviting"},
    {"id": "nuclear_winter", "name": "Nuclear Winter", "emoji": "❄️", "category": "Dystopia",
     "keywords": "frozen nuclear wasteland, ash-gray snow, dead skeletal trees, desolate cold apocalypse, bleak muted palette",
     "avoids": "warm, lush, vibrant, alive, bright"},
    {"id": "corporate_hellscape", "name": "Corporate Hellscape", "emoji": "🏢", "category": "Dystopia",
     "keywords": "endless corporate gray towers, drone-filled sky, dehumanizing megacorp dystopia, cold sterile oppression, overcast",
     "avoids": "warm, natural, free, colorful, human"},
    {"id": "flooded_world", "name": "Flooded World", "emoji": "🌊", "category": "Dystopia",
     "keywords": "submerged drowned city, climate-collapse flood, half-sunken buildings, murky still water, desolate abandonment",
     "avoids": "dry, pristine, thriving, bright, clean"},
    {"id": "dust_bowl", "name": "Dust Bowl", "emoji": "🌫️", "category": "Dystopia",
     "keywords": "sepia dust-choked desolation, dried cracked land, swirling dust storms, depression-era bleakness, faded grim tones",
     "avoids": "vibrant, lush, clean, colorful, fresh"},
    {"id": "underground_bunker", "name": "Underground Bunker", "emoji": "🔦", "category": "Dystopia",
     "keywords": "claustrophobic underground bunker, dim flickering lights, grimy concrete tunnels, oppressive enclosure, survival dread",
     "avoids": "open, bright, airy, natural, spacious"},
    {"id": "dystopian_slums", "name": "Dystopian Slums", "emoji": "🏘️", "category": "Dystopia",
     "keywords": "overcrowded dystopian slums, makeshift shanty sprawl, tangled wires, grimy oppressive neon, dense urban decay",
     "avoids": "clean, spacious, affluent, bright, orderly"},
    {"id": "ash_metropolis", "name": "Ash Metropolis", "emoji": "🌆", "category": "Dystopia",
     "keywords": "ash-covered ruined metropolis, smoldering broken skyline, gray apocalyptic haze, abandoned grandeur, somber desolation",
     "avoids": "thriving, clean, bright, lush, intact"},

    # ----- Horror -----
    {"id": "cosmic_horror", "name": "Cosmic Horror", "emoji": "🐙", "category": "Horror",
     "keywords": "lovecraftian cosmic horror, eldritch presence, impossible geometry, cosmic dread, looming tentacled silhouette, eerie sickly glow",
     "avoids": "cute, bright, safe, cheerful, colorful"},
    {"id": "gothic_horror", "name": "Gothic Horror", "emoji": "🦇", "category": "Horror",
     "keywords": "victorian gothic horror, foggy moonlit graveyard, decaying mansion, dread atmosphere, candlelit shadows, ominous mood",
     "avoids": "bright, modern, cheerful, colorful, safe"},
    {"id": "haunted_mansion", "name": "Haunted Mansion", "emoji": "🕯️", "category": "Horror",
     "keywords": "decrepit haunted mansion, ghostly apparitions, dust and cobwebs, flickering candlelight, eerie creeping dread",
     "avoids": "new, bright, cheerful, lively, clean"},
    {"id": "eldritch_void", "name": "Eldritch Void", "emoji": "🌑", "category": "Horror",
     "keywords": "abyssal eldritch void, writhing shadow tendrils, oppressive consuming darkness, otherworldly horror, dim sickly light",
     "avoids": "bright, safe, colorful, cute, cheerful"},
    {"id": "psychological_horror", "name": "Psychological Horror", "emoji": "🩻", "category": "Horror",
     "keywords": "unsettling psychological horror, surreal wrongness, distorted uncanny figure, creeping dread, muted desaturated palette",
     "avoids": "cheerful, normal, bright, safe, comforting"},
    {"id": "folk_horror", "name": "Folk Horror", "emoji": "🌾", "category": "Horror",
     "keywords": "rural folk horror, eerie pagan ritual, isolated countryside dread, overcast unease, ominous masks and effigies",
     "avoids": "urban, bright, cheerful, modern, safe"},
    {"id": "liminal_dread", "name": "Liminal Dread", "emoji": "🚪", "category": "Horror",
     "keywords": "liminal space dread, empty fluorescent backrooms, uncanny endless hallways, unsettling emptiness, eerie still silence",
     "avoids": "lively, warm, populated, cheerful, cozy"},
    {"id": "cursed_forest", "name": "Cursed Forest", "emoji": "🌲", "category": "Horror",
     "keywords": "twisted cursed forest, gnarled dead trees, creeping fog, sickly moonlight, lurking dread, dark fairytale horror",
     "avoids": "sunny, lush, cheerful, safe, vibrant"},
    {"id": "abandoned_asylum", "name": "Abandoned Asylum", "emoji": "🏥", "category": "Horror",
     "keywords": "decaying abandoned asylum, peeling walls, rusted gurneys, cold institutional dread, grimy flickering horror",
     "avoids": "clean, bright, modern, cheerful, safe"},
    {"id": "nightmare_fuel", "name": "Nightmare Fuel", "emoji": "😱", "category": "Horror",
     "keywords": "surreal nightmare, distorted melting forms, irrational dread, dark dreamlike horror, disturbing uncanny atmosphere",
     "avoids": "calm, normal, bright, pleasant, comforting"},
    {"id": "slasher_night", "name": "Slasher Night", "emoji": "🔪", "category": "Horror",
     "keywords": "moonlit slasher horror, ominous lurking menace, suburban night dread, deep shadows, suspenseful tension, eerie quiet",
     "avoids": "bright, safe, cheerful, daylight, calm"},
    {"id": "witch_hollow", "name": "Witch's Hollow", "emoji": "🧹", "category": "Horror",
     "keywords": "eerie witch's hollow, misty swamp, hanging moss, candle-lit ritual, dark folk dread, ominous green gloom",
     "avoids": "bright, clean, cheerful, modern, safe"},
    {"id": "abyssal_deep", "name": "Abyssal Deep", "emoji": "🦑", "category": "Horror",
     "keywords": "deep-sea horror, bioluminescent abyss, unknown lurking leviathan, crushing dark depths, eerie cold dread",
     "avoids": "bright, shallow, safe, cheerful, sunny"},
]

HORROR_PHOTO_KEYWORDS = (
    "photorealistic horror photography, documentary realism, grainy analog film, "
    "visible film grain, low-light practical lighting, natural lens texture, "
    "real-world physical detail, unsettling candid realism"
)
HORROR_PHOTO_AVOIDS = (
    "illustration, painting, cartoon, anime, CGI, plastic render, overly clean digital art, "
    "airbrushed fantasy art"
)

for _m in MOODS:
    if _m["category"] == "Horror":
        _m["keywords"] = f"{HORROR_PHOTO_KEYWORDS}, {_m['keywords']}"
        _m["avoids"] = f"{_m['avoids']}, {HORROR_PHOTO_AVOIDS}"

_BY_ID = {m["id"]: m for m in MOODS}


def get_mood(mood_id: str) -> dict | None:
    return _BY_ID.get(mood_id)


def apply_mood(prompt: str, negative: str, mood_id: str) -> tuple[str, str]:
    """Return (prompt, negative) augmented with the mood's keywords/avoids."""
    ids = [m.strip() for m in mood_id.split(",") if m.strip()]
    moods = [m for m in (get_mood(mid) for mid in ids) if m]
    if not moods:
        return prompt, negative
    keywords = ", ".join(m["keywords"] for m in moods)
    avoids = ", ".join(m["avoids"] for m in moods)
    new_prompt = f"{keywords}, {prompt}" if prompt.strip() else keywords
    new_neg = f"{negative}, {avoids}" if negative.strip() else avoids
    return new_prompt, new_neg


if __name__ == "__main__":
    from collections import Counter
    cats = Counter(m["category"] for m in MOODS)
    ids = [m["id"] for m in MOODS]
    assert len(ids) == len(set(ids)), "duplicate mood id!"
    p, n = apply_mood("a city", "", "solarpunk")
    assert "solarpunk" in p and "dystopian" in n
    print(f"{len(MOODS)} moods, no dup ids. by category: {dict(cats)}")
