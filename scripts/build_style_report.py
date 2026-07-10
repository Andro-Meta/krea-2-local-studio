"""Build a self-contained HTML report of the moodboard / reference-image style study.

Reads the result composites (and a few full-res examples) produced by the
experiment scripts, base64-embeds them (so the file opens anywhere with no broken
links), and writes an easy-to-read report with the prompts, reference images,
results, and findings.

Run:  venv\\Scripts\\python.exe scripts\\build_style_report.py
Output: outputs/moodboard_style_report/report.html
"""
from __future__ import annotations
import base64, io, sys, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
OUT_DIR = ROOT / "outputs"
REPORT_DIR = OUT_DIR / "moodboard_style_report"


def data_uri(path: Path, max_w: int = 1700, quality: int = 84) -> str | None:
    """Load an image, downscale to max_w, return a JPEG data URI (or None if missing)."""
    if not path or not path.exists():
        return None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if img.width > max_w:
            img = img.resize((max_w, round(img.height * max_w / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[report] embed failed for {path}: {e}")
        return None


def pil_data_uri(img, max_w: int = 220, quality: int = 82) -> str:
    from PIL import Image  # noqa
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def hydrate_refs(board_id: int, n: int = 5):
    """Return a list of PIL reference thumbnails for a board (network only, no GPU)."""
    try:
        import asyncio
        from moodboards_catalog import get_moodboard, _moodboard_image_urls, fetch_moodboard_image_b64
        from PIL import Image
        item = asyncio.run(get_moodboard(board_id))
        if not item:
            return [], ""
        title = str(item.get("title", ""))
        urls = _moodboard_image_urls([item.get("primary_image_url", ""), *(item.get("image_urls") or [])])
        out = []
        for url in urls:
            if len(out) >= n:
                break
            try:
                b = fetch_moodboard_image_b64(url)
                out.append(Image.open(io.BytesIO(base64.b64decode(b.split(",")[-1]))).convert("RGB"))
            except Exception:
                pass
        return out, title
    except Exception as e:
        print(f"[report] hydrate failed for board {board_id}: {e}")
        return [], ""


def refs_html(board_id: int, fallback_glob: str = "") -> str:
    imgs, title = hydrate_refs(board_id)
    uris = [pil_data_uri(im) for im in imgs]
    if not uris and fallback_glob:
        for p in sorted((OUT_DIR / "moodboard_channels" / "refs").glob(fallback_glob)):
            u = data_uri(p, max_w=220)
            if u:
                uris.append(u)
    if not uris:
        return "<p class='muted'>(reference thumbnails unavailable)</p>"
    thumbs = "".join(f"<img class='ref' src='{u}' alt='reference'/>" for u in uris)
    cap = f"Board #{board_id}" + (f" — “{html.escape(title)}”" if title else "")
    return f"<div class='refrow'>{thumbs}</div><div class='refcap'>{cap} · these are the reference images fed to the engine</div>"


def fig(path: Path, caption: str, max_w: int = 1700) -> str:
    u = data_uri(path, max_w=max_w)
    if not u:
        return f"<p class='muted'>(missing: {html.escape(str(path.name))})</p>"
    return f"<figure><img src='{u}' alt='{html.escape(caption)}'/><figcaption>{caption}</figcaption></figure>"


def row(paths_caps: list[tuple[Path, str]], max_w: int = 640) -> str:
    cells = []
    for p, cap in paths_caps:
        u = data_uri(p, max_w=max_w)
        if u:
            cells.append(f"<figure class='cell'><img src='{u}'/><figcaption>{cap}</figcaption></figure>")
    return f"<div class='rowfigs'>{''.join(cells)}</div>"


def prompt_box(label: str, text: str) -> str:
    return f"<div class='prompt'><span class='plabel'>{label}</span><code>{html.escape(text)}</code></div>"


def finding(text: str) -> str:
    return f"<div class='finding'><strong>Finding.</strong> {text}</div>"


def note(text: str) -> str:
    return f"<div class='note'>{text}</div>"


MC = OUT_DIR / "moodboard_channels"
STY = OUT_DIR / "moodboard_style_experiments"
ADV = STY / "advanced"
ZEL = STY / "zelda"
FP = STY / "fullpicture"
FP2 = STY / "fullpicture2"

SUBJECT_PORTRAIT = "a portrait of a young woman standing on a city street, looking at the camera"
SUBJECT_PUPPY = "a golden retriever puppy sitting in a sunny green meadow, bright daylight"
SUBJECT_CAR = "a red vintage sports car on a desert highway at noon"
SUBJECT_BERRIES = "a bowl of fresh strawberries on a white marble kitchen counter"
STYLE_SYS = ("Describe ONLY the visual style of the image: color palette, lighting, contrast, artistic "
             "medium and technique, brushwork or rendering, texture, and overall mood. Do NOT describe "
             "or mention the specific subjects, people, animals, objects, or their spatial composition.")


def build() -> str:
    css = """
    :root{--bg:#15131f;--panel:#1e1b2e;--ink:#ece8f5;--muted:#a49fb5;--line:rgba(202,196,208,.14);
      --accent:#d0bcff;--good:#7ee0a2;--warn:#ffd479;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
    .wrap{max-width:1120px;margin:0 auto;padding:32px 20px 96px}
    h1{font-size:30px;margin:0 0 6px} h2{font-size:23px;margin:38px 0 6px;color:var(--accent)}
    h3{font-size:18px;margin:22px 0 6px} .sub{color:var(--muted);margin:0 0 20px}
    p{margin:10px 0} a{color:var(--accent)}
    .toc{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 20px;margin:18px 0 28px}
    .toc ol{margin:6px 0;padding-left:22px} .toc li{margin:3px 0}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:18px 0}
    figure{margin:14px 0;text-align:center} figure img{max-width:100%;border-radius:10px;border:1px solid var(--line)}
    figcaption{color:var(--muted);font-size:13px;margin-top:6px}
    .rowfigs{display:flex;flex-wrap:wrap;gap:12px;justify-content:center}
    .rowfigs .cell{flex:1 1 260px;max-width:340px;margin:0}
    .refrow{display:flex;flex-wrap:wrap;gap:8px}
    img.ref{height:120px;border-radius:8px;border:1px solid var(--line)}
    .refcap{color:var(--muted);font-size:13px;margin-top:6px}
    .prompt{background:#0e0c17;border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:8px 0}
    .prompt .plabel{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}
    .prompt code{color:#e6dcff;font:14px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap}
    .finding{background:rgba(126,224,162,.10);border-left:3px solid var(--good);border-radius:6px;padding:10px 14px;margin:14px 0}
    .note{background:rgba(255,212,121,.09);border-left:3px solid var(--warn);border-radius:6px;padding:10px 14px;margin:14px 0;font-size:14px;color:#f0e6cf}
    .muted{color:var(--muted)} table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
    th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
    th{background:#26223a;color:var(--accent)} code.k{background:#0e0c17;padding:1px 6px;border-radius:5px;color:#e6dcff}
    .rec{background:linear-gradient(180deg,rgba(208,188,255,.14),rgba(208,188,255,.05));border:1px solid rgba(208,188,255,.4);
      border-radius:14px;padding:18px 22px;margin:20px 0}
    """
    P = []
    P.append(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>")
    P.append(f"<title>Krea 2 — Moodboard & Reference-Image Style Study</title><style>{css}</style></head><body><div class='wrap'>")

    P.append("<h1>Krea 2 — Moodboard &amp; Reference-Image Style Study</h1>")
    P.append("<p class='sub'>How reference images influence Krea 2, why they copied content, and the recipe that transfers <em>style</em> instead. The goal throughout is to carry the <b>feel of a chosen moodboard — any style, not “realism”</b> — into the image you're making. All images below are real outputs from the experiment scripts.</p>")

    # TOC
    P.append("""<div class='toc'><strong>Contents</strong><ol>
      <li><a href='#bg'>Background: two different things</a></li>
      <li><a href='#mech'>How the reference path actually works</a></li>
      <li><a href='#e1'>Experiment 1 — Moodboard channels (text vs image vs fusion)</a></li>
      <li><a href='#e2'>Experiment 2 — Why image refs made a split image</a></li>
      <li><a href='#e3'>Experiment 3 — Style vs content: the levers</a></li>
      <li><a href='#e4'>Experiment 4 — Refining &amp; confirming across subjects/boards</a></li>
      <li><a href='#e5'>Experiment 5 — Megapixel sweep (0.30–1.00)</a></li>
      <li><a href='#e6'>Experiment 6 — Full sweep (0.10–1.00) + anchors</a></li>
      <li><a href='#r2'>Round 2 — Reference count, strength±, anti-style, board blend, attribute isolation</a></li>
      <li><a href='#tr'>Transforming a style-locked subject (Link &amp; Zelda + realism)</a></li>
      <li><a href='#mash'>Meaningful mashups of clashing styles</a></li>
      <li><a href='#fp'>Full picture — combine, restyle, engine, consistency</a></li>
      <li><a href='#fp2'>Feel transfer II — resolution, your-own-image blend, text economy, per-region</a></li>
      <li><a href='#ideas'>Unique workflow ideas</a></li>
      <li><a href='#rec'>Conclusion &amp; recommendation</a></li>
    </ol></div>""")

    # Background
    P.append("<div class='card' id='bg'><h2>1 · Background: two different things</h2>")
    P.append("<p>Two features were being conflated. They work completely differently:</p>")
    P.append("""<table><tr><th>Feature</th><th>Direction</th><th>What it does</th></tr>
      <tr><td><b>Create&nbsp;from&nbsp;image</b></td><td>image → <b>text</b></td><td>Captions the image into a text prompt you can edit. A separate local Qwen3-VL.</td></tr>
      <tr><td><b>Reference image</b></td><td>image → <b>generation</b></td><td>Feeds the image through Qwen3-VL's vision path as conditioning; steers the output directly. No text is written.</td></tr>
      <tr><td><b>Catalog moodboard (before)</b></td><td>→ <b>text only</b></td><td>Only its Qwen-authored words were appended to the prompt; its images were never used at generation.</td></tr></table>""")
    P.append(finding("A catalog moodboard's <em>images</em> were dead weight at generation — only its text guidance was used. This study is about making the images work as a <em>style</em> reference."))
    P.append("</div>")

    # Mechanism
    P.append("<div class='card' id='mech'><h2>2 · How the reference path actually works</h2>")
    P.append("<p><code class='k'>TextEncodeKrea2</code> is essentially a <b>VLM captioner into conditioning</b>: it runs the reference image through Qwen3-VL under a <b>system prompt</b> and turns the description into the 12-layer conditioning the DiT consumes. That yields three levers:</p>")
    P.append("""<ul>
      <li><b>System prompt</b> — the default says “describe color, shape, size, texture, <em>objects, background</em>” → it encodes <b>content</b>. A style-only instruction encodes <b>look</b>.</li>
      <li><b>vision_megapixels</b> — a size cap. A tiny thumbnail conveys palette/mood but not fine composition. (Node floor = 0.10.)</li>
      <li><b>Multiple images in one encode</b> — the node writes “Picture 1: … Picture 2: …”, so the model paints a <b>collage</b>.</li>
      <li><b>Per-layer weights</b> — layer index 8 (Qwen tap 26) is the “subject/content” layer; the rest carry style.</li>
    </ul>""")
    P.append("</div>")

    # E1
    P.append("<div class='card' id='e1'><h2>3 · Experiment 1 — Moodboard channels</h2>")
    P.append("<p>Same subject + same board + same seeds across every way a moodboard could influence a generation.</p>")
    P.append(prompt_box("Subject prompt", SUBJECT_PORTRAIT))
    P.append(refs_html(3194, fallback_glob="ref*.png"))
    P.append("<p class='muted'>Channels: <b>Baseline</b> (subject only) · <b>A</b> text guidance · <b>B</b> image refs · <b>C</b> text+refs · <b>D</b> magic-prompt fusion · <b>E</b> fusion+refs · <b>F</b> image→prompt+subject.</p>")
    P.append(fig(MC / "_compare_seed111.png", "Seed 111 — all channels"))
    P.append(fig(MC / "_compare_seed222.png", "Seed 222 — all channels"))
    P.append(finding("Text guidance (A) gives reliable style but drifts the subject. Raw image refs (B/C) are strongest but <b>copy the reference's content</b> (benches, trees, figures) and even <b>collage</b>. Blending the board guidance into the prompt — <b>fusion (D)</b> — best kept the subject <em>and</em> applied the style."))
    P.append("</div>")

    # E2
    P.append("<div class='card' id='e2'><h2>4 · Experiment 2 — Why image refs made a split image</h2>")
    P.append(prompt_box("Subject prompt", SUBJECT_PORTRAIT))
    P.append("<p>Swept number-of-refs and strength to isolate the cause of the collage.</p>")
    P.append(fig(MC / "imageref_sweep" / "_compare.png", "1 ref (various strengths) vs 3 refs (various) — seed 111"))
    P.append(finding("The split is caused by feeding <b>multiple</b> reference images (the multi-image node tiles “Picture 1/2/3”), <b>not</b> by strength — lowering strength only partly helped. <b>One</b> reference never splits, but a single full-detail ref <b>copies the reference's content</b>."))
    P.append("</div>")

    # E3
    P.append("<div class='card' id='e3'><h2>5 · Experiment 3 — Style vs content: the levers</h2>")
    P.append("<p>Subject deliberately <b>different</b> from the references (a bright puppy vs. the refs' teal solitary figures) so style-transfer vs content-copy is obvious.</p>")
    P.append(prompt_box("Subject prompt", SUBJECT_PUPPY))
    P.append(prompt_box("Style-only system prompt (used by the winning recipe)", STYLE_SYS))
    P.append(fig(STY / "_compare.png", "Levers: text-only · content-copy · collage · style system-prompt · low-MP · suppress-subject · combo · averaged"))
    P.append(finding("<b>Averaging separate single-image style-encodes (“chaining”) is the winner</b> — it preserved the puppy <em>and</em> transferred the teal painterly style, with no collage and no content copy. Averaging cancels each image's unique content while reinforcing the <em>shared</em> style. System-prompt + low MP help; per-layer subject-suppression was weak on its own."))
    P.append("</div>")

    # E4
    P.append("<div class='card' id='e4'><h2>6 · Experiment 4 — Refining &amp; confirming</h2>")
    P.append("<p>Swept the averaging knobs (#images, MP, optional per-layer reweight) across two subjects and a <b>second board</b>.</p>")
    P.append(prompt_box("Subjects", f"{SUBJECT_PUPPY}\n{SUBJECT_CAR}\n{SUBJECT_BERRIES}"))
    P.append(fig(STY / "refine" / "_compare_s1_puppy.png", "Puppy · board #3194 (Abyssal Gothic)"))
    P.append(fig(STY / "refine" / "_compare_s2_car.png", "Red sports car · board #3194"))
    P.append(fig(STY / "refine" / "_compare_s3_berries.png", "Strawberries · board #3831 (Abyssal Storm Gothic) — a different board"))
    P.append(finding("<b>avg 4 @ MP 0.3</b> / <b>avg 5 @ MP 0.25</b> (no per-layer rebalance) were robust across all subjects and both boards. The per-layer tricks (+style layers / +suppress subject) consistently <b>weakened</b> the style, so they were dropped."))
    P.append("</div>")

    # E5
    P.append("<div class='card' id='e5'><h2>7 · Experiment 5 — Megapixel sweep (0.30–1.00)</h2>")
    P.append(prompt_box("Subject prompt", SUBJECT_PUPPY))
    P.append(fig(STY / "mp_sweep" / "_compare_mp_sweep.png", "Averaged style (5 refs) — MP 0.30 → 1.00"))
    P.append(row([(STY / "mp_sweep" / "mp030.png", "MP 0.30 — full mood"),
                  (STY / "mp_sweep" / "mp060.png", "MP 0.60 — mood fading"),
                  (STY / "mp_sweep" / "mp100.png", "MP 1.00 — style washed out")]))
    P.append(finding("Higher MP <b>weakens</b> style — it is an <b>inverted</b> style-strength dial. A tiny thumbnail forces the VLM to encode the dominant global impression (dark, painterly, teal); at high MP the averaged detail dilutes into just “oil painting,” and the prompt's own palette takes over."))
    P.append("</div>")

    # E6
    P.append("<div class='card' id='e6'><h2>8 · Experiment 6 — Full sweep (0.10–1.00) + anchors</h2>")
    P.append(prompt_box("Subject prompt", SUBJECT_PUPPY))
    P.append("<p>Includes a <b>text-only</b> (no-reference) anchor and the reference images, across the full supported range.</p>")
    P.append(fig(STY / "full_sweep" / "_compare_full_sweep.png", "References · text-only · MP 0.10 → 1.00"))
    P.append(row([(STY / "full_sweep" / "mp010.png", "MP 0.10 — max style"),
                  (STY / "full_sweep" / "mp015.png", "MP 0.15 — strong"),
                  (STY / "full_sweep" / "mp020.png", "MP 0.20 — safe/clean")]))
    P.append(note("The node's hard floor is <b>MP 0.10</b> (values below 0.1 are rejected), so 0.05 isn't possible."))
    P.append(finding("Clean monotonic gradient. <b>~0.10–0.15</b> = maximum mood (tiny chance of a faint content wisp). <b>~0.20–0.25</b> = strong and reliably leak-free. <b>0.5+</b> = subtle medium only."))
    P.append("</div>")

    # Round 2
    P.append("<div class='card' id='r2'><h2>9 · Round 2 — going deeper</h2>")
    P.append("<p>Extra experiments to map the edges of the technique. Same subject (puppy) unless noted.</p>")
    P.append(prompt_box("Subject prompt", SUBJECT_PUPPY))

    P.append("<h3>9.1 · Reference count strung together (1 → 6)</h3>")
    P.append(fig(ADV / "_compare_num_refs.png", "Averaging 1..6 refs (MP 0.20)"))
    P.append(finding("Number of averaged refs is a <b>“style consensus” dial</b>: <b>1 ref</b> = that single image's specific vibe <em>and</em> the strongest content copy (its bench/tree/figure); <b>3–4 refs</b> = the subject (puppy) survives with a coherent board style; <b>5–6 refs</b> = softer, more <em>generic</em> style. The subject reliably survives from ~3+ refs. More refs = safer against content copy, at the cost of specificity."))

    P.append("<h3>9.2 · Strength multiplier, including negative</h3>")
    P.append(fig(ADV / "_compare_strength.png", "ConditioningKrea2Rebalance multiplier −1.0 → +4.0 (uniform weights, MP 0.20)"))
    P.append(finding("A secondary <b>intensity</b> knob: +0.5 soft → +1 normal → +2 punchy/saturated → +4 overcooked. <b>Negative multipliers (−0.5, −1.0) are degenerate</b> — they invert the whole conditioning (subject included) into psychedelic noise. So don't expose negative on this dial; keep it ~0.5–2.0."))

    P.append("<h3>9.3 · Style as a NEGATIVE prompt (anti-style)</h3>")
    P.append(fig(ADV / "_compare_negative_slot.png", "Style in positive vs negative slot vs plain (CFG 3.0)"))
    P.append(finding("Putting the averaged style in the <b>negative</b> slot is an <b>“anti-style”</b>: it pushes toward the <em>opposite</em> aesthetic (the board's cool/dark → warm/blown-out here). Requires CFG&gt;1 to have any effect (turbo's default CFG 1 ignores negatives) and tends to overcook. Niche, but a real way to say “<em>not</em> this palette/mood.”"))

    P.append("<h3>9.4 · Two-board style blend</h3>")
    P.append(fig(ADV / "_compare_board_blend.png", "Averaging refs across board #3194 and #3831 at different ratios"))
    P.append(finding("Blending <b>works and is ratio-controllable</b>: 2+2 = a genuine hybrid; 1+3 lets board B's storm dominate; 3+1 keeps board A's painterly meadow. This is a first-class <b>“mix two moodboards”</b> capability — and it would work the same for blending a board with a user's own uploaded image."))

    P.append("<h3>9.5 · Attribute isolation via system prompt</h3>")
    P.append(fig(ADV / "_compare_attributes.png", "content (default) vs full-style vs palette-only vs lighting/mood-only vs medium/brushwork-only"))
    P.append(finding("Targeted system prompts (palette-only / lighting-only / medium-only) <b>nudge</b> which facet transfers, but the effect is <b>modest</b> — a cohesive board's traits bleed across facets, so isolation is partial. Useful as a bias (“lean on the palette”), not a clean separator."))
    P.append("</div>")

    # Transform (style-locked subject)
    P.append("<div class='card' id='tr'><h2>10 · Transforming a style-locked subject</h2>")
    P.append("<p>What if the <em>subject itself</em> carries a strong trained style — like a cartoon IP? We asked a realism board to de-cartoon Link &amp; Princess Zelda.</p>")
    P.append(prompt_box("Subject prompt", "Link and Princess Zelda from The Legend of Zelda standing together in a grand fantasy landscape, full body"))
    P.append(refs_html(1348))
    P.append(fig(ZEL / "_compare_transform.png", "Baseline · text guidance · style-avg subtle/balanced/strong · strong+mult2 — realism board #1348"))
    P.append(finding("<b>Text guidance won, clearly.</b> The realism board's <b>text</b> (2nd panel) turned two characters that are <em>hard to render realistically</em> into believable, realistic-looking people. The <b>vision style-average did not</b> — it stayed anime/game-art at every strength, and at <b>“strong” (MP 0.10) it actively degraded the result</b>: it dropped Link, duplicated Zelda, and still wasn't realistic. Cranking vision strength on a <b>style-locked</b> subject corrupts the composition without buying realism, because the vision path carries palette/mood — not “photoreal rendering.”"))
    P.append(note("The two channels have <b>different jobs</b>. <b>Style-averaging</b> shines on a <b>style-neutral</b> subject (a puppy, a car) — it applies a look. But it <b>cannot transform a style-locked</b> subject (a named IP): in this test <em>every</em> style-avg setting stayed cartoon, and pushing strength only broke the subject. <b>Text guidance is the only thing that transformed Link &amp; Zelda</b>, and it works best <b>on its own</b> — adding style-avg muddied it (see §12.1). Rule: style-locked subject → use <b>text</b>; don't reach for the image/style channel."))
    P.append("</div>")

    # Mashups
    P.append("<div class='card' id='mash'><h2>11 · Meaningful mashups of clashing styles</h2>")
    P.append("<p>Two boards that clash hard: <b>#1348 Somber Realism</b> (muted, photographic) × <b>#3089 Bold Graphic Neon Surrealism</b> (flat vector, bold outlines, neon).</p>")
    P.append(fig(ZEL / "_compare_mashup.png", "realism only · neon only · naive 2+2 · mostly realism · mostly neon · palette=neon+light/medium=realism · palette=realism+light/medium=neon"))
    P.append(finding("<b>Ratio controls dominance</b> (mostly-realism = muted/soft; mostly-neon = bright/saturated). A <b>naive 50/50 blend of clashing boards muddies/darkens</b> — the styles average into something neither intended. The meaningful fix is a <b>facet-split</b>: take the <b>palette from one board</b> and the <b>lighting/medium from the other</b> (via targeted system prompts). That yields a coherent, intentional hybrid — e.g. neon palette + realistic soft light — instead of mud."))
    P.append(note("Rule of thumb for mashups: <b>don't blend clashing boards wholesale — split them by facet.</b> Pick which board owns color vs. which owns light/medium. Boards that are already compatible blend fine by simple ratio."))
    P.append("</div>")

    # Full picture
    P.append("<div class='card' id='fp'><h2>12 · Full picture — combine, restyle, engine, consistency</h2>")

    P.append("<h3>12.1 · Combine text + style-avg on a style-locked IP</h3>")
    P.append(fig(FP / "_compare_combine.png", "baseline · board text · explicit realism words · style-avg only · text+style-avg · explicit+style-avg (Link & Zelda)"))
    P.append(finding("<b>Only the board's realism TEXT produced clean realistic characters.</b> Generic explicit words (“photorealistic, 85mm”) stayed cartoon, and <b>every style-avg variant stayed cartoon</b>. Critically, <b>adding style-avg on top of the text did not help</b> — it muddied and darkened the image and shrank the figures. For a style-locked subject, <b>text guidance ALONE is best</b>; the image/style channel does not transform it and can actively degrade it."))

    P.append("<h3>12.2 · Restyle an existing photo (img2img)</h3>")
    P.append(fig(FP / "_compare_restyle.png", "original photo · restyle at denoise 0.35 / 0.5 / 0.65 / 0.8 (board #3194 style-avg)"))
    P.append(finding("<b>Weak.</b> Plain img2img + style-avg barely restyled the photo — its bright daylight palette persisted even at denoise 0.8. The source image's own style anchors the result; to truly restyle you'd need very high denoise (which loses the subject) or to add text guidance. <b>Style-avg is a text-to-image tool, not a photo filter.</b>"))

    P.append("<h3>12.3 · RAW vs Turbo</h3>")
    P.append(fig(FP / "_compare_raw_turbo.png", "Turbo 8 steps (CFG~1) vs RAW 28 steps (CFG 3.5) — same style-avg"))
    P.append(finding("Counterintuitive: <b>Turbo transfers the style more strongly than RAW.</b> RAW's high CFG (3.5) pushes <em>literal prompt adherence</em> (“bright sunny meadow”), which overrides the style conditioning and washes out the mood. <b>Low-CFG (Turbo) is the favorable regime for style transfer</b> — high guidance fights the style."))

    P.append("<h3>12.4 · Consistency across seeds</h3>")
    P.append(fig(FP / "_compare_seeds.png", "Same style-avg recipe across 4 seeds"))
    P.append(finding("<b>Very stable.</b> Palette, medium, lighting, and mood hold across seeds; only composition varies. This validates a <b>“style preset”</b> — a saved style-avg gives a consistent look across many generations."))
    P.append("</div>")

    # Full picture II
    P.append("<div class='card' id='fp2'><h2>13 · Feel transfer II — the point is the moodboard's <em>feel</em></h2>")
    P.append("<p>Reframed on purpose: this isn't about realism, it's about carrying a <b>chosen moodboard's feel</b> into a style-neutral subject. Expressive boards used here: painterly #3194, neon-vector #3089.</p>")

    P.append("<h3>13.1 · Does the feel hold at higher resolution?</h3>")
    P.append(fig(FP2 / "_compare_highres.png", "Same style-avg at 1024 / 1536 / 2048 (painterly board #3194)"))
    P.append(finding("The <b>feel (palette + mood) holds</b> at 2K with no artifacts, but the <b>painterly medium/brushwork weakens</b> — more pixels push toward photographic rendering. For medium-heavy looks (oil, brushwork), generate near <b>1K</b> for the strongest feel; the color/mood still carries at 2K."))

    P.append("<h3>13.2 · Blend a board with your OWN image</h3>")
    P.append(fig(FP2 / "_compare_userblend.png", "board only · your image only · board+yours 3:1 · board+yours 1:1 (your image = a neon piece)"))
    P.append(finding("Works and is <b>ratio-controllable</b>: 3:1 keeps the board dominant with a hint of your image; <b>1:1 is a true hybrid</b> (the board's painterly medium + your image's neon palette). So a user can fuse a personal reference with a catalog board to make a bespoke look."))

    P.append("<h3>13.3 · How little text conveys the feel?</h3>")
    P.append(fig(FP2 / "_compare_textecon.png", "no style · +'painterly' · +short phrase · +full board text · style-avg (image)"))
    P.append(finding("One word (“painterly”) is too weak, but a <b>short 6–8 word phrase</b> (“dark teal painterly oil painting, stormy, high contrast”) already carries most of the feel and is <b>composition-safe</b>. Full board text is strongest. The <b>image channel adds authentic texture but can pull the board's own subject matter</b> (a seascape crept into the “city street”) — a reason to keep image-style strength moderate, or lean on a text phrase when composition must stay put."))

    P.append("<h3>13.4 · Two moodboards, one image (per-region)</h3>")
    P.append(fig(FP2 / "_compare_region.png", "gothic only · neon only · split (gothic left / neon right)"))
    P.append(finding("<b>Per-region style works.</b> Masking two moodboard style-conditionings to different regions (gothic left, neon right) yields one coherent image with <b>distinct styles per area</b>. Opens spatial style control — different boards in different parts of a composition."))
    P.append("</div>")

    # Workflow ideas
    P.append("<div class='card' id='ideas'><h2>14 · Unique workflow ideas</h2>")
    P.append("<p>Ways to turn these findings into features:</p>")
    P.append("""<table>
      <tr><th>Idea</th><th>How</th><th>Why it's useful</th></tr>
      <tr><td><b>Style strength slider</b></td><td>MP inverted (0.10 strong → 0.5 subtle)</td><td>One intuitive dial for “how much of this look.”</td></tr>
      <tr><td><b>Style consensus dial</b></td><td># of averaged refs (1 → 6)</td><td>1 = “this exact image's vibe,” 4+ = “the board's general style.”</td></tr>
      <tr><td><b>Style blending / mashup</b></td><td>Average refs across 2+ boards at a chosen ratio</td><td>Invent new hybrid looks.</td></tr>
      <tr><td><b>Bring-your-own-image blend</b></td><td>Average a catalog board with the user's uploaded image (ratio-controlled)</td><td>Fuse a personal reference with a board for a bespoke look (1:1 = true hybrid).</td></tr>
      <tr><td><b>Feel in a phrase</b></td><td>A short 6–8 word style phrase in the prompt</td><td>Composition-safe feel with no image needed; pairs well with modest image-style.</td></tr>
      <tr><td><b>Facet-split mashup</b></td><td>Palette from board A + lighting/medium from board B (targeted system prompts)</td><td>Make <em>clashing</em> boards meaningful instead of muddy — each board owns one dimension.</td></tr>
      <tr><td><b>Transform lock-breaker</b></td><td>Text guidance <b>alone</b> for style-locked subjects — do <em>not</em> add image-style</td><td>Only text de-cartoons a named IP; the image/style channel can't, and adding it degrades the result.</td></tr>
      <tr><td><b>Anti-style (negative)</b></td><td>Averaged style in the negative slot, CFG&gt;1</td><td>“Avoid this palette/mood” (e.g. not washed-out, not cartoonish).</td></tr>
      <tr><td><b>Attribute nudge</b></td><td>palette-only / mood-only / medium-only system prompts</td><td>Transfer mostly the color, or mostly the lighting, or mostly the brushwork.</td></tr>
      <tr><td><b>Style intensity</b></td><td>Rebalance multiplier 0.5–2.0 (never negative)</td><td>Secondary “punch” on top of the strength slider.</td></tr>
      <tr><td><b>Style presets / library</b></td><td>Cache the averaged style conditioning per board or per saved “look”</td><td>Instant reuse; a personal style library; no re-encode cost.</td></tr>
      <tr><td><b>Restyle a photo</b> <span class='muted'>(weak)</span></td><td>img2img + style-average — but needs high denoise or added text; source palette resists</td><td>Limited: style-avg is a text-to-image tool, not a photo filter.</td></tr>
      <tr><td><b>Prefer low CFG for style</b></td><td>Turbo / CFG≈1 rather than high-CFG RAW</td><td>High guidance overrides the style; low-CFG lets the look come through.</td></tr>
      <tr><td><b>Per-region style</b></td><td>Masked style-average per region (ConditioningSetMask + Combine) — proven</td><td>Different board styles in different parts of one image.</td></tr>
    </table>""")
    P.append("</div>")

    # Recommendation
    P.append("<div class='rec' id='rec'><h2 style='margin-top:0'>15 · Conclusion &amp; recommendation</h2>")
    P.append("<p><b>Final synthesis:</b> keep the current <b>moodboard text guidance</b> as the default/primary transfer path. It is composition-safe, works on style-locked subjects, and the full default generation recipe validated prompt adherence. Add image-based style averaging as an <b>optional</b> second channel for neutral subjects, texture/medium transfer, personal-image blends, mashups, and regional styling.</p>")
    P.append("<p><b>Optional image-style recipe:</b></p>")
    P.append("""<ol>
      <li>Encode <b>each</b> reference image <b>separately</b> with the <b>style-only system prompt</b>.</li>
      <li>Cap <code class='k'>vision_megapixels</code> low (this is the strength dial).</li>
      <li><b>Running-mean average</b> the encodes (<code class='k'>ConditioningAverage</code>) — never feed them into one multi-image encode (that collages).</li>
      <li>No per-layer rebalance.</li>
    </ol>""")
    P.append("<p>Expose one <b>“Image style strength”</b> slider mapped (inverted) to MP:</p>")
    P.append("""<table><tr><th>Slider</th><th>MP</th><th>Effect</th></tr>
      <tr><td>Strong</td><td>0.10</td><td>Full palette + mood + medium</td></tr>
      <tr><td><b>Balanced</b></td><td><b>0.20</b></td><td>Strong style, leak-free</td></tr>
      <tr><td>Subtle</td><td>0.50+</td><td>Light painterly touch only</td></tr></table>""")
    P.append("<p>Recommended product behavior: moodboards default to <b>Text guidance</b>. Add an opt-in <b>Use board images</b> mode with <b>Match style</b> (averaged) vs. <b>Copy composition</b> (existing multi-image compose). Do not use image-style to transform style-locked subjects; use text/fusion for that.</p>")
    P.append("</div>")

    P.append("<p class='muted' style='margin-top:30px'>Generated by <code>scripts/build_style_report.py</code> from real outputs in <code>outputs/moodboard_*</code>. Board #3194 “Abyssal Gothic Surrealism”, #3831 “Abyssal Storm Gothic”. Engine: Krea 2 Turbo via ComfyUI, 8 steps, seed 111/222.</p>")
    P.append("</div></body></html>")
    return "".join(P)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_str = build()
    out = REPORT_DIR / "report.html"
    out.write_text(html_str, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"[report] wrote {out}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
