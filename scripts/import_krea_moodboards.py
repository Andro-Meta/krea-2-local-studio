from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from moodboards_catalog import (  # noqa: E402
    KREA_MOODBOARD_GALLERY_URL,
    KreaMoodboardCrawler,
    MOODBOARD_SEED_PATH,
    export_moodboard_seed,
    init_moodboard_db,
    upsert_moodboard,
)


def discover_with_browser(url: str, *, scrolls: int = 24) -> list[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Browser discovery requires Playwright. Use seeded URLs or install Playwright.") from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=60_000)
        for _ in range(scrolls):
            page.mouse.wheel(0, 2400)
            page.wait_for_timeout(750)
        hrefs = page.eval_on_selector_all(
            "a[href*='moodboard-feed/']",
            "els => els.map(a => a.href)",
        )
        browser.close()
    crawler = KreaMoodboardCrawler(delay_seconds=0)
    return crawler.discover_gallery_urls(
        "".join(f'<a href="{href}"></a>' for href in hrefs),
        url,
    )


async def import_records(
    urls: list[str],
    *,
    max_pages: int,
    browser_discovery: bool,
    export_seed: bool = False,
    seed_path: Path = MOODBOARD_SEED_PATH,
    export_every: int = 50,
) -> tuple[int, int]:
    await init_moodboard_db()
    seeds = list(urls)
    if not seeds:
        seeds = [KREA_MOODBOARD_GALLERY_URL]
    if browser_discovery:
        seeds = discover_with_browser(KREA_MOODBOARD_GALLERY_URL) + seeds

    crawler = KreaMoodboardCrawler()
    imported = 0
    for record in crawler.iter_crawl(seeds, max_pages=max_pages):
        await upsert_moodboard(record)
        imported += 1
        if imported % 10 == 0:
            print(f"Imported or updated {imported} moodboards...", flush=True)
        if export_seed and export_every > 0 and imported % export_every == 0:
            await export_moodboard_seed(seed_path)
    exported = await export_moodboard_seed(seed_path) if export_seed else 0
    return imported, exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Import public Krea moodboards into the local catalog.")
    parser.add_argument("urls", nargs="*", help="Krea moodboard-feed URLs to seed the crawl.")
    parser.add_argument("--max-pages", type=int, default=200, help="Maximum moodboard detail pages to import.")
    parser.add_argument("--browser", action="store_true", help="Use Playwright to discover lazy-loaded gallery links first.")
    parser.add_argument("--export-seed", action="store_true", help="Write the imported catalog to the portable seed JSON file.")
    parser.add_argument("--seed-path", type=Path, default=MOODBOARD_SEED_PATH, help="Path for --export-seed output.")
    parser.add_argument("--export-every", type=int, default=50, help="When exporting a seed, refresh it after every N imported records.")
    args = parser.parse_args()

    imported, exported = asyncio.run(import_records(
        args.urls,
        max_pages=args.max_pages,
        browser_discovery=args.browser,
        export_seed=args.export_seed,
        seed_path=args.seed_path,
        export_every=args.export_every,
    ))
    print(f"Imported or updated {imported} moodboards.")
    if args.export_seed:
        print(f"Exported {exported} moodboards to {args.seed_path}.")


if __name__ == "__main__":
    main()
