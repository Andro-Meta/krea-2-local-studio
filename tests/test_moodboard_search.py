from __future__ import annotations

import asyncio
import threading
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from moodboard_search import (  # noqa: E402
    MoodboardSearchIndex,
    format_matched_reason,
    get_cached_moodboard_search,
    invalidate_moodboard_search_cache,
    moodboard_search_cache_build_count,
    normalize_tokens,
)
from moodboards_catalog import (  # noqa: E402
    _hydrate_moodboard_items,
    _load_search_documents_sync,
    MoodboardRecord,
    init_moodboard_db,
    list_moodboards,
    set_moodboard_favorite,
    set_moodboard_qwen_guidance,
    suggest_moodboards,
    upsert_moodboard,
)


def item(
    item_id: int,
    title: str,
    *,
    keywords: list[str] | None = None,
    taste: str = "",
    axes: list[str] | None = None,
    notes: list[str] | None = None,
    prompt: str = "",
    negative: str = "",
    summary: str = "",
) -> dict:
    return {
        "id": item_id,
        "uuid": f"uuid-{item_id}",
        "title": title,
        "taste_profile": taste,
        "keywords": keywords or [],
        "preview_image_urls": [f"/preview/{item_id}.webp"],
        "source": "official",
        "qwen_guidance": {
            "style_axes": axes or [],
            "conditioning_notes": notes or [],
            "prompt_guidance": prompt,
            "negative_guidance": negative,
            "source_summary": summary,
            "guidance_version": 2,
        },
        "qwen_guidance_version": 2,
    }


class MoodboardSearchIndexTests(unittest.TestCase):
    def test_normalizes_unicode_punctuation_and_preserves_hyphen_phrase(self) -> None:
        tokens = normalize_tokens("CINÉMA—old-school, low-key!")
        self.assertIn("cinema", tokens)
        self.assertIn("old", tokens)
        self.assertIn("old-school", tokens)
        self.assertIn("low-key", tokens)

    def test_old_never_matches_bold_but_matches_exact_and_hyphen_token(self) -> None:
        index = MoodboardSearchIndex([
            item(1, "Bold Geometry", keywords=["bold"]),
            item(2, "Old Film", keywords=["old"]),
            item(3, "Archive", keywords=["old-school"]),
        ])
        self.assertEqual([result.item_id for result in index.search("old")], [2, 3])

    def test_another_exact_query_token_can_include_bold_board(self) -> None:
        index = MoodboardSearchIndex([
            item(1, "Bold Geometry", keywords=["bold", "geometry"]),
            item(2, "Old Film", keywords=["old"]),
        ])
        self.assertEqual(
            {result.item_id for result in index.search("old geometry")},
            {1, 2},
        )

    def test_phrase_boost_and_duplicate_title_tie_break_are_deterministic(self) -> None:
        index = MoodboardSearchIndex([
            item(3, "Same", keywords=["teal", "grain"]),
            item(2, "Same", keywords=["muted teal", "analog grain"]),
            item(1, "Other", keywords=["muted teal", "grain"]),
        ])
        first = index.search("muted teal")
        second = index.search("muted teal")
        self.assertEqual([r.item_id for r in first], [1, 2, 3])
        self.assertEqual(first, second)
        duplicate_index = MoodboardSearchIndex([
            item(5, "Duplicate", keywords=["violet"]),
            item(4, "Duplicate", keywords=["violet"]),
        ])
        self.assertEqual([r.item_id for r in duplicate_index.search("violet")], [4, 5])

    def test_basic_plural_prefix_and_conservative_fuzzy_matching(self) -> None:
        index = MoodboardSearchIndex([
            item(1, "Textures", keywords=["textures"]),
            item(2, "Cinematic", keywords=["cinematic"]),
            item(3, "Chiaroscuro", keywords=["chiaroscuro"]),
        ])
        self.assertEqual(index.search("texture")[0].item_id, 1)
        self.assertEqual(index.search("cinem")[0].item_id, 2)
        self.assertEqual(index.search("chiaroscoro")[0].item_id, 3)
        self.assertEqual(index.search("old"), [])

    def test_negative_guidance_and_source_summary_do_not_create_matches(self) -> None:
        index = MoodboardSearchIndex([
            item(1, "Neutral", negative="old neon", summary="spaceship portrait"),
            item(2, "Grain", axes=["analog grain"]),
        ])
        self.assertEqual(index.search("spaceship"), [])
        self.assertEqual(index.search("old"), [])

    def test_original_style_cues_outweigh_expanded_cues(self) -> None:
        index = MoodboardSearchIndex([
            item(1, "Original", axes=["muted teal"]),
            item(2, "Expanded", axes=["golden baroque"]),
        ])
        results = index.suggest("muted teal portrait", "golden baroque portrait")
        self.assertEqual(results[0].item_id, 1)

    def test_expanded_style_can_help_minimal_original_and_subject_summary_cannot(self) -> None:
        index = MoodboardSearchIndex([
            item(1, "Style", axes=["analog grain"], summary="cat portrait"),
            item(2, "Subject Leak", summary="cat portrait"),
        ])
        results = index.suggest("cat", "cat portrait with analog grain")
        self.assertEqual([result.item_id for result in results], [1])
        self.assertIn("analog grain", results[0].matched_cues)

    def test_subject_catalog_fields_are_manual_search_only(self) -> None:
        subject_index = MoodboardSearchIndex([
            item(
                1,
                "Cat Portrait",
                keywords=["cat"],
                taste="A cat portrait in a quiet room.",
            ),
        ])
        self.assertEqual(
            [result.item_id for result in subject_index.search("cat")], [1]
        )
        self.assertEqual(subject_index.suggest("cat", "cat portrait"), [])

        safe_index = MoodboardSearchIndex([
            item(
                2,
                "Safe Lighting",
                axes=["portrait lighting", "analog grain"],
                prompt="Lighting: soft portrait lighting. Medium and texture: analog grain.",
            ),
        ])
        safe = safe_index.suggest(
            "portrait lighting", "portrait lighting analog grain"
        )
        self.assertEqual([result.item_id for result in safe], [2])
        reason = format_matched_reason(safe[0].matched_cues).lower()
        self.assertNotIn("cat", reason)
        self.assertNotIn("portrait of", reason)
        self.assertIn("portrait lighting", reason)

    def test_suggestion_index_uses_positive_style_vocabulary_only(self) -> None:
        index = MoodboardSearchIndex([
            item(
                1,
                "Unsafe",
                axes=[
                    "spaceship cinematic",
                    "cat noir",
                    "woman editorial",
                    "car analog",
                    "novel texture",
                ],
                prompt="spaceship cat woman car novel",
            ),
            item(
                2,
                "Recognized",
                axes=["bauhaus", "surrealism", "film noir", "cottagecore"],
                prompt="Bauhaus composition. Surrealism. Film noir lighting.",
            ),
        ])
        for unknown in ("spaceship", "cat", "woman", "car", "novel", "unknowncore"):
            self.assertEqual(index.suggest(unknown, unknown), [], unknown)
        for recognized in ("bauhaus", "surrealism", "film noir", "cottagecore"):
            results = index.suggest(recognized, recognized)
            self.assertTrue(results, recognized)
            self.assertEqual(results[0].item_id, 2, recognized)
        mixed = index.suggest(
            "spaceship bauhaus cat",
            "spaceship bauhaus cat surrealism",
        )
        reason = format_matched_reason(mixed[0].matched_cues).lower()
        self.assertIn("bauhaus", reason)
        self.assertIn("surrealism", reason)
        self.assertNotIn("spaceship", reason)
        self.assertNotIn("cat", reason)

    def test_named_style_reviewer_probes_all_match(self) -> None:
        probes = (
            "cyberpunk",
            "steampunk",
            "solarpunk",
            "afrofuturism",
            "synthwave",
            "vaporwave",
            "maximalism",
            "minimalism",
            "brutalism",
            "art nouveau",
            "art deco",
            "neo-expressionism",
            "expressionism",
            "impressionism",
            "retrofuturism",
            "photorealism",
            "hyperrealism",
            "low-poly",
            "pixel art",
            "collage",
            "risograph",
            "ukiyo-e",
        )
        index = MoodboardSearchIndex([
            item(index, probe, axes=[probe])
            for index, probe in enumerate(probes, 1)
        ])
        for expected_id, probe in enumerate(probes, 1):
            results = index.suggest(probe, probe)
            self.assertTrue(results, probe)
            self.assertEqual(results[0].item_id, expected_id, probe)

    def test_mixed_fragments_keep_only_known_style_cues(self) -> None:
        index = MoodboardSearchIndex([
            item(
                1,
                "Mixed",
                axes=[
                    "cinematic spaceship art",
                    "heavy film grain",
                    "high contrast",
                    "soft focus",
                    "muted palette",
                ],
            )
        ])
        result = index.suggest(
            "spaceship cinematic art",
            "spaceship cinematic art heavy film grain high contrast soft focus muted palette",
        )[0]
        reason = format_matched_reason(result.matched_cues).lower()
        self.assertIn("cinematic", reason)
        self.assertIn("art", reason)
        self.assertNotIn("spaceship", reason)
        for probe in ("heavy film grain", "high contrast", "soft focus", "muted palette"):
            self.assertEqual(index.suggest(probe, probe)[0].item_id, 1, probe)
        for unknown in ("spaceship", "cat", "woman", "car", "novel"):
            self.assertEqual(index.suggest(unknown, unknown), [], unknown)

    def test_generic_modifiers_require_an_anchor_cue(self) -> None:
        index = MoodboardSearchIndex([
            item(1, "Watercolor", axes=["soft watercolor", "muted palette"]),
            item(2, "Noir", axes=["high contrast", "film noir"]),
            item(3, "Cinema", axes=["cinematic", "low-key lighting"]),
            item(4, "Generic", axes=["soft focus", "negative space", "dramatic lighting"]),
        ])
        for generic in (
            "a man in space",
            "a light bulb",
            "a soft pillow",
            "a field of flowers",
            "lighting",
        ):
            self.assertEqual(index.suggest(generic, generic), [], generic)

        probes = {
            "soft watercolor": 1,
            "high contrast film noir": 2,
            "cinematic low-key lighting": 3,
        }
        for probe, expected_id in probes.items():
            result = index.suggest(probe, probe)[0]
            self.assertEqual(result.item_id, expected_id, probe)
            reason = format_matched_reason(result.matched_cues).lower()
            for word in probe.split():
                self.assertIn(word.replace("-", " "), reason)

        expanded = index.suggest(
            "a soft pillow",
            "a soft pillow rendered in watercolor",
        )
        self.assertEqual(expanded[0].item_id, 1)

    def test_manual_search_scores_v2_and_legacy_guidance_once(self) -> None:
        legacy = item(1, "Legacy", axes=["analog grain"], prompt="Analog grain")
        legacy["qwen_guidance_version"] = 1
        legacy["qwen_guidance"]["guidance_version"] = 1
        modern = item(2, "Modern", axes=["analog grain"], prompt="Analog grain")
        index = MoodboardSearchIndex([legacy, modern])
        scores = {result.item_id: result.score for result in index.search("analog grain")}
        self.assertEqual(scores[1], scores[2])

    def test_suggestions_are_confident_and_diversified(self) -> None:
        index = MoodboardSearchIndex([
            item(1, "Teal One", axes=["muted teal", "analog grain"]),
            item(2, "Teal Two", axes=["muted teal", "analog grain"]),
            item(3, "Noir", axes=["low-key lighting", "cinematic noir"]),
            item(4, "Irrelevant", axes=["bright watercolor"]),
        ])
        results = index.suggest(
            "muted teal low-key lighting",
            "muted teal analog grain low-key lighting cinematic noir",
            limit=3,
        )
        self.assertEqual(len(results), 3)
        self.assertIn(3, [result.item_id for result in results[:2]])
        self.assertNotIn(4, [result.item_id for result in results])
        self.assertEqual(index.suggest("a dog in a park", "a dog in a park"), [])

    def test_suggestion_reason_is_bounded_deduped_and_deterministic(self) -> None:
        cues = [
            " ".join([f"cue-{index}", "descriptive"] * 30)
            for index in range(200)
        ]
        cues.insert(1, cues[0].upper())
        first = format_matched_reason(cues)
        second = format_matched_reason(cues)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("Matched: "))
        self.assertLessEqual(len(first), 180)
        rendered_cues = first.removeprefix("Matched: ").split(" · ")
        self.assertLessEqual(len(rendered_cues), 3)
        self.assertEqual(len(rendered_cues), len({cue.casefold() for cue in rendered_cues}))
        self.assertTrue(all(len(cue) <= 48 for cue in rendered_cues))


class MoodboardSearchCatalogTests(unittest.TestCase):
    @staticmethod
    def _record(item_id: int, title: str, keywords: list[str]) -> MoodboardRecord:
        uuid = f"11111111-1111-5111-9111-{item_id:012d}"
        return MoodboardRecord(
            url=f"https://www.krea.ai/moodboard-feed/board-{item_id}-{uuid}",
            slug=f"board-{item_id}-{uuid}",
            uuid=uuid,
            title=title,
            taste_profile="",
            keywords=keywords,
        )

    def test_catalog_old_regression_and_cache_invalidation_after_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> None:
                await init_moodboard_db(db_path)
                bold_id = await upsert_moodboard(self._record(1, "Bold", ["bold"]), db_path)
                old_id = await upsert_moodboard(self._record(2, "Old School", ["old-school"]), db_path)
                first = await list_moodboards(query="old", db_path=db_path)
                self.assertIn(old_id, [row["id"] for row in first["items"]])
                self.assertNotIn(bold_id, [row["id"] for row in first["items"]])
                builds = moodboard_search_cache_build_count(db_path)
                await list_moodboards(query="old", db_path=db_path)
                self.assertEqual(moodboard_search_cache_build_count(db_path), builds)
                await set_moodboard_qwen_guidance(
                    bold_id,
                    {"style_axes": ["old print"], "guidance_version": 2},
                    db_path=db_path,
                )
                second = await list_moodboards(query="old", db_path=db_path)
                self.assertEqual(moodboard_search_cache_build_count(db_path), builds + 1)
                self.assertIn(bold_id, [row["id"] for row in second["items"]])
                self.assertIn(old_id, [row["id"] for row in second["items"]])

            asyncio.run(run())

    def test_lightweight_loader_and_ordered_page_hydration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> None:
                await init_moodboard_db(db_path)
                ids: list[int] = []
                for index, title in enumerate(("Alpha", "Beta"), 1):
                    record = self._record(index, title, ["manual keyword"])
                    record.image_urls = [
                        f"https://optim-images.krea.ai/{index}-{image}.webp"
                        for image in range(80)
                    ]
                    board_id = await upsert_moodboard(record, db_path)
                    await set_moodboard_qwen_guidance(
                        board_id,
                        {
                            "style_axes": ["analog film grain"],
                            "prompt_guidance": "Medium and texture: analog film grain.",
                            "conditioning_notes": ["Preserve cinematic texture."],
                            "guidance_version": 2,
                        },
                        db_path=db_path,
                    )
                    ids.append(board_id)

                documents = _load_search_documents_sync(db_path)
                official = [
                    document
                    for document in documents
                    if document["source"] == "official"
                ]
                self.assertEqual(len(official), 2)
                for document in official:
                    self.assertNotIn("image_urls", document)
                    self.assertNotIn("primary_image_url", document)
                    self.assertNotIn("related_urls", document)
                    self.assertNotIn("qwen_guidance_json", document)

                with patch(
                    "moodboards_catalog._hydrate_moodboard_items",
                    wraps=_hydrate_moodboard_items,
                ) as hydrate:
                    page = await list_moodboards(
                        query="analog film grain",
                        source="official",
                        page=2,
                        page_size=1,
                        db_path=db_path,
                    )
                self.assertEqual(hydrate.await_count, 1)
                self.assertEqual(hydrate.await_args.args[0], [ids[1]])
                self.assertEqual(page["total"], 2)
                self.assertEqual([row["id"] for row in page["items"]], [ids[1]])
                self.assertEqual(page["items"][0]["title"], "Beta")
                self.assertEqual(len(page["items"][0]["image_urls"]), 80)

            asyncio.run(run())

    def test_favorite_boost_is_user_specific(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> None:
                await init_moodboard_db(db_path)
                first = await upsert_moodboard(self._record(1, "First", ["muted teal"]), db_path)
                second = await upsert_moodboard(self._record(2, "Second", ["muted teal"]), db_path)
                for board_id in (first, second):
                    await set_moodboard_qwen_guidance(
                        board_id,
                        {
                            "style_axes": ["muted teal"],
                            "prompt_guidance": "Palette: muted teal.",
                            "guidance_version": 2,
                        },
                        db_path=db_path,
                    )
                await set_moodboard_favorite(second, True, db_path, username="alice")
                alice = await suggest_moodboards("muted teal", "", "alice", db_path=db_path)
                bob = await suggest_moodboards("muted teal", "", "bob", db_path=db_path)
                self.assertEqual(alice[0]["id"], second)
                self.assertEqual(bob[0]["id"], first)
                self.assertFalse(any(row["id"] == second and row.get("favorite") for row in bob))

            asyncio.run(run())

    def test_warm_3549_document_search_reuses_index_under_generous_budget(self) -> None:
        items = [
            item(
                index,
                f"Board {index}",
                keywords=[f"texture-{index}", "common style"],
                axes=[
                    "analog grain"
                    if index % 11 == 0
                    else "cinematic art"
                ],
            )
            for index in range(1, 3550)
        ]
        started = time.perf_counter()
        index = MoodboardSearchIndex(items)
        build_elapsed = time.perf_counter() - started
        identity = id(index)
        started = time.perf_counter()
        for _ in range(5):
            results = index.search("analog grain")
        warm_elapsed = (time.perf_counter() - started) / 5
        started = time.perf_counter()
        suggestions = index.suggest(
            "cinematic art",
            "cinematic art analog film grain",
            limit=12,
        )
        suggestion_elapsed = time.perf_counter() - started
        self.assertEqual(id(index), identity)
        self.assertTrue(results)
        self.assertEqual(len(suggestions), 12)
        self.assertLess(warm_elapsed, 0.15)
        self.assertLess(suggestion_elapsed, 0.5)
        self.assertLess(build_elapsed, 3.0)

    def test_production_shaped_3609_document_build_and_suggestion_budget(self) -> None:
        long_guidance = (
            "Palette: muted amber and teal with monochrome shadow harmony. "
            "Lighting: cinematic low-key rim lighting with soft atmospheric diffusion. "
            "Medium and texture: analog film grain, distressed risograph halftone, "
            "and painterly watercolor finish. Composition: editorial framing with "
            "negative space and geometric balance."
        )
        named_styles = (
            "bauhaus",
            "surrealism",
            "film noir",
            "editorial risograph",
        )
        items = [
            item(
                index,
                f"Production Board {index}",
                axes=[
                    "cinematic low-key lighting",
                    "heavy analog film grain",
                    "muted teal palette",
                    "editorial geometric framing",
                    named_styles[index % len(named_styles)],
                ],
                notes=[
                    "Use soft focus and atmospheric haze for filmic depth.",
                    "Preserve high contrast and tactile print texture. "
                    + "Analog diffusion. " * (index % 3),
                ],
                prompt=long_guidance
                + " "
                + "Cinematic editorial color treatment. " * (index % 5),
            )
            for index in range(1, 3610)
        ]
        started = time.perf_counter()
        index = MoodboardSearchIndex(items)
        build_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        suggestions = index.suggest(
            "cinematic lighting",
            "cinematic low-key lighting with analog film grain",
            limit=12,
        )
        suggestion_elapsed = time.perf_counter() - started
        self.assertEqual(len(suggestions), 12)
        self.assertLess(build_elapsed, 5.0)
        self.assertLess(suggestion_elapsed, 0.5)


class MoodboardSearchCacheTests(unittest.TestCase):
    @staticmethod
    def _cache_item(item_id: int) -> dict:
        return item(item_id, f"Board {item_id}", axes=["analog grain"])

    def test_cold_cache_build_is_single_flight_for_eight_threads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "single-flight.db"
            invalidate_moodboard_search_cache(db_path)
            calls = 0
            call_lock = threading.Lock()
            start = threading.Barrier(8)

            def loader() -> list[dict]:
                nonlocal calls
                with call_lock:
                    calls += 1
                time.sleep(0.05)
                return [self._cache_item(1)]

            def load() -> object:
                start.wait()
                return get_cached_moodboard_search(db_path, loader)

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _index: load(), range(8)))
            self.assertEqual(calls, 1)
            self.assertEqual(len({id(result) for result in results}), 1)

    def test_invalidation_during_build_never_publishes_stale_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "invalidate.db"
            invalidate_moodboard_search_cache(db_path)
            first_started = threading.Event()
            release_first = threading.Event()
            calls = 0
            call_lock = threading.Lock()

            def loader() -> list[dict]:
                nonlocal calls
                with call_lock:
                    calls += 1
                    call_number = calls
                if call_number == 1:
                    first_started.set()
                    release_first.wait(timeout=2)
                return [self._cache_item(call_number)]

            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(get_cached_moodboard_search, db_path, loader)
                self.assertTrue(first_started.wait(timeout=2))
                waiter = pool.submit(get_cached_moodboard_search, db_path, loader)
                invalidate_moodboard_search_cache(db_path)
                release_first.set()
                first_result = first.result(timeout=2)
                waiter_result = waiter.result(timeout=2)

            self.assertEqual(calls, 2)
            self.assertEqual(first_result.items[0]["id"], 2)
            self.assertIs(first_result, waiter_result)

    def test_loader_exception_wakes_waiter_and_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "retry.db"
            invalidate_moodboard_search_cache(db_path)
            first_started = threading.Event()
            release_first = threading.Event()
            calls = 0
            call_lock = threading.Lock()

            def loader() -> list[dict]:
                nonlocal calls
                with call_lock:
                    calls += 1
                    call_number = calls
                if call_number == 1:
                    first_started.set()
                    release_first.wait(timeout=2)
                    raise RuntimeError("loader failed")
                return [self._cache_item(2)]

            with ThreadPoolExecutor(max_workers=2) as pool:
                failed = pool.submit(get_cached_moodboard_search, db_path, loader)
                self.assertTrue(first_started.wait(timeout=2))
                waiter = pool.submit(get_cached_moodboard_search, db_path, loader)
                release_first.set()
                with self.assertRaisesRegex(RuntimeError, "loader failed"):
                    failed.result(timeout=2)
                recovered = waiter.result(timeout=2)

            retried = get_cached_moodboard_search(db_path, loader)
            self.assertEqual(calls, 2)
            self.assertEqual(recovered.items[0]["id"], 2)
            self.assertIs(recovered, retried)


if __name__ == "__main__":
    unittest.main()
