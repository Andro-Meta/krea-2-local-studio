from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from moodboards_catalog import (  # noqa: E402
    DB_PATH,
    MOODBOARD_SEED_PATH,
    export_moodboard_seed,
    generate_and_store_moodboard_qwen_guidance,
    import_moodboard_seed,
    init_moodboard_db,
)


def missing_guidance_ids(db_path: Path, *, source: str = "official", limit: int = 1, include_existing: bool = False) -> list[int]:
    where = ["source = ?"]
    params: list[object] = [source]
    if not include_existing:
        where.append("COALESCE(qwen_guidance_json, '') = ''")
    query = f"""
        SELECT id
        FROM moodboards
        WHERE {' AND '.join(where)}
        ORDER BY title
        LIMIT ?
    """
    params.append(max(1, int(limit)))
    with sqlite3.connect(str(db_path)) as con:
        return [int(row[0]) for row in con.execute(query, params).fetchall()]


async def enrich(args: argparse.Namespace) -> int:
    await init_moodboard_db(args.db_path)
    if args.import_seed and args.seed_path.exists():
        imported = await import_moodboard_seed(args.seed_path, db_path=args.db_path)
        print(f"Imported/updated {imported} moodboards from {args.seed_path}")

    ids = missing_guidance_ids(
        args.db_path,
        source=args.source,
        limit=args.limit,
        include_existing=args.include_existing,
    )
    if not ids:
        print("No moodboards need Qwen guidance.")
        if args.export_seed:
            exported = await export_moodboard_seed(args.seed_path, db_path=args.db_path)
            print(f"Exported {exported} moodboards to {args.seed_path}")
        return 0

    processed = 0
    for board_id in ids:
        print(f"[{processed + 1}/{len(ids)}] Enriching moodboard id={board_id} ...", flush=True)
        if args.dry_run:
            processed += 1
            continue
        try:
            item = await generate_and_store_moodboard_qwen_guidance(board_id, db_path=args.db_path)
        except Exception as exc:
            print(f"  FAILED id={board_id}: {exc}", flush=True)
            if args.stop_on_error:
                raise
            continue
        processed += 1
        print(f"  OK: {item.get('title', board_id)}", flush=True)
        if args.export_seed and args.export_every > 0 and processed % args.export_every == 0:
            exported = await export_moodboard_seed(args.seed_path, db_path=args.db_path)
            print(f"  Exported {exported} moodboards to {args.seed_path}", flush=True)

    if args.export_seed and not args.dry_run:
        exported = await export_moodboard_seed(args.seed_path, db_path=args.db_path)
        print(f"Exported {exported} moodboards to {args.seed_path}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Qwen guidance for Krea moodboard catalog entries.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Moodboard SQLite database path.")
    parser.add_argument("--seed-path", type=Path, default=MOODBOARD_SEED_PATH, help="Portable seed JSON path.")
    parser.add_argument("--import-seed", action="store_true", help="Import the seed before enriching.")
    parser.add_argument("--export-seed", action="store_true", help="Export enriched seed after processing.")
    parser.add_argument("--export-every", type=int, default=10, help="Refresh seed after every N successful enrichments.")
    parser.add_argument("--limit", type=int, default=1, help="Number of moodboards to enrich this run.")
    parser.add_argument("--source", choices=["official", "custom"], default="official", help="Moodboard source to enrich.")
    parser.add_argument("--include-existing", action="store_true", help="Regenerate guidance even when present.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected IDs without running Qwen.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed enrichment.")
    args = parser.parse_args()
    processed = asyncio.run(enrich(args))
    print(f"Processed {processed} moodboards.")


if __name__ == "__main__":
    main()
