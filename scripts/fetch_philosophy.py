"""Fetch Kant + Nietzsche raw text from Project Gutenberg.

Downloads a curated set of Gutenberg .txt files into ``--out`` (default
``data/philosophy/``). Idempotent: skips files that already exist with
non-trivial size. Tolerates 404s — a missing book is logged and skipped, the
overall corpus stays usable.

The strip-license step is handled later by ``prepare_uncensored_data.py``;
this script writes raw Gutenberg .txt verbatim.

All IDs below were verified to point at the right Kant/Nietzsche work via
``https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt``. The mirror URL
pattern is stable; bare ID changes upstream are rare but possible.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# (gutenberg_id, filename_stem, human_title)
BOOKS: list[tuple[int, str, str]] = [
    # --- Kant ---
    (4280,  "kant_critique_of_pure_reason",         "Kant — Critique of Pure Reason (Meiklejohn)"),
    (5683,  "kant_critique_of_practical_reason",    "Kant — Critique of Practical Reason (Abbott)"),
    (48433, "kant_critique_of_judgement",           "Kant — Critique of Judgement (Bernard)"),
    (5682,  "kant_metaphysic_of_morals_groundwork", "Kant — Fundamental Principles of the Metaphysic of Morals (Abbott)"),
    (50922, "kant_perpetual_peace",                 "Kant — Perpetual Peace"),
    # --- Nietzsche ---
    (1998,  "nietzsche_zarathustra",                "Nietzsche — Thus Spake Zarathustra (Common)"),
    (4363,  "nietzsche_beyond_good_and_evil",       "Nietzsche — Beyond Good and Evil (Zimmern)"),
    (19322, "nietzsche_antichrist",                 "Nietzsche — The Antichrist (Mencken)"),
    (52190, "nietzsche_ecce_homo",                  "Nietzsche — Ecce Homo"),
    (51356, "nietzsche_birth_of_tragedy",           "Nietzsche — The Birth of Tragedy (Haussmann)"),
    (52319, "nietzsche_genealogy_of_morals",        "Nietzsche — The Genealogy of Morals (Samuel)"),
    (52263, "nietzsche_twilight_of_the_idols",      "Nietzsche — Twilight of the Idols + The Antichrist"),
    (52881, "nietzsche_joyful_wisdom",              "Nietzsche — The Joyful Wisdom (The Gay Science)"),
    (38145, "nietzsche_human_all_too_human_v1",     "Nietzsche — Human, All Too Human, Vol. I (Harvey)"),
    (39955, "nietzsche_dawn_of_day",                "Nietzsche — The Dawn of Day (Kennedy)"),
]

URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
MIN_BYTES = 20_000  # smaller than this almost certainly means a Gutenberg error page

USER_AGENT = "lrd-reason-fetch/1.0 (+https://github.com/teddytennant/lrd-reason)"


def fetch_one(gid: int, out_path: Path, retries: int = 2, timeout: int = 60) -> tuple[bool, int]:
    """Download one Gutenberg book. Returns (success, bytes_written).

    Skips download if ``out_path`` already exists with size >= MIN_BYTES.
    On HTTP error or partial read, returns (False, 0) without raising.
    """
    if out_path.exists() and out_path.stat().st_size >= MIN_BYTES:
        return True, out_path.stat().st_size

    url = URL_TEMPLATE.format(id=gid)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < MIN_BYTES:
                last_err = f"too small ({len(data)} bytes)"
                continue
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.rename(out_path)
            return True, len(data)
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = f"URL error: {e.reason}"
        except TimeoutError:
            last_err = "timeout"
        if attempt < retries:
            time.sleep(2 ** attempt)

    print(f"  ! skip id={gid}: {last_err}", file=sys.stderr)
    return False, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/philosophy"),
                    help="output directory for .txt files")
    ap.add_argument("--list", action="store_true",
                    help="print the curated book list and exit")
    args = ap.parse_args()

    if args.list:
        for gid, stem, title in BOOKS:
            print(f"{gid:6d}  {stem:42s}  {title}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    ok_count = 0
    for gid, stem, title in BOOKS:
        target = args.out / f"{stem}.txt"
        cached = target.exists() and target.stat().st_size >= MIN_BYTES
        ok, nbytes = fetch_one(gid, target)
        if ok:
            ok_count += 1
            total_bytes += nbytes
            tag = "cached" if cached else "fetched"
            print(f"  [{tag}] {nbytes/1024:7.1f} KB  {title}")

    print(f"\n[fetch] {ok_count}/{len(BOOKS)} books, {total_bytes/1024/1024:.1f} MB total in {args.out}/")
    if ok_count == 0:
        print("[fetch] ERROR: no books downloaded — check network", file=sys.stderr)
        return 1
    if ok_count < len(BOOKS) // 2:
        print("[fetch] WARNING: fewer than half the books downloaded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
