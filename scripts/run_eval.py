"""Run all ablations and emit results/ablations.md."""

from __future__ import annotations

import argparse
from pathlib import Path

from lrd_reason.eval.ablations import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--stage1-ckpt", type=str, default=None)
    ap.add_argument("--stage2-ckpt", type=str, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    if out.is_dir() or args.out.endswith("/"):
        out = out / "ablations.md"
    rows = run(args.config, out, stage1_ckpt=args.stage1_ckpt, stage2_ckpt=args.stage2_ckpt)
    print(f"[done] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
