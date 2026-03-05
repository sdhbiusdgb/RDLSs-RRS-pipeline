from __future__ import annotations
import argparse
from .config import load_config
from .pipeline import segment_dataset, analyze_dataset, run_all

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tile-anno", description="De-identified tile annotation pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "segment", "analyze"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True, help="Path to YAML config.")
    return p

def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.cmd == "segment":
        segment_dataset(cfg)
    elif args.cmd == "analyze":
        analyze_dataset(cfg)
    else:
        run_all(cfg)

if __name__ == "__main__":
    main()
