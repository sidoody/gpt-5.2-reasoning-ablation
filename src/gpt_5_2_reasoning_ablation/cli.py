from __future__ import annotations

import argparse

from .analysis import analyze_overthinking, summarize_runs
from .grading import grade_variants
from .reporting import export_discordant_cases, generate_final_artifacts
from .runner import run_variants
from .settings import StudySettings, SUPPORTED_REASONING_LEVELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt52-ablation",
        description="GPT-5.2 reasoning-ablation benchmark runner.",
    )
    parser.add_argument("--config", default=None, help="Optional path to a JSON settings file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run GPT-5.2 inference for one or more reasoning settings.")
    run_parser.add_argument("--variants", nargs="+", choices=SUPPORTED_REASONING_LEVELS, default=None)
    run_parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke tests.")
    run_parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing results for the selected variants.")

    grade_parser = subparsers.add_parser("grade", help="Grade saved runs with GPT-4.1.")
    grade_parser.add_argument("--variants", nargs="+", choices=SUPPORTED_REASONING_LEVELS, default=None)
    grade_parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing grades for the selected variants.")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize completed runs.")
    summarize_parser.add_argument("--write", default=None, help="Optional JSON output path for the summary rows.")

    overthinking_parser = subparsers.add_parser("analyze-overthinking", help="Pairwise analysis across reasoning settings.")
    overthinking_parser.add_argument("--write", default=None, help="Optional JSON output path for the pairwise analysis.")

    report_parser = subparsers.add_parser(
        "report",
        help="Generate publishable report artifacts under reports/ from saved results and scores.",
    )
    report_parser.add_argument(
        "--discordant-limit",
        type=int,
        default=30,
        help="How many none-vs-high discordant examples to export (default: 30).",
    )

    discordant_parser = subparsers.add_parser(
        "export-discordant",
        help="Export discordant diagnosis cases for a paired comparison.",
    )
    discordant_parser.add_argument("--a", choices=SUPPORTED_REASONING_LEVELS, default="none")
    discordant_parser.add_argument("--b", choices=SUPPORTED_REASONING_LEVELS, default="high")
    discordant_parser.add_argument("--limit", type=int, default=30)
    discordant_parser.add_argument("--write", default=None, help="Optional output path (JSON).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = StudySettings.from_json(args.config)

    if args.command == "run":
        run_variants(settings, requested_variants=args.variants, limit=args.limit, overwrite=args.overwrite)
    elif args.command == "grade":
        grade_variants(settings, requested_variants=args.variants, overwrite=args.overwrite)
    elif args.command == "summarize":
        summarize_runs(settings, write_path=args.write)
    elif args.command == "analyze-overthinking":
        analyze_overthinking(settings, write_path=args.write)
    elif args.command == "report":
        generate_final_artifacts(settings, discordant_limit=args.discordant_limit)
    elif args.command == "export-discordant":
        rows = export_discordant_cases(
            settings,
            a_level=args.a,
            b_level=args.b,
            limit=args.limit,
            write_path=args.write,
        )
        if rows:
            print(f"Exported {len(rows)} discordant cases for {args.a} vs {args.b}.")
        else:
            print(f"No discordant cases available for {args.a} vs {args.b}.")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
