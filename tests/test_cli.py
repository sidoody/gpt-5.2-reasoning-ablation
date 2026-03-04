import pytest

from gpt_5_2_reasoning_ablation.cli import build_parser


def test_analyze_pairs_command_is_available():
    parser = build_parser()
    args = parser.parse_args(["analyze-pairs"])
    assert args.command == "analyze-pairs"


def test_analyze_overthinking_is_supported_as_deprecated_alias():
    parser = build_parser()
    args = parser.parse_args(["analyze-overthinking"])
    assert args.command in {"analyze-pairs", "analyze-overthinking"}


def test_xhigh_is_rejected_by_cli_variant_choices():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--variants", "none", "xhigh"])
