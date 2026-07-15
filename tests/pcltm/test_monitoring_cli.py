from __future__ import annotations

from pcltm.cli import build_parser


def test_webui_parser_defaults_are_local_and_read_only() -> None:
    args = build_parser().parse_args(["webui", "--no-open-browser"])
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.no_open_browser is True
    assert args.refresh_seconds == 5.0
