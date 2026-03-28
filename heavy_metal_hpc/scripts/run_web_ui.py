#!/usr/bin/env python
"""Run the small browser UI for Auth0 login and Gemini brief generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.web import create_app


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the browser UI demo.")
    parser.add_argument("--config", required=True, help="Path to YAML run configuration.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    app = create_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
