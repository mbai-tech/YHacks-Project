"""Flask application factory for the arsenic digital twin demo."""

from __future__ import annotations

from pathlib import Path

from flask import Flask

from ..frontend import create_frontend_blueprint
from ..utils.config import load_config


def create_app(config_path: str | Path) -> Flask:
    """Create the Flask app and register the browser-facing frontend."""
    app = Flask(__name__, static_folder=None)
    app.secret_key = "arsenic-digital-twin-demo-secret"
    cfg = load_config(config_path)
    app.register_blueprint(create_frontend_blueprint(cfg))
    return app
