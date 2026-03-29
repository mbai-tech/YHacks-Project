"""Frontend blueprint for the Water Risk Explorer dashboard."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

from ..utils.config import RunConfig
from ..web.water_risk_service import DEFAULT_SCENARIO, WaterRiskService


def create_frontend_blueprint(cfg: RunConfig) -> Blueprint:
    """Create a frontend blueprint backed by the local data service."""
    frontend = Blueprint(
        "frontend",
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/frontend-static",
    )
    data_dir = Path(__file__).resolve().parents[3] / "data"
    service = WaterRiskService(data_dir)

    @frontend.get("/")
    def index():
        return render_template("index.html")

    @frontend.get("/api/search")
    def search():
        query = request.args.get("q", "")
        return jsonify({"results": service.search(query)})

    @frontend.get("/api/risk")
    def risk():
        latitude = float(request.args["lat"])
        longitude = float(request.args["lon"])
        name = request.args.get("name")
        return jsonify(service.get_risk(latitude, longitude, name=name))

    @frontend.post("/api/scenario")
    def scenario():
        payload = request.get_json(force=True) or {}
        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])
        name = payload.get("name")
        return jsonify(service.apply_scenario(latitude, longitude, payload.get("scenario", {}), name=name))

    @frontend.post("/api/compare")
    def compare():
        payload = request.get_json(force=True) or {}
        result = service.compare(payload["locationA"], payload["locationB"], payload.get("scenario", DEFAULT_SCENARIO))
        return jsonify(result)

    @frontend.get("/api/map-data")
    def map_data():
        scenario_payload = {
            key: request.args.get(key, default, type=float)
            for key, default in DEFAULT_SCENARIO.items()
        }
        return jsonify(service.build_map_payload(scenario_payload))

    @frontend.post("/api/brief")
    def brief():
        payload = request.get_json(force=True) or {}
        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])
        brief_text = service.build_brief(
            latitude,
            longitude,
            payload.get("scenario", DEFAULT_SCENARIO),
            name=payload.get("name"),
        )
        output_path = Path(cfg.output_dir) / "web_remediation_brief.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(brief_text)
        return jsonify({"brief": brief_text})

    return frontend
