"""Small Flask web UI for Auth0 login and Gemini brief generation."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for

from ..agents.auth0 import Auth0AgentContext, Auth0ConfigurationError
from ..agents.reporting import build_brief_prompt
from ..ai.gemini import GeminiClient
from ..api.hydrology import HydrologyAPI
from ..api.loader import DataLoader
from ..api.weather import WeatherAPI
from ..utils.config import load_config
from .map_data import build_map_payload


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Arsenic Digital Twin</title>
    <style>
      :root {
        --bg: #dfe8db;
        --panel: rgba(247, 244, 234, 0.9);
        --panel-strong: #f6f1e5;
        --ink: #12322b;
        --accent: #0f6a73;
        --accent-2: #b7652a;
        --accent-3: #405e3f;
        --muted: #60706a;
        --line: rgba(18, 50, 43, 0.12);
      }
      body {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        background:
          radial-gradient(circle at top left, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0) 35%),
          linear-gradient(135deg, #edf4e5 0%, #d7e5d7 45%, #d8d7c9 100%);
        color: var(--ink);
      }
      .shell {
        max-width: 1220px;
        margin: 0 auto;
        padding: 36px 20px 64px;
      }
      .hero {
        display: grid;
        grid-template-columns: 1.25fr 0.95fr;
        gap: 22px;
        align-items: stretch;
      }
      .card {
        background: var(--panel);
        backdrop-filter: blur(12px);
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 26px;
        box-shadow: 0 18px 50px rgba(35, 55, 49, 0.09);
      }
      .eyebrow {
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 10px;
      }
      h1 {
        font-size: clamp(2.2rem, 4vw, 4rem);
        line-height: 0.98;
        margin: 0 0 14px;
      }
      p {
        line-height: 1.58;
        margin: 0;
      }
      .row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 20px 0 0;
      }
      button, a.button {
        border: 0;
        border-radius: 999px;
        padding: 12px 18px;
        font-size: 1rem;
        background: var(--accent);
        color: white;
        text-decoration: none;
        cursor: pointer;
      }
      button.secondary, a.secondary {
        background: var(--accent-2);
      }
      button.tertiary {
        background: var(--accent-3);
      }
      .stack {
        display: grid;
        gap: 18px;
      }
      .status, .brief, .insight {
        padding: 18px;
        border-radius: 18px;
        background: rgba(255,255,255,0.65);
        border: 1px solid var(--line);
      }
      .muted {
        color: var(--muted);
      }
      pre {
        white-space: pre-wrap;
        font-family: "SFMono-Regular", Menlo, monospace;
      }
      .map-card {
        position: relative;
        overflow: hidden;
      }
      .map-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
          radial-gradient(circle at 20% 15%, rgba(255,255,255,0.55), transparent 30%),
          linear-gradient(180deg, rgba(15,106,115,0.08), rgba(183,101,42,0.07));
        pointer-events: none;
      }
      .map-wrap {
        position: relative;
        aspect-ratio: 1.08 / 1;
        border-radius: 22px;
        overflow: hidden;
        background:
          linear-gradient(180deg, rgba(204, 223, 200, 0.95), rgba(164, 198, 173, 0.98)),
          #b6d0bc;
        border: 1px solid rgba(18, 50, 43, 0.14);
      }
      #map {
        width: 100%;
        height: 100%;
        display: block;
      }
      .hud {
        position: absolute;
        left: 18px;
        right: 18px;
        bottom: 18px;
        display: flex;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
      }
      .pill {
        background: rgba(247, 244, 234, 0.92);
        border: 1px solid rgba(18, 50, 43, 0.12);
        border-radius: 999px;
        padding: 10px 14px;
        font-size: 0.92rem;
      }
      .legend {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 14px;
        color: var(--muted);
        font-size: 0.92rem;
      }
      .legend-bar {
        flex: 1;
        height: 12px;
        border-radius: 999px;
        background: linear-gradient(90deg, #1b6f72 0%, #e0ab32 52%, #b53a1d 100%);
      }
      .metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 18px;
      }
      .metric {
        background: rgba(255,255,255,0.7);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px;
      }
      .metric strong {
        display: block;
        font-size: 1.4rem;
        margin-top: 6px;
      }
      .split {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
        margin-top: 20px;
      }
      ul {
        margin: 10px 0 0;
        padding-left: 18px;
      }
      li + li {
        margin-top: 6px;
      }
      .error {
        color: #8a2d12;
      }
      @media (max-width: 960px) {
        .hero,
        .split,
        .metrics {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="hero">
        <div class="card map-card">
          <div class="eyebrow">Digital Twin / Bangladesh Surface Water</div>
          <h1>Arsenic Spread Map</h1>
          <p>This view simulates dissolved arsenic spread across a Bangladesh delta-style water network using the PDF’s transport terms: groundwater loading, runoff, upstream inflow, sediment exchange, attenuation, and phased remediation.</p>
          <div class="legend">
            <span>Lower concentration</span>
            <div class="legend-bar"></div>
            <span>Hotspot</span>
          </div>
          <div class="map-wrap" style="margin-top:18px;">
            <canvas id="map"></canvas>
            <div class="hud">
              <div class="pill" id="time-pill">Loading timeline...</div>
              <div class="pill" id="stat-pill">Computing plume...</div>
            </div>
          </div>
          <div class="row">
            <button class="tertiary" id="toggle-map">Pause Animation</button>
            <button class="secondary" id="reload-map">Reload Scenario</button>
          </div>
          <div class="metrics">
            <div class="metric">
              <div class="muted">Peak dissolved arsenic</div>
              <strong id="peak-metric">--</strong>
            </div>
            <div class="metric">
              <div class="muted">Final mean concentration</div>
              <strong id="mean-metric">--</strong>
            </div>
            <div class="metric">
              <div class="muted">Saved snapshots</div>
              <strong id="snap-metric">--</strong>
            </div>
          </div>
        </div>

        <div class="stack">
          <div class="card">
            <div class="eyebrow">Copilot</div>
            <h1 style="font-size: clamp(1.8rem, 3vw, 2.8rem);">Bangladesh Arsenic Operations Copilot</h1>
            <p>Sign in with Auth0, pull forcing data, and generate a Gemini remediation brief from the browser.</p>
            <div class="row">
              <a class="button" href="{{ url_for('start_login') }}">Sign In With Auth0</a>
              <button class="secondary" id="generate" {% if not user %}disabled{% endif %}>Generate Brief</button>
              {% if user %}
              <a class="button secondary" href="{{ url_for('logout') }}">Log Out</a>
              {% endif %}
            </div>
            <div class="status" style="margin-top:20px;">
              <strong>Status</strong>
              <div id="status-text">
                {% if user %}
                  Signed in as {{ user.get("email", user.get("sub", "unknown user")) }}
                {% elif error %}
                  {{ error }}
                {% else %}
                  Not signed in yet.
                {% endif %}
              </div>
            </div>
            <div class="insight" style="margin-top:18px;">
              <strong>Scenario Notes</strong>
              <ul id="scenario-notes">
                <li>Loading simulation assumptions...</li>
              </ul>
            </div>
          </div>

          <div class="card brief" id="brief-box">
            <strong>Brief Output</strong>
            <pre id="brief-text">{% if brief %}{{ brief }}{% else %}No brief generated yet.{% endif %}</pre>
          </div>
        </div>
      </div>
    </div>

    <script>
      let animationHandle = null;
      let frameIndex = 0;
      let playing = true;
      let mapPayload = null;

      async function generateBrief() {
        const button = document.getElementById("generate");
        const briefText = document.getElementById("brief-text");
        button.disabled = true;
        briefText.textContent = "Generating remediation brief...";
        const response = await fetch("{{ url_for('generate_brief') }}", { method: "POST" });
        const data = await response.json();
        briefText.textContent = data.brief || data.error || "No response.";
        button.disabled = false;
      }

      function blendColor(stops, t) {
        const clamped = Math.max(0, Math.min(1, t));
        const scaled = clamped * (stops.length - 1);
        const low = Math.floor(scaled);
        const high = Math.min(stops.length - 1, low + 1);
        const frac = scaled - low;
        const a = stops[low];
        const b = stops[high];
        return [
          Math.round(a[0] + (b[0] - a[0]) * frac),
          Math.round(a[1] + (b[1] - a[1]) * frac),
          Math.round(a[2] + (b[2] - a[2]) * frac),
        ];
      }

      function drawMap() {
        if (!mapPayload) return;
        const canvas = document.getElementById("map");
        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const width = rect.width;
        const height = rect.height;
        const frame = mapPayload.frames[frameIndex];
        const mask = mapPayload.mask;
        const nx = frame.length;
        const ny = frame[0].length;
        const cellW = width / nx;
        const cellH = height / ny;
        const maxValue = Math.max(mapPayload.stats.max_concentration_ugL, 1);
        const palette = [
          [24, 88, 95],
          [45, 135, 122],
          [221, 171, 57],
          [209, 98, 39],
          [158, 41, 24],
        ];

        ctx.clearRect(0, 0, width, height);
        const background = ctx.createLinearGradient(0, 0, 0, height);
        background.addColorStop(0, "#d9ead7");
        background.addColorStop(1, "#a7c5ae");
        ctx.fillStyle = background;
        ctx.fillRect(0, 0, width, height);

        for (let i = 0; i < nx; i++) {
          for (let j = 0; j < ny; j++) {
            const x = i * cellW;
            const y = height - (j + 1) * cellH;
            if (!mask[i][j]) {
              ctx.fillStyle = "rgba(114, 149, 109, 0.14)";
              ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
              continue;
            }
            const ratio = Math.max(0, Math.min(1, frame[i][j] / maxValue));
            const [r, g, b] = blendColor(palette, ratio);
            ctx.fillStyle = "rgba(" + r + "," + g + "," + b + "," + (0.25 + ratio * 0.72) + ")";
            ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
          }
        }

        ctx.save();
        ctx.globalAlpha = 0.18;
        ctx.strokeStyle = "#0f6a73";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let yStep = 0; yStep <= 40; yStep++) {
          const yn = yStep / 40;
          const xn = 0.6 - 0.14 * Math.sin(2.6 * (yn - 0.1)) + 0.03 * Math.cos(7.0 * yn);
          const px = xn * width;
          const py = height - yn * height;
          if (yStep === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.restore();

        document.getElementById("time-pill").textContent =
          "Simulation hour " + mapPayload.times[frameIndex].toFixed(1);
        document.getElementById("stat-pill").textContent =
          "Peak " + mapPayload.stats.max_concentration_ugL.toFixed(1) + " ug/L";
      }

      function renderNotes() {
        if (!mapPayload) return;
        const notes = [
          "Groundwater discharge hotspots are seeded in the northwest and west-central floodplain.",
          "Runoff pulses intensify during high-precipitation periods, matching the PDF's monsoon transport emphasis.",
          "North-to-south inflow strengthens the river corridor and carries arsenic through connected surface water.",
          "Remediation ramps up in downstream cells later in the run to mimic phased intervention deployment.",
        ];
        if (mapPayload.forcing_summary && mapPayload.forcing_summary.monsoon_risk) {
          notes[1] = "Forcing summary reports monsoon risk as " + mapPayload.forcing_summary.monsoon_risk + ", and runoff strength is scaled to that precipitation signal.";
        }
        const list = document.getElementById("scenario-notes");
        list.innerHTML = notes.map((item) => "<li>" + item + "</li>").join("");
        document.getElementById("peak-metric").textContent =
          mapPayload.stats.max_concentration_ugL.toFixed(1) + " ug/L";
        document.getElementById("mean-metric").textContent =
          mapPayload.stats.mean_concentration_ugL.toFixed(1) + " ug/L";
        document.getElementById("snap-metric").textContent = String(mapPayload.stats.snapshot_count);
      }

      function startAnimation() {
        if (animationHandle) {
          clearInterval(animationHandle);
        }
        animationHandle = setInterval(() => {
          if (!playing || !mapPayload) return;
          frameIndex = (frameIndex + 1) % mapPayload.frames.length;
          drawMap();
        }, 420);
      }

      async function loadMapSimulation() {
        const notes = document.getElementById("scenario-notes");
        notes.innerHTML = "<li>Building simulation...</li>";
        document.getElementById("time-pill").textContent = "Loading timeline...";
        document.getElementById("stat-pill").textContent = "Computing plume...";
        const response = await fetch("{{ url_for('map_simulation') }}");
        const data = await response.json();
        if (!response.ok) {
          notes.innerHTML = "<li class='error'>" + (data.error || "Unable to load map.") + "</li>";
          document.getElementById("stat-pill").textContent = "Map unavailable";
          return;
        }
        mapPayload = data;
        frameIndex = 0;
        renderNotes();
        drawMap();
        startAnimation();
      }

      document.getElementById("generate").addEventListener("click", generateBrief);
      document.getElementById("toggle-map").addEventListener("click", () => {
        playing = !playing;
        document.getElementById("toggle-map").textContent = playing ? "Pause Animation" : "Resume Animation";
      });
      document.getElementById("reload-map").addEventListener("click", loadMapSimulation);
      window.addEventListener("resize", drawMap);
      loadMapSimulation();
    </script>
  </body>
</html>
"""


def create_app(config_path: str | Path) -> Flask:
    """Create the Flask app."""
    app = Flask(__name__)
    app.secret_key = "arsenic-digital-twin-demo-secret"
    cfg = load_config(config_path)
    map_cache: dict[str, object] = {}

    def _auth0() -> Auth0AgentContext:
        return Auth0AgentContext.from_env(
            domain=cfg.auth0.domain,
            audience=cfg.auth0.audience,
            client_id_env=cfg.auth0.client_id_env,
            client_secret_env=cfg.auth0.client_secret_env,
            token_vault_audience=cfg.auth0.token_vault_audience,
        )

    def _gemini() -> GeminiClient:
        return GeminiClient.from_env(
            api_key_env=cfg.gemini.api_key_env,
            model=cfg.gemini.model,
            base_url=cfg.gemini.base_url,
        )

    def _loader(gemini: GeminiClient) -> DataLoader:
        return DataLoader(
            weather_api=WeatherAPI(
                base_url=cfg.data_sources.weather_base_url,
                api_key="",
                cache_dir=cfg.data_sources.cache_dir,
            ),
            hydrology_api=HydrologyAPI(
                base_url=cfg.data_sources.hydrology_base_url,
                api_key="",
                cache_dir=cfg.data_sources.cache_dir,
            ),
            cache_dir=cfg.data_sources.cache_dir,
            gemini_client=gemini,
        )

    @app.get("/")
    def index():
        return render_template_string(
            INDEX_HTML,
            user=session.get("user_profile"),
            error=session.pop("auth_error", None),
            brief=session.get("brief"),
        )

    @app.get("/login")
    def start_login():
        try:
            redirect_uri = url_for("auth_callback", _external=True)
            authorize_url, state = _auth0().build_authorize_url(
                redirect_uri=redirect_uri,
                include_audience=False,
            )
            session["oauth_state"] = state
            return redirect(authorize_url)
        except Auth0ConfigurationError as exc:
            session["auth_error"] = str(exc)
            return redirect(url_for("index"))

    @app.get("/callback")
    def auth_callback():
        if request.args.get("error"):
            session["auth_error"] = request.args.get("error_description", request.args["error"])
            return redirect(url_for("index"))

        expected_state = session.get("oauth_state")
        actual_state = request.args.get("state")
        if not expected_state or expected_state != actual_state:
            session["auth_error"] = "Auth0 login failed because the OAuth state did not match."
            return redirect(url_for("index"))

        code = request.args.get("code")
        if not code:
            session["auth_error"] = "Auth0 did not return an authorization code."
            return redirect(url_for("index"))

        redirect_uri = url_for("auth_callback", _external=True)
        token = _auth0().exchange_authorization_code(code=code, redirect_uri=redirect_uri)
        session["access_token"] = token["access_token"]
        session["user_profile"] = _auth0().fetch_user_profile(token["access_token"])
        session.pop("oauth_state", None)
        return redirect(url_for("index"))

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("index"))

    @app.post("/generate")
    def generate_brief():
        user_profile = session.get("user_profile")
        if not user_profile:
            return jsonify({"error": "Please sign in first."}), 401
        if cfg.time is None:
            return jsonify({"error": "Config must define a time window."}), 400

        gemini = _gemini()
        forcing = _loader(gemini).load(cfg.time.start_date, cfg.time.end_date)
        brief = gemini.generate_text(
            prompt=build_brief_prompt(user_profile, forcing, cfg.name),
            system_instruction=(
                "Return practical, safety-aware markdown for environmental remediation planning. "
                "Keep the tone executive and action-oriented."
            ),
        )
        session["brief"] = brief
        output_path = Path(cfg.output_dir) / "web_remediation_brief.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(brief)
        return jsonify({"brief": brief})

    @app.get("/api/map-simulation")
    def map_simulation():
        try:
            if "payload" not in map_cache:
                loader = DataLoader(
                    weather_api=WeatherAPI(
                        base_url=cfg.data_sources.weather_base_url,
                        api_key="",
                        cache_dir=cfg.data_sources.cache_dir,
                    ),
                    hydrology_api=HydrologyAPI(
                        base_url=cfg.data_sources.hydrology_base_url,
                        api_key="",
                        cache_dir=cfg.data_sources.cache_dir,
                    ),
                    cache_dir=cfg.data_sources.cache_dir,
                )
                map_cache["payload"] = build_map_payload(cfg, loader)
            return jsonify(map_cache["payload"])
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            return jsonify({"error": str(exc)}), 500

    return app
