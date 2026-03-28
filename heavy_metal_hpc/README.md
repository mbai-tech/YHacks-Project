# heavy_metal_hpc

High-performance digital twin scaffold for arsenic transport and remediation
optimization in bodies of water in Bangladesh.

The project couples a physics-based advection-diffusion-reaction model with
inverse parameter estimation, ensemble uncertainty quantification, Gemini-based
forcing interpretation, and Auth0-secured AI-agent workflows.

---

## Project layout

```
heavy_metal_hpc/
├── data/               # raw inputs, processed intermediates, cached outputs
├── src/
│   ├── api/            # external data loaders (weather, hydrology APIs)
│   ├── grid/           # mesh generation and lake geometry
│   ├── physics/        # PDE operators, transport, sediment exchange
│   ├── model/          # simulator, state vector, parameter containers
│   ├── inference/      # loss functions, gradient-based parameter estimation
│   ├── ensemble/       # Monte-Carlo / UQ ensemble runner
│   ├── optimization/   # remediation objective, constraints, solvers
│   ├── viz/            # heatmaps, uncertainty ribbons, diagnostic plots
│   ├── parallel/       # MPI helpers and distributed task scheduling
│   └── utils/          # configuration loading and generic I/O
├── notebooks/          # exploratory Jupyter notebooks
├── tests/              # pytest unit & integration tests
└── scripts/            # CLI entry-points for simulation / ensemble / optimization
```

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd heavy_metal_hpc

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a toy forward simulation
python scripts/run_simulation.py --config config/default.yaml

# 5. Run the ensemble (parallel, 64 members)
mpirun -n 8 python scripts/run_ensemble.py --n-members 64

# 6. Solve the remediation optimization problem
python scripts/run_optimization.py --budget 1e6
```

---

## Gemini and Auth0

The repo now includes:

- `src/ai/gemini.py`: a lightweight Gemini REST client for structured planning
  and forcing summaries.
- `src/agents/auth0.py`: Auth0 context helpers for user-scoped API access in AI
  agents.
- `src/api/loader.py`: a unified forcing loader that can synthesize hydrology,
  fetch Open-Meteo weather, and ask Gemini for an operational summary.

Set these environment variables before running secured agent flows:

```bash
export GEMINI_API_KEY=...
export AUTH0_CLIENT_ID=...
export AUTH0_CLIENT_SECRET=...
```

For the browser UI demo:

```bash
python scripts/run_web_ui.py --config config/default.yaml
```

Then open `http://127.0.0.1:5001`, click `Sign In With Auth0`, finish login in the browser, and click `Generate Brief`.

## Key dependencies

| Package | Purpose |
|---------|---------|
| NumPy / SciPy | Array math, sparse solvers |
| Numba / CuPy | JIT & GPU acceleration for stencil kernels |
| JAX / PyTorch | Automatic differentiation for parameter inference |
| mpi4py / Dask | Distributed ensemble runs |
| Pyomo / NLopt | Remediation optimization |
| xarray / netCDF4 | Structured geospatial data I/O |
| Pydantic | Validated configuration schemas |

---

## License

MIT
