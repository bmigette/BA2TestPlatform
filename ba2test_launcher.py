"""Console-script entry point for the BA2 Test Platform (ML / backtest API).

Installed as the ``ba2-test`` command (see ``pyproject.toml`` ``[project.scripts]``).
It launches the FastAPI backend via uvicorn exactly like the documented
``uvicorn app.main:app`` invocation, after putting ``backend/`` on the path — a thin,
behaviour-preserving wrapper.

Works for an editable/source install (the dev setup): the repo root is resolved
from this module's location. Flags: ``--host`` / ``--port`` / ``--reload``.
"""
from __future__ import annotations


def main() -> None:
    import argparse
    import os
    import sys

    repo_root = os.path.dirname(os.path.abspath(__file__))
    backend = os.path.join(repo_root, "backend")
    if not os.path.isdir(backend):
        sys.exit(
            f"ba2-test: backend dir not found at {backend}. The console command "
            f"requires an editable/source install of the test-platform repo."
        )
    sys.path.insert(0, backend)
    os.chdir(backend)

    p = argparse.ArgumentParser(prog="ba2-test", description="Launch the BA2 Test Platform API.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Auto-reload (dev).")
    args = p.parse_args()

    try:
        import uvicorn
    except ImportError:
        sys.exit(
            "ba2-test: uvicorn is not installed in this environment. Install the test "
            "platform backend requirements (backend/requirements.txt) into this venv."
        )
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
