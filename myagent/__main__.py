"""Entry point / CLI.

  openbrain                  run the daemon (default)
  openbrain run|daemon       run the daemon
  openbrain connect [...]    detect & wire installed AI tools
  openbrain init             set up your persona
  openbrain on | off         turn memory on/off (global switch)
  openbrain status           show brain status
  openbrain dashboard        open the web UI

Installed as the `openbrain` console script via pyproject.toml; the bash shim
in bin/openbrain remains for dev use.
"""

import json
import sys
import urllib.request

from . import __version__


def _url(path: str = "") -> str:
    from .config import load_config
    return f"http://127.0.0.1:{load_config().memory_port}{path}"


def _get(path: str) -> dict:
    with urllib.request.urlopen(_url(path), timeout=5) as r:
        return json.loads(r.read())


def _post(path: str) -> dict:
    req = urllib.request.Request(_url(path), method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _run_daemon() -> None:
    import logging
    import uvicorn
    from .config import load_config
    from .server import create_app

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    # Loopback only: this is a local-first daemon, never exposed on a network.
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=cfg.memory_port,
                log_level="info")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"

    if cmd in ("run", "daemon"):
        _run_daemon()
    elif cmd == "init":
        from .seed import run_init
        run_init()
    elif cmd == "connect":
        from .connect import run_connect
        run_connect()
    elif cmd in ("off", "pause"):
        try:
            _post("/pause"); print("🔕 memory OFF — tools use only their own context")
        except Exception:
            print("couldn't reach the daemon (is it running?)")
    elif cmd in ("on", "resume"):
        try:
            _post("/resume"); print("🧠 memory ON — your brain is active again")
        except Exception:
            print("couldn't reach the daemon (is it running?)")
    elif cmd == "status":
        try:
            d = _get("/stats")
            state = "OFF (paused)" if d.get("paused") else "ON"
            print(f"🧠 brain {state} — {d['total']} memories "
                  f"({d['core']} core, {d['auto']} auto) · "
                  f"used {d['injections']}x this session")
        except Exception:
            print("brain unreachable — daemon not responding")
    elif cmd in ("dashboard", "ui", "open"):
        import webbrowser
        webbrowser.open(_url("/"))
    elif cmd in ("-v", "--version", "version"):
        print(f"openbrain {__version__}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
