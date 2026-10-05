"""CLI entry point for running Adam's WebUI sidecar standalone.

Usage:
    python -m sidecar [options]
    python sidecar/main.py [options]
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add project root to sys.path if not already present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import WebUIConfig, load_config
from sidecar.bridge import StandaloneBridge, DisconnectedBridge
from sidecar.server import start_sidecar, _is_loopback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Adam's optional WebUI sidecar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--config", default="config.yaml", help="Path to Adam configuration file")
    parser.add_argument("--auth-token", default="", help="Optional authentication bearer token")
    return parser.parse_args()


async def run_server(args: argparse.Namespace) -> None:
    if not _is_loopback(args.host):
        print(
            f"[ERROR] Insecure host '{args.host}': Adam WebUI server must bind loopback only "
            "(127.0.0.1 or ::1) unless a future explicit secure remote design exists.",
            file=sys.stderr,
        )
        sys.exit(1)

    app_config = load_config(args.config)
    web_config = WebUIConfig(
        enabled=True,
        host=args.host,
        port=args.port,
        auth_token=args.auth_token or getattr(app_config.webui, "auth_token", ""),
    )

    bridge = DisconnectedBridge(
        reason=(
            "Standalone sidecar running in disconnected mode. To connect WebUI to Adam, "
            "start the daemon with WebUI enabled (e.g. `webui.enabled: true` in config "
            "or `python -m src.main --webui`). Standalone sidecar does not instantiate "
            "a duplicate brain or fake connection to the live daemon."
        )
    )
    print("[Sidecar] Running standalone WebUI sidecar in disconnected mode.")
    print("[Sidecar] Standalone processes do not instantiate duplicate brain engines.")
    print("[Sidecar] To run with live voice and tools, launch Adam with `--webui`: python -m src.main --webui")

    runner = await start_sidecar(web_config, bridge)

    print("=" * 60)
    print(f" ADAM WEBUI SIDECAR READY (DISCONNECTED STANDALONE MODE)")
    print(f" URL: http://{web_config.host}:{web_config.port}")
    print(f" Binding: Loopback only ({web_config.host})")
    print(f" Auth: {'Enabled' if web_config.auth_token else 'Disabled (Loopback trust)'}")
    print(f" Voice Mode: Primary default preserved")
    print("=" * 60)
    print("Press Ctrl+C to shut down.")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    await stop_event.wait()
    print("\n[Sidecar] Shutting down sidecar runner...")
    await runner.cleanup()
    print("[Sidecar] Clean shutdown complete.")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
