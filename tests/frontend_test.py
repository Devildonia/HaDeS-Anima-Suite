"""Frontend test: the HaDeS widgets in a real ComfyUI frontend, in a headless browser (needs `pip install playwright`).

Creates all 24 nodes in the classic canvas and in Nodes 2.0 (Vue) and runs tests/frontend_checks.js on them: HaDeS UI
installed, native widgets hidden, links dropped on a row reach its input, sockets on their rows, vanilla view complete,
dropdown filter box. Any browser error from the pack's scripts also fails the test.

  python tests/frontend_test.py --comfy-dir /path/to/ComfyUI                    # Playwright's Chromium
  python tests/frontend_test.py --comfy-dir /path/to/ComfyUI --channel msedge   # an installed Edge / Chrome
"""
import argparse
import os
import sys

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_test import PACK_NAME, Server  # noqa: E402

CHECKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend_checks.js")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-dir", required=True, help="ComfyUI root (main.py); this pack must be in its custom_nodes")
    ap.add_argument("--python", default=sys.executable, help="interpreter of the ComfyUI environment")
    ap.add_argument("--port", type=int, default=8198)
    ap.add_argument("--channel", default=None, help="browser channel instead of Playwright's Chromium (msedge, chrome)")
    args, extra = ap.parse_known_args()
    with open(CHECKS, encoding="utf-8") as f:
        checks = f.read()

    server = Server(os.path.abspath(args.comfy_dir), args.python, args.port, extra)
    failures = []
    try:
        server.wait_ready()
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            browser_errors = []
            page.on("pageerror", lambda e: browser_errors.append(f"page error: {e}"))
            page.on("console", lambda m: m.type == "error" and browser_errors.append(f"console: {m.text} @ {m.location.get('url', '')}"))
            page.goto(server.url)
            page.wait_for_function("() => window.comfyAPI?.app?.app?.graph && window.LiteGraph?.registered_node_types?.HaDeSAnimaKSampler", timeout=120000)
            for mode in ("classic", "vue"):
                mode_failures = page.evaluate(checks, mode)
                failures += mode_failures
                print(f"[{'OK' if not mode_failures else 'FAIL'}] {mode}: 24 nodes checked")
            browser.close()
        failures += [e for e in browser_errors if "hades" in e.lower() or PACK_NAME.lower() in e.lower()]
    except Exception:
        print(f"--- ComfyUI log ---\n{server.tail()}", file=sys.stderr)
        raise
    finally:
        server.stop()

    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print("FRONTEND TEST PASSED")


if __name__ == "__main__":
    main()
