"""The upload process: `python -m bazt_lab.features.sync.adapters.child '<config json>'`. It reads "stop" (or end of
input, which is what a dead parent looks like) from stdin, does a final push, prints one JSON summary line on stdout
and exits. Uploading is I/O bound here, so the experiment process never waits for it."""
from __future__ import annotations

import importlib
import json
import signal
import sys
import threading

from bazt_lab.features.sync.adapters import env
from bazt_lab.features.sync.private import loop
from bazt_lab.features.sync.private.settings import SyncSettings


def _uploader(config: dict):
    name = config.get("uploader")                    # "module:callable", for tests; the Hub uploader when absent
    module, _, attribute = (name or "bazt_lab.features.sync.adapters.hub:make_uploader").partition(":")
    return getattr(importlib.import_module(module), attribute)(config)


def main(argv: list) -> int:
    config = json.loads(argv[1])
    env.load()
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # the parent's Ctrl-C ends the run; the final push is stop()'s
    fields = config["settings"]
    settings = SyncSettings(**{**fields, "include": tuple(fields["include"]), "exclude": tuple(fields["exclude"])})
    uploads = loop.Uploads(settings, config["paths"], _uploader(config), redact=env.redact)
    stop = threading.Event()

    def wait_for_stop() -> None:
        sys.stdin.readline()                         # "stop\n" or end of input
        stop.set()

    threading.Thread(target=wait_for_stop, daemon=True).start()
    loop.run(uploads, stop)
    print(json.dumps({"commits": uploads.commits, "failures": uploads.failures}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
