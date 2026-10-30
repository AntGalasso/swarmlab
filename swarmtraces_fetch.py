#!/usr/bin/env python3
"""swarmtraces_fetch.py -- PRE-EXISTING SCAFFOLDING. Download and PROFILE the public Swarm Traces dataset.

Source (public, per the organisers): https://swarmtraces.org/data/final/redacted.jsonl.gz  (JSONL, gzipped, ~16 MB)
Report: https://swarmtraces.org/  |  Viewer: https://swarmtraces.org/viewer/

What this does NOT do, on purpose (dual-use):
  * never prints, copies or re-publishes payload text; never executes anything from the data;
  * never tries to un-redact anything (credentials, hosts, usernames are redacted by the authors);
  * keeps the raw file under data/ (git-ignored) so the repo does not redistribute attack content.

What it DOES: schema discovery (the exact field names are not documented in the report, so nothing is
hard-coded), counts, low-cardinality field distributions, text-length statistics, and keyword PREVALENCE
for coordination-related vocabulary. Keyword prevalence is a crude lexical signal, not a behaviour
classifier: say so in the paper.

  python swarmtraces_fetch.py download
  python swarmtraces_fetch.py profile [--max-rows 0] [--file data/redacted.jsonl.gz]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import urllib.request

URL = "https://swarmtraces.org/data/final/redacted.jsonl.gz"
DEFAULT = "data/redacted.jsonl.gz"
MAX_BYTES = 200 * 1024 * 1024  # refuse anything unexpectedly large (expected ~16 MB)

# Lexical vocabulary taken from the public report's descriptions of coordination machinery.
# Each entry is a regex; we only COUNT matching rows, we never store the matched text.
VOCAB = {
    "duplicate_suppression_lock_claim": r"\b(claim|lockfile|lock file|elect one|only one (worker|controller)|dedup|duplicate)\b",
    "signed_or_authenticated_command": r"\b(signature|signed|verify|hmac|rsa)\b",
    "encryption_of_results": r"\b(encrypt|aes|fernet|cipher)\b",
    "cleanup_or_trace_removal": r"\b(cleanup|clean up|delete|remove only files|restore original|neutraliz)\w*",
    "controller_or_relay_naming": r"\b(controller|bridge|relay|bot|beacon|c2)\b",
    "polling_command_channel": r"\b(poll|command channel|loadcmd|mailbox|board)\b",
    "dns_exfil_vocabulary": r"\b(getaddrinfo|dns|b32encode)\b",
    "kubernetes_vocabulary": r"\b(kubernetes|kube-system|namespace|configmap|serviceaccount)\b",
    "evaluation_or_grader_vocabulary": r"\b(cybergym|exploit ?gym|grader|flag|scorer)\b",
    "uses_other_llm_apis": r"\b(chat/completions|inference|completion)\b",
}
VOCAB_RE = {k: re.compile(v, re.I) for k, v in VOCAB.items()}


def download(dest: str = DEFAULT) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "swarmlab-research-fetch/1.0"})
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            n += len(chunk)
            if n > MAX_BYTES:
                f.close()
                os.remove(dest)
                sys.exit("Refusing: file larger than expected (>200 MB).")
            h.update(chunk)
            f.write(chunk)
    print(f"saved {dest}  {n / 1e6:.1f} MB  sha256={h.hexdigest()}")
    print("Keep it local: data/ is git-ignored. Do not redistribute it with your submission.")


def _flatten(obj, prefix="", depth=0):
    """Yield (path, value) for scalars; lists are summarised as <path>[]."""
    if isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k), depth + 1)
    elif isinstance(obj, list):
        yield (prefix + "[]", len(obj))
        for v in obj[:3]:
            if isinstance(v, (dict, list)) and depth < 3:
                yield from _flatten(v, prefix + "[]", depth + 1)
    else:
        yield (prefix, obj)


def profile(path: str = DEFAULT, max_rows: int = 0, out: str = "data/swarmtraces_profile.json") -> dict:
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run: python swarmtraces_fetch.py download")
    n = 0
    types = collections.defaultdict(collections.Counter)
    present = collections.Counter()
    distinct = collections.defaultdict(collections.Counter)
    lens = collections.defaultdict(list)
    vocab_hits = collections.Counter()
    text_rows = 0
    bad = 0
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            n += 1
            blob_parts = []
            seen_keys = set()
            for key, val in _flatten(row):
                seen_keys.add(key)
                types[key][type(val).__name__] += 1
                if isinstance(val, str):
                    lens[key].append(len(val))
                    if len(val) <= 64 and len(distinct[key]) <= 300:
                        distinct[key][val] += 1
                    elif len(val) > 64:
                        blob_parts.append(val)
                elif isinstance(val, (int, float, bool)) and len(distinct[key]) <= 300:
                    distinct[key][str(val)] += 1
            for k in seen_keys:
                present[k] += 1
            if blob_parts:
                text_rows += 1
                blob = "\n".join(blob_parts)
                for name, rx in VOCAB_RE.items():
                    if rx.search(blob):
                        vocab_hits[name] += 1
            if max_rows and n >= max_rows:
                break

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None

    fields = {}
    for k in sorted(types, key=lambda k: -present[k]):
        info = {"present_rows": present[k], "present_pct": round(100 * present[k] / max(1, n), 2),
                "types": dict(types[k])}
        if lens[k]:
            info["str_len"] = {"p50": pct(lens[k], .5), "p90": pct(lens[k], .9), "max": max(lens[k])}
        if 0 < len(distinct[k]) <= 300:
            info["low_cardinality_top10"] = distinct[k].most_common(10)  # short labels only (<=64 chars)
        fields[k] = info
    rep = {"file": path, "rows": n, "unparseable_lines": bad, "rows_with_long_text": text_rows,
           "vocab_prevalence_rows": dict(vocab_hits),
           "vocab_prevalence_pct_of_text_rows": {k: round(100 * v / max(1, text_rows), 2) for k, v in vocab_hits.items()},
           "fields": fields,
           "caveats": ["lexical prevalence only, not behaviour classification",
                       "report: ~80% of payloads are outbound, 97% carry no timestamp, agent names are self-chosen",
                       "dataset may include traffic not from the swarm (see the report's Limitations)"]}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"rows={n}  unparseable={bad}  rows_with_long_text={text_rows}")
    print("top fields:")
    for k, v in list(fields.items())[:15]:
        extra = f"  top={v['low_cardinality_top10'][:5]}" if "low_cardinality_top10" in v else ""
        print(f"  {k:<32}{v['present_pct']:>6}%  {v['types']}{extra}")
    print("vocabulary prevalence (% of rows with long text):")
    for k, v in sorted(rep["vocab_prevalence_pct_of_text_rows"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<36}{v:>6}%")
    print("wrote", out)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "profile"])
    ap.add_argument("--file", default=DEFAULT)
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    download(a.file) if a.cmd == "download" else profile(a.file, a.max_rows)
