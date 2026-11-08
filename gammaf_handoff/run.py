#!/usr/bin/env python3
"""Run the offline GAMMAF audit, or rebuild it from pinned public source files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import zipfile

from compact_check import check

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest():
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    for relative, digest in manifest['sha256'].items():
        path = HERE / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError('Bundle checksum mismatch: ' + relative)
    return manifest


def command(arguments):
    subprocess.run([sys.executable, *map(str, arguments)], check=True)


def equal_json(actual, expected):
    if json.loads(actual.read_text()) != json.loads(expected.read_text()):
        raise ValueError('Recomputed result differs from reference: ' + actual.name)


def fetch_sources(root, workers):
    voting = json.loads((HERE / 'gamma_sources.json').read_text())
    detection = json.loads((HERE / 'detection_credit/sources.json').read_text())
    pins = {pin['path']: (pin['bytes'], pin['blob']) for pin in voting['files'].values()}
    for pin in detection['source_files']:
        previous = pins.setdefault(pin['path'], (pin['bytes'], pin['git_blob']))
        if previous != (pin['bytes'], pin['git_blob']):
            raise ValueError('Conflicting source pins')

    def get(pin):
        relative, (size, digest) = pin
        path = root / relative
        if path.exists():
            data = path.read_bytes()
        else:
            url = 'https://raw.githubusercontent.com/' + voting['repo'] + '/' + voting['commit'] + '/' + relative
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if len(data) != size or blob != digest:
            raise ValueError('Public source checksum mismatch: ' + relative)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print('Verified source:', relative, flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(get, sorted(pins.items())))
    return len(pins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-source', action='store_true', help='Verify/rebuild from public originals (~122 MB; NumPy required)')
    parser.add_argument('--raw', type=Path, help='Public GAMMAF checkout/cache root (only with --from-source)')
    parser.add_argument('--out', type=Path, help='A new output directory; default: gammaf_handoff/_runs/<UTC timestamp>')
    parser.add_argument('--workers', type=int, default=4, help='Concurrent public-file downloads (1-8)')
    args = parser.parse_args()
    if args.raw and not args.from_source:
        parser.error('--raw requires --from-source')
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    if args.from_source and importlib.util.find_spec('numpy') is None:
        parser.error('Source replay needs NumPy. Install gammaf_handoff/requirements.txt in a virtual environment.')
    manifest = verify_manifest()
    out = (args.out or HERE / '_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print('Output:', out, flush=True)
    projection = HERE / 'gamma_debates.jsonl.gz'
    sources_checked = 0
    if args.from_source:
        raw = (args.raw or HERE / '_source').resolve()
        sources_checked = fetch_sources(raw, args.workers)
        command([HERE / 'gamma_votes.py', 'develop', '--raw', raw, '--out', out])
        command([HERE / 'gamma_votes.py', 'validate', '--raw', raw, '--out', out])
        projection = out / 'gamma_debates.jsonl.gz'
        command([HERE / 'verify_votes.py', '--raw', raw, '--debates', projection,
                 '--out', out / 'VOTE_VERIFICATION.json'])
        for name in ('MMLU_SELECTION.json', 'GAMMAF_RESULTS.json', 'VOTE_VERIFICATION.json'):
            equal_json(out / name, HERE / 'expected' / name)
        command([HERE / 'detection_credit/audit.py', '--checkout', raw,
                 '--output', out / 'DETECTION_CHECK.json'])
    else:
        command([HERE / 'detection_credit/audit.py', '--output', out / 'DETECTION_CHECK.json'])
    equal_json(out / 'DETECTION_CHECK.json', HERE / 'detection_credit/results.json')
    voting = check(projection, HERE / 'expected', out)
    receipt = {'status': 'PASS', 'mode': 'public_source_rebuild' if args.from_source else 'bundled_projection',
               'python': platform.python_version(), 'platform': platform.platform(),
               'public_commit': manifest['public_commit'], 'source_files_verified': sources_checked,
               'terminal_decisions_recomputed': voting['decisions_recomputed'],
               'detection_credit_matches_reference': True,
               'bootstrap_intervals_recomputed': bool(args.from_source),
               'frozen_source_results_match_reference': bool(args.from_source),
               'manifest_sha256': sha256(HERE / 'MANIFEST.json'),
               'no_new_model_calls': True}
    if args.from_source:
        import numpy
        receipt['numpy'] = numpy.__version__
    (out / 'CHECKS.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    archive = out / 'reproduction.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.iterdir()):
            if path.is_file() and path != archive:
                bundle.write(path, arcname=path.name)
    print('PASS: 11,200 terminal decisions; voting and detection checks match references.', flush=True)
    print('Defended-label disagreements:', voting['defended']['disagreement'], '/ 9600', flush=True)
    print('Frozen narrow-alphabet sensitivity:', voting['frozen_narrow_alphabet_no_ties'], flush=True)
    print('Dataset-alphabet sensitivity:', voting['dataset_alphabet_no_ties'], flush=True)
    print('Return this file to Ricardo:', archive, flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('FAIL:', error, file=sys.stderr)
        sys.exit(1)
