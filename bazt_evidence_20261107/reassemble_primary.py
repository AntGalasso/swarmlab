"""Reassemble byte-identical original Actions ZIPs; no extraction or inference."""
from pathlib import Path
import argparse, hashlib, json
ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('topology', choices=['barabasi_albert', 'chain', 'small_world', 'star_hub', 'star_leaf'])
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    rec = json.loads((ROOT / 'primary/RAW_ARCHIVES.json').read_text())[a.topology]
    target = a.output_dir / ('bazt-headline-evaluation-' + a.topology + '.zip')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == rec['sha256']:
            print('Already exact:', target)
            return
        raise RuntimeError('Refusing to replace a different file: ' + str(target))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.zip.partial')
    digest = hashlib.sha256()
    with tmp.open('xb') as out:
        for item in rec['parts']:
            raw = (ROOT / item['path']).read_bytes()
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise RuntimeError('Part differs')
            digest.update(raw)
            out.write(raw)
    if digest.hexdigest() != rec['sha256']:
        raise RuntimeError('Combined archive differs')
    tmp.rename(target)
    print('Verified original ZIP:', target)
if __name__ == '__main__':
    main()
