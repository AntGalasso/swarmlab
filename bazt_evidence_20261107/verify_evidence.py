"""Read-only stdlib audit of retained evidence; never imports a provider client."""
from pathlib import Path, PurePosixPath
import gzip, hashlib, json, tarfile
ROOT = Path(__file__).resolve().parent

def require(value, message):
    if not value:
        raise RuntimeError(message)

def main():
    manifest = json.loads((ROOT / 'FILE_MANIFEST.json').read_text())
    for name, record in manifest.items():
        p = ROOT / name
        require(p.is_file(), 'Missing file ' + name)
        require(p.stat().st_size == record['bytes'], 'Size mismatch ' + name)
        require(hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256'], 'Hash mismatch ' + name)
    raw_manifest = json.loads((ROOT / 'primary/RAW_ARCHIVES.json').read_text())
    require(len(raw_manifest) == 5, 'Missing primary topology archive')
    for topology, record in raw_manifest.items():
        whole = hashlib.sha256()
        size = 0
        for part in record['parts']:
            data = (ROOT / part['path']).read_bytes()
            whole.update(data)
            size += len(data)
            require(hashlib.sha256(data).hexdigest() == part['sha256'], 'Changed raw ZIP part')
            require(hashlib.sha1(b'blob ' + str(len(data)).encode() + bytes([0]) + data).hexdigest() == part['git_blob_sha'], 'Raw part Git identity differs')
        require(size == record['bytes'] and whole.hexdigest() == record['sha256'], 'Combined primary ZIP differs ' + topology)
    audit = {'original_primary_archives_preserved': 5}
    for name in ['observation_parity_raw_results_20261107.tar.gz', 'observability_extension_complete_20261107.tar.gz']:
        with tarfile.open(ROOT / 'observation' / name, 'r:gz') as t:
            members = {m.name: m for m in t.getmembers()}
            require(all((not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in members)), 'Unsafe archive path')
            rows = 0
            pairs = 0
            groups = {}
            for n, m in members.items():
                if not n.endswith('.jsonl') or '/units/' not in n:
                    continue
                done = n[:-6] + '.done'
                require(done in members, 'Uncommitted raw unit ' + n)
                records = [json.loads(line) for line in t.extractfile(m).read().splitlines() if line.strip()]
                ids = [r['id'] for r in records]
                committed = json.loads(t.extractfile(members[done]).read())
                require(ids == committed, 'Row/done identity mismatch ' + n)
                require(len(ids) == len(set(ids)), 'Duplicate unit in ' + n)
                group = n.rsplit('/units/', 1)[0]
                seen = groups.setdefault(group, set())
                require(not seen & set(ids), 'Duplicate raw ID across batches')
                seen.update(ids)
                require(all((r.get('violations', 0) == 0 for r in records)), 'Enforcement violation ' + n)
                rows += len(ids)
                pairs += 1
            require(pairs > 0, 'Missing raw committed records')
            audit[name] = {'committed_pairs': pairs, 'raw_rows_including_calibration': rows, 'record_groups': {g: len(v) for g, v in groups.items()}}
    rows = [json.loads(line) for line in gzip.open(ROOT / 'semantic/LLM_CANONICAL_ROWS_42391846181.jsonl.gz', 'rt') if line.strip()]
    require(len(rows) == 320, 'Wrong semantic row count')
    keys = [(r['block'], r['arm'], r['seed'], r['conversation'], r['attack']) for r in rows]
    require(len(set(keys)) == 320, 'Duplicate semantic observation')
    require(sum((r['attack'] == 'none' for r in rows)) == 160, 'Wrong clean count')
    require({r['seed'] for r in rows} == set(range(1001, 1011)), 'Semantic seed grid changed')
    require(all((r.get('violations', 0) == 0 for r in rows)), 'Semantic budget violation')
    review = json.loads((ROOT / 'semantic/INDEPENDENT_REVIEW.json').read_text())
    require(review['resource_final']['explicitly_acknowledged_unknown_calls'] == [998], 'Unknown998 was discarded')
    require(review['resource_final']['unknown_reservations_remain_fully_charged'], 'Unknown cost not conserved')
    comparison = json.loads((ROOT / 'semantic/BLINDED_TRANSFER_RESULT.json').read_text())
    require(comparison['team_level_blinded_validation'] is False, 'Incorrect blinding claim')
    audit.update(files_hashed=len(manifest), semantic_rows=320, semantic_seeds=10, provider_requests=0, team_level_blinded_validation=False)
    print(json.dumps(audit, indent=2))
    return audit
if __name__ == '__main__':
    main()
