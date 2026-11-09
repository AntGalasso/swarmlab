#!/usr/bin/env python3
"""Run only the reserved CPU N12/N100 diagnostic in an isolated pinned worktree."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
PIN = "a88f146675134133b85ae7eaf16d38048096c7c6"
BLOCKS = ("mock_calibration", "mock_epidemiology")


def checked(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, **kw)


def output(cmd):
    return checked(cmd, capture_output=True).stdout.strip()


def validate(settings=HERE / "n100_frozen.toml", protocol=HERE / "protocol.json"):
    raw = settings.read_bytes()
    p, s = json.loads(protocol.read_text()), tomllib.loads(raw.decode())
    if hashlib.sha256(raw).hexdigest() != p["settings_sha256"] or p["source_commit"] != PIN:
        raise ValueError("Frozen settings/source digest changed; register a separate protocol first.")
    m = s["mock"]
    assert m["main_seeds"] == [1,30] and m["tune_seeds"] == [9001,9150] and m["heldout_seeds"] == [31,60]
    assert m["epidemiology_agents"] == [12,100] and m["every_sink"] is False
    assert m["families"] == ["insider_pi","insider_ta","insider_ma","infection","aitm"]
    assert m["epidemiology"] == {"topologies":["barabasi_albert","star_hub","small_world"],
        "arms":["C","BAZT","BAZT_upstream"],"cells":[[10,10],["uncapped","uncapped"]]}
    assert s["calibration"]["mode"] == "closed_loop" and s["calibration"]["max_passes"] == 40
    assert s["analysis"]["pair_unit"]["mock"] == "seed" and s["analysis"]["utility_endpoint"] == "accuracy_clean"
    assert p["evaluation_reserved_for_owner"] and p["no_llm_calls"]
    count = 2*3*6*3*2*30
    assert count == p["expected_evaluation_units"] == 6480
    return s, p


def ensure_runtime(repo, runtime, pin=PIN):
    if runtime.exists():
        if output(["git","-C",str(runtime),"rev-parse","HEAD"]) != pin:
            raise ValueError("Existing runtime worktree has a different commit; refusing to replace it.")
    else:
        have = subprocess.run(["git","-C",str(repo),"cat-file","-e",pin+"^{commit}"],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if have.returncode:
            checked(["git","-C",str(repo),"fetch","--no-tags","origin",pin])
        runtime.parent.mkdir(parents=True,exist_ok=True)
        checked(["git","-C",str(repo),"worktree","add","--detach",str(runtime),pin])
    if output(["git","-C",str(runtime),"status","--porcelain","--untracked-files=all"]):
        raise ValueError("Pinned runtime has local changes; refusing to overwrite or mix source versions.")
    if output(["git","-C",str(runtime),"rev-parse","HEAD"]) != pin:
        raise ValueError("Runtime commit verification failed.")


def commands(runtime, settings, results, workers):
    base=["uv","run","--project",str(runtime),"--frozen","python",str(runtime/"run_experiments.py"),
          "--settings",str(settings),"--device","cpu","--workers",str(workers),
          "--set","mock.batch_size=32","--results-root",str(results)]
    return [base+["--run","--block",block] for block in BLOCKS]


def calibration_gate(path, p):
    data=json.loads(path.read_text())
    needed={(a,t,n) for a in p["evaluation_arms"] for t in p["topologies"] for n in p["agents"]}
    cells={(c["arm"],c["topology"],c["agents"]):c for c in data["cells"]}
    if not needed <= cells.keys():
        raise ValueError("Calibration is incomplete for the requested evaluation grid; resume calibration.")
    flagged=[]
    for key in sorted(needed):
        q=cells[key].get("diagnostics",{})
        rate=q.get("rate")
        if not isinstance(rate,(int,float)) or not 0 <= rate <= p["empirical_alpha_upper_cap"]:
            raise ValueError("No admissible calibration at the declared empirical cap for "+str(key))
        if not q.get("within_tolerance",False):flagged.append({"cell":key,**q})
    return {"required_cells":len(needed),"flagged_unmatched_cells":flagged,
            "interpretation":"Below-target cells remain exploratory; not exact equal-FPR matching."}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--results-root",type=Path,default=Path.cwd()/"results/antonio_n100_20261107")
    ap.add_argument("--runtime-root",type=Path)
    ap.add_argument("--dry-run",action="store_true",help="Validate and print commands; no fetch, worktree, inference or episodes.")
    a=ap.parse_args(argv)
    if not 1 <= a.workers <= 8:ap.error("--workers must be1..8")
    s,p=validate();settings=(HERE/"n100_frozen.toml").resolve();results=a.results_root.resolve()
    repo=HERE.parent.resolve();runtime=(a.runtime_root or repo.parent/"swarmlab_antonio_n100_runtime_a88f").resolve()
    if results == repo or results == repo/"results" or results.is_relative_to(repo/"results/experiments"):
        raise ValueError("Use a new dedicated output folder, never the existing experiments/live folder.")
    cmds=commands(runtime,settings,results,a.workers)
    if a.dry_run:
        print(json.dumps({"dry_run":True,"source":PIN,"expected_units":6480,"commands":cmds},indent=2));return 0
    results.mkdir(parents=True,exist_ok=True)
    marker=results/"ANTONIO_N100_MANIFEST.json"
    identity={"source_commit":PIN,"settings_sha256":p["settings_sha256"],"blocks":list(BLOCKS),"owner":p["owner"]}
    if marker.exists() and json.loads(marker.read_text())["identity"] != identity:
        raise ValueError("This output folder belongs to another protocol; use a new folder.")
    if not marker.exists() and any(p.name != "ANTONIO_N100.lock" for p in results.iterdir()):
        raise ValueError("Output folder is nonempty and has no matching handoff manifest; nothing overwritten.")
    from filelock import FileLock
    with FileLock(str(results/"ANTONIO_N100.lock"),timeout=0):
        marker.write_text(json.dumps({"identity":identity,"protocol":p,"runtime_worktree":str(runtime)},indent=2)+"\n")
        ensure_runtime(repo,runtime)
        env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
             "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","WANDB_MODE":"offline"}
        checked(cmds[0],cwd=runtime,env=env)
        profile=s["active_profile"]
        gate=calibration_gate(results/profile/"mock_calibration/thresholds.json",p)
        (results/"CALIBRATION_GATE.json").write_text(json.dumps(gate,indent=2)+"\n")
        checked(cmds[1],cwd=runtime,env=env)
        checked(["uv","run","--project",str(runtime),"--frozen","python",str(HERE/"summarize.py"),
                 "--results",str(results/profile),"--expected","6480"],cwd=runtime,env=env)
        print("Complete CPU diagnostic:",results,"\nRerun the same command to resume/check existing units.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
