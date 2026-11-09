"""Validate committed mock outputs and write a small descriptive CSV; no hypothesis changes or network."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("--results",type=Path,required=True);p.add_argument("--expected",type=int,required=True)
a=p.parse_args();units=a.results/"mock_epidemiology/units"
seen=set();groups={};violations=0
for file in units.rglob("*.jsonl"):
    if not file.with_suffix(".done").exists():continue
    for line in file.open():
        r=json.loads(line)
        if r["id"] in seen:raise SystemExit("Duplicate unit found")
        seen.add(r["id"]);violations+=r["violations"]
        key=(r["agents"],r["topology"],r["arm"],r["audit"],r["intervention"],r["attack"])
        g=groups.setdefault(key,{"seeds":set(),"sums":collections.defaultdict(float),"counts":collections.Counter()})
        if r["seed"] in g["seeds"]:raise SystemExit("Repeated seed inside a one-sink cell")
        g["seeds"].add(r["seed"])
        ref,got=r.get("accuracy_clean_reference"),r.get("accuracy_attack")
        r["excess_task_error"]=(ref-got if isinstance(ref,(int,float)) and isinstance(got,(int,float)) else math.nan)
        for m in ("asr_final","accuracy_clean","accuracy_attack","accuracy_clean_reference","excess_task_error","targeted_asr","protected_sink_harm","agents_reached_auc","spend_total","fpr_message","benign_delivery_rate"):
            v=r[m]
            if isinstance(v,(int,float)) and math.isfinite(v):g["sums"][m]+=v;g["counts"][m]+=1
valid=len(seen)==a.expected and violations==0 and len(groups)==216 and all(g["seeds"]==set(range(1,31)) for g in groups.values())
report={"complete":valid,"committed_units":len(seen),"expected_units":a.expected,"cells":len(groups),
        "budget_violations":violations,"independent_seeds_per_cell":30,"interpretation":"Exploratory one-target scalability diagnostic; no full-frontier/H-topology confirmation."}
(a.results/"COMPLETENESS.json").write_text(json.dumps(report,indent=2)+"\n")
if not valid:raise SystemExit("Output incomplete or violated an invariant; inspect COMPLETENESS.json and resume.")
metrics=("asr_final","accuracy_clean","accuracy_attack","accuracy_clean_reference","excess_task_error","targeted_asr","protected_sink_harm","agents_reached_auc","spend_total","fpr_message","benign_delivery_rate")
with (a.results/"n100_summary.csv").open("w",newline="") as out:
    w=csv.writer(out);w.writerow(["agents","topology","arm","audit","intervention","attack","seed_count",*[m+"_mean" for m in metrics],*[m+"_n" for m in metrics]])
    for k,g in sorted(groups.items(),key=lambda kv:str(kv[0])):
        w.writerow([*k,len(g["seeds"]),*[g["sums"][m]/g["counts"][m] if g["counts"][m] else "" for m in metrics],*[g["counts"][m] for m in metrics]])
print(json.dumps(report,indent=2))
