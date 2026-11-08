"""Reproduce an exploratory audit of the public GAMMAF experiment 1 records."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean


HERE = Path(__file__).resolve().parent
FIELDS = (
    "dataset", "method", "question", "topology", "round", "included",
    "padded", "correct_consensus", "attackers", "flags", "released_adr"
)
TOPOLOGIES = {"chain", "random", "star", "tree"}
BASELINE = "no_defense_baseline"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def upstream_projection(checkout):
    """Verify pinned original bytes and retain only quantities used by this audit."""
    sources = json.loads((HERE / "sources.json").read_text())
    rows = []
    reference = {}
    for item in sources["source_files"]:
        raw = (checkout / item["path"]).read_bytes()
        object_header = f"blob {len(raw)}\0".encode()
        digest = hashlib.sha1(object_header + raw).hexdigest()
        require(len(raw) == item["bytes"] and digest == item["git_blob"], item["path"])
        data = json.loads(raw)
        if "dataset" not in item:
            for source in sources["source_files"]:
                if "dataset" in source:
                    method, ds = source["method"], source["dataset"]
                    cells = data["runs"][method]["datasets"][ds]["results"]
                    for cell in cells:
                        reference[(ds, method, cell["topology"])] = cell["rounds_rates"][2]["ADR"]
            continue
        ds, method = item["dataset"], item["method"]
        require((ds, method) == (data["dataset_tag"], data["model"]), item["path"])
        for row in data["items"]:
            n_agents = row["n_agents"]
            bad = row["malicious_agent_indexes"]
            require(n_agents == 10 and row["n_malicious"] == len(set(bad)) == 3,
                    "Unexpected agent counts")
            require(all(0 <= index < n_agents for index in bad), "Bad role indices")
            flags = row["flags"]
            require(len(flags) in (0, n_agents) and set(flags) <= {0, 1}, "Bad flags")
            attackers = "".join("1" if i in bad else "0" for i in range(n_agents))
            rows.append((ds, method, str(row["loader_index"]), row["topology"],
                         str(row["round"]), str(int(row["included"])), str(int(row["padded"])),
                         str(int(row["consensus"] and row["is_correct"])), attackers,
                         "".join(map(str, flags)), str(row["metrics"]["ADR"])))
    return sorted(rows), reference


def measured_adr(row):
    roles, flags = row["attackers"], row["flags"]
    require(len(roles) == len(flags) == 10 and roles.count("1") == 3,
            "Missing actual flags on an executed round")
    hits = sum(a == f == "1" for a, f in zip(roles, flags))
    return 100 * hits / roles.count("1")


def summarise(rows, reference):
    debates = defaultdict(list)
    for values in rows:
        row = dict(zip(FIELDS, values))
        require(row["included"] == "1", "Excluded rows need an explicit denominator decision")
        key = row["dataset"], row["method"], row["question"], row["topology"]
        debates[key].append(row)

    cells = defaultdict(list)
    questions = defaultdict(dict)
    for (ds, method, question, topo), rounds in sorted(debates.items()):
        rounds.sort(key=lambda row: int(row["round"]))
        require([r["round"] for r in rounds] == ["0", "1", "2"], "Incomplete or duplicated debate")
        observed = [r for r in rounds if r["padded"] == "0"]
        require(observed and observed == rounds[:len(observed)], "Nonterminal padding")
        require(len({r["attackers"] for r in rounds}) == 1, "Changed role labels")
        for r in observed:
            require(abs(measured_adr(r) - float(r["released_adr"])) < 0.0051,
                    "ADR disagrees with executed flags")
        for r in rounds[len(observed):]:
            require(r["flags"] == "", "Padded row with executed flags")
            if ds in {"MMLU", "MMLUPRO", "CSQA", "GSM8K"}:
                require(float(r["released_adr"]) == 100 * int(r["correct_consensus"]),
                        "Outcome-based padding differs from recorded convention")
        final = rounds[-1]
        released = float(final["released_adr"])
        measured = measured_adr(observed[-1])
        stats = (released, measured, final["padded"] == "1", len(observed),
                 sum(r["flags"].count("1") for r in observed),
                 final["padded"] == "1" and released == 100)
        cells[ds, method, topo].append(stats)
        cells[ds, method, "all"].append(stats)
        questions[ds, method].setdefault(question, {})[topo] = (released, measured)

    results = {}
    for (ds, method, topo), cell in sorted(cells.items()):
        released, measured, padded, rounds, flags, perfect_padding = zip(*cell)
        source_mean = fmean(released)
        total_flags = sum(flags)
        if method == BASELINE:
            require(total_flags == 0, f"Disabled detector emitted a flag in {ds}/{topo}")
        if topo != "all":
            require(math.isclose(source_mean, reference[ds, method, topo], abs_tol=1e-7),
                    f"Public aggregate differs for {ds}/{method}/{topo}")
        results.setdefault(ds, {}).setdefault(method, {})[topo] = {
            "debates": len(cell), "padded_third_rounds": sum(padded),
            "padded_third_rounds_with_100_ADR": sum(perfect_padding),
            "executed_rounds_all": sum(rounds), "positive_flags_all_executed_rounds": total_flags,
            "released_round3_ADR_percent": round(source_mean, 6),
            "last_executed_ADR_percent": round(fmean(measured), 6),
        }
    for qs in questions.values():
        require(len(qs) == 100 and all(set(graphs) == TOPOLOGIES for graphs in qs.values()),
                "The four graphs must repeat the same 100 questions")
    return results, questions


def percentile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def paired_comparisons(questions):
    """Exploratory paired comparisons across all six named MMLU detectors."""
    methods = sorted(m for ds, m in questions if ds == "MMLU" and m != BASELINE)
    ids = sorted(questions["MMLU", methods[0]], key=int)
    by_method = {}
    for method in methods:
        qs = questions["MMLU", method]
        require(set(qs) == set(ids), "Methods did not evaluate the same questions")
        by_method[method] = [
            tuple(fmean(cell[k] for cell in qs[question].values()) for k in (0, 1))
            for question in ids
        ]
    # Resample whole questions. The four graph measurements stay together.
    rng = random.Random(20261107)
    n = len(ids)
    weights = [rng.choices(range(n), k=n) for _ in range(5000)]
    comparisons = []
    max_deviation = [0.0] * len(weights)
    for first, second in itertools.combinations(methods, 2):
        values = [(a[0]-b[0], a[1]-b[1]) for a, b in zip(by_method[first], by_method[second])]
        columns = {"released_diff": [v[0] for v in values],
                   "last_executed_diff": [v[1] for v in values],
                   "change_in_contrast": [v[1]-v[0] for v in values]}
        out = {"first": first, "second": second}
        for label, column in columns.items():
            samples = [sum(column[i] for i in sample)/n for sample in weights]
            mean = fmean(column)
            max_deviation = [max(current, abs(stat-mean)) for current, stat
                             in zip(max_deviation, samples)]
            out[label] = {"mean_pp": round(mean, 6),
                          "pointwise_95_percentile_interval_pp":
                              [round(percentile(samples, p), 6) for p in (.025, .975)]}
        comparisons.append(out)
    # Common radius in percentage points over the 15 pairs and all three contrasts.
    radius = percentile(max_deviation, .95)
    for comparison in comparisons:
        for label in ("released_diff", "last_executed_diff", "change_in_contrast"):
            mean = comparison[label]["mean_pp"]
            comparison[label]["simultaneous_95_max_deviation_interval_pp"] = [
                round(mean-radius, 6), round(mean+radius, 6)]
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, help="Check every value against the pinned public Git files")
    parser.add_argument("--output", type=Path, help="Write the full recomputed JSON summary")
    args = parser.parse_args()
    with gzip.open(HERE / "records.csv.gz", "rt", newline="") as stream:
        table = csv.reader(stream)
        require(tuple(next(table)) == FIELDS, "Unexpected projection schema")
        rows = sorted(tuple(r) for r in table)
    reference = {(r["dataset"], r["method"], r["topology"]): r["round3_adr"]
                 for r in json.loads((HERE / "official_aggregates.json").read_text())}
    if args.checkout:
        original_rows, original_reference = upstream_projection(args.checkout)
        require((original_rows, original_reference) == (rows, reference),
                "Bundled projection differs from public source")
    results, questions = summarise(rows, reference)
    output = {"cells": results, "mmlu_detector_pairs": paired_comparisons(questions)}
    if args.output:
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Dataset / method                    Reported round 3    Last executed")
    for ds, by_method in sorted(results.items()):
        for method, summary in sorted(by_method.items()):
            cell = summary["all"]
            print(f"{ds+'/'+method:<35} {cell['released_round3_ADR_percent']:>8.2f}%"
                  f"         {cell['last_executed_ADR_percent']:>8.2f}%")


if __name__ == "__main__":
    main()
