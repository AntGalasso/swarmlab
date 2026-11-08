"""Render the descriptive audit figure. Requires matplotlib."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "results.json").read_text())["cells"]
blue, orange, ink = "#14587a", "#a84426", "#172a36"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.labelcolor": ink, "text.color": ink,
                     "xtick.color": ink, "ytick.color": ink, "pdf.fonttype": 42})
fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.75),
                                 gridspec_kw={"width_ratios": [0.87, 1.22]})
fig.subplots_adjust(left=.12, right=.975, bottom=.26, top=.78, wspace=.40)

def draw(ax, cells, xmax, ticks):
    for y, (_, cell) in enumerate(cells):
        released = cell["released_round3_ADR_percent"]
        executed = cell["last_executed_ADR_percent"]
        ax.plot([released, executed], [y, y], color="#9daeb7", lw=1.6, zorder=2)
        # Vertical separation keeps coincident zeroes visible in the negative controls.
        ax.scatter(released, y-.060, marker="o", s=38, facecolors="white",
                   edgecolors=orange, linewidths=1.6, zorder=4)
        ax.scatter(executed, y+.060, marker="D", s=23, color=blue, zorder=5)
    ax.set_yticks(range(len(cells)), [label for label, _ in cells])
    ax.set_ylim(len(cells)-.42, -.55)
    ax.set_xlim(-xmax*.035, xmax*1.02)
    ax.set_xticks(ticks)
    ax.set_xlabel("Adversarial detection rate (%)", labelpad=7)
    ax.grid(axis="x", color="#dce2e5", linewidth=.7)
    ax.tick_params(length=0, pad=4)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#a5b4bc")

order = ["MMLU", "MMLUPRO", "CSQA", "GSM8K", "MA", "TA"]
labels = {"MMLUPRO": "MMLU-Pro", "MA": "Memory attack", "TA": "Tool attack"}
draw(left, [(labels.get(ds, ds), data[ds]["no_defense_baseline"]["all"]) for ds in order],
     8.5, [0, 2, 4, 6, 8])
methods = ["XG-Guard", "CASPIAN", "DOMINANT", "BlindGuard", "PREM", "TAM"]
draw(right, [(m, data["MMLU"][m]["all"]) for m in methods], 100, [0, 25, 50, 75, 100])
left.set_title("A   No detector", loc="left", weight="bold", fontsize=11, pad=13)
right.set_title("B   Six detectors on MMLU", loc="left", weight="bold", fontsize=11, pad=13)
handles = [Line2D([], [], ls="", marker="o", mfc="white", mec=orange, mew=1.6,
                  label="Released third-round score"),
           Line2D([], [], ls="", marker="D", color=blue, markersize=5.3,
                  label="Flags in last executed round")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .008),
           ncol=2, frameon=False, handletextpad=.5, columnspacing=2.1)
fig.suptitle("Detection credit after early consensus", y=.99, fontsize=15, weight="bold")
fig.savefig(HERE / "detection_credit.pdf", bbox_inches="tight", pad_inches=.12)
