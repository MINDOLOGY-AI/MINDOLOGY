# the LSCL grid's results in one table and one plot.
#   summary.md    one row per run: steps, learned (wrong_train), leak (wrong_heldout), prior knowledge (correct) and every
#                 GENERAL_TEST score, each as the change from the base model
#   tradeoff.png  per consolidation method (one panel each): prior knowledge kept (x) vs general ability kept (y);
#                 color = strength (light weak -> dark strong), shape = learning method, gray = no consolidation
# general ability kept = the mean over mmlu, gsm8k, humaneval, bbh of (run score / base score), in %.
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.summarize

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from evoke.Qwen3_5_4b.lscl.common import RESULTS

GENERAL_METRICS = ("mmlu", "gsm8k", "humaneval", "bbh")
CONSOLIDATIONS = ("mix", "kl", "ewc")
# sequential blue, weak -> strong (dataviz reference ramp steps 250, 450, 650); gray = none
STRENGTH_COLORS = ("#86b6ef", "#2a78d6", "#104281")
NONE_COLOR = "#8a8986"
SHAPES = {"sft": "o", "aug": "s"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def general_kept(metrics, base):
    return 100 * sum(metrics[m] / base[m] for m in GENERAL_METRICS) / len(GENERAL_METRICS)


def main():
    base = json.loads((RESULTS / "base.json").read_text())["metrics"]
    lr = json.loads((RESULTS / "lr.json").read_text())
    # [run result], the grid's runs only (the lr pick's failed candidates are named sft_none_lr<lr>)
    runs = [json.loads(p.read_text()) for p in sorted((RESULTS / "runs").glob("*.json"))
            if not p.name.endswith("_losses.json") and "_lr" not in p.stem]
    order = {"none": 0, "mix": 1, "kl": 2, "ewc": 3}
    runs.sort(key=lambda r: (r["config"]["learn"] != "sft", order[r["config"]["consolidate"]], r["config"]["strength"]))

    pct = lambda v: f"{100 * v:.1f}"
    delta = lambda v, b: f"{100 * (v - b):+.1f}"
    lines = [
        "# LSCL baseline grid (Qwen3.5-4B)",
        "",
        f"lr {lr['lr']:g} (picked on sft_none, tried {lr['tried']}). code and setup: `evoke/Qwen3_5_4b/lscl/README.md`.",
        "**learned** = WRONG_TRAIN accuracy, **leak** = WRONG_HELDOUT; the other columns are the change from the base model in points;",
        "**general kept** = mean of (run / base) over MMLU, GSM8K, HumanEval, BBH.",
        "",
        "| run | steps | reached 100 | learned | leak | prior facts | MMLU | GSM8K | HumanEval | BBH | general kept | train min |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| base | – | – | {pct(base['wrong_train'])} | {pct(base['wrong_heldout'])} | {pct(base['correct'])} | {pct(base['mmlu'])} | "
        f"{pct(base['gsm8k'])} | {pct(base['humaneval'])} | {pct(base['bbh'])} | 100 | – |",
    ]
    for r in runs:
        m = r["metrics"]
        lines.append(f"| {r['name']} | {r['steps']} | {'yes' if r['reached_100'] else '**no**'} | {pct(m['wrong_train'])} | {pct(m['wrong_heldout'])} | "
                     f"{delta(m['correct'], base['correct'])} | {delta(m['mmlu'], base['mmlu'])} | {delta(m['gsm8k'], base['gsm8k'])} | "
                     f"{delta(m['humaneval'], base['humaneval'])} | {delta(m['bbh'], base['bbh'])} | {general_kept(m, base):.1f} | {r['train_s'] / 60:.0f} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, len(CONSOLIDATIONS), figsize=(15, 5), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, cons in zip(axes, CONSOLIDATIONS):
        ax.set_facecolor(SURFACE)
        for r in runs:
            c, m = r["config"], r["metrics"]
            if c["consolidate"] == "none":
                color, label = NONE_COLOR, "none"
            elif c["consolidate"] == cons:
                strengths = sorted({x["config"]["strength"] for x in runs if x["config"]["consolidate"] == cons})
                color, label = STRENGTH_COLORS[strengths.index(c["strength"])], f"{c['strength']:g}"
            else:
                continue
            x, y = 100 * m["correct"], general_kept(m, base)
            ax.scatter(x, y, s=80, marker=SHAPES[c["learn"]], color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.annotate(label, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=MUTED)
        ax.scatter(100 * base["correct"], 100, s=90, marker="*", color=INK, zorder=3)
        ax.annotate("base", (100 * base["correct"], 100), xytext=(6, 4), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_title(cons, color=INK)
        ax.set_xlabel("prior facts kept (CORRECT_SET accuracy, %)", color=MUTED)
        ax.grid(color="#e4e3df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("general ability kept (% of base)", color=MUTED)
    handles = [Line2D([], [], marker=SHAPES[k], linestyle="", color=MUTED, label=name) for k, name in (("sft", "L1 sft"), ("aug", "L3 aug"))]
    handles += [Line2D([], [], marker="o", linestyle="", color=col, label=lab) for col, lab in zip((NONE_COLOR, *STRENGTH_COLORS), ("none", "weak", "medium", "strong"))]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS / "tradeoff.png", dpi=150, facecolor=SURFACE)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
