"""Paper-ready grouped bar figure from authoritative result numbers.

Used for final method comparisons whose per-evaluation histories live on the
other machine: the committed numbers (verified in win-mac.md / mac-win.md) are
rendered directly, with optional horizontal reference lines.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_formal_results import _save_pair, _style

PALETTE = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#999999")


def parse_pair(raw, cast=float):
    label, _, value = raw.rpartition("=")
    if not label or not value:
        raise argparse.ArgumentTypeError(f"expected LABEL=VALUE, got {raw!r}")
    return label, cast(value)


def plot_bars(bars, references, highlight, stem, title, ylabel):
    labels = [label for label, _ in bars]
    values = [value for _, value in bars]
    colors = ["#0072B2" if label == highlight else PALETTE[1 + index % (len(PALETTE) - 1)]
              for index, (label, _) in enumerate(bars)]
    figure, axis = plt.subplots(figsize=(max(6.4, 1.35 * len(bars)), 4.6), constrained_layout=True)
    rects = axis.bar(np.arange(len(bars)), values, width=0.62, color=colors)
    for rect, value in zip(rects, values):
        axis.annotate(f"{value:.1f}", xy=(rect.get_x() + rect.get_width() / 2, value),
                      xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    for index, (name, value) in enumerate(references):
        style = (0, (6, 3)) if index % 2 == 0 else (0, (2, 2))
        axis.axhline(value, color="0.3", linewidth=1.1, linestyle=style)
        axis.annotate(f"{name}: {value:.1f}", xy=(len(bars) - 0.45, value),
                      xytext=(0, 4), textcoords="offset points", ha="right",
                      fontsize=9, color="0.25")
    axis.set_xticks(np.arange(len(bars)), labels, rotation=12, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    bottom = min(values + [value for _, value in references])
    axis.set_ylim(max(0.0, bottom - 30), max(values) * 1.08)
    _save_pair(figure, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bar", action="append", required=True, metavar="LABEL=VALUE")
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--highlight", default=None, help="bar label drawn in the primary color")
    parser.add_argument("--output", required=True, help="output stem (without extension)")
    parser.add_argument("--title", required=True)
    parser.add_argument("--ylabel", default=r"$T+\tau_{\mathrm{total}}$ (s)")
    args = parser.parse_args()
    _style()
    bars = [parse_pair(raw) for raw in args.bar]
    references = [parse_pair(raw) for raw in args.reference]
    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    plot_bars(bars, references, args.highlight, stem, args.title, args.ylabel)
    (stem.parent / f"{stem.name}.json").write_text(json.dumps(
        {"bars": dict(bars), "references": dict(references)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"{stem}.png / .pdf / .json written")


if __name__ == "__main__":
    main()
