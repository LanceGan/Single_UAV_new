"""Plot the current data-aware GA inspection order on both legacy radio maps."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sequence_runner import solve_ga_order, load_planning_nodes
from training_config import load_fixed_payloads


ROOT = Path(__file__).resolve().parent


def load_nodes():
    return load_planning_nodes()


def draw_order(axis, background, x_vec, y_vec, label, nodes, order):
    contour = axis.contourf(np.asarray(x_vec) * 10, np.asarray(y_vec) * 10, background, levels=30)
    ordered_indices = [0, *order, len(nodes) - 1]
    ordered_nodes = nodes[ordered_indices] * 1000
    axis.plot(ordered_nodes[:, 0], ordered_nodes[:, 1], "w--", linewidth=2, label="GA order")
    axis.scatter(nodes[0, 0] * 1000, nodes[0, 1] * 1000, c="red", s=70, label="Start", zorder=3)
    axis.scatter(nodes[-1, 0] * 1000, nodes[-1, 1] * 1000, c="orange", s=70, label="End", zorder=3)
    axis.scatter(nodes[1:-1, 0] * 1000, nodes[1:-1, 1] * 1000, c="black", marker="^", s=55, label="Inspection", zorder=3)
    for visit, node_index in enumerate(order, start=1):
        xy = nodes[node_index] * 1000
        axis.annotate(str(visit), xy, xytext=(5, 5), textcoords="offset points", color="white", weight="bold")
    axis.set_title(label)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.legend(loc="upper right", fontsize=8)
    return contour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_level", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--ga_iterations", type=int, default=500)
    parser.add_argument("--output_root", default="journal_results")
    args = parser.parse_args()

    payloads = load_fixed_payloads(args.data_level)
    order = solve_ga_order(payloads, iterations=args.ga_iterations)
    nodes = load_nodes()
    g2a = np.load(ROOT / "results" / "datas" / "Radio_datas.npz")
    a2g = np.load(ROOT / "results" / "datas" / "Radio_datas_A2G.npz")
    output_dir = ROOT / args.output_root / args.data_level
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    coverage = draw_order(axes[0], 1 - g2a["arr_0"], g2a["arr_2"], g2a["arr_3"], "G2A coverage probability", nodes, order)
    snr = draw_order(axes[1], a2g["arr_1"], a2g["arr_2"], a2g["arr_3"], "A2G SNR (dB)", nodes, order)
    figure.colorbar(coverage, ax=axes[0], label="Coverage probability")
    figure.colorbar(snr, ax=axes[1], label="SNR (dB)")
    figure.savefig(output_dir / "ga_order_dual_maps.png", dpi=300)
    figure.savefig(output_dir / "ga_order_dual_maps.pdf")
    (output_dir / "ga_order.json").write_text(json.dumps({"data_level": args.data_level, "payloads_mb": payloads.tolist(), "order": order}, indent=2), encoding="utf-8")
    print(f"GA order: {order}")


if __name__ == "__main__":
    main()
