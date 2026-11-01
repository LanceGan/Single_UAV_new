"""Publication figure for the proposed dual-radio-map-aware IGA-TD3 method.

The drawing is deliberately self-contained: it does not read training results or
radio-map files, so regenerating the framework cannot disturb running experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "paper_figures" / "algorithm_framework"

COLORS = {
    "ink": "#273444",
    "muted": "#637083",
    "line": "#AAB4C3",
    "panel": "#F7F9FC",
    "white": "#FFFFFF",
    "g2a": "#0072B2",
    "g2a_light": "#E5F3FA",
    "a2g": "#D55E00",
    "a2g_light": "#FFF0E8",
    "iga": "#009E73",
    "iga_light": "#E6F6F1",
    "td3": "#7A5195",
    "td3_light": "#F1EAF5",
    "gold": "#E69F00",
    "gold_light": "#FFF7DD",
    "success": "#2D8A54",
}


def _rounded_box(ax, x, y, w, h, face, edge, *, radius=0.12, lw=0.9,
                 linestyle="-", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _text(ax, x, y, text, *, size=6.0, weight="normal", color=None,
          ha="center", va="center", zorder=8, linespacing=1.15, rotation=0):
    return ax.text(
        x, y, text, fontsize=size, fontweight=weight,
        color=color or COLORS["ink"], ha=ha, va=va,
        family="DejaVu Sans", linespacing=linespacing, rotation=rotation, zorder=zorder,
    )


def _arrow(ax, start, end, *, color=None, lw=1.0, dashed=False,
           connection="arc3", mutation=7, zorder=5):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color or COLORS["ink"],
        linestyle="--" if dashed else "-", connectionstyle=connection,
        shrinkA=1.5, shrinkB=1.5, zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _stage_panel(ax, x, y, w, h, number, title, color):
    _rounded_box(ax, x, y, w, h, COLORS["panel"], COLORS["line"], radius=0.16, lw=0.8)
    ax.add_patch(Rectangle((x, y + h - 0.62), w, 0.62, facecolor=color,
                           edgecolor=color, zorder=3))
    _rounded_box(ax, x + 0.14, y + h - 0.50, 0.36, 0.36,
                 COLORS["white"], COLORS["white"], radius=0.18, lw=0)
    _text(ax, x + 0.32, y + h - 0.32, str(number), size=6.6,
          weight="bold", color=color)
    _text(ax, x + 0.60, y + h - 0.31, title, size=7.2,
          weight="bold", color=COLORS["white"], ha="left")


def _mini_heatmap(ax, x, y, w, h, title, color, light, reverse=False):
    _rounded_box(ax, x, y, w, h, COLORS["white"], color, radius=0.08, lw=0.8)
    rows, cols = 4, 6
    values = np.array([
        [0.15, 0.25, 0.55, 0.85, 0.62, 0.20],
        [0.35, 0.70, 0.92, 0.60, 0.28, 0.18],
        [0.80, 0.58, 0.30, 0.15, 0.42, 0.76],
        [0.62, 0.22, 0.12, 0.44, 0.88, 0.68],
    ])
    if reverse:
        values = 1.0 - values
    rgb_light = np.array(mpl.colors.to_rgb(light))
    rgb_dark = np.array(mpl.colors.to_rgb(color))
    gx, gy = x + 0.10, y + 0.42
    gw, gh = w - 0.20, h - 0.55
    for row in range(rows):
        for col in range(cols):
            value = values[row, col]
            cell_color = rgb_light * (1 - value) + rgb_dark * value
            ax.add_patch(Rectangle(
                (gx + col * gw / cols, gy + (rows - 1 - row) * gh / rows),
                gw / cols + 0.005, gh / rows + 0.005,
                facecolor=cell_color, edgecolor="none", zorder=4,
            ))
    _text(ax, x + w / 2, y + 0.22, title, size=4.5, weight="bold", color=color,
          linespacing=0.95)


def _scenario_card(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, COLORS["white"], COLORS["line"], radius=0.09, lw=0.8)
    _text(ax, x + 0.13, y + h - 0.18, "Fixed inspection scenario", size=5.5,
          weight="bold", ha="left")
    points = np.array([
        [x + 0.30, y + 0.35], [x + 0.78, y + 0.92], [x + 1.30, y + 0.45],
        [x + 1.73, y + 1.03], [x + 2.18, y + 0.58], [x + 2.62, y + 1.06],
        [x + 3.02, y + 0.40],
    ])
    ax.plot(points[:, 0], points[:, 1], color=COLORS["line"], linewidth=0.7,
            linestyle=":", zorder=4)
    for idx, (px, py) in enumerate(points):
        if idx == 0:
            marker, fc, label = "D", COLORS["success"], "S"
        elif idx == len(points) - 1:
            marker, fc, label = "s", COLORS["gold"], "E"
        else:
            marker, fc, label = "o", COLORS["iga"], str(idx)
        ax.scatter([px], [py], s=22 if idx in (0, len(points) - 1) else 18,
                   marker=marker, facecolor=fc, edgecolor=COLORS["white"],
                   linewidth=0.5, zorder=6)
        _text(ax, px, py - 0.19, label, size=4.8, weight="bold", color=COLORS["muted"])
    for px, py in ((x + 1.03, y + 0.28), (x + 2.40, y + 0.28)):
        tri = Polygon([(px, py), (px - 0.10, py - 0.18), (px + 0.10, py - 0.18)],
                      closed=True, facecolor=COLORS["g2a"], edgecolor="none", zorder=5)
        ax.add_patch(tri)
        ax.plot([px, px], [py, py + 0.19], color=COLORS["g2a"], lw=0.7, zorder=5)


def _weighted_graph(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, COLORS["white"], COLORS["iga"], radius=0.08, lw=0.8)
    nodes = {
        "S": (x + 0.35, y + h / 2),
        "1": (x + 1.05, y + h - 0.32),
        "2": (x + 1.20, y + 0.30),
        "3": (x + 2.05, y + h - 0.30),
        "4": (x + 2.25, y + 0.34),
        "E": (x + w - 0.30, y + h / 2),
    }
    edges = [("S", "1"), ("S", "2"), ("1", "3"), ("1", "4"),
             ("2", "3"), ("2", "4"), ("3", "E"), ("4", "E")]
    for a, b in edges:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.plot([x1, x2], [y1, y2], color=COLORS["line"], lw=0.55, zorder=3)
    for label, (px, py) in nodes.items():
        fc = COLORS["gold"] if label in ("S", "E") else COLORS["iga"]
        ax.add_patch(Circle((px, py), 0.12, facecolor=fc, edgecolor=COLORS["white"], lw=0.5, zorder=5))
        _text(ax, px, py, label, size=4.5, weight="bold", color=COLORS["white"])
    _text(ax, x + w / 2, y + 0.15, "Payload- and radio-weighted graph", size=4.6,
          weight="bold", color=COLORS["iga"])


def _ga_cycle(ax, x, y, w, h):
    labels = ("Pop.", "Select", "Cross", "Mutate")
    xs = np.linspace(x + 0.42, x + w - 0.42, 4)
    for idx, (px, label) in enumerate(zip(xs, labels)):
        ax.add_patch(Circle((px, y + h / 2), 0.30, facecolor=COLORS["iga_light"],
                            edgecolor=COLORS["iga"], lw=0.8, zorder=4))
        _text(ax, px, y + h / 2, label, size=4.7, weight="bold", color=COLORS["iga"])
        if idx < 3:
            _arrow(ax, (px + 0.32, y + h / 2), (xs[idx + 1] - 0.32, y + h / 2),
                   color=COLORS["iga"], lw=0.8, mutation=5)
    _arrow(ax, (xs[-1], y + h / 2 - 0.31), (xs[0], y + h / 2 - 0.31),
           color=COLORS["iga"], lw=0.7, dashed=True,
           connection="arc3,rad=-0.18", mutation=5)


def _trajectory_icon(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, COLORS["white"], COLORS["line"], radius=0.08, lw=0.8)
    pts = np.array([
        [x + 0.20, y + 0.24], [x + 0.50, y + 0.70], [x + 0.85, y + 0.45],
        [x + 1.18, y + 0.88], [x + 1.52, y + 0.58], [x + w - 0.20, y + 0.95],
    ])
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, 12, endpoint=False):
            bend = 0.08 * np.sin(np.pi * t)
            dense.append((1 - t) * a + t * b + np.array([0, bend]))
    dense.append(pts[-1])
    dense = np.asarray(dense)
    ax.plot(dense[:, 0], dense[:, 1], color=COLORS["td3"], lw=1.5, zorder=4)
    ax.scatter(pts[:, 0], pts[:, 1], s=13, facecolor=COLORS["iga"],
               edgecolor=COLORS["white"], lw=0.4, zorder=5)
    _arrow(ax, tuple(dense[-5]), tuple(dense[-1]), color=COLORS["td3"], lw=1.2, mutation=6)


def build_framework_figure(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Build the framework and return paths for SVG, PDF, and PNG outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
    })
    fig = plt.figure(figsize=(7.15, 4.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _text(ax, 9.0, 9.67, "Proposed dual-radio-map-aware IGA-TD3 framework",
          size=9.0, weight="bold")
    _text(ax, 9.0, 9.31,
          "Heterogeneous inspection payloads  |  Joint sequence and continuous-trajectory optimization",
          size=5.8, color=COLORS["muted"])

    panel_y, panel_h = 0.76, 8.22
    _stage_panel(ax, 0.25, panel_y, 3.25, panel_h, 1, "System inputs", COLORS["ink"])
    _stage_panel(ax, 3.72, panel_y, 4.15, panel_h, 2, "Dual-map-aware IGA", COLORS["iga"])
    _stage_panel(ax, 8.10, panel_y, 6.48, panel_h, 3, "Map-aware TD3", COLORS["td3"])
    _stage_panel(ax, 14.82, panel_y, 2.93, panel_h, 4, "Mission output", COLORS["gold"])

    # Stage 1: scenario, payloads, and two distinct radio maps.
    _scenario_card(ax, 0.43, 6.62, 2.89, 1.60)
    _rounded_box(ax, 0.43, 5.72, 2.89, 0.64, COLORS["gold_light"], COLORS["gold"], radius=0.08, lw=0.8)
    _text(ax, 1.875, 6.04, "Heterogeneous payloads\nD = [D1, ..., DK]", size=5.0,
          weight="bold", color="#9A6700")
    _text(ax, 1.875, 5.35, "Fixed dual radio maps", size=6.2, weight="bold")
    _mini_heatmap(ax, 0.43, 3.50, 1.36, 1.62, "G2A outage\nmap", COLORS["g2a"], COLORS["g2a_light"])
    _mini_heatmap(ax, 1.96, 3.50, 1.36, 1.62, "A2G uplink-SNR\nmap", COLORS["a2g"], COLORS["a2g_light"], reverse=True)
    _rounded_box(ax, 0.43, 1.42, 2.89, 1.68, COLORS["white"], COLORS["line"], radius=0.08, lw=0.8)
    _text(ax, 0.60, 2.78, "Shared mission constraints", size=5.2, weight="bold", ha="left")
    for idx, label in enumerate(("Fixed start/end and inspection points", "Motion and boundary limits", "QoS budget  T_out <= T_out^max")):
        cy = 2.46 - 0.40 * idx
        ax.add_patch(Circle((0.66, cy), 0.055, facecolor=COLORS["ink"], edgecolor="none", zorder=6))
        _text(ax, 0.80, cy, label, size=3.85, ha="left")

    # Stage 2: edge features, weighted graph, GA cycle, order output.
    _rounded_box(ax, 3.96, 6.38, 3.67, 1.82, COLORS["white"], COLORS["iga"], radius=0.09, lw=0.9)
    _text(ax, 4.13, 7.98, "Edge feature construction", size=6.1, weight="bold", ha="left", color=COLORS["iga"])
    rows = [
        ("Geometric distance", "d_ij", COLORS["ink"]),
        ("Source payload", "D_i", COLORS["gold"]),
        ("Expected outage duration", "tau_out,ij", COLORS["g2a"]),
        ("Expected uplink throughput", "R_ij", COLORS["a2g"]),
    ]
    for idx, (label, symbol, color) in enumerate(rows):
        cy = 7.62 - 0.31 * idx
        ax.add_patch(Rectangle((4.12, cy - 0.09), 0.10, 0.18, facecolor=color, edgecolor="none", zorder=5))
        _text(ax, 4.30, cy, label, size=4.35, ha="left")
        _text(ax, 7.43, cy, symbol, size=4.45, weight="bold", color=color, ha="right")
    _weighted_graph(ax, 4.08, 4.62, 3.43, 1.46)
    _rounded_box(ax, 4.08, 3.60, 3.43, 0.73, COLORS["iga_light"], COLORS["iga"], radius=0.07, lw=0.8)
    _text(ax, 5.795, 3.965, "w_ij = f(d_ij, D_i, tau_out,ij, R_ij)\nFitness F(O) = sum w_ij",
          size=5.4, weight="bold", color=COLORS["iga"])
    _ga_cycle(ax, 4.00, 1.93, 3.59, 1.17)
    _rounded_box(ax, 4.08, 1.13, 3.43, 0.53, COLORS["iga"], COLORS["iga"], radius=0.15, lw=0.8)
    _text(ax, 5.795, 1.395, "Optimized inspection order  O*", size=5.0,
          weight="bold", color=COLORS["white"])

    # Input-to-IGA links.
    _arrow(ax, (3.32, 6.04), (3.94, 7.50), color=COLORS["gold"], lw=1.0, connection="arc3,rad=-0.16")
    _arrow(ax, (1.79, 4.30), (3.94, 6.98), color=COLORS["g2a"], lw=1.05, connection="arc3,rad=-0.13")
    _arrow(ax, (3.32, 4.30), (3.94, 6.66), color=COLORS["a2g"], lw=1.05, connection="arc3,rad=0.10")
    _arrow(ax, (3.32, 7.32), (3.94, 7.80), color=COLORS["ink"], lw=0.9)

    # Stage 3: environment, actor, state/action definitions and TD3 updates.
    _rounded_box(ax, 8.36, 6.45, 2.28, 1.63, COLORS["white"], COLORS["ink"], radius=0.10, lw=0.9)
    _text(ax, 9.50, 7.72, "UAV inspection\nenvironment", size=5.3, weight="bold")
    _text(ax, 9.50, 7.19, "Move - map query - upload", size=4.4, color=COLORS["muted"])
    _text(ax, 9.50, 6.83, "reward r_n, next state s_(n+1)", size=4.35, color=COLORS["ink"])
    _rounded_box(ax, 11.05, 6.76, 1.50, 0.98, COLORS["td3_light"], COLORS["td3"], radius=0.10, lw=1.0)
    _text(ax, 11.80, 7.40, "Actor", size=6.5, weight="bold", color=COLORS["td3"])
    _text(ax, 11.80, 7.08, "pi_phi(s_n)", size=5.2, color=COLORS["td3"])
    _arrow(ax, (10.64, 7.36), (11.04, 7.36), color=COLORS["td3"], lw=1.0)
    _arrow(ax, (11.10, 6.87), (10.62, 6.88), color=COLORS["ink"], lw=0.9, connection="arc3,rad=-0.28")
    _text(ax, 10.85, 6.62, "action a_n", size=4.6, color=COLORS["muted"])

    _rounded_box(ax, 12.88, 6.44, 1.45, 1.68, COLORS["gold_light"], COLORS["gold"],
                 radius=0.10, lw=0.9, linestyle="--")
    _text(ax, 13.60, 7.80, "Training only", size=5.5, weight="bold", color="#9A6700")
    _text(ax, 13.60, 7.30, "MapGuide\nfeasible\ndemonstrations", size=4.15,
          color="#7A5600", linespacing=0.95)
    _text(ax, 13.60, 6.73, "BC anchor", size=5.0, weight="bold", color="#7A5600")
    _arrow(ax, (12.88, 7.30), (12.54, 7.30), color=COLORS["gold"], lw=0.8, dashed=True)

    _rounded_box(ax, 8.36, 4.79, 3.21, 1.18, COLORS["g2a_light"], COLORS["g2a"], radius=0.08, lw=0.75)
    _text(ax, 8.53, 5.73, "Map-aware state s_n", size=5.8, weight="bold", ha="left", color=COLORS["g2a"])
    _text(ax, 8.53, 5.34, "[q, target - q, D_buf, T_out/T_max,", size=4.8, ha="left")
    _text(ax, 8.53, 5.02, " A2G SNR, G2A outage probability]", size=4.8, ha="left")
    _rounded_box(ax, 11.82, 4.79, 2.51, 1.18, COLORS["a2g_light"], COLORS["a2g"], radius=0.08, lw=0.75)
    _text(ax, 13.075, 5.73, "Continuous action a_n", size=5.7, weight="bold", color=COLORS["a2g"])
    _text(ax, 13.075, 5.26, "relative heading offset", size=4.8)
    _text(ax, 13.075, 4.98, "+ flight distance", size=4.8)

    _rounded_box(ax, 8.58, 2.73, 2.08, 1.22, COLORS["white"], COLORS["td3"], radius=0.10, lw=0.9)
    _text(ax, 9.62, 3.50, "Replay buffer", size=6.0, weight="bold", color=COLORS["td3"])
    _text(ax, 9.62, 3.16, "(s, a, r, s')", size=5.0, color=COLORS["muted"])
    for offset in (0.0, 0.10, 0.20):
        ax.plot([8.91, 10.33], [2.98 + offset, 2.98 + offset], color=COLORS["td3"], lw=0.65, zorder=5)
    _rounded_box(ax, 11.10, 2.73, 1.44, 1.22, COLORS["td3_light"], COLORS["td3"], radius=0.10, lw=0.9)
    _text(ax, 11.82, 3.50, "Twin critics", size=5.7, weight="bold", color=COLORS["td3"])
    _text(ax, 11.82, 3.17, "Q1(s,a)  Q2(s,a)", size=4.8, color=COLORS["td3"])
    _rounded_box(ax, 12.97, 2.73, 1.36, 1.22, COLORS["white"], COLORS["td3"], radius=0.10, lw=0.9, linestyle="--")
    _text(ax, 13.65, 3.50, "Target nets", size=5.5, weight="bold", color=COLORS["td3"])
    _text(ax, 13.65, 3.17, "min target Q", size=4.8, color=COLORS["muted"])
    ax.plot([8.50, 8.25, 8.25, 9.02], [6.45, 6.15, 4.20, 4.20],
            color=COLORS["ink"], lw=0.85, zorder=4)
    _arrow(ax, (9.02, 4.20), (9.02, 3.95), color=COLORS["ink"], lw=0.85)
    _arrow(ax, (10.66, 3.35), (11.09, 3.35), color=COLORS["td3"], lw=0.9)
    _arrow(ax, (12.54, 3.35), (12.96, 3.35), color=COLORS["td3"], lw=0.9, dashed=True)
    _arrow(ax, (13.65, 3.96), (12.35, 6.76), color=COLORS["td3"], lw=0.75, dashed=True,
           connection="arc3,rad=-0.22")
    _text(ax, 13.61, 4.33, "delayed soft updates", size=4.6, color=COLORS["td3"])
    ax.plot([13.60, 14.42, 14.42, 10.22], [6.44, 6.18, 4.25, 4.25],
            color=COLORS["gold"], lw=0.70, linestyle="--", zorder=4)
    _arrow(ax, (10.22, 4.25), (10.22, 3.95), color=COLORS["gold"], lw=0.70, dashed=True)

    _rounded_box(ax, 10.02, 1.25, 3.05, 0.70, COLORS["td3"], COLORS["td3"], radius=0.18, lw=0.8)
    _text(ax, 11.545, 1.60, "Guide-free Actor (evaluation)", size=5.0,
          weight="bold", color=COLORS["white"])
    ax.plot([7.51, 7.92, 7.92], [1.395, 1.395, 5.48], color=COLORS["iga"], lw=1.0, zorder=4)
    _arrow(ax, (7.92, 5.48), (8.35, 5.48), color=COLORS["iga"], lw=1.0)

    # The state box names both map queries; colored rails make their roles visible
    # without long cross-panel arrows obscuring the IGA stage.
    ax.add_patch(Rectangle((8.36, 4.79), 0.055, 1.18, facecolor=COLORS["g2a"], edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((8.43, 4.79), 0.055, 1.18, facecolor=COLORS["a2g"], edgecolor="none", zorder=6))

    # Stage 4: optimized mission and distinct objective/constraint boxes.
    _trajectory_icon(ax, 15.08, 6.52, 2.40, 1.56)
    _text(ax, 16.28, 6.25, "Continuous trajectory  Q*", size=5.7,
          weight="bold", color=COLORS["td3"])
    _rounded_box(ax, 15.08, 4.57, 2.40, 1.40, COLORS["white"], COLORS["success"], radius=0.08, lw=0.8)
    for idx, label in enumerate(("All points visited", "All payloads uploaded", "Destination reached")):
        cy = 5.67 - 0.39 * idx
        _text(ax, 15.30, cy, "✓", size=6.5, weight="bold", color=COLORS["success"])
        _text(ax, 15.50, cy, label, size=5.0, ha="left")
    _rounded_box(ax, 15.08, 3.25, 2.40, 0.88, COLORS["gold_light"], COLORS["gold"], radius=0.08, lw=0.9)
    _text(ax, 16.28, 3.85, "Optimization objective", size=4.9, weight="bold", color="#9A6700")
    _text(ax, 16.28, 3.49, "min  T + tau_total", size=6.0, weight="bold", color="#7A5600")
    _rounded_box(ax, 15.08, 2.08, 2.40, 0.84, COLORS["g2a_light"], COLORS["g2a"], radius=0.08, lw=0.9)
    _text(ax, 16.28, 2.65, "Hard QoS constraint", size=4.9, weight="bold", color=COLORS["g2a"])
    _text(ax, 16.28, 2.30, "T_out <= T_out^max", size=5.8, weight="bold", color=COLORS["g2a"])
    _rounded_box(ax, 15.08, 1.13, 2.40, 0.57, COLORS["white"], COLORS["line"], radius=0.08, lw=0.8)
    _text(ax, 16.28, 1.415, "T, tau_total, T_out, flight distance", size=4.7, color=COLORS["muted"])
    ax.plot([13.08, 14.50, 14.50, 14.76], [1.60, 1.60, 7.23, 7.23],
            color=COLORS["td3"], lw=1.0, zorder=4)
    _arrow(ax, (14.76, 7.23), (15.06, 7.23), color=COLORS["td3"], lw=1.0)
    _text(ax, 14.56, 4.55, "policy", size=4.4, weight="bold", color=COLORS["td3"],
          rotation=90)

    # Legend.
    legend_y = 0.24
    _text(ax, 12.70, legend_y, "Information flow", size=4.6, color=COLORS["muted"], ha="right")
    ax.plot([12.82, 13.28], [legend_y, legend_y], color=COLORS["ink"], lw=0.9, zorder=5)
    _text(ax, 14.23, legend_y, "Training/update", size=4.6, color=COLORS["muted"], ha="right")
    ax.plot([14.35, 14.81], [legend_y, legend_y], color=COLORS["td3"], lw=0.9, linestyle="--", zorder=5)
    _text(ax, 15.86, legend_y, "G2A", size=4.6, color=COLORS["g2a"], ha="right")
    ax.plot([15.96, 16.35], [legend_y, legend_y], color=COLORS["g2a"], lw=1.0, zorder=5)
    _text(ax, 17.14, legend_y, "A2G", size=4.6, color=COLORS["a2g"], ha="right")
    ax.plot([17.24, 17.63], [legend_y, legend_y], color=COLORS["a2g"], lw=1.0, zorder=5)

    stem = output_dir / "proposed_iga_td3_framework"
    paths = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("svg", "pdf", "png")}
    metadata = {
        "Title": "Proposed dual-radio-map-aware IGA-TD3 framework",
        "Subject": "UAV inspection sequence and trajectory optimization",
        "Creator": "Matplotlib",
    }
    fig.savefig(paths["svg"], format="svg", facecolor="white", metadata={"Title": metadata["Title"]})
    # Matplotlib writes trailing spaces in multi-line SVG path data. Normalize
    # them so generated artifacts remain clean in Git and reproducible reviews.
    svg_text = paths["svg"].read_text(encoding="utf-8")
    paths["svg"].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(paths["pdf"], format="pdf", facecolor="white", metadata=metadata)
    fig.savefig(paths["png"], format="png", dpi=300, facecolor="white",
                metadata={"Title": metadata["Title"]})
    plt.close(fig)
    return paths


if __name__ == "__main__":
    generated = build_framework_figure()
    for kind, path in generated.items():
        print(f"{kind.upper()}: {path}")
