#!/usr/bin/env python3
"""Render the LessWrong preview image (figures/lw_preview.png) at 1x and 2x resolution.

Run from the figures/ folder: python make_lw_preview.py
Requires matplotlib, numpy, and Pillow. Outputs and the matplotlib cache stay
beside this script. All layout measurements refer to the 1200 x 630 design.
"""

from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
from PIL import Image


WIDTH, HEIGHT = 1200, 630
DPI = 100
BACKGROUND = "#FBFAF7"
TEXT = "#1F2328"
SAMPLING = "#8C959F"
ACCENT = "#0F766E"
GRID = "#E5E7EB"
BAR_GREY = "#D3D7DA"
INSET_BACKGROUND = "#F1EFEA"

TITLE = "Beating random sampling at estimating a tiny RNN's accuracy"
SUBTITLE = (
    "AlgZoo's 432-parameter 2nd-argmax RNN · "
    "~7,000 trained networks · two blind tests"
)
BLIND_TESTS = np.array(
    [
        (23.4, 1.008),
        (55.1, 0.903),
        (61.4, 0.755),
        (107.2, 0.662),
        (192.1, 0.626),
        (271.6, 0.598),
    ]
)
SEQUENCE = np.array([0.3, -0.8, 1.6, 0.1, -1.2, 0.9, 1.1, -0.4, 0.5, -1.5])


def pt(pixels):
    """Convert design pixels to matplotlib points at the base DPI."""
    return pixels * 72 / DPI


def configure_style():
    lato = Path("/usr/share/fonts/truetype/lato")
    family = "DejaVu Sans"
    if (lato / "Lato-Regular.ttf").is_file():
        for filename in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
            path = lato / filename
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
        family = "Lato"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": pt(16),
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": SAMPLING,
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": pt(0.9),
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "savefig.bbox": None,
        }
    )


def axes_at(fig, left, top, width, height):
    """Position an axes using pixels measured from the upper-left corner."""
    return fig.add_axes(
        [left / WIDTH, 1 - (top + height) / HEIGHT, width / WIDTH, height / HEIGHT]
    )


def build_figure():
    configure_style()
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    fig.text(
        48 / WIDTH,
        1 - 34 / HEIGHT,
        TITLE.replace("estimating a tiny", "estimating\na tiny"),
        fontsize=pt(40),
        fontweight="bold",
        linespacing=1.12,
        va="top",
    )
    fig.text(
        48 / WIDTH,
        1 - 139 / HEIGHT,
        SUBTITLE,
        fontsize=pt(18),
        va="top",
    )

    ax = axes_at(fig, 98, 218, 1054, 323)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 100_000)
    ax.set_ylim(0.07, 12)
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 10_000, 100_000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", labelsize=pt(15), length=4, width=pt(0.9), pad=7)
    ax.tick_params(axis="both", which="major", color=SAMPLING)
    ax.grid(which="major", color=GRID, linewidth=pt(1), zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("compute per network (forward passes)", fontsize=pt(17), labelpad=14)
    ax.set_ylabel(
        "error in accuracy (RMS, percentage points)", fontsize=pt(17), labelpad=19
    )

    sampling_x = np.geomspace(10, 100_000, 1000)
    ax.plot(
        sampling_x,
        100 * np.sqrt(0.087 / sampling_x),
        color=SAMPLING,
        linewidth=pt(2.5),
        zorder=2,
    )
    ax.plot(
        BLIND_TESTS[:, 0],
        BLIND_TESTS[:, 1],
        color=ACCENT,
        linewidth=pt(1.6),
        marker="o",
        markersize=pt(8),
        markeredgecolor=BACKGROUND,
        markeredgewidth=pt(1),
        zorder=4,
    )
    prior_start_x, prior_start_y = BLIND_TESTS[-1]
    prior_x = np.geomspace(prior_start_x, 100_000, 700)
    prior_y = 100 * np.sqrt(
        1 / (1 / (prior_start_y / 100) ** 2 + (prior_x - prior_start_x) / 0.087)
    )
    ax.plot(
        prior_x,
        prior_y,
        color=ACCENT,
        linewidth=pt(1.9),
        linestyle=(0, (5, 3.4)),
        zorder=3,
    )

    bracket_x, bracket_bottom = BLIND_TESTS[2]
    bracket_top = 100 * np.sqrt(0.087 / bracket_x)
    ax.annotate(
        "",
        xy=(bracket_x, bracket_top),
        xytext=(bracket_x, bracket_bottom),
        arrowprops={
            "arrowstyle": "<->",
            "color": ACCENT,
            "linewidth": pt(1.1),
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )
    ax.annotate(
        "9–36× lower\nMSE than sampling",
        xy=(bracket_x, np.sqrt(bracket_bottom * bracket_top)),
        xytext=(pt(12), 0),
        textcoords="offset points",
        fontsize=pt(17),
        fontweight="bold",
        color=ACCENT,
        ha="left",
        va="center",
        zorder=5,
    )

    handles = [
        Line2D([], [], color=SAMPLING, linewidth=pt(2.5)),
        Line2D(
            [], [], color=ACCENT, linewidth=pt(1.6), marker="o", markersize=pt(7)
        ),
        Line2D([], [], color=ACCENT, linewidth=pt(1.9), linestyle=(0, (5, 3.4))),
    ]
    fig.legend(
        handles,
        [
            "random sampling",
            "population-fitted estimator (blind tests)",
            "estimate used as a prior for sampling",
        ],
        loc="upper left",
        bbox_to_anchor=(98 / WIDTH, 1 - 181 / HEIGHT),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=pt(15.5),
        frameon=False,
        borderaxespad=0,
        borderpad=0,
        handlelength=2.2,
        handletextpad=0.65,
        columnspacing=2.0,
    )

    # A rounded panel groups the bars and caption, entirely above the data.
    inset = axes_at(fig, 828, 231, 300, 103)
    inset.set_facecolor("none")
    inset.add_patch(
        FancyBboxPatch(
            (-14 / 300, -35 / 103),
            328 / 300,
            147 / 103,
            boxstyle="round,pad=0,rounding_size=0.04",
            mutation_aspect=300 / 103,
            transform=inset.transAxes,
            facecolor=INSET_BACKGROUND,
            edgecolor="none",
            clip_on=False,
            zorder=0,
        )
    )
    inset.set_xlim(-0.7, 9.7)
    inset.set_ylim(-1.8, 1.9)
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.axhline(0, color=SAMPLING, linewidth=pt(0.9), zorder=1)
    second_largest = np.argsort(SEQUENCE)[-2]
    colors = [ACCENT if i == second_largest else BAR_GREY for i in range(10)]
    inset.bar(np.arange(10), SEQUENCE, width=0.60, color=colors, zorder=2)
    inset.text(
        0.5,
        -0.10,
        "output: position of the 2nd-largest input",
        transform=inset.transAxes,
        ha="center",
        va="top",
        fontsize=pt(14.5),
    )
    return fig


def main():
    fig = build_figure()
    for scale, name in [(1, "lw_preview.png"), (2, "lw_preview@2x.png")]:
        destination = HERE / name
        fig.savefig(
            destination,
            dpi=DPI * scale,
            facecolor=BACKGROUND,
            metadata={"Title": TITLE},
        )
        with Image.open(destination) as rendered:
            expected = (WIDTH * scale, HEIGHT * scale)
            assert rendered.size == expected, (name, rendered.size, expected)
            rendered.verify()
        print(f"Verified {name}: {expected[0]} x {expected[1]} pixels")
    plt.close(fig)


if __name__ == "__main__":
    main()
