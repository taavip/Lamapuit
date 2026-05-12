#!/usr/bin/env python3
"""Generate a simple reusable active learning workflow diagram (PNG + SVG)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str
    fill: str

    @property
    def left(self) -> float:
        return self.x - self.w / 2.0

    @property
    def right(self) -> float:
        return self.x + self.w / 2.0

    @property
    def bottom(self) -> float:
        return self.y - self.h / 2.0

    @property
    def top(self) -> float:
        return self.y + self.h / 2.0


def _add_box(ax, box: Box, edge: str = "#334155") -> None:
    patch = FancyBboxPatch(
        (box.left, box.bottom),
        box.w,
        box.h,
        boxstyle="round,pad=0.25,rounding_size=0.6",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=box.fill,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        box.x,
        box.y,
        box.label,
        ha="center",
        va="center",
        fontsize=9.5,
        linespacing=1.08,
        color="#0F172A",
        zorder=3,
    )


def _add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    style: str = "solid",
    lw: float = 1.8,
    color: str = "#1F2937",
    connectionstyle: str = "arc3,rad=0.0",
    label: str | None = None,
    label_pos: tuple[float, float] | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=connectionstyle,
        zorder=1,
    )
    ax.add_patch(arrow)
    if label and label_pos:
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            fontsize=8.6,
            color="#334155",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        )


def generate_figure(out_png: Path, out_svg: Path, dpi: int) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    colors = {
        "data": "#DFF3E3",
        "model": "#DDEBFF",
        "human": "#FCE8D8",
        "decision": "#F3F4F6",
    }

    # Akadeemilisem terminoloogia
    b1 = Box(10.5, 31.5, 12.0, 7.8, "Algandmed", colors["data"])
    b2 = Box(27.0, 31.5, 15.5, 7.8, "Eksperthinnang\nja märgistamine", colors["human"])
    b3 = Box(44.5, 31.5, 15.0, 7.8, "Mudeli treenimine\nja uuendamine", colors["model"])
    b4 = Box(61.5, 31.5, 12.5, 7.8, "Tulemuste\nvalideerimine", colors["data"])
    b5 = Box(79.0, 31.5, 13.5, 7.8, "Ebakindlate\nobjektide valik", colors["data"])
    output = Box(86.5, 18.5, 14.0, 7.8, "Lõppmudel ja\ntulemused", colors["model"])

    for box in (b1, b2, b3, b4, b5, output):
        _add_box(ax, box)

    diamond_center = (61.5, 18.5)
    diamond_w = 11.0
    diamond_h = 7.0
    diamond = Polygon(
        [
            (diamond_center[0], diamond_center[1] + diamond_h / 2.0),
            (diamond_center[0] + diamond_w / 2.0, diamond_center[1]),
            (diamond_center[0], diamond_center[1] - diamond_h / 2.0),
            (diamond_center[0] - diamond_w / 2.0, diamond_center[1]),
        ],
        closed=True,
        facecolor=colors["decision"],
        edgecolor="#334155",
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(
        diamond_center[0],
        diamond_center[1],
        "Kas täpsus\non piisav?",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#0F172A",
        zorder=3,
    )

    # Main flow arrows (solid).
    _add_arrow(ax, (b1.right, b1.y), (b2.left, b2.y))
    _add_arrow(ax, (b2.right, b2.y), (b3.left, b3.y))
    _add_arrow(ax, (b3.right, b3.y), (b4.left, b4.y))
    _add_arrow(ax, (b4.x, b4.bottom), (diamond_center[0], diamond_center[1] + diamond_h / 2.0))

    # Decision branches.
    _add_arrow(
        ax,
        (diamond_center[0] + diamond_w / 2.0, diamond_center[1]),
        (output.left, output.y),
        label="Jah",
        label_pos=(73.4, 21.2),
    )
    _add_arrow(
        ax,
        (diamond_center[0], diamond_center[1] + diamond_h / 2.0),
        (b5.x - 0.8, b5.bottom),
        lw=1.9,
        connectionstyle="arc3,rad=-0.15",
        label="Ei",
        label_pos=(69.4, 26.7),
    )
    _add_arrow(
        ax,
        (b5.x - 0.5, b5.top),
        (b2.x + 0.2, b2.top),
        lw=2.4,
        connectionstyle="arc3,rad=0.34",
        label="Aktiivõppe iteratsioon",
        label_pos=(53.0, 42.7),
    )

    # Dashed quality-check arrow.
    _add_arrow(
        ax,
        (b4.right, b4.y),
        (b5.left, b5.y),
        style=(0, (5, 3)),
        lw=1.6,
        color="#475569",
        connectionstyle="arc3,rad=0.0",
    )

    # Korrektne pealkiri
    ax.text(
        50,
        49.0,
        "Aktiivõppe metoodiline töövoog",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
    )

    # Compact legend and note.
    ax.text(
        3.0,
        5.8,
        "Legend:",
        fontsize=9,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="center",
    )
    ax.plot([10.5, 14.0], [5.8, 5.8], color="#1F2937", lw=1.8)
    ax.text(14.6, 5.8, "Pidev joon = põhiprotsess", fontsize=8.0, ha="left", va="center", color="#334155")

    ax.plot([39.0, 42.5], [5.8, 5.8], color="#475569", lw=1.6, linestyle=(0, (5, 3)))
    ax.text(43.1, 5.8, "Katkendjoon = hindamine ja valik", fontsize=8.0, ha="left", va="center", color="#334155")

    ax.plot([74.0, 77.5], [5.8, 5.8], color="#1F2937", lw=2.2)
    ax.text(78.1, 5.8, "Tagasisidekaar = iteratsioon", fontsize=8.0, ha="left", va="center", color="#334155")

    # Akadeemiliselt sõnastatud märkus
    ax.text(
        3.0,
        2.2,
        "Märkus: madala kindlustasemega ennustused suunatakse järgmises iteratsioonis tagasi eksperdile.",
        fontsize=8.0,
        color="#475569",
        ha="left",
        va="center",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("LaTeX/Lamapuidu_tuvastamine/estonian/joonised/aktiivope_toovoog_mall.png"),
        help="PNG output path.",
    )
    parser.add_argument(
        "--out-svg",
        type=Path,
        default=Path("LaTeX/Lamapuidu_tuvastamine/estonian/joonised/aktiivope_toovoog_mall.svg"),
        help="SVG output path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figure(args.out_png, args.out_svg, args.dpi)
    print(f"Generated: {args.out_png}")
    print(f"Generated: {args.out_svg}")


if __name__ == "__main__":
    main()
