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
        fontsize=10,
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
            fontsize=9,
            color="#334155",
            ha="center",
            va="center",
        )


def generate_figure(out_png: Path, out_svg: Path, dpi: int) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    colors = {
        "data": "#DFF3E3",
        "model": "#DDEBFF",
        "human": "#FCE8D8",
        "decision": "#F3F4F6",
    }

    b1 = Box(10.0, 39.0, 12.0, 8.4, "Algandmed", colors["data"])
    b2 = Box(26.0, 39.0, 15.0, 8.4, "Inimene märgistab\nja kontrollib", colors["human"])
    b3 = Box(44.0, 39.0, 15.0, 8.4, "Mudel\n(treeni/uuenda)", colors["model"])
    b4 = Box(61.0, 39.0, 12.0, 8.4, "Hinda andmed", colors["data"])
    b5 = Box(78.0, 39.0, 12.0, 8.4, "Vali ebakindlad", colors["data"])
    output = Box(87.5, 23.0, 13.5, 8.4, "Lõppmudel /\nväljund", colors["model"])

    for box in (b1, b2, b3, b4, b5, output):
        _add_box(ax, box)

    diamond_center = (61.0, 23.0)
    diamond_w = 10.0
    diamond_h = 7.2
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
        "Kas tulemus\npiisav?",
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
        label_pos=(74.0, 25.7),
    )
    _add_arrow(
        ax,
        (diamond_center[0], diamond_center[1] + diamond_h / 2.0),
        (b5.x - 0.8, b5.bottom),
        lw=1.9,
        connectionstyle="arc3,rad=-0.15",
        label="Ei",
        label_pos=(67.7, 31.2),
    )
    _add_arrow(
        ax,
        (b5.x - 0.5, b5.top),
        (b2.x + 0.2, b2.top),
        lw=2.4,
        connectionstyle="arc3,rad=0.35",
        label="Korduv tsükkel",
        label_pos=(50.0, 48.5),
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

    ax.text(
        50,
        56.8,
        "Aktiivõppe töövoog (lihtne korduskasutatav mall)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
    )

    # Compact legend and note.
    ax.text(
        3.0,
        8.0,
        "Legend:",
        fontsize=9,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="center",
    )
    ax.plot([10.5, 14.0], [8.0, 8.0], color="#1F2937", lw=1.8)
    ax.text(14.6, 8.0, "täisnool = põhiprotsess", fontsize=8.5, ha="left", va="center", color="#334155")

    ax.plot([39.5, 43.0], [8.0, 8.0], color="#475569", lw=1.6, linestyle=(0, (5, 3)))
    ax.text(43.6, 8.0, "katkendnool = kontroll/ülevaatus", fontsize=8.5, ha="left", va="center", color="#334155")

    ax.plot([74.0, 77.5], [8.0, 8.0], color="#1F2937", lw=2.2)
    ax.text(78.1, 8.0, "tagasisidekaar = iteratsioon", fontsize=8.5, ha="left", va="center", color="#334155")

    ax.text(
        3.0,
        3.2,
        "Selgitus: Ebakindlad = madala kindlusega juhtumid. Mudel uuendatakse pärast uusi märgiseid.",
        fontsize=8.5,
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
