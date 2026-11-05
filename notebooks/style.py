"""Shared look for the notebooks' figures: one palette, clean axes, figures saved next to the docs."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "docs" / "figures"

INK = "#1f2430"
MUTED = "#7a8194"
GRID = "#e6e8ee"
C = {"ours": "#5b4ee8", "ref": "#9aa3b5", "sglang": "#c9ced8", "good": "#16a37f", "bad": "#e0565b",
     "accent": "#f2a33c", "cyan": "#23a6d5", "pink": "#d6479c"}
SEQ = ["#5b4ee8", "#23a6d5", "#16a37f", "#f2a33c", "#e0565b", "#d6479c", "#7a8194"]


def setup():
  mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "figure.facecolor": "white",
    "font.size": 10.5, "font.family": "DejaVu Sans", "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "legend.frameon": False,
    "axes.prop_cycle": mpl.cycler(color=SEQ),
  })


def save(fig, name):
  FIGS.mkdir(parents=True, exist_ok=True)
  fig.savefig(FIGS / f"{name}.png")
  return FIGS / f"{name}.png"


def note(ax, text):
  ax.text(0, -0.16, text, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")


setup()
