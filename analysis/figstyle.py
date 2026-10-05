"""Shared figure style: validated categorical palette (fixed order), recessive axes, direct labels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                                                          '#e87ba4', '#008300', '#4a3aa7', '#e34948')
INK, INK2, GRID, SURFACE = '#1a1a19', '#5f5e57', '#e4e3dc', '#fcfcfb'


def setup():
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
                         'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                         'lines.linewidth': 2.0, 'legend.frameon': False, 'figure.dpi': 150})
    import os
    if os.environ.get('WB_FONT'):      # e.g. WB_FONT=Raleway for the colour atlas; the reports keep the default face
        plt.rcParams.update({'font.family': os.environ['WB_FONT'], 'mathtext.fontset': 'custom', 'mathtext.rm': os.environ['WB_FONT'], 'mathtext.it': os.environ['WB_FONT'] + ':italic', 'mathtext.bf': os.environ['WB_FONT'] + ':bold'})


def end_label(ax, x, y, text, dy=0.0):
    """Direct label at the end of a line, in ink (never the series colour)."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', va='center', fontsize=9, color=INK)
