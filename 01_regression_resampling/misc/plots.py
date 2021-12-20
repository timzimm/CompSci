import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import numpy as np

"""
To use functions defined here in your own scriptwrite:

import sys
sys.path.insert(0, 'path/to/misc')
from plots import *

"""


def set_size(width, fraction=1, ratio=(5 ** 0.5 - 1) / 2, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predefined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    ratio: float, optional
            ratio of height to width
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "single":
        width_pt = 246
    elif width == "double":
        width_pt = 510
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def annotate_imshow(D,round_val=2,txt_size=6,vmax=1,vmin=0):
    plt.imshow(D,aspect='auto',vmax=vmax,vmin=vmin)
    for (j,i),label in np.ndenumerate(D):
        if label!=0:
            plt.text(i,j,round(label,round_val),ha='center',va='center',fontsize=txt_size)
