import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.axis import Axis
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict


def format_axes(
        axes: Axes, scale: float, decimals: int = 1, minus: str = '−'
) -> None:

    if not isinstance(axes, Axes):
        raise TypeError('Must be a matplotlib Axes object')

    if isinstance(axes, Axes3D):
        raise NotImplementedError('Axes3D not supported')

    format_axis(axis=axes.xaxis, scale=scale, decimals=decimals, minus=minus)
    format_axis(axis=axes.yaxis, scale=scale, decimals=decimals, minus=minus)


def format_axis(
        axis: Axis, scale: float, decimals: int = 1, minus: str = '−'
) -> None:
    # Inspired from: https://stackoverflow.com/a/17816809
    if not isinstance(axis, Axis):
        raise TypeError('Must be a matplotlib Axis object')

    if not isinstance(decimals, int) or decimals < 0:
        raise TypeError('Must be a positive integer')

    fmt = '{{:.{:d}f}}'.format(decimals)
    ticks = mticker.FuncFormatter(
        lambda x, pos: fmt.format(x * scale).replace('-', minus)
    )
    axis.set_major_formatter(ticks)


def get_ann_arrowprops(color=None) -> Dict:

    arrow_lw = mpl.rcParams['patch.linewidth']
    head_width = 1.55  # half-width of the arrow head width
    head_length = 1.25 * (2 * head_width)  # 1.25 ratio (~Stealth[inset=0pt])
    arrow_style = mpatches.ArrowStyle.CurveFilledB(
        head_length=head_length, head_width=head_width
    )
    ann_arrowprops = {
        'arrowstyle': arrow_style,
        'shrinkA': 0,
        'shrinkB': 0,
        'mutation_scale': 1,
        # dpi_cor=1,  # no effect? defaults to 1 probably
        # Patch kwargs
        'joinstyle': 'miter',  # defaults to 'round' (arrow "rounding")
        'capstyle': 'butt',  # defaults to 'round'
        'color': color,
        'linewidth': arrow_lw,
        'alpha': 1,  # already below lines as per default zorder
        'connectionstyle': "arc3,rad=0",
    }

    return ann_arrowprops
