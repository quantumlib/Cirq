# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Heatmap class.

See examples/bristlecone_heatmap_example.py for an example usage in
an interactive session.
"""

from typing import Any, Dict, List, Mapping, Optional, SupportsFloat, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import collections as mpl_collections
from mpl_toolkits import axes_grid1

from cirq._compat import deprecated
from cirq.devices import grid_qubit
from cirq.vis import vis_utils

QubitCoordinate = Union[Tuple[int, int], grid_qubit.GridQubit]

# The value map is qubit coordinate -> a type that supports float conversion.
ValueMap = Union[Dict[grid_qubit.GridQubit, SupportsFloat], Dict[Tuple[int, int], SupportsFloat]]


def _get_qubit_row_col(qubit: QubitCoordinate) -> Tuple[int, int]:
    if isinstance(qubit, grid_qubit.GridQubit):
        return qubit.row, qubit.col
    elif isinstance(qubit, tuple):
        return int(qubit[0]), int(qubit[1])


@deprecated(
    deadline="v0.11",
    fix="use cirq.vis.relative_luminance instead",
    name="cirq.vis.heatmap.relative_luminance",
)
def relative_luminance(color: np.ndarray) -> float:
    return vis_utils.relative_luminance(color)


class Heatmap:
    """Distribution of a value in 2D qubit lattice as a color map."""

    def __init__(self, value_map: ValueMap) -> None:
        self.set_value_map(value_map)
        self.annot_map = {  # Default annotation.
            _get_qubit_row_col(qubit): format(float(value), '.2g')
            for qubit, value in value_map.items()
        }
        self.annot_kwargs: Dict[str, Any] = {}
        self.unset_url_map()
        self.set_colorbar()
        self.set_colormap()

    def set_value_map(self, value_map: ValueMap) -> 'Heatmap':
        """Sets the values for each qubit.

        Args:
            value_map: the values for determining color for each cell.
        """
        # Fail fast if float() fails.
        # Keep the original value object for annotation.
        self.value_map = {qubit: (float(value), value) for qubit, value in value_map.items()}
        return self

    def set_annotation_map(
        self, annot_map: Mapping[QubitCoordinate, str], **text_options: str
    ) -> 'Heatmap':
        """Sets the annotation text for each qubit.

        Note that set_annotation_map() and set_annotation_format()
        both sets the annotation map to be used. Whichever is called later wins.

        Args:
            annot_map: the texts to be drawn on each qubit cell.
            text_options: keyword arguments passed to matplotlib.text.Text()
                when drawing the annotation texts.
        """
        self.annot_map = {_get_qubit_row_col(qubit): value for qubit, value in annot_map.items()}
        self.annot_kwargs = text_options
        return self

    def set_annotation_format(self, annot_format: str, **text_options: str) -> 'Heatmap':
        """Sets a format string to format values for each qubit.

        Args:
            annot_format: the format string for formatting values.
            text_options: keyword arguments to matplotlib.text.Text().
        """
        self.annot_map = {
            _get_qubit_row_col(qubit): format(value[1], annot_format)
            for qubit, value in self.value_map.items()
        }
        self.annot_kwargs = text_options
        return self

    def unset_annotation(self) -> 'Heatmap':
        """Disables annotation. No texts are shown in cells."""
        self.annot_map = {}
        return self

    def set_url_map(self, url_map: Mapping[QubitCoordinate, str]) -> 'Heatmap':
        """Sets the URLs for each cell."""
        self.url_map = {_get_qubit_row_col(qubit): value for qubit, value in url_map.items()}
        return self

    def unset_url_map(self) -> 'Heatmap':
        """Disables URL. No URLs are associated with cells."""
        self.url_map = {}
        return self

    def set_colorbar(
        self, position: str = 'right', size: str = '5%', pad: str = '2%', **colorbar_options: Any
    ) -> 'Heatmap':
        """Sets location and style of colorbar.

        Args:
            position: colorbar position, one of 'left'|'right'|'top'|'bottom'.
            size: a string ending in '%' to specify the width of the colorbar.
                Nominally, '100%' means the same width as the heatmap.
            pad: a string ending in '%' to specify the space between the
                colorbar and the heatmap.
            colorbar_options: keyword arguments passed to
                matplotlib.Figure.colorbar().
        """
        self.plot_colorbar = True
        self.colorbar_location_options = {'position': position, 'size': size, 'pad': pad}
        self.colorbar_options = colorbar_options
        return self

    def unset_colorbar(self) -> 'Heatmap':
        """Disables colorbar. No colorbar is drawn."""
        self.plot_colorbar = False
        return self

    def set_colormap(
        self,
        colormap: Union[str, mpl.colors.Colormap] = 'viridis',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> 'Heatmap':
        """Sets the colormap.

        Args:
            colormap: either a colormap name or a Colormap instance.
            vmin: the minimum value to map to the minimum color. Default is
                the minimum value in value_map.
            vmax: the maximum value to map to the maximum color. Default is
                the maximum value in value_map.
        """
        self.colormap = colormap
        self.vmin = vmin
        self.vmax = vmax
        return self

    def plot(
        self, ax: Optional[plt.Axes] = None, **pcolor_options: Any
    ) -> Tuple[plt.Axes, mpl_collections.Collection, pd.DataFrame]:
        """Plots the heatmap on the given Axes.

        Args:
            ax: the Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            pcolor_options: keyword arguments passed to ax.pcolor().

        Returns:
            A 3-tuple ``(ax, mesh, value_table)``. ``ax`` is the `plt.Axes` that
            is plotted on. ``mesh`` is the collection of paths drawn and filled.
            ``value_table`` is the 2-D pandas DataFrame of values constructed
            from the value_map.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
        # Find the boundary and size of the heatmap.
        coordinate_list = [_get_qubit_row_col(qubit) for qubit in self.value_map.keys()]
        rows = [row for row, _ in coordinate_list]
        cols = [col for _, col in coordinate_list]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        height, width = max_row - min_row + 1, max_col - min_col + 1
        # Construct the (height x width) table of values. Cells with no values
        # are filled with np.nan.
        value_table = pd.DataFrame(
            np.nan, index=range(min_row, max_row + 1), columns=range(min_col, max_col + 1)
        )
        for qubit, (float_value, _) in self.value_map.items():
            row, col = _get_qubit_row_col(qubit)
            value_table[col][row] = float_value
        # Construct the (height + 1) x (width + 1) cell boundary tables.
        x_table = np.array([np.arange(min_col - 0.5, max_col + 1.5)] * (height + 1))
        y_table = np.array([np.arange(min_row - 0.5, max_row + 1.5)] * (width + 1)).transpose()

        # Construct the URL array as an ordered list of URLs for non-nan cells.
        url_array: List[str] = []
        if self.url_map:
            url_array = [self.url_map.get((row, col), '') for row, col in value_table.stack().index]

        # Plot the heatmap.
        mesh = ax.pcolor(
            x_table,
            y_table,
            value_table,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=self.colormap,
            urls=url_array,
            **pcolor_options,
        )
        mesh.update_scalarmappable()
        ax.set(xlabel='column', ylabel='row')
        ax.xaxis.set_ticks(np.arange(min_col, max_col + 1))
        ax.yaxis.set_ticks(np.arange(min_row, max_row + 1))
        ax.set_ylim((max_row + 0.5, min_row - 0.5))

        if self.plot_colorbar:
            self._plot_colorbar(mesh, ax)

        if self.annot_map:
            self._write_annotations(mesh, ax)

        if show_plot:
            fig.show()

        return ax, mesh, value_table

    def _plot_colorbar(
        self, mappable: mpl.cm.ScalarMappable, ax: plt.Axes
    ) -> mpl.colorbar.Colorbar:
        """Plots the colorbar. Internal."""
        colorbar_ax = axes_grid1.make_axes_locatable(ax).append_axes(
            **self.colorbar_location_options
        )
        position = self.colorbar_location_options.get('position', 'right')
        orien = 'vertical' if position in ('left', 'right') else 'horizontal'
        colorbar = ax.figure.colorbar(
            mappable, colorbar_ax, ax, orientation=orien, **self.colorbar_options
        )
        colorbar_ax.tick_params(axis='y', direction='out')
        return colorbar

    def _write_annotations(self, mesh: mpl_collections.Collection, ax: plt.Axes) -> None:
        """Writes annotations to the center of cells. Internal."""
        for path, facecolor in zip(mesh.get_paths(), mesh.get_facecolors()):
            # Calculate the center of the cell, assuming that it is a square
            # centered at (x=col, y=row).
            vertices = path.vertices[:4]
            row = int(round(np.mean([v[1] for v in vertices])))
            col = int(round(np.mean([v[0] for v in vertices])))
            annotation = self.annot_map.get((row, col), '')
            if not annotation:
                continue
            face_luminance = vis_utils.relative_luminance(facecolor)
            text_color = 'black' if face_luminance > 0.4 else 'white'
            text_kwargs = dict(color=text_color, ha="center", va="center")
            text_kwargs.update(self.annot_kwargs)
            ax.text(col, row, annotation, **text_kwargs)
