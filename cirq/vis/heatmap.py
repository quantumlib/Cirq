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

from typing import (Any, Dict, List, Mapping, Optional, SupportsFloat, Tuple,
                    Iterable, Union)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import collections as mpl_collections
from mpl_toolkits import axes_grid1

from cirq.devices import grid_qubit

QubitCoordinate = Union[Tuple[float, float], grid_qubit.GridQubit]

QubitPair = Union[Tuple[grid_qubit.GridQubit, grid_qubit.GridQubit],
                  Tuple[Tuple[int, int], Tuple[int, int]]]

Hashable = Union[QubitCoordinate, QubitPair]

# The value map is qubit coordinate -> a type that supports float conversion.
ValueMap = Dict[Hashable, SupportsFloat]


def relative_luminance(color: np.ndarray) -> float:
    """Returns the relative luminance according to W3C specification.

    Spec: https://www.w3.org/TR/WCAG21/#dfn-relative-luminance.

    Args:
        color: a numpy array with the first 3 elements red, green, and blue
            with values in [0, 1].
    Returns:
        relative luminance of color in [0, 1].
    """
    rgb = color[:3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
    return rgb.dot([.2126, .7152, .0722])


def _get_qubit_row_col(qubit: QubitCoordinate) -> Tuple[int, int]:
    if isinstance(qubit, grid_qubit.GridQubit):
        return qubit.row, qubit.col
    elif isinstance(qubit, tuple):
        return qubit[0], qubit[1]


def _ave_qubit_row_col(qubitpair: QubitPair) -> Tuple[float, float]:
    if isinstance(qubitpair[0], grid_qubit.GridQubit):
        return (qubitpair[0].row+qubitpair[1].row)/2, (qubitpair[0].col+qubitpair[1].col)/2
    elif isinstance(qubitpair[1], tuple):
        return (qubitpair[0][0]+qubitpair[0][1])/2, (qubitpair[1][0]+qubitpair[1][1])/2



class Heatmap_base:
    """Distribution of a value in 2D qubit lattice as a color map."""

    def __init__(self, value_map: ValueMap, title: Optional[str] = None) -> None:
        self.set_value_map(value_map)
        self.annot_map = {  # Default annotation.
            _get_qubit_row_col(hashable): format(float(value[0]), '.2g')
            for hashable, value in self.value_map.items()
        }
        self.annot_kwargs: Dict[str, Any] = {}
        self.unset_url_map()
        self.set_colorbar()
        self.set_colormap()
        self.title = title

    def set_annotation_map(self, annot_map: Mapping[QubitCoordinate, str],
                           **text_options: str) -> 'Heatmap_base':
        """Sets the annotation text for each qubit.

        Note that set_annotation_map() and set_annotation_format()
        both sets the annotation map to be used. Whichever is called later wins.

        Args:
            annot_map: the texts to be drawn on each qubit cell.
            text_options: keyword arguments passed to matplotlib.text.Text()
                when drawing the annotation texts.
        """
        self.annot_map = {
            _get_qubit_row_col(hashable): value
            for hashable, value in annot_map.items()
        }
        self.annot_kwargs = text_options
        return self

    def set_annotation_format(self, annot_format: str,
                              **text_options: str) -> 'Heatmap_base':
        """Sets a format string to format values for each qubit.

        Args:
            annot_format: the format string for formatting values.
            text_options: keyword arguments to matplotlib.text.Text().
        """
        self.annot_map = {
            _get_qubit_row_col(hashable): format(value[1], annot_format)
            for hashable, value in self.value_map.items()
        }
        self.annot_kwargs = text_options
        return self

    def unset_annotation(self) -> 'Heatmap_base':
        """Disables annotation. No texts are shown in cells."""
        self.annot_map = {}
        return self

    def set_url_map(self, url_map: Mapping[QubitCoordinate, str]) -> 'Heatmap_base':
        """Sets the URLs for each cell."""
        self.url_map = {
            _get_qubit_row_col(hashable): value
            for hashable, value in url_map.items()
        }
        return self

    def unset_url_map(self) -> 'Heatmap':
        """Disables URL. No URLs are associated with cells."""
        self.url_map = {}
        return self

    def set_colorbar(self,
                     position: str = 'right',
                     size: str = '5%',
                     pad: str = '2%',
                     **colorbar_options: Any) -> 'Heatmap_base':
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
        self.colorbar_location_options = {
            'position': position,
            'size': size,
            'pad': pad
        }
        self.colorbar_options = colorbar_options
        return self

    def unset_colorbar(self) -> 'Heatmap':
        """Disables colorbar. No colorbar is drawn."""
        self.plot_colorbar = False
        return self

    def set_colormap(self,
                     colormap: Union[str, mpl.colors.Colormap] = 'viridis',
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None) -> 'Heatmap_base':
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

    def _plot_colorbar(self, mappable: mpl.cm.ScalarMappable,
                       ax: plt.Axes) -> mpl.colorbar.Colorbar:
        """Plots the colorbar. Internal."""
        colorbar_ax = axes_grid1.make_axes_locatable(ax).append_axes(
            **self.colorbar_location_options)
        position = self.colorbar_location_options.get('position', 'right')
        orien = 'vertical' if position in ('left', 'right') else 'horizontal'
        colorbar = ax.figure.colorbar(mappable,
                                      colorbar_ax,
                                      ax,
                                      orientation=orien,
                                      **self.colorbar_options)
        colorbar_ax.tick_params(axis='y', direction='out')
        return colorbar

    def _write_annotations(self, ax: plt.Axes) -> None:
        """Writes annotations to the center of cells. Internal."""
        for row, col in self.value_map.keys():
            annotation = self.annot_map.get((row, col), '')
            if not annotation:
                continue
            text_color = 'black' if self.value_map[(row, col)][0] > (self.vmax+self.vmin)/2 else 'white'
            text_kwargs = dict(color=text_color, ha="center", va="center")
            text_kwargs.update(self.annot_kwargs)
            ax.text(col, row, annotation, **text_kwargs)

class Heatmap(Heatmap_base):
    """Distribution of a value in 2D qubit lattice as a color map."""

    def set_value_map(self, value_map: ValueMap) -> 'Heatmap':
        """Sets the values for each qubit.

        Args:
            value_map: the values for determining color for each cell.
        """
        # Fail fast if float() fails.
        # Keep the original value object for annotation.
        self.value_map = {
            hashable: (float(value), value) for hashable, value in value_map.items()
        }
        return self

    def plot(self, ax: Optional[plt.Axes] = None, **pcolor_options: Any
             ) -> Tuple[plt.Axes, mpl_collections.Collection, pd.DataFrame]:
        """Plots the heatmap on the given Axes.

        Args:
            ax: the Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            filepath: the path to save the produced image file
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
        coordinate_list = [
            _get_qubit_row_col(qubit) for qubit in self.value_map.keys()
        ]
        rows = [row for row, _ in coordinate_list]
        cols = [col for _, col in coordinate_list]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        height, width = max_row - min_row + 1, max_col - min_col + 1
        # Construct the (height x width) table of values. Cells with no values
        # are filled with np.nan.
        value_table = pd.DataFrame(np.nan,
                                   index=range(min_row, max_row + 1),
                                   columns=range(min_col, max_col + 1))
        for qubit, (float_value, _) in self.value_map.items():
            row, col = _get_qubit_row_col(qubit)
            value_table[col][row] = float_value
        # Construct the (height + 1) x (width + 1) cell boundary tables.
        x_table = np.array([np.arange(min_col - 0.5, max_col + 1.5)] *
                           (height + 1))
        y_table = np.array([np.arange(min_row - 0.5, max_row + 1.5)] *
                           (width + 1)).transpose()

        # Construct the URL array as an ordered list of URLs for non-nan cells.
        url_array: List[str] = []
        if self.url_map:
            url_array = [
                self.url_map.get((row, col), '')
                for row, col in value_table.stack().index
            ]

        # Plot the heatmap.
        mesh = ax.pcolor(x_table,
                         y_table,
                         value_table,
                         vmin=self.vmin,
                         vmax=self.vmax,
                         cmap=self.colormap,
                         urls=url_array,
                         **pcolor_options)
        mesh.update_scalarmappable()
        ax.set(xlabel='column', ylabel='row')
        ax.xaxis.set_ticks(np.arange(min_col, max_col + 1))
        ax.yaxis.set_ticks(np.arange(min_row, max_row + 1))
        ax.set_ylim((max_row + 0.5, min_row - 0.5))

        if self.plot_colorbar:
            self._plot_colorbar(mesh, ax)

        if self.annot_map:
            self._write_annotations(ax)

        if show_plot:
            fig.show()

        return ax, mesh, value_table


class InterHeatmap(Heatmap_base):
    """Distribution of a value in 2D qubit lattice as a color map."""

    def set_value_map(self, inter_value_map: ValueMap) -> 'Heatmap':
        """Sets the values for each qubit.

        Args:
            inter_value_map: the values for determining color for each cell.
        """
        # Fail fast if float() fails.
        # Keep the original value object for annotation.

        if isinstance(list(inter_value_map.values())[0], Iterable):
            self.value_map = {
                _ave_qubit_row_col(hashable): (float(value[0]), value[0]) for hashable, value in inter_value_map.items()
            }
        else:
            self.value_map = {
                _ave_qubit_row_col(hashable): (float(value), value) for hashable, value in inter_value_map.items()
            }
        return self

    def plot(self, ax: Optional[plt.Axes] = None, **pcolor_options: Any
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
        coordinate_list = [
            _get_qubit_row_col(qubit) for qubit in self.value_map.keys()
        ]
        rows = [row for row, _ in coordinate_list]
        cols = [col for _, col in coordinate_list]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        height, width = max_row - min_row + 1, max_col - min_col + 1
        # Construct the (height x width) table of values. Cells with no values
        # are filled with np.nan.
        value_table = pd.DataFrame(np.nan,
                                   index=np.arange(min_row, max_row + 0.1, 0.5),
                                   columns=np.arange(min_col, max_col + 0.1, 0.5))
        self.set_colormap('viridis', min(self.value_map.values())[0], max(self.value_map.values())[0])
        for qubit, (float_value, _) in self.value_map.items():
            row, col = _get_qubit_row_col(qubit)
            value_table[col][row] = float_value
        # Construct the (height + 1) x (width + 1) cell boundary tables.
        x_table = np.arange(min_col - 0.25, max_col + 0.75, 0.5)
        y_table = np.arange(min_row - 0.25, max_row + 0.75, 0.5)

        # Construct the URL array as an ordered list of URLs for non-nan cells.
        url_array: List[str] = []
        if self.url_map:
            url_array = [
                self.url_map.get((row, col), '')
                for row, col in value_table.stack().index
            ]

        # Plot the heatmap.
        mesh = ax.pcolor(x_table,
                         y_table,
                         value_table,
                         vmin=self.vmin,
                         vmax=self.vmax,
                         cmap=self.colormap,
                         urls=url_array,
                         **pcolor_options)
        mesh.update_scalarmappable()
        ax.set(xlabel='column', ylabel='row')
        ax.set_xticks(np.arange(min_col - 0.5, max_col + 1.5))
        ax.set_yticks(np.arange(min_row - 0.5, max_row + 1.5))
        ax.set_xticks(np.arange(min_col, max_col + 1), minor='true')
        ax.set_yticks(np.arange(min_row, max_row + 1), minor='true')
        ax.grid(b=True, which='minor', linestyle='--')
        ax.set_xlim((max_col + 0.5, min_col - 0.5))
        ax.set_ylim((max_row + 0.5, min_row - 0.5))
        plt.title(self.title)

        if self.plot_colorbar:
            self._plot_colorbar(mesh, ax)

        if self.annot_map:
            self._write_annotations(ax)

        if show_plot:
            fig.show()

        plt.show()

        return ax, mesh, value_table

import cirq

def main():
    title = 'Two Qubit Sycamore Gate Xeb Cycle Total Error'
    value_map = {
        (cirq.GridQubit(3, 2), cirq.GridQubit(4, 2)):[0.004619111460557768],
        (cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)):[0.0076079162393482835],
        (cirq.GridQubit(4, 1), cirq.GridQubit(5, 1)):[0.010323903068646778],
        (cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)):[0.00729037246947839],
        (cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)):[0.008226663382640803],
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)):[0.01504682356081491],
        (cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)):[0.00673880216745637],
        (cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)):[0.01020380985719993],
        (cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)):[0.005713058677283056],
        (cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)):[0.006431698844451689],
        (cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)):[0.004676551878404933],
        (cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)):[0.009471810549265769],
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)):[0.003834724159559072],
        (cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)):[0.010423354216218345],
        (cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)):[0.0062515002303844824],
        (cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)):[0.005419247075412775],
        (cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)):[0.02236774155039517],
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)):[0.006116965562115412],
        (cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)):[0.005300336755683754],
        (cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)):[0.012849356290539266],
        (cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)):[0.007785990142364307],
        (cirq.GridQubit(6, 5), cirq.GridQubit(7, 5)):[0.008790971346696541],
        (cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)):[0.004104719338404117],
        (cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)):[0.009236765681133435],
        (cirq.GridQubit(7, 3), cirq.GridQubit(8, 3)):[0.024921853294157192],
        (cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)):[0.0059072812181635015],
        (cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)):[0.004990546867455203],
        (cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)):[0.007852170748540305],
        (cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)):[0.006424831182351348],
        (cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)):[0.005248674988741292],
        (cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)):[0.014301577907262525],
        (cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)):[0.0038720100369923904]
    }
    heatmap = InterHeatmap(value_map, title)
    heatmap.plot()


if __name__ == '__main__':
    main()
