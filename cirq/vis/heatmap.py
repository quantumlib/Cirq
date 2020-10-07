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
"""Heatmap class
See examples/heatmap_example.py for an example usage in
an interactive session.
"""

from typing import (Any, Dict, List, Mapping, Optional, SupportsFloat, Tuple,
                    Union)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import collections as mpl_collections
from mpl_toolkits import axes_grid1

from cirq.devices import grid_qubit

QubitCoordinate = Union[Tuple[float, float], grid_qubit.GridQubit]

QubitPair = Union[Tuple[grid_qubit.GridQubit, grid_qubit.GridQubit],
                  Tuple[Tuple[float, float], Tuple[float, float]]]

Hashable = Union[QubitCoordinate, QubitPair]

# The value map is qubit coordinate -> a type that supports float conversion.
ValueMap = Union[Dict[grid_qubit.GridQubit, SupportsFloat],
                 Dict[Tuple[float, float], SupportsFloat]]
InterValueMap = Union[
    Dict[Tuple[grid_qubit.GridQubit, grid_qubit.GridQubit], SupportsFloat],
    Dict[Tuple[Tuple[float, float], Tuple[float, float]], SupportsFloat]]


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


def _get_qubit_row_col(qubit: QubitCoordinate) -> Tuple[float, float]:
    if isinstance(qubit, grid_qubit.GridQubit):
        return qubit.row, qubit.col
    else:
        return qubit[0], qubit[1]


def _ave_qubit_row_col(qubitpair: QubitPair) -> Tuple[float, float]:
    if isinstance(qubitpair[0], grid_qubit.GridQubit):
        return (qubitpair[0].row+qubitpair[1].row)/2, \
               (qubitpair[0].col+qubitpair[1].col)/2
    else:
        return (qubitpair[0][0]+qubitpair[0][1])/2, \
               (qubitpair[1][0]+qubitpair[1][1])/2


class HeatmapBase:
    """Distribution of a value in 2D qubit lattice as a color map."""

    def __init__(self, title: Optional[str] = None) -> None:
        # coverage: ignore
        self.value_map: Dict[Tuple[float, float],
                             Tuple[float, SupportsFloat]] = {}
        self.annot_map = {  # Default annotation.
            _get_qubit_row_col(hashable): format(float(value[0]), '.2g')
            for hashable, value in self.value_map.items()
        }
        self.annot_kwargs: Dict[str, Any] = {}
        self._unset_url_map()
        self._set_colorbar()
        value_list = [num for num, _ in list(self.value_map.values())]
        self._set_colormap(min(value_list), max(value_list))
        self.title = title

    def _set_annotation_map(self, annot_map: Mapping[QubitCoordinate, str],
                            **text_options: str) -> None:
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

    def _set_annotation_format(self, annot_format: str,
                               **text_options: str) -> None:
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

    def _unset_annotation(self) -> None:
        """Disables annotation. No texts are shown in cells."""
        self.annot_map = {}

    def _set_url_map(self, url_map: Mapping[QubitCoordinate, str]) \
            -> None:
        """Sets the URLs for each cell."""
        self.url_map = {
            _get_qubit_row_col(hashable): value
            for hashable, value in url_map.items()
        }

    def _unset_url_map(self) -> None:
        """Disables URL. No URLs are associated with cells."""
        self.url_map = {}

    def _set_colorbar(self,
                      position: str = 'right',
                      size: str = '5%',
                      pad: str = '2%',
                      **colorbar_options: Any) -> None:
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

    def _unset_colorbar(self) -> None:
        """Disables colorbar. No colorbar is drawn."""
        self.plot_colorbar = False

    def _set_colormap(self,
                      vmin: float,
                      vmax: float,
                      colormap: Union[str, mpl.colors.Colormap] = 'viridis'
                     ) -> None:
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
            text_color = 'black' if self.value_map[(
                row, col)][0] > (self.vmax + self.vmin) / 2 else 'white'
            text_kwargs = dict(color=text_color, ha="center", va="center")
            text_kwargs.update(self.annot_kwargs)
            ax.text(col, row, annotation, **text_kwargs)


class Heatmap(HeatmapBase):
    """Distribution of a value in 2D qubit lattice as a color map."""

    def __init__(self, value_map: ValueMap,
                 title: Optional[str] = None) -> None:
        self.value_map: Dict[Tuple[float, float],
                             Tuple[float, SupportsFloat]] = {}
        self.set_value_map(value_map)
        self.annot_map = {  # Default annotation.
            _get_qubit_row_col(hashable): format(float(value[0]), '.2g')
            for hashable, value in self.value_map.items()
        }
        self.annot_kwargs: Dict[str, Any] = {}
        self._unset_url_map()
        self._set_colorbar()
        value_list = [num for num, _ in list(self.value_map.values())]
        self._set_colormap(min(value_list), max(value_list))
        self.title = title

    def set_value_map(self, value_map: ValueMap) -> 'Heatmap':
        """Sets the values for each qubit.
        Args:
            value_map: the values for determining color for each cell.
        """
        # Fail fast if float() fails.
        # Keep the original value object for annotation.
        self.value_map = {
            _get_qubit_row_col(hashable): (float(value), value)
            for hashable, value in value_map.items()
        }
        return self

    def set_annotation_map(self, annot_map: Mapping[QubitCoordinate, str],
                           **text_options: str) -> 'Heatmap':
        self._set_annotation_map(annot_map, **text_options)
        return self

    def set_annotation_format(self, annot_format: str,
                              **text_options: str) -> 'Heatmap':
        self._set_annotation_format(annot_format, **text_options)
        return self

    def unset_annotation(self) -> 'Heatmap':
        self._unset_annotation()
        return self

    def set_url_map(self, url_map: Mapping[QubitCoordinate, str]) \
            -> 'Heatmap':
        self._set_url_map(url_map)
        return self

    def unset_url_map(self) -> 'Heatmap':
        self._unset_url_map()
        return self

    def set_colorbar(self,
                     position: str = 'right',
                     size: str = '5%',
                     pad: str = '2%',
                     **colorbar_options: Any) -> 'Heatmap':
        self._set_colorbar(position, size, pad, **colorbar_options)
        return self

    def unset_colorbar(self) -> 'Heatmap':
        """Disables colorbar. No colorbar is drawn."""
        self._unset_colorbar()
        return self

    def set_colormap(self,
                     vmin: float,
                     vmax: float,
                     colormap: Union[str, mpl.colors.Colormap] = 'viridis'
                    ) -> 'Heatmap':
        self._set_colormap(vmin, vmax, colormap)
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
        # Construct table of values.
        # Cells with no values are filled with np.nan.
        value_table = pd.DataFrame(np.nan,
                                   index=np.arange(min_row, max_row + 1),
                                   columns=np.arange(min_col, max_col + 1))
        for qubit, (float_value, _) in self.value_map.items():
            row, col = _get_qubit_row_col(qubit)
            value_table[col][row] = float_value
        # Construct the (height + 1) x (width + 1) cell boundary tables.

        x_table = np.array([np.arange(min_col - 0.5, max_col + 1.5)])
        y_table = np.array([np.arange(min_row - 0.5,
                                      max_row + 1.5)]).transpose()

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


class InterHeatmap(HeatmapBase):
    # coverage: ignore
    """Distribution of a value in 2D qubit lattice as a color map."""

    def __init__(self, value_map: InterValueMap,
                 title: Optional[str] = None) -> None:
        self.value_map: Dict[Tuple[float, float],
                             Tuple[float, SupportsFloat]] = {}
        self.set_value_map(value_map)
        self.annot_map = {  # Default annotation.
            _get_qubit_row_col(hashable): format(float(value[0]), '.2g')
            for hashable, value in self.value_map.items()
        }
        self.annot_kwargs: Dict[str, Any] = {}
        self._unset_url_map()
        self._set_colorbar()
        value_list = [num for num, _ in list(self.value_map.values())]
        self._set_colormap(min(value_list), max(value_list))
        self.title = title

    def set_value_map(self, inter_value_map: InterValueMap) -> 'InterHeatmap':
        """Sets the values for each qubit.
        Args:
            inter_value_map: the values for determining color for each cell.
        """
        # Fail fast if float() fails.
        # Keep the original value object for annotation.

        self.value_map = {
            _ave_qubit_row_col(hashable): (float(value), value)
            for hashable, value in inter_value_map.items()
        }
        return self

    def set_annotation_map(self, annot_map: Mapping[QubitCoordinate, str],
                           **text_options: str) -> 'InterHeatmap':
        self._set_annotation_map(annot_map, **text_options)
        return self

    def set_annotation_format(self, annot_format: str,
                              **text_options: str) -> 'InterHeatmap':
        self._set_annotation_format(annot_format, **text_options)
        return self

    def unset_annotation(self) -> 'InterHeatmap':
        self._unset_annotation()
        return self

    def set_url_map(self, url_map: Mapping[QubitCoordinate, str]) \
            -> 'InterHeatmap':
        self._set_url_map(url_map)
        return self

    def unset_url_map(self) -> 'InterHeatmap':
        self._unset_url_map()
        return self

    def set_colorbar(self,
                     position: str = 'right',
                     size: str = '5%',
                     pad: str = '2%',
                     **colorbar_options: Any) -> 'InterHeatmap':
        self._set_colorbar(position, size, pad, **colorbar_options)
        return self

    def unset_colorbar(self) -> 'InterHeatmap':
        """Disables colorbar. No colorbar is drawn."""
        self._unset_colorbar()
        return self

    def set_colormap(self,
                     vmin: float,
                     vmax: float,
                     colormap: Union[str, mpl.colors.Colormap] = 'viridis'
                    ) -> 'InterHeatmap':
        self._set_colormap(vmin, vmax, colormap)
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
        # Construct the table of values.
        # Cells with no values are filled with np.nan.
        value_table = pd.DataFrame(np.nan,
                                   index=np.arange(min_row, max_row + 0.1, 0.5),
                                   columns=np.arange(min_col, max_col + 0.1,
                                                     0.5))

        for qubit, (float_value, _) in self.value_map.items():
            row, col = _get_qubit_row_col(qubit)
            value_table[col][row] = float_value
        # Construct the (height + 1) x (width + 1) cell boundary tables.
        x_table = np.arange(min_col - 0.25, max_col + 0.75, 0.5)
        y_table = np.arange(min_row - 0.25, max_row + 0.75, 0.5).transpose()

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

        return ax, mesh, value_table
