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
import abc
from typing import Any, Dict, List, Mapping, Optional, SupportsFloat, Tuple, Union, TypeVar, Generic

import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import collections as mpl_collections
from mpl_toolkits import axes_grid1

from cirq.devices import grid_qubit

GraphicalCoordinate = Tuple[float, float]

TupleQubitCoord = Tuple[int, int]
QubitCoordinate = Union[TupleQubitCoord, grid_qubit.GridQubit]

GridQubitPair = Tuple[grid_qubit.GridQubit, grid_qubit.GridQubit]
TupleQubitPair = Tuple[TupleQubitCoord, TupleQubitCoord]

QubitPair = Union[
    GridQubitPair,
    TupleQubitPair,
]

# The value map is qubit coordinate -> a type that supports float conversion.
ValueMap = Union[Dict[grid_qubit.GridQubit, SupportsFloat], Dict[TupleQubitCoord, SupportsFloat]]

# The value map that maps a qubit pair to a type that supports float conversion.
InterValueMap = Union[
    Dict[GridQubitPair, SupportsFloat],
    Dict[TupleQubitPair, SupportsFloat],
]

# The potential targets
TargetType = Union[QubitPair, QubitCoordinate]

GValueMap = Dict[TargetType, SupportsFloat]

QubitPairAnnotation = Mapping[QubitPair, str]
QubitCoordAnnotation = Mapping[QubitCoordinate, str]


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
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return rgb.dot([0.2126, 0.7152, 0.0722])


def _get_qubit_row_col(qubit: QubitCoordinate) -> TupleQubitCoord:
    if isinstance(qubit, grid_qubit.GridQubit):
        return qubit.row, qubit.col
    else:
        return qubit[0], qubit[1]


# self type for HeatmapBase
T = TypeVar("T", bound='HeatmapBase')
# Key type (either a qubit or a qubit pair)
K = TypeVar("K")
# Mapping type qubit/qubit pair -> float
M = TypeVar("M", bound=Mapping)


class HeatmapBase(Generic[T, K, M], abc.ABC):
    """Base class for heatmaps."""

    def __init__(self, value_map: M, title: Optional[str] = None) -> None:
        self.set_value_map(value_map)
        self.annot_map = {  # Default annotation.
            self._target_to_coordinate(target): format(float(value), '.2g')
            for target, value in value_map.items()
        }
        self.annot_kwargs: Dict[str, Any] = {}
        self.unset_url_map()
        self.set_colorbar()
        values = [num for num, _ in self.value_map.values()]
        self.set_colormap(vmin=min(values), vmax=max(values))
        self.title = title

    def set_value_map(self: T, value_map: M) -> T:
        """Sets the values for each target.

        Args:
            value_map: the values for determining color for each cell.
        """
        # Fail fast if float() fails.
        # Keep the original value object for annotation.
        self.value_map = {target: (float(value), value) for target, value in value_map.items()}
        return self

    def _to_coord(self, target: Union[GraphicalCoordinate, K]):
        if isinstance(target, tuple) and (
            isinstance(target[0], float) or isinstance(target[1], float)
        ):
            return target
        return self._target_to_coordinate(target)  # type: ignore

    @abc.abstractmethod
    def _target_to_coordinate(self, target: K) -> GraphicalCoordinate:
        pass

    def set_annotation_map(
        self: T,
        annot_map: Union[Mapping[GraphicalCoordinate, str], Mapping[K, str]],
        **text_options: str,
    ) -> T:
        """Sets the annotation text.

        This method supports setting the annotation map directly via coordinates, or by the "target"
        object (qubit or qubit pair) of the heatmap.
        Note that set_annotation_map() and set_annotation_format()
        both sets the annotation map to be used. Whichever is called later wins.

        Args:
            annot_map: the texts to be drawn on each coordinate.
            text_options: keyword arguments passed to matplotlib.text.Text()
                when drawing the annotation texts.
        """
        self.annot_kwargs = text_options

        self.annot_map = {self._to_coord(key): value for key, value in annot_map.items()}
        return self

    def set_annotation_format(self: T, annot_format: str, **text_options: str) -> T:
        """Sets a format string to format values for each qubit.

        Args:
            annot_format: the format string for formatting values.
            text_options: keyword arguments to matplotlib.text.Text().
        """
        self.annot_map = {
            self._to_coord(target): format(value[1], annot_format)
            for target, value in self.value_map.items()
        }
        self.annot_kwargs = text_options
        return self

    def unset_annotation(self: T) -> T:
        """Disables annotation. No texts are shown in cells."""
        self.annot_map = {}
        return self

    def set_url_map(
        self: T, url_map: Union[Mapping[GraphicalCoordinate, str], Mapping[K, str]]
    ) -> T:
        """Sets the URLs for each cell."""

        self.url_map = {self._to_coord(key): value for key, value in url_map.items()}
        return self

    def unset_url_map(self: T) -> T:
        """Disables URL. No URLs are associated with cells."""
        self.url_map = {}
        return self

    def set_colorbar(
        self: T, position: str = 'right', size: str = '5%', pad: str = '2%', **colorbar_options: Any
    ) -> T:
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

    def unset_colorbar(self: T) -> T:
        """Disables colorbar. No colorbar is drawn."""
        self.plot_colorbar = False
        return self

    def set_colormap(
        self: T,
        colormap: Union[str, mpl.colors.Colormap] = 'viridis',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> T:
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

    def _write_annotations(self, ax: plt.Axes) -> None:
        """Writes annotations to the center of cells. Internal."""
        for target in self.value_map.keys():
            row, col = self._target_to_coordinate(target)
            annotation = self.annot_map.get((row, col), '')
            if not annotation:
                continue
            medium_brightness = (
                (self.vmax + self.vmin) / 2
                if (self.vmax is not None and self.vmin is not None)
                else 0.5
            )
            text_color = 'black' if self.value_map[target][0] > medium_brightness else 'white'
            text_kwargs = dict(color=text_color, ha="center", va="center")
            text_kwargs.update(self.annot_kwargs)
            ax.text(col, row, annotation, **text_kwargs)


class Heatmap(HeatmapBase['Heatmap', QubitCoordinate, ValueMap]):
    """Distribution of a value in 2D qubit lattice as a color map."""

    def __init__(self, value_map: ValueMap, title: Optional[str] = None) -> None:
        super().__init__(value_map, title)

    def _target_to_coordinate(self, target: QubitCoordinate) -> Tuple[float, float]:
        r, c = _get_qubit_row_col(target)
        return float(r), float(c)

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
        coordinate_list = [self._target_to_coordinate(target) for target in self.value_map.keys()]
        rows = [row for row, _ in coordinate_list]
        cols = [col for _, col in coordinate_list]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        # Construct table of values.
        # Cells with no values are filled with np.nan.
        value_table = pd.DataFrame(
            np.nan, index=np.arange(min_row, max_row + 1), columns=np.arange(min_col, max_col + 1)
        )
        for qubit, (float_value, _) in self.value_map.items():
            row, col = _get_qubit_row_col(qubit)
            value_table[col][row] = float_value
        # Construct the (height + 1) x (width + 1) cell boundary tables.

        x_table = np.array([np.arange(min_col - 0.5, max_col + 1.5)])
        y_table = np.array([np.arange(min_row - 0.5, max_row + 1.5)]).transpose()

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
            self._write_annotations(ax)

        if show_plot:
            fig.show()

        return ax, mesh, value_table


class TwoQubitInteractionHeatmap(
    HeatmapBase['TwoQubitInteractionHeatmap', QubitPair, InterValueMap]
):
    """Visualizing interactions between neighboring qubits on a 2D grid."""

    def __init__(self, value_map: InterValueMap, title: Optional[str] = None) -> None:
        super().__init__(value_map, title)

    def _target_to_coordinate(self, target: QubitPair) -> Tuple[float, float]:
        r1, c1 = _get_qubit_row_col(target[0])
        r2, c2 = _get_qubit_row_col(target[1])
        return float(r1 + r2) / 2, float(c1 + c2) / 2

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
        coordinate_list = [
            self._target_to_coordinate(qubit_pair) for qubit_pair in self.value_map.keys()
        ]
        rows = [row for row, _ in coordinate_list]
        cols = [col for _, col in coordinate_list]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        # Construct the table of values.
        # Cells with no values are filled with np.nan.
        value_table = pd.DataFrame(
            np.nan,
            index=np.arange(min_row, max_row + 0.1, 0.5),
            columns=np.arange(min_col, max_col + 0.1, 0.5),
        )

        for qubit, (float_value, _) in self.value_map.items():
            row, col = self._target_to_coordinate(qubit)
            value_table[col][row] = float_value
        # Construct the (height + 1) x (width + 1) cell boundary tables.
        x_table = np.arange(min_col - 0.25, max_col + 0.75, 0.5)
        y_table = np.arange(min_row - 0.25, max_row + 0.75, 0.5).transpose()

        # Construct the URL array as an ordered list of URLs for non-nan cells.
        url_array: List[str] = []
        if self.url_map:
            url_array = [self.url_map.get((row, col), '') for row, col in value_table.stack().index]

        def get_qubit_map(vm):
            # TODO: clean this up later
            return {**{q0: 0.0 for q0, _ in vm.keys()}, **{q1: 0.0 for _, q1 in vm.keys()}}

        hm = Heatmap(get_qubit_map(self.value_map))
        hm.set_colormap('binary')
        hm.unset_colorbar()
        hm.plot(
            ax=ax,
            linewidths=2,
            edgecolor='darkgrey',
            linestyle='dashed',
            # annot_kws={'alpha': 0.0},  # A hack: transparent texts in qubit cells.
        )

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
        ax.set_xticks(np.arange(min_col - 0.5, max_col + 1.5))
        ax.set_yticks(np.arange(min_row - 0.5, max_row + 1.5))
        # ax.set_xticks(np.arange(min_col, max_col + 1), minor='true')
        # ax.set_yticks(np.arange(min_row, max_row + 1), minor='true')
        ax.grid(b=True, which='minor', linestyle='--')
        ax.set_xlim((min_col - 1, max_col + 0.5))
        ax.set_ylim((max_row + 0.5, min_row - 0.5))
        plt.title(self.title)

        # if self.plot_colorbar:
        #     self._plot_colorbar(mesh, ax)

        if self.annot_map:
            self._write_annotations(ax)

        if show_plot:
            fig.show()

        return ax, None, value_table
