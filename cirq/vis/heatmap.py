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
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    SupportsFloat,
    Tuple,
    Union,
    TypeVar,
    Generic,
    NamedTuple,
)

from matplotlib import collections as mcoll
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import collections as mpl_collections
from matplotlib.patches import Polygon
from mpl_toolkits import axes_grid1

from cirq.devices import grid_qubit

Point = Tuple[float, float]

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
InteractionValueMap = Union[
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

    def _to_coord(self, target: Union[Point, K]):
        if isinstance(target, tuple) and (
            isinstance(target[0], float) or isinstance(target[1], float)
        ):
            return target
        return self._target_to_coordinate(target)  # type: ignore

    @abc.abstractmethod
    def _target_to_coordinate(self, target: K) -> Point:
        pass

    def set_annotation_map(
        self: T,
        annot_map: Union[Mapping[Point, str], Mapping[K, str]],
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

    def set_url_map(self: T, url_map: Union[Mapping[Point, str], Mapping[K, str]]) -> T:
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

    def _write_annotations(
        self, centers: List[Point], mesh: mpl_collections.Collection, ax: plt.Axes
    ) -> None:
        """Writes annotations to the center of cells. Internal."""
        for center, path, facecolor in zip(centers, mesh.get_paths(), mesh.get_facecolors()):
            # Calculate the center of the cell, assuming that it is a square
            # centered at (x=col, y=row).
            col, row = center
            annotation = self.annot_map.get((row, col), 'none')
            if not annotation:
                continue
            face_luminance = relative_luminance(facecolor)
            text_color = 'black' if face_luminance > 0.4 else 'white'
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
            centers = [
                np.mean([np.array(v) for v in p.vertices[:4]], axis=0) for p in mesh.get_paths()
            ]
            self._write_annotations(
                centers,
                mesh,
                ax,
            )

        if show_plot:
            fig.show()

        return ax, mesh, value_table


class TwoQubitInteractionHeatmap(
    HeatmapBase['TwoQubitInteractionHeatmap', QubitPair, InteractionValueMap]
):
    """Visualizing interactions between neighboring qubits on a 2D grid."""

    def __init__(self, value_map: InteractionValueMap, title: Optional[str] = None) -> None:
        super().__init__(value_map, title)

    def _target_to_coordinate(self, target: QubitPair) -> Tuple[float, float]:
        r1, c1 = _get_qubit_row_col(target[0])
        r2, c2 = _get_qubit_row_col(target[1])
        return float(r1 + r2) / 2, float(c1 + c2) / 2

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        coupler_margin: float = 0.03,
        coupler_width: float = 0.6,
        **pcolor_options: Any,
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
        rows = {row for row, _ in coordinate_list}
        cols = {col for _, col in coordinate_list}
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
        # x_table = np.arange(min_col - 0.25, max_col + 0.75, 0.5)
        # y_table = np.arange(min_row - 0.25, max_row + 0.75, 0.5).transpose()

        # Construct the URL array as an ordered list of URLs for non-nan cells.
        url_array: List[str] = []
        if self.url_map:
            url_array = [self.url_map.get((row, col), '') for row, col in value_table.stack().index]

        hm = Heatmap({q: 0.0 for qubits in self.value_map.keys() for q in qubits})
        hm.set_colormap('binary')
        hm.unset_colorbar()
        hm.unset_annotation()
        hm.plot(
            ax=ax,
            linewidths=2,
            edgecolor='lightgrey',
            linestyle='dashed',
        )

        coupler_list = _extract_pair_data(self.value_map, coupler_margin, coupler_width)

        # Make a blank heatmap of qubits. Pop out (vmin, vmax) to ensure it's blank.
        vmin = pcolor_options.pop('vmin', None)
        vmax = pcolor_options.pop('vmax', None)

        # Make the heatmap for two-qubit metrics.
        collection = mcoll.PolyCollection([c.polygon for c in coupler_list], cmap=self.colormap)
        collection.set_clim(vmin, vmax)
        collection.set_array(np.array([c.value for c in coupler_list]))
        ax.add_collection(collection)
        collection.update_scalarmappable()  # Populate facecolors.
        if self.annot_map:
            centers = [c.center for c in coupler_list]
            self._write_annotations(centers, collection, ax)

        ax.set(xlabel='column', ylabel='row')

        min_xtick = np.floor(min_col)
        max_xtick = np.ceil(max_col)
        ax.set_xticks(np.arange(min_xtick, max_xtick + 1))
        min_ytick = np.floor(min_row)
        max_ytick = np.ceil(max_row)
        ax.set_yticks(np.arange(min_ytick, max_ytick + 1))
        ax.set_xlim((min_xtick - 0.6, max_xtick + 0.6))
        ax.set_ylim((max_ytick + 0.6, min_ytick - 0.6))
        plt.title(self.title)

        if self.plot_colorbar:
            self._plot_colorbar(collection, ax)

        if show_plot:
            fig.show()

        return ax, collection, value_table


class Coupler(NamedTuple('Coupler', [('polygon', Polygon), ('center', Point), ('value', Any)])):
    """A Coupler contains all data necessary to plot a qubit-pair coupling.

    Note that all coordinates are relative and un-normalized, i.e., x is in
    [0, max_col + 1 - min_col] and y is in [0, max_row + 1 - min_row]. It is
    a translation from the data coordinates [min_col, max_col + 1] x
    [min_row, max_row + 1] by (-min_col, -min_row). This is the coordinates
    of the axes after a pcolor() call.

    Attributes:
        polygon: a polygon for the vertices of the coupler.
        center: the center coordinates for drawing the annotation.
        value: the value associated with the coupler.
    """


def _extract_pair_data(
    pair_value_map: Mapping[QubitPair, float], coupler_margin: float, coupler_width: float
) -> List[Coupler]:
    """Extracts data from pair_value_map and returns them.

    Args:
        pair_value_map: the map of a pair of qubit names to a value.
        coupler_margin: the margin between a coupler polygon and the center of
            qubit squares, zero means 2 couplers of the same qubit touch each
            other.
        coupler_width: the full width of the coupler polygon. Setting it to 0 or
            a negative number removes the polygon.

    Returns:
        A tuple of 3 containers:
        - A set of qubit names needed for plotting the background qubit map.
        - A list of Coupler objects for plotting the qubit-pair couplers.
        - A tuple of (minimum column, minimum row, maximum column, maximum row).
    """
    coupler_list: List[Coupler] = []
    cwidth = coupler_width / 2.0
    setback = 0.5 - cwidth
    for (q1, q2) in sorted(pair_value_map.keys()):
        value = pair_value_map[(q1, q2)]
        row1, col1 = map(float, _get_qubit_row_col(q1))
        row2, col2 = map(float, _get_qubit_row_col(q2))

        if abs(row1 - row2) + abs(col1 - col2) != 1:
            raise ValueError(f"{q1}-{q2} is not supported because they are not nearest neighbors")

        if coupler_width <= 0:
            polygon: Polygon = []
        elif row1 == row2:  # horizontal
            col1, col2 = min(col1, col2), max(col1, col2)
            col_center = (col1 + col2) / 2.0
            polygon = [
                (col1 + coupler_margin, row1),
                (col_center - setback, row1 + cwidth - coupler_margin),
                (col_center + setback, row1 + cwidth - coupler_margin),
                (col2 - coupler_margin, row2),
                (col_center + setback, row1 - cwidth + coupler_margin),
                (col_center - setback, row1 - cwidth + coupler_margin),
            ]
        elif col1 == col2:  # vertical
            row1, row2 = min(row1, row2), max(row1, row2)
            row_center = (row1 + row2) / 2.0
            polygon = [
                (col1, row1 + coupler_margin),
                (col1 + cwidth - coupler_margin, row_center - setback),
                (col1 + cwidth - coupler_margin, row_center + setback),
                (col2, row2 - coupler_margin),
                (col1 - cwidth + coupler_margin, row_center + setback),
                (col1 - cwidth + coupler_margin, row_center - setback),
            ]
        coupler_list.append(
            Coupler(
                polygon=[(c, r) for c, r in polygon],
                center=((col1 + col2) / 2.0, (row1 + row2) / 2.0),
                value=value[0],
            )
        )

    return coupler_list
