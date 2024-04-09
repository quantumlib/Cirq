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
import copy
from dataclasses import astuple, dataclass
from typing import (
    Any,
    cast,
    Dict,
    List,
    Mapping,
    Optional,
    overload,
    Sequence,
    SupportsFloat,
    Tuple,
    Union,
)

import matplotlib as mpl
import matplotlib.collections as mpl_collections
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import axes_grid1

from cirq.devices import grid_qubit
from cirq.vis import vis_utils

QubitTuple = Tuple[grid_qubit.GridQubit, ...]

Polygon = Sequence[Tuple[float, float]]


@dataclass
class Point:
    x: float
    y: float

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class PolygonUnit:
    """Dataclass to store information about a single polygon unit to plot on the heatmap

    For single (grid) qubit heatmaps, the polygon is a square.
    For two (grid) qubit interaction heatmaps, the polygon is a hexagon.

    Args:
        polygon: Vertices of the polygon to plot.
        value: The value for the heatmap coloring.
        center: The center point of the polygon where annotation text should be printed.
        annot: The annotation string to print on the coupler.

    """

    polygon: Polygon
    value: float
    center: Point
    annot: Optional[str]


class Heatmap:
    """Distribution of a value in 2D qubit lattice as a color map."""

    # pylint: disable=function-redefined
    @overload
    def __init__(self, value_map: Mapping[QubitTuple, SupportsFloat], **kwargs):
        pass

    @overload
    def __init__(self, value_map: Mapping[grid_qubit.GridQubit, SupportsFloat], **kwargs):
        pass

    def __init__(
        self,
        value_map: Union[
            Mapping[QubitTuple, SupportsFloat], Mapping[grid_qubit.GridQubit, SupportsFloat]
        ],
        **kwargs,
    ):
        """2D qubit grid Heatmaps

        Draw 2D qubit grid heatmap with Matplotlib with parameters to configure the properties of
        the plot.

        Args:
            value_map: A dictionary of qubits or QubitTuples as keys and corresponding magnitude
                as float values. It corresponds to the data which should be plotted as a heatmap.
            **kwargs: Optional kwargs including
                title: str, default = None
                plot_colorbar: bool, default = True

                annotation_map: dictionary,
                    A dictionary of QubitTuples as keys and corresponding annotation str as values.
                    It corresponds to the text that should be added on top of each heatmap
                    polygon unit.
                annotation_format: str, default = '.2g'
                    Formatting string using which annotation_map will be implicitly constructed by
                    applying format(value, annotation_format) for each key in value_map.
                    This is ignored if annotation_map is explicitly specified.
                annotation_text_kwargs: Matplotlib Text **kwargs,

                colorbar_position: {'right', 'left', 'top', 'bottom'}, default = 'right'
                colorbar_size: str, default = '5%'
                colorbar_pad: str, default = '2%'
                colorbar_options: Matplotlib colorbar **kwargs, default = None,


                collection_options: Matplotlib PolyCollection **kwargs, default
                                    {"cmap" : "viridis"}
                vmin, vmax: colormap scaling floats, default = None
        """
        self._value_map: Mapping[QubitTuple, SupportsFloat] = {
            k if isinstance(k, tuple) else (k,): v for k, v in value_map.items()
        }
        self._validate_kwargs(kwargs)
        if '_config' not in self.__dict__:
            self._config: Dict[str, Any] = {}
        self._config.update(
            {
                "plot_colorbar": True,
                "colorbar_position": "right",
                "colorbar_size": "5%",
                "colorbar_pad": "2%",
                "collection_options": {"cmap": "viridis"},
                "annotation_format": ".2g",
            }
        )
        self._config.update(kwargs)

    def _extra_valid_kwargs(self) -> List[str]:
        return []

    def _validate_kwargs(self, kwargs) -> None:
        valid_colorbar_kwargs = [
            "plot_colorbar",
            "colorbar_position",
            "colorbar_size",
            "colorbar_pad",
            "colorbar_options",
        ]
        valid_collection_kwargs = ["collection_options", "vmin", "vmax"]
        valid_heatmap_kwargs = [
            "title",
            "annotation_map",
            "annotation_text_kwargs",
            "annotation_format",
        ]
        valid_kwargs = (
            valid_colorbar_kwargs
            + valid_collection_kwargs
            + valid_heatmap_kwargs
            + self._extra_valid_kwargs()
        )
        if any([k not in valid_kwargs for k in kwargs]):
            invalid_args = ", ".join([k for k in kwargs if k not in valid_kwargs])
            raise ValueError(f"Received invalid argument(s): {invalid_args}")

    def update_config(self, **kwargs) -> 'Heatmap':
        """Add/Modify **kwargs args passed during initialisation."""
        self._validate_kwargs(kwargs)
        self._config.update(kwargs)
        return self

    def _qubits_to_polygon(self, qubits: QubitTuple) -> Tuple[Polygon, Point]:
        qubit = qubits[0]
        x, y = float(qubit.row), float(qubit.col)
        return (
            [(y - 0.5, x - 0.5), (y - 0.5, x + 0.5), (y + 0.5, x + 0.5), (y + 0.5, x - 0.5)],
            Point(y, x),
        )

    def _get_annotation_value(self, key, value) -> Optional[str]:
        if self._config.get('annotation_map'):
            return self._config['annotation_map'].get(key)
        elif self._config.get('annotation_format'):
            try:
                return format(value, self._config['annotation_format'])
            except:
                return format(float(value), self._config['annotation_format'])
        else:
            return None

    def _get_polygon_units(self) -> List[PolygonUnit]:
        polygon_unit_list: List[PolygonUnit] = []
        for qubits, value in sorted(self._value_map.items()):
            polygon, center = self._qubits_to_polygon(qubits)
            polygon_unit_list.append(
                PolygonUnit(
                    polygon=polygon,
                    center=center,
                    value=float(value),
                    annot=self._get_annotation_value(qubits, value),
                )
            )
        return polygon_unit_list

    def _plot_colorbar(
        self, mappable: mpl.cm.ScalarMappable, ax: plt.Axes
    ) -> mpl.colorbar.Colorbar:
        """Plots the colorbar. Internal."""
        colorbar_ax = axes_grid1.make_axes_locatable(ax).append_axes(
            position=self._config['colorbar_position'],
            size=self._config['colorbar_size'],
            pad=self._config['colorbar_pad'],
        )
        position = self._config['colorbar_position']
        orien = 'vertical' if position in ('left', 'right') else 'horizontal'
        colorbar = cast(plt.Figure, ax.figure).colorbar(
            mappable, colorbar_ax, ax, orientation=orien, **self._config.get("colorbar_options", {})
        )
        colorbar_ax.tick_params(axis='y', direction='out')
        return colorbar

    def _write_annotations(
        self,
        centers_and_annot: List[Tuple[Point, Optional[str]]],
        collection: mpl_collections.Collection,
        ax: plt.Axes,
    ) -> None:
        """Writes annotations to the center of cells. Internal."""
        for (center, annotation), facecolor in zip(centers_and_annot, collection.get_facecolor()):
            # Calculate the center of the cell, assuming that it is a square
            # centered at (x=col, y=row).
            if not annotation:
                continue
            x, y = center
            face_luminance = vis_utils.relative_luminance(facecolor)  # type: ignore
            text_color = 'black' if face_luminance > 0.4 else 'white'
            text_kwargs: Dict[str, Any] = dict(color=text_color, ha="center", va="center")
            text_kwargs.update(self._config.get('annotation_text_kwargs', {}))
            ax.text(x, y, annotation, **text_kwargs)

    def _plot_on_axis(self, ax: plt.Axes) -> mpl_collections.Collection:
        # Step-1: Convert value_map to a list of polygons to plot.
        polygon_list = self._get_polygon_units()
        collection: mpl_collections.Collection = mpl_collections.PolyCollection(
            [c.polygon for c in polygon_list], **self._config.get('collection_options', {})
        )
        collection.set_clim(self._config.get('vmin'), self._config.get('vmax'))
        collection.set_array(np.array([c.value for c in polygon_list]))
        # Step-2: Plot the polygons
        ax.add_collection(collection)
        collection.update_scalarmappable()
        # Step-3: Write annotation texts
        if self._config.get('annotation_map') or self._config.get('annotation_format'):
            self._write_annotations([(c.center, c.annot) for c in polygon_list], collection, ax)
        ax.set(xlabel='column', ylabel='row')
        # Step-4: Draw colorbar if applicable
        if self._config.get('plot_colorbar'):
            self._plot_colorbar(collection, ax)
        # Step-5: Set min/max limits of x/y axis on the plot.
        rows = set([q.row for qubits in self._value_map.keys() for q in qubits])
        cols = set([q.col for qubits in self._value_map.keys() for q in qubits])
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        min_xtick = np.floor(min_col)
        max_xtick = np.ceil(max_col)
        ax.set_xticks(np.arange(min_xtick, max_xtick + 1))
        min_ytick = np.floor(min_row)
        max_ytick = np.ceil(max_row)
        ax.set_yticks(np.arange(min_ytick, max_ytick + 1))
        ax.set_xlim((min_xtick - 0.6, max_xtick + 0.6))
        ax.set_ylim((max_ytick + 0.6, min_ytick - 0.6))
        # Step-6: Set title
        if self._config.get("title"):
            ax.set_title(self._config["title"], fontweight='bold')
        return collection

    def plot(
        self, ax: Optional[plt.Axes] = None, **kwargs: Any
    ) -> Tuple[plt.Axes, mpl_collections.Collection]:
        """Plots the heatmap on the given Axes.
        Args:
            ax: the Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **kwargs: The optional keyword arguments are used to temporarily
                override the values present in the heatmap config. See
                __init__ for more details on the allowed arguments.
        Returns:
            A 2-tuple ``(ax, collection)``. ``ax`` is the `plt.Axes` that
            is plotted on. ``collection`` is the collection of paths drawn and filled.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax = cast(plt.Axes, ax)
        original_config = copy.deepcopy(self._config)
        self.update_config(**kwargs)
        collection = self._plot_on_axis(ax)
        if show_plot:
            fig.show()
        self._config = original_config
        return (ax, collection)


class TwoQubitInteractionHeatmap(Heatmap):
    """Visualizing interactions between neighboring qubits on a 2D grid."""

    def __init__(self, value_map: Mapping[QubitTuple, SupportsFloat], **kwargs):
        """Heatmap to display two-qubit interaction fidelities.

        Draw 2D qubit-qubit interaction heatmap with Matplotlib with arguments to configure the
        properties of the plot. The valid argument list includes all arguments of cirq.vis.Heatmap()
        plus the following.

        Args:
            value_map: A map from a qubit tuple location to a value.
            **kwargs: Optional kwargs including
                coupler_margin: float, default = 0.03
                coupler_width: float, default = 0.6
        """
        self._config: Dict[str, Any] = {"coupler_margin": 0.03, "coupler_width": 0.6}
        super().__init__(value_map, **kwargs)

    def _extra_valid_kwargs(self) -> List[str]:
        return ["coupler_margin", "coupler_width"]

    def _qubits_to_polygon(self, qubits: QubitTuple) -> Tuple[Polygon, Point]:
        coupler_margin = self._config["coupler_margin"]
        coupler_width = self._config["coupler_width"]
        cwidth = coupler_width / 2.0
        setback = 0.5 - cwidth
        row1, col1 = map(float, (qubits[0].row, qubits[0].col))
        row2, col2 = map(float, (qubits[1].row, qubits[1].col))
        if abs(row1 - row2) + abs(col1 - col2) != 1:
            raise ValueError(
                f"{qubits[0]}-{qubits[1]} is not supported because they are not nearest neighbors"
            )
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

        return (polygon, Point((col1 + col2) / 2.0, (row1 + row2) / 2.0))

    def plot(
        self, ax: Optional[plt.Axes] = None, **kwargs: Any
    ) -> Tuple[plt.Axes, mpl_collections.Collection]:
        """Plots the heatmap on the given Axes.
        Args:
            ax: the Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **kwargs: The optional keyword arguments are used to temporarily
                override the values present in the heatmap config. See
                __init__ for more details on the allowed arguments.
        Returns:
            A 2-tuple ``(ax, collection)``. ``ax`` is the `plt.Axes` that
            is plotted on. ``collection`` is the collection of paths drawn and filled.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax = cast(plt.Axes, ax)
        original_config = copy.deepcopy(self._config)
        self.update_config(**kwargs)
        qubits = set([q for qubits in self._value_map.keys() for q in qubits])
        Heatmap({q: 0.0 for q in qubits}).plot(
            ax=ax,
            collection_options={
                'cmap': 'binary',
                'linewidths': 2,
                'edgecolor': 'lightgrey',
                'linestyle': 'dashed',
            },
            plot_colorbar=False,
            annotation_format=None,
        )
        collection = self._plot_on_axis(ax)
        if show_plot:
            fig.show()
        self._config = original_config
        return (ax, collection)
