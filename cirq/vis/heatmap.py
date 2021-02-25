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
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, SupportsFloat, Tuple

import numpy as np
from matplotlib import collections as mcoll
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from matplotlib import collections as mpl_collections
from mpl_toolkits import axes_grid1
from matplotlib.patches import Polygon

from cirq._compat import deprecated
from cirq.devices import grid_qubit
from cirq.vis import vis_utils


QubitTuple = Tuple[grid_qubit.GridQubit, ...]
ValueMap = Dict[QubitTuple, SupportsFloat]
Point = Tuple[float, float]

Point = NamedTuple('Point', [('x', SupportsFloat), ('y', SupportsFloat)])
PolygonUnit = NamedTuple(
    'PolygonUnit',
    [('polygon', Polygon), ('center', Point), ('value', SupportsFloat), ('annot', str)],
)


class Heatmap:
    _config = None

    def __init__(self, value_map: Mapping[QubitTuple, SupportsFloat], **kwargs):
        self._value_map = value_map
        self._validate_kwargs(kwargs)
        self._config = {
            "plot_colorbar": True,
            "colorbar_position": "right",
            "colorbar_size": "5%",
            "colorbar_pad": "2%",
            "colormap": "viridis",
            "annotation_format": ".2g",
        }
        self._config.update(kwargs)

    def _validate_kwargs(self, kwargs):
        valid_colorbar_kwargs = [
            "plot_colorbar",
            "colorbar_position",
            "colorbar_size",
            "colorbar_pad",
            "colorbar_options",
        ]
        valid_colormap_kwargs = [
            "colormap",
            "vmin",
            "vmax",
        ]
        valid_heatmap_kwargs = [
            "title",
            "annotation_map",
            "annotation_text_kwargs",
            "annotation_format",
        ]
        valid_kwargs = valid_colorbar_kwargs + valid_colormap_kwargs + valid_heatmap_kwargs
        if any([k not in valid_kwargs for k in kwargs]):
            invalid_args = ", ".join([k for k in kwargs if k not in valid_kwargs])
            raise ValueError(f"Received invalid argument(s): {invalid_args}")

    def update_config(self, **kwargs):
        self.__validate_kwargs__(kwargs)
        self._config.update(kwargs)

    def _qubits_to_polygon(self, qubits: Tuple[grid_qubit.GridQid]) -> Tuple[Polygon, float]:
        print(qubits)
        qubit = qubits[0]
        print(qubit)
        x, y = map(float, (qubit.row, qubit.col))
        print(x, y)
        return (
            [
                (y - 0.5, x - 0.5),
                (y - 0.5, x + 0.5),
                (y + 0.5, x + 0.5),
                (y + 0.5, x - 0.5),
            ],
            (y, x),
        )

    def _get_annotation_value(self, key):
        if self._config['annotation_format']:
            print(key)
            return format(float(self._value_map[key][0]), self._config['annotation_format'])
        elif self._config.get('annotation_map', None):
            return self._config['annotation_map'].get(key, None)
        else:
            return None

    # @abc.abstractmethod
    def _get_polygon_units(self) -> List[PolygonUnit]:
        polygon_unit_list: List[PolygonUnit] = []
        for qubits, value in sorted(self._value_map.items()):
            polygon, center = self._qubits_to_polygon(qubits)
            print(qubits, polygon, center)
            polygon_unit_list.append(
                PolygonUnit(
                    polygon=polygon,
                    center=center,
                    value=value[0],
                    annot=self._get_annotation_value(qubits),
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
        colorbar = ax.figure.colorbar(
            mappable, colorbar_ax, ax, orientation=orien, **self._config.get("colorbar_options", {})
        )
        colorbar_ax.tick_params(axis='y', direction='out')
        return colorbar

    def _write_annotations(
        self,
        centers_and_annot: List[Tuple[Point, str]],
        collection: mpl_collections.Collection,
        ax: plt.Axes,
    ) -> None:
        """Writes annotations to the center of cells. Internal."""
        for (center, annotation), facecolor in zip(centers_and_annot, collection.get_facecolors()):
            # Calculate the center of the cell, assuming that it is a square
            # centered at (x=col, y=row).
            if not annotation:
                continue
            x, y = center
            face_luminance = vis_utils.relative_luminance(facecolor)
            text_color = 'black' if face_luminance > 0.4 else 'white'
            text_kwargs = dict(color=text_color, ha="center", va="center")
            text_kwargs.update(self._config.get('annotation_text_kwargs', {}))
            ax.text(x, y, annotation, **text_kwargs)

    def _plot_on_axis(self, ax: Optional[plt.Axes], **collection_options: Any):
        # Step-1: Convert value_map to a list of polygons to plot.
        polygon_list = self._get_polygon_units()
        collection = mcoll.PolyCollection(
            [c.polygon for c in polygon_list], cmap=self._config['colormap'], **collection_options
        )
        collection.set_clim(self._config.get('vmin', None), self._config.get('vmax', None))
        collection.set_array(np.array([c.value for c in polygon_list]))
        # Step-2: Plot the polygons
        ax.add_collection(collection)
        collection.update_scalarmappable()
        # Step-3: Write annotation texts
        if self._config.get('annotation_map', None) or self._config.get('annotation_format', None):
            self._write_annotations([(c.center, c.annot) for c in polygon_list], collection, ax)
        ax.set(xlabel='column', ylabel='row')
        # Step-4: Draw colorbar if applicable
        if self._config.get('plot_colorbar', None):
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
        if self._config.get("title", None):
            ax.set_title(self._config["title"], fontweight='bold')

    def plot(self, ax: Optional[plt.Axes] = None, **collection_options: Any):
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
        self._plot_on_axis(ax, **collection_options)
        if show_plot:
            fig.show()


class TwoQubitInteractionHeatmap(Heatmap):
    def _qubits_to_polygon(self, qubits: Tuple[grid_qubit.GridQid]) -> Tuple[Polygon, float]:
        coupler_margin: float = 0.03
        coupler_width: float = 0.6
        cwidth = coupler_width / 2.0
        setback = 0.5 - cwidth
        row1, col1 = map(float, (qubits[0].row, qubits[0].col))
        row2, col2 = map(float, (qubits[1].row, qubits[1].col))
        print(row1, col1, row2, col2, coupler_margin)
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

        return (polygon, ((col1 + col2) / 2.0, (row1 + row2) / 2.0))

    def plot(self, ax: Optional[plt.Axes] = None, **collection_options: Any):
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
        qubits = set([q for qubits in self._value_map.keys() for q in qubits])
        Heatmap(
            {(q,): [0.0] for q in qubits},
            colormap='binary',
            plot_colorbar=False,
            annotation_format=None,
        ).plot(ax=ax, linewidths=2, edgecolor='lightgrey', linestyle='dashed')
        self._plot_on_axis(ax, **collection_options)
        if show_plot:
            fig.show()
