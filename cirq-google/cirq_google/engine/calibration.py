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
"""Calibration wrapper for calibrations returned from the Quantum Engine."""

from collections import abc, defaultdict
import datetime
from itertools import cycle
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import google.protobuf.json_format as json_format

import cirq
from cirq_google.api import v2


# Calibration Metric types
METRIC_KEY = Tuple[Union[cirq.GridQubit, str], ...]
METRIC_VALUE = List[Union[str, int, float]]
METRIC_DICT = Dict[METRIC_KEY, METRIC_VALUE]
ALL_METRICS = Dict[str, METRIC_DICT]


class Calibration(abc.Mapping):
    """A convenience wrapper for calibrations that acts like a dictionary.

    Calibrations act as dictionaries whose keys are the names of the metric, and whose values are
    the metric values.  The metric values themselves are represented as a dictionary. These metric
    value dictionaries have keys that are tuples of `cirq.GridQubit`s and values that are lists of
    the metric values for those qubits. If a metric acts globally and is attached to no specified
    number of qubits, the map will be from the empty tuple to the metrics values.

    Calibrations act just like a python dictionary. For example you can get
    a list of all of the metric names using

        `calibration.keys()`

    and query a single value by looking up the name by index:

        `calibration['t1']`

    This class can be instantiated either from a `MetricsSnapshot` proto
    or from a dictionary of metric values.

    Attributes:
        timestamp: The time that this calibration was run, in milliseconds since
            the epoch.
    """

    def __init__(
        self,
        calibration: v2.metrics_pb2.MetricsSnapshot = v2.metrics_pb2.MetricsSnapshot(),
        metrics: Optional[ALL_METRICS] = None,
    ) -> None:
        self.timestamp = calibration.timestamp_ms
        if metrics is None:
            self._metric_dict = self._compute_metric_dict(calibration.metrics)
        else:
            self._metric_dict = metrics

    def _compute_metric_dict(self, metrics: v2.metrics_pb2.MetricsSnapshot) -> ALL_METRICS:
        results: ALL_METRICS = defaultdict(dict)
        for metric in metrics:
            name = metric.name
            # Flatten the values to a list, removing keys containing type names
            # (e.g. proto version of each value is {<type>: value}).
            flat_values = [getattr(v, v.WhichOneof('val')) for v in metric.values]
            if metric.targets:
                qubits = tuple(self.str_to_key(t) for t in metric.targets)
                results[name][qubits] = flat_values
            else:
                assert len(results[name]) == 0, (
                    'Only one metric of a given name can have no targets. '
                    'Found multiple for key {}'.format(name)
                )
                results[name][()] = flat_values
        return results

    def __getitem__(self, key: str) -> METRIC_DICT:
        """Supports getting calibrations by index.

        Calibration may be accessed by key:

            `calibration['t1']`.

        This returns a map from tuples of `cirq.GridQubit`s to a list of the
        values of the metric. If there are no targets, the only key will only
        be an empty tuple.
        """
        if not isinstance(key, str):
            raise TypeError(f'Calibration metrics only have string keys. Key was {key}')
        if key not in self._metric_dict:
            raise KeyError(f'Metric named {key} not in calibration')
        return self._metric_dict[key]

    def __iter__(self) -> Iterator:
        return iter(self._metric_dict)

    def __len__(self) -> int:
        return len(self._metric_dict)

    def __str__(self) -> str:
        return f'Calibration(keys={list(sorted(self.keys()))})'

    def __repr__(self) -> str:
        return f'cirq_google.Calibration(metrics={dict(self._metric_dict)!r})'

    def to_proto(self) -> v2.metrics_pb2.MetricsSnapshot:
        """Reconstruct the protobuf message represented by this class."""
        proto = v2.metrics_pb2.MetricsSnapshot()
        for key in self._metric_dict:
            for targets, value_list in self._metric_dict[key].items():
                current_metric = proto.metrics.add()
                current_metric.name = key
                current_metric.targets.extend(
                    [
                        target if isinstance(target, str) else v2.qubit_to_proto_id(target)
                        for target in targets
                    ]
                )
                for value in value_list:
                    current_value = current_metric.values.add()
                    if isinstance(value, float):
                        current_value.double_val = value
                    elif isinstance(value, int):
                        current_value.int64_val = value
                    elif isinstance(value, str):
                        current_value.str_val = value
                    else:
                        raise ValueError(
                            f'Unsupported metric value {value}. '
                            'Must be int, float, or str to '
                            'convert to proto.'
                        )
        return proto

    @classmethod
    def _from_json_dict_(cls, metrics: str, **kwargs) -> 'Calibration':
        """Magic method for the JSON serialization protocol."""
        metric_proto = v2.metrics_pb2.MetricsSnapshot()
        return cls(json_format.ParseDict(metrics, metric_proto))

    def _json_dict_(self) -> Dict[str, Any]:
        """Magic method for the JSON serialization protocol."""
        return {'metrics': json_format.MessageToDict(self.to_proto())}

    def timestamp_str(self, tz: Optional[datetime.tzinfo] = None, timespec: str = 'auto') -> str:
        """Return a string for the calibration timestamp.

        Args:
            tz: The timezone for the string. If None, the method uses the
                platform's local timezone.
            timespec: See datetime.isoformat for valid values.

        Returns:
            The string in ISO 8601 format YYYY-MM-DDTHH:MM:SS.ffffff.
        """
        dt = datetime.datetime.fromtimestamp(self.timestamp / 1000, tz)
        dt += datetime.timedelta(microseconds=self.timestamp % 1000000)
        return dt.isoformat(sep=' ', timespec=timespec)

    def str_to_key(self, target: str) -> Union[cirq.GridQubit, str]:
        """Turns a string into a calibration key.

        Attempts to parse it as a GridQubit.  If this fails,
        returns the string itself.
        """
        try:
            return v2.grid_qubit_from_proto_id(target)
        except ValueError:
            return target

    @staticmethod
    def key_to_qubit(target: METRIC_KEY) -> cirq.GridQubit:
        """Returns a single qubit from a metric key.

        Raises:
           ValueError: If the metric key is a tuple of strings.
        """
        if target and isinstance(target, tuple) and isinstance(target[0], cirq.GridQubit):
            return target[0]
        raise ValueError(f'The metric target {target} was not a tuple of qubits')

    @staticmethod
    def key_to_qubits(target: METRIC_KEY) -> Tuple[cirq.GridQubit, ...]:
        """Returns a tuple of qubits from a metric key.

        Raises:
           ValueError: If the metric key is a tuple of strings.
        """
        if (
            target
            and isinstance(target, tuple)
            and all(isinstance(q, cirq.GridQubit) for q in target)
        ):
            return target  # type: ignore
        raise ValueError(f'The metric target {target} was not a tuple of grid qubits.')

    @staticmethod
    def value_to_float(value: METRIC_VALUE) -> float:
        """Returns a single float from a metric value.

        Metric values can be a list of strings, ints, or floats.
        However, the typical case is that they are a single float.
        This converts the metric value to a single float.

        If the metric value has multiple values, only the first will be
        returned.  If the value is empty or a string that cannot be converted,
        this function will raise a ValueError.
        """
        if not value:
            raise ValueError('Metric Value was empty')
        return float(value[0])

    def heatmap(self, key: str) -> cirq.Heatmap:
        """Return a heatmap for metrics that target single qubits.

        Args:
            key: The metric key to return a heatmap for.

        Returns:
            A `cirq.Heatmap` for the metric.

        Raises:
            ValueError: If the heatmap is not for one/two qubits or the metric
                values are not single floats.
        """
        metrics = self[key]
        if not all(len(k) == 1 for k in metrics.values()):
            raise ValueError(
                'Heatmaps are only supported if all values in a metric are single metric values.'
                + f'{key} has metric values {metrics.values()}'
            )
        value_map = {self.key_to_qubits(k): self.value_to_float(v) for k, v in metrics.items()}
        if all(len(k) == 1 for k in value_map.keys()):
            return cirq.Heatmap(value_map, title=key.replace('_', ' ').title())
        elif all(len(k) == 2 for k in value_map.keys()):
            return cirq.TwoQubitInteractionHeatmap(value_map, title=key.replace('_', ' ').title())
        raise ValueError(
            'Heatmaps are only supported if all the targets in a metric are one or two qubits.'
            + f'{key} has target qubits {value_map.keys()}'
        )

    def plot_histograms(
        self,
        keys: Sequence[str],
        ax: Optional[plt.Axes] = None,
        *,
        labels: Optional[Sequence[str]] = None,
    ) -> plt.Axes:
        """Plots integrated histograms of metric values corresponding to keys

        Args:
            keys: List of metric keys for which an integrated histogram should be plot
            ax: The axis to plot on. If None, we generate one.
            labels: Optional label that will be used in the legend.

        Returns:
            The axis that was plotted on.

        Raises:
            ValueError: If the metric values are not single floats.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1)

        if isinstance(keys, str):
            keys = [keys]
        if not labels:
            labels = keys
        colors = ['b', 'r', 'k', 'g', 'c', 'm']
        for key, label, color in zip(keys, labels, cycle(colors)):
            metrics = self[key]
            if not all(len(k) == 1 for k in metrics.values()):
                raise ValueError(
                    'Histograms are only supported if all values in a metric '
                    + 'are single metric values.'
                    + f'{key} has metric values {metrics.values()}'
                )
            cirq.integrated_histogram(
                [self.value_to_float(v) for v in metrics.values()],
                ax,
                label=label,
                color=color,
                title=key.replace('_', ' ').title(),
            )
        if show_plot:
            fig.show()

        return ax

    def plot(
        self, key: str, fig: Optional[mpl.figure.Figure] = None
    ) -> Tuple[mpl.figure.Figure, List[plt.Axes]]:
        """Plots a heatmap and an integrated histogram for the given key.

        Args:
            key: The metric key to plot a heatmap and integrated histogram for.
            fig: The figure to plot on. If none, we generate one.

        Returns:
            The figure and list of axis that was plotted on.

        Raises:
            ValueError if the key is not for one/two qubits metric or the metric
            values are not single floats.
        """
        show_plot = not fig
        if not fig:
            fig = plt.figure()
        axs = fig.subplots(1, 2)
        self.heatmap(key).plot(axs[0])
        self.plot_histograms(key, axs[1])
        if show_plot:
            fig.show()
        return fig, axs
