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

from typing import Any, Dict, Iterator, Optional, Tuple, TYPE_CHECKING

from cirq import devices, vis
from cirq.google.api import v2

if TYPE_CHECKING:
    import cirq


class Calibration(abc.Mapping):
    """A convenience wrapper for calibrations that acts like a dictionary.

    Calibrations act as dictionaries whose keys are the names of the metric,
    and whose values are the metric values.  The metric values themselves are
    represented as a dictionary.  These metric value dictionaries have
    keys that are tuples of `cirq.GridQubit`s and values that are lists of the
    metric values for those qubits. If a metric acts globally and is attached
    to no specified number of qubits, the map will be from the empty tuple
    to the metrics values.

    Calibrations act just like a python dictionary. For example you can get
    a list of all of the metric names using

        `calibration.keys()`

    and query a single value by looking up the name by index:

        `calibration['t1']`

    Attributes:
        timestamp: The time that this calibration was run, in milliseconds since
            the epoch.
    """

    def __init__(self, calibration: v2.metrics_pb2.MetricsSnapshot) -> None:
        self.timestamp = calibration.timestamp_ms
        self._metric_dict = self._compute_metric_dict(calibration.metrics)

    def _compute_metric_dict(
            self, metrics: v2.metrics_pb2.MetricsSnapshot
    ) -> Dict[str, Dict[Tuple['cirq.GridQubit', ...], Any]]:
        results: Dict[str, Dict[Tuple[devices.
                                      GridQubit, ...], Any]] = defaultdict(dict)
        for metric in metrics:
            name = metric.name
            # Flatten the values to a list, removing keys containing type names
            # (e.g. proto version of each value is {<type>: value}).
            flat_values = [
                getattr(v, v.WhichOneof('val')) for v in metric.values
            ]
            if metric.targets:
                qubits = tuple(
                    v2.grid_qubit_from_proto_id(t) for t in metric.targets)
                results[name][qubits] = flat_values
            else:
                assert len(results[name]) == 0, (
                    'Only one metric of a given name can have no targets. '
                    'Found multiple for key {}'.format(name))
                results[name][()] = flat_values
        return results

    def __getitem__(self, key: str) -> Dict[Tuple['cirq.GridQubit', ...], Any]:
        """Supports getting calibrations by index.

        Calibration may be accessed by key:

            `calibration['t1']`.

        This returns a map from tuples of `cirq.GridQubit`s to a list of the
        values of the metric. If there are no targets, the only key will only
        be an empty tuple.
        """
        if not isinstance(key, str):
            raise TypeError(
                'Calibration metrics only have string keys. Key was {}'.format(
                    key))
        if key not in self._metric_dict:
            raise KeyError('Metric named {} not in calibration'.format(key))
        return self._metric_dict[key]

    def __iter__(self) -> Iterator:
        return iter(self._metric_dict)

    def __len__(self) -> int:
        return len(self._metric_dict)

    def __str__(self) -> str:

        return 'Calibration(keys={})'.format(list(sorted(self.keys())))

    def timestamp_str(self,
                      tz: Optional[datetime.tzinfo] = None,
                      timespec: str = 'auto') -> str:
        """Return a string for the calibration timestamp.

        Args:
            tz: The timezone for the string. If None, the method uses the
                platform's local date and time.
            timespec: See datetime.isoformat for valid values.

        Returns:
            The string in ISO 8601 format YYYY-MM-DDTHH:MM:SS.ffffff.
        """
        dt = datetime.datetime.fromtimestamp(self.timestamp / 1000, tz)
        dt += datetime.timedelta(microseconds=self.timestamp % 1000000)
        return dt.isoformat(sep=' ', timespec=timespec)

    def heatmap(self, key: str) -> vis.Heatmap:
        """Return a heatmap for metrics that target single qubits.

        Args:
            key: The metric key to return a heatmap for.

        Returns:
            A `cirq.Heatmap` for the metric.

        Raises:
            AssertionError if the heatmap is not for single qubits or the metric
            values are not single floats.
        """
        metrics = self[key]
        assert all(len(k) == 1 for k in metrics.keys()), (
            'Heatmaps are only supported if all the targets in a metric'
            ' are single qubits.')
        assert all(len(k) == 1 for k in metrics.values()), (
            'Heatmaps are only supported if all the values in a metric'
            ' are single metric values.')
        value_map = {qubit: value for (qubit,), (value,) in metrics.items()}
        return vis.Heatmap(value_map)
