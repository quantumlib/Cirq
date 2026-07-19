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

from __future__ import annotations

import datetime
from collections import abc, defaultdict
from collections.abc import Iterator, Sequence
from itertools import cycle
from typing import Any, cast

import google.protobuf.json_format as json_format
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import cirq
from cirq_google.api import v2

# Calibration Metric types
METRIC_KEY = tuple[cirq.GridQubit | str, ...]
METRIC_VALUE = list[str | int | float]
METRIC_DICT = dict[METRIC_KEY, METRIC_VALUE]
ALL_METRICS = dict[str, METRIC_DICT]


# Reference values corresponding to the spec sheet (https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf)
SQ_RB_PAULI_ERROR_REF = [
    0.0031868597369656415,
    0.00036178648393073165,
    0.0004887606077522144,
    0.0006040772358906765,
    0.000322892948463549,
    0.0008148057935692399,
    0.00025008948010105114,
    0.00040461714515024894,
    0.00029825950873643703,
    0.0002444904854654595,
    0.0003072893550247202,
    0.0007121792060773291,
    0.0006420648617254665,
    0.0002205156296834765,
    0.00040814087101267993,
    0.0002290054486225046,
    0.00041635971284453843,
    0.000980801239580692,
    0.00024991367800134,
    0.0005167603584379787,
    0.00023162546953722774,
    0.00024410898739907205,
    0.00024847322196197563,
    0.0003657513483480146,
    0.001126706267556954,
    0.0005985513203474457,
    0.00031169720771931364,
    0.00035521831464618847,
    0.0008581870397833535,
    0.00047340040624557767,
    0.0003817465481152915,
    0.00025163798963007133,
    0.0004081852654946039,
    0.0003160177573825196,
    0.0005004801405303627,
    0.0009669499192906938,
    0.00048369987290503014,
    0.0002757695972772922,
    0.000417907686138419,
    0.00032332194302492345,
    0.001135350986707473,
    0.00027063219619549717,
    0.00024252902473112514,
    0.0003786980230370829,
    0.0003114153044067036,
    0.0004958878607125294,
    0.00037122455461705006,
    0.0005491245410546852,
    0.0007041440201674409,
    0.000367081288938087,
    0.001329271492369083,
    0.00024013472571149386,
    0.0006659649610605001,
    0.00060854066853816,
    0.0004940313795656348,
    0.000247685289105265,
    0.00026768941194046936,
    0.0007438317790320892,
    0.00038831860511864824,
    0.0005399230524786025,
    0.0006201361224233981,
    0.0003485719164970924,
    0.000345307853954091,
    0.0005434698922620607,
    0.00029821613689107185,
    0.0005755745135990475,
    0.0007363633893783805,
    0.0002572029993438696,
    0.0005591360020817115,
    0.0027431201466925625,
    0.00040254898293826114,
    0.00025361055502309826,
    0.0002510999767969535,
    0.00034263884334106987,
    0.00024458533945670435,
    0.00042536358215208847,
    0.0006891049966780693,
    0.00043507025189740145,
    0.00035562518476245364,
    0.0005701709800401966,
    0.001117660880557536,
    0.0003232043677245111,
    0.0002444290196206289,
    0.00034729044317521884,
    0.0004996200924115568,
    0.0011857737305895344,
    0.0005129480743508164,
    0.0005361239730846123,
    0.0009356269600868106,
    0.00031878692565162625,
    0.0002634941156226256,
    0.0006557446389774402,
    0.0006747823028516176,
    0.0007575555540696444,
    0.00029626398096466167,
    0.00036280234094998587,
    0.0002973145928892029,
]
CZ_INFERRED_GATE_ERROR_PAULI_REF = [
    0.004821403840952025,
    0.00674773852968491,
    0.0030733471923675287,
    0.007665937691956337,
    0.005293062513656958,
    0.003234970683726468,
    0.004810767371198413,
    0.0031536649101319633,
    0.0031387072149252626,
    0.004274700149341226,
    0.004611188545745673,
    0.006130416772795778,
    0.0009834798308460888,
    0.0036491899700493574,
    0.0035114308265759425,
    0.0033527284061338203,
    0.007226346452381231,
    0.0021919511487526744,
    0.003991523195470102,
    0.003514936151504597,
    0.006353741730719918,
    0.0025782454054173315,
    0.0034428202674625583,
    0.003909178416045163,
    0.0043747426258812505,
    0.0036949816628364593,
    0.002278459465631065,
    0.0019129896949395334,
    0.010621765229495393,
    0.0005634627665435099,
    0.007577350771780442,
    0.005760459151642888,
    0.002429528942384622,
    0.00593328905444699,
    0.003395561070163901,
    0.011568702257493922,
    0.003917415461466428,
    0.0022757938190133553,
    0.005263578760671232,
    0.0035443257402501677,
    0.0017776894437276924,
    0.006305948072165592,
    0.00557330747135952,
    0.0033097318182283286,
    0.003313795074763315,
    0.004155314653255526,
    0.0038880209202285868,
    0.004309032011767834,
    0.0016576729549275596,
    0.004574258932775743,
    0.0034095671113806153,
    0.011128771897815047,
    0.003404182960924408,
    0.002609372022474299,
    0.002330901726564881,
    0.003209977606147274,
    0.002284952940204109,
    0.010376495742512454,
    0.008563307205943123,
    0.0020823836064906565,
    0.002983215894429292,
    0.007978794915165753,
    0.0018062848377727383,
    0.0023857992661561675,
    0.004259647398728687,
    0.0037054711237010513,
    0.005621218474436904,
    0.006529354330762176,
    0.002412283409525054,
    0.005412767358030264,
    0.006302366384870299,
    0.002252299109956335,
    0.00626719910769704,
    0.005745221815108177,
    0.00810704754867296,
    0.0035800820588209603,
    0.0016224333871928467,
    0.0029249397654772247,
    0.0015254721188271564,
    0.003392059697547578,
    0.00415763203072482,
    0.002232110883922528,
    0.00287978084033913,
    0.003638893666900525,
    0.00325903787618792,
    0.005139488792650394,
    0.0025369081104610267,
    0.0020392319291625063,
    0.005612427077702972,
    0.0014312554378833817,
    0.002200982197255452,
    0.002245461857785548,
    0.0030519528596392037,
    0.002285729076062608,
    0.003107737954316063,
    0.004369538272267354,
    0.007580044979224876,
    0.0015629766290153543,
    0.009506163458856617,
    0.003943005274524049,
    0.002576686409651724,
    0.0015839094385694183,
    0.005310405222690821,
    0.0015211584091653907,
    0.0024757356357620747,
    0.00757305351869457,
    0.0045889715056701,
    0.0032594142411813617,
    0.004768656657637531,
    0.007509524539830691,
    0.004786168950359736,
    0.0022631576265771947,
    0.001790132666171576,
    0.01210644605111992,
    0.003411358891572229,
    0.00314305550276308,
    0.001955104705742561,
    0.002563810312913202,
    0.0022627659747841145,
    0.0026642893172993884,
    0.007549845421772408,
    0.0021500860553417406,
    0.004703601558181773,
    0.005469576259666126,
    0.005705804301337873,
    0.003307887239110234,
    0.0025192589600889215,
    0.0027554794113542405,
    0.0015175685735338215,
    0.0035979465588229648,
    0.010385772877817417,
    0.0032160830178715877,
    0.0034996160134964996,
    0.002798540506943094,
    0.002106854333924109,
    0.0026787735966211576,
    0.0029174997307627668,
    0.005706671758001457,
    0.002932049385647624,
    0.006582965538765237,
    0.0031377730922498015,
    0.0031863307440048103,
    0.002666337040078022,
    0.002680153300369907,
    0.007954942509081735,
    0.0043074083203261185,
    0.0025619967221566717,
    0.0023231681414369668,
    0.002042269973138687,
    0.0009630382607278207,
    0.0029262181323464614,
    0.0014598428284962398,
    0.0036963267182626644,
    0.00426417969466391,
    0.005886185257380286,
    0.0035880126921097877,
    0.0050580478573374055,
    0.002072906149301508,
    0.0058378592923987935,
    0.0030466048850733007,
    0.004881489408415408,
    0.0008516980247764636,
    0.004202922447359876,
    0.00449487596067874,
    0.008283046924843322,
    0.0038924003568547444,
    0.003763330627765925,
]
READOUT_REF = [
    0.007789999999999982,
    0.005889999999999975,
    0.006329999999999987,
    0.004929999999999976,
    0.008369999999999982,
    0.008589999999999985,
    0.00541999999999998,
    0.00839999999999998,
    0.006849999999999981,
    0.006069999999999977,
    0.011279999999999984,
    0.008019999999999975,
    0.008049999999999984,
    0.014079999999999978,
    0.008239999999999978,
    0.007029999999999976,
    0.007319999999999976,
    0.006919999999999982,
    0.007859999999999982,
    0.005349999999999972,
    0.00840999999999998,
    0.00721999999999998,
    0.005709999999999979,
    0.010549999999999981,
    0.008189999999999985,
    0.008149999999999984,
    0.007899999999999971,
    0.00686999999999998,
    0.0061499999999999845,
    0.009009999999999978,
    0.005739999999999973,
    0.0060399999999999725,
    0.011239999999999972,
    0.006119999999999981,
    0.006089999999999971,
    0.008069999999999976,
    0.0077199999999999665,
    0.006059999999999979,
    0.005369999999999975,
    0.012079999999999976,
    0.00790999999999998,
    0.006709999999999976,
    0.00560999999999998,
    0.0070599999999999795,
    0.005239999999999976,
    0.006499999999999979,
    0.013079999999999984,
    0.010809999999999983,
]


KEY_TO_REF = {
    'sq_rb_pauli_error': SQ_RB_PAULI_ERROR_REF,
    'cz_inferred_gate_error_pauli': CZ_INFERRED_GATE_ERROR_PAULI_REF,
    'readout': READOUT_REF,
}


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
        metrics: ALL_METRICS | None = None,
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
                    f'Found multiple for key {name}'
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
        return f'Calibration(keys={sorted(self.keys())})'

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
    def _from_json_dict_(cls, metrics: str, **kwargs) -> Calibration:
        """Magic method for the JSON serialization protocol."""
        metric_proto = v2.metrics_pb2.MetricsSnapshot()
        return cls(json_format.ParseDict(metrics, metric_proto))

    def _json_dict_(self) -> dict[str, Any]:
        """Magic method for the JSON serialization protocol."""
        return {'metrics': json_format.MessageToDict(self.to_proto())}

    def timestamp_str(self, tz: datetime.tzinfo | None = None, timespec: str = 'auto') -> str:
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

    def str_to_key(self, target: str) -> cirq.GridQubit | str:
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
    def key_to_qubits(target: METRIC_KEY) -> tuple[cirq.GridQubit, ...]:
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
        if (
            key == 'readout'
            and 'readout' not in self
            and 'zero_error' in self
            and 'one_error' in self
        ):
            metrics = {
                q: (
                    (
                        self.value_to_float(self['zero_error'][q])
                        + self.value_to_float(self['one_error'][q])
                    )
                    / 2,
                )
                for q in self['zero_error']
            }
        else:
            metrics = self[key]
        if not all(len(k) == 1 for k in metrics.values()):
            raise ValueError(
                'Heatmaps are only supported if all values in a metric are single metric values.'
                f'{key} has metric values {metrics.values()}'
            )
        value_map = {self.key_to_qubits(k): self.value_to_float(v) for k, v in metrics.items()}
        if all(len(k) == 1 for k in value_map.keys()):
            return cirq.Heatmap(value_map, title=key.replace('_', ' ').title())
        elif all(len(k) == 2 for k in value_map.keys()):
            return cirq.TwoQubitInteractionHeatmap(value_map, title=key.replace('_', ' ').title())
        raise ValueError(
            'Heatmaps are only supported if all the targets in a metric are one or two qubits.'
            f'{key} has target qubits {value_map.keys()}'
        )

    def plot_histograms(
        self,
        keys: Sequence[str],
        ax: plt.Axes | None = None,
        *,
        labels: Sequence[str] | None = None,
        show_ref: bool = True,
    ) -> plt.Axes:
        """Plots integrated histograms of metric values corresponding to keys

        Args:
            keys: List of metric keys for which an integrated histogram should be plot
            ax: The axis to plot on. If None, we generate one.
            labels: Optional label that will be used in the legend.
            show_ref: Whether to show the reference from the spec sheet.

        Returns:
            The axis that was plotted on.

        Raises:
            ValueError: If the metric values are not single floats.
        """
        show_plot = not ax
        if ax is None:
            fig, ax = plt.subplots(1, 1, facecolor='white', dpi=150, figsize=(10, 5))

        if isinstance(keys, str):
            keys = [keys]
        if not labels:
            labels = keys
        colors = ['b', 'r', 'k', 'g', 'c', 'm']
        for key, label, color in zip(keys, labels, cycle(colors)):
            if (
                key == 'readout'
                and 'readout' not in self
                and 'zero_error' in self
                and 'one_error' in self
            ):
                data_to_plot = [
                    (
                        self.value_to_float(self['zero_error'][q])
                        + self.value_to_float(self['one_error'][q])
                    )
                    / 2
                    for q in self['zero_error']
                ]
            else:
                metrics = self[key]
                if not all(len(k) == 1 for k in metrics.values()):
                    raise ValueError(
                        'Histograms are only supported if all values in a metric '
                        'are single metric values.'
                        f'{key} has metric values {metrics.values()}'
                    )
                data_to_plot = [self.value_to_float(v) for v in metrics.values()]
            cirq.integrated_histogram(
                data_to_plot,
                ax,
                label=label,
                color=color,
                title=key.replace('_', ' ').title(),
                median_line=False,
            )

            if show_ref and key in KEY_TO_REF:
                cirq.integrated_histogram(
                    KEY_TO_REF[key], ax, color=color, alpha=0.3, label='ref', median_line=False
                )

        ax.set_title('')
        ax.set_ylabel('Percentile')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlabel('Error rate')
        ax.tick_params(direction='in', which='both', top=True, right=True)

        if show_plot:
            fig.show()

        return ax

    def plot(
        self,
        key: str,
        fig: mpl.figure.Figure | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> tuple[mpl.figure.Figure, list[plt.Axes]]:
        """Plots a heatmap and an integrated histogram for the given key.

        Args:
            key: The metric key to plot a heatmap and integrated histogram for.
            fig: The figure to plot on. If none, we generate one.
            vmin: Min for the heatmap.
            vmax: Max for the heatmap.

        Returns:
            The figure and list of axis that was plotted on.

        Raises:
            ValueError if the key is not for one/two qubits metric or the metric
            values are not single floats.
        """
        show_plot = not fig
        if fig is None:
            fig = plt.figure(figsize=(8, 4), dpi=150)
        axs = cast(list[plt.Axes], fig.subplots(1, 2, width_ratios=(2, 1)))
        self.heatmap(key).plot(
            axs[0],
            annotation_text_kwargs={'fontsize': 5},
            annotation_format='.2%',
            vmin=vmin,
            vmax=vmax,
        )
        self.plot_histograms(key, axs[1], labels=['current'])
        axs[0].set_xlabel('')
        axs[0].set_ylabel('')
        axs[0].tick_params(direction='in', top=True, right=True)
        axs[0].set_title('')
        axs[1].set_xlabel(key)
        fig.tight_layout()
        for ax in fig.axes:
            if hasattr(ax, '_colorbar'):
                cbar = ax._colorbar
                cbar.ax.tick_params(direction='in')
                break

        if show_plot:
            fig.show()
        return fig, axs
