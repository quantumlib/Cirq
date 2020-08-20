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

from typing import (cast, Dict, Iterable, Iterator, List, NamedTuple, Optional,
                    Set, TYPE_CHECKING)

from collections import OrderedDict
import numpy as np

from cirq.google.api import v2
from cirq.google.api.v2 import result_pb2
from cirq import circuits
from cirq import devices
from cirq import ops
from cirq import study

if TYPE_CHECKING:
    import cirq


class MeasureInfo(
        NamedTuple('MeasureInfo', [
            ('key', str),
            ('qubits', List['cirq.GridQubit']),
            ('slot', int),
            ('invert_mask', List[bool]),
        ])):
    """Extra info about a single measurement within a circuit.

    Attributes:
        key: String identifying this measurement.
        qubits: List of measured qubits, in order.
        slot: The location of this measurement within the program. For circuits,
            this is just the moment index; for schedules it is the start time
            of the measurement. This is used internally when scheduling on
            hardware so that we can combine measurements that occupy the same
            slot.
        invert_mask: a list of booleans describing whether the results should
            be flipped for each of the qubits in the qubits field.
    """


def find_measurements(program: 'cirq.Circuit') -> List[MeasureInfo]:
    """Find measurements in the given program (circuit).

    Returns:
        List of Measurement objects with named measurements in this program.
    """
    measurements: List[MeasureInfo] = []
    keys: Set[str] = set()

    if isinstance(program, circuits.Circuit):
        measure_iter = _circuit_measurements(program)
    else:
        raise NotImplementedError(f'Unrecognized program type: {type(program)}')

    for m in measure_iter:
        if m.key in keys:
            raise ValueError('Duplicate measurement key: {}'.format(m.key))
        keys.add(m.key)
        measurements.append(m)

    return measurements


def _circuit_measurements(circuit: 'cirq.Circuit') -> Iterator[MeasureInfo]:
    for i, moment in enumerate(circuit):
        for op in moment:
            if isinstance(op.gate, ops.MeasurementGate):
                yield MeasureInfo(key=op.gate.key,
                                  qubits=_grid_qubits(op),
                                  slot=i,
                                  invert_mask=list(op.gate.full_invert_mask()))


def _grid_qubits(op: 'cirq.Operation') -> List['cirq.GridQubit']:
    if not all(isinstance(q, devices.GridQubit) for q in op.qubits):
        raise ValueError('Expected GridQubits: {}'.format(op.qubits))
    return cast(List['cirq.GridQubit'], list(op.qubits))


def pack_bits(bits: np.ndarray) -> bytes:
    """Pack bits given as a numpy array of bools into bytes."""
    # Pad length to multiple of 8 if needed.
    pad = -len(bits) % 8
    if pad:
        bits = np.pad(bits, (0, pad), 'constant')

    # Pack in little-endian bit order.
    bits = bits.reshape((-1, 8))[:, ::-1]
    byte_arr = np.packbits(bits, axis=1).reshape(-1)

    return byte_arr.tobytes()


def unpack_bits(data: bytes, repetitions: int) -> np.ndarray:
    """Unpack bits from a byte array into numpy array of bools."""
    byte_arr = np.frombuffer(data, dtype='uint8').reshape((len(data), 1))
    bits = np.unpackbits(byte_arr, axis=1)[:, ::-1].reshape(-1).astype(bool)
    return bits[:repetitions]


def results_to_proto(
        trial_sweeps: Iterable[Iterable[study.TrialResult]],
        measurements: List[MeasureInfo],
        *,
        out: Optional[result_pb2.Result] = None,
) -> result_pb2.Result:
    """Converts trial results from multiple sweeps to v2 protobuf message.

    Args:
        trial_sweeps: Iterable over sweeps and then over trial results within
            each sweep.
        measurements: List of info about measurements in the program.
        out: Optional message to populate. If not given, create a new message.
    """
    if out is None:
        out = result_pb2.Result()
    for trial_sweep in trial_sweeps:
        sweep_result = out.sweep_results.add()
        for i, trial_result in enumerate(trial_sweep):
            if i == 0:
                sweep_result.repetitions = trial_result.repetitions
            elif trial_result.repetitions != sweep_result.repetitions:
                raise ValueError(
                    'different numbers of repetitions in one sweep')
            pr = sweep_result.parameterized_results.add()
            pr.params.assignments.update(trial_result.params.param_dict)
            for m in measurements:
                mr = pr.measurement_results.add()
                mr.key = m.key
                m_data = trial_result.measurements[m.key]
                for i, qubit in enumerate(m.qubits):
                    qmr = mr.qubit_measurement_results.add()
                    qmr.qubit.id = v2.qubit_to_proto_id(qubit)
                    qmr.results = pack_bits(m_data[:, i])
    return out

def results_from_proto(
        msg: result_pb2.Result,
        measurements: List[MeasureInfo] = None,
) -> List[List[study.TrialResult]]:
    """Converts a v2 result proto into List of list of trial results.

    Args:
        msg: v2 Result message to convert.
        measurements: List of info about expected measurements in the program.
            This may be used for custom ordering of the result. If no
            measurement config is provided, then all results will be returned
            in the order specified within the result.

    Returns:
        A list containing a list of trial results for each sweep.
    """

    measure_map = {m.key: m for m in measurements} if measurements else None
    return [
        _trial_sweep_from_proto(sweep_result, measure_map)
        for sweep_result in msg.sweep_results
    ]


def _trial_sweep_from_proto(
        msg: result_pb2.SweepResult,
        measure_map: Dict[str, MeasureInfo] = None,
) -> List[study.TrialResult]:
    """Converts a SweepResult proto into List of list of trial results.

    Args:
        msg: v2 Result message to convert.
        measure_map: A mapping of measurement keys to a measurement
            configuration containing qubit ordering. If no measurement config is
            provided, then all results will be returned in the order specified
            within the result.

    Returns:
        A list containing a list of trial results for the sweep.
    """

    trial_sweep: List[study.TrialResult] = []
    for pr in msg.parameterized_results:
        m_data: Dict[str, np.ndarray] = {}
        for mr in pr.measurement_results:
            qubit_results: OrderedDict[devices.GridQubit, np.
                                       ndarray] = OrderedDict()
            for qmr in mr.qubit_measurement_results:
                qubit = v2.grid_qubit_from_proto_id(qmr.qubit.id)
                if qubit in qubit_results:
                    raise ValueError('qubit already exists: {}'.format(qubit))
                qubit_results[qubit] = unpack_bits(qmr.results, msg.repetitions)
            if measure_map:
                ordered_results = [
                    qubit_results[qubit] for qubit in measure_map[mr.key].qubits
                ]
            else:
                ordered_results = list(qubit_results.values())
            m_data[mr.key] = np.array(ordered_results).transpose()
        trial_sweep.append(
            study.TrialResult.from_single_parameter_set(
                params=study.ParamResolver(dict(pr.params.assignments)),
                measurements=m_data,
            ))
    return trial_sweep
