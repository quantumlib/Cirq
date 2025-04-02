# Copyright 2018 The Cirq Developers
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

import json
from typing import Any, cast, Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import sympy

import cirq
from cirq_google.api.v1 import operations_pb2

if TYPE_CHECKING:
    import cirq_google


def _load_json_bool(b: Any):
    """Converts a json field to bool.  If already a bool, pass through."""
    if isinstance(b, bool):
        return b
    return json.loads(b)


def gate_to_proto(
    gate: cirq.Gate, qubits: Tuple[cirq.Qid, ...], delay: int
) -> operations_pb2.Operation:
    if isinstance(gate, cirq.MeasurementGate):
        return operations_pb2.Operation(
            incremental_delay_picoseconds=delay, measurement=_measure_to_proto(gate, qubits)
        )

    if isinstance(gate, cirq.XPowGate):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')  # pragma: no cover
        return operations_pb2.Operation(
            incremental_delay_picoseconds=delay, exp_w=_x_to_proto(gate, qubits[0])
        )

    if isinstance(gate, cirq.YPowGate):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')  # pragma: no cover
        return operations_pb2.Operation(
            incremental_delay_picoseconds=delay, exp_w=_y_to_proto(gate, qubits[0])
        )

    if isinstance(gate, cirq.PhasedXPowGate):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')  # pragma: no cover
        return operations_pb2.Operation(
            incremental_delay_picoseconds=delay, exp_w=_phased_x_to_proto(gate, qubits[0])
        )

    if isinstance(gate, cirq.ZPowGate):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')  # pragma: no cover
        return operations_pb2.Operation(
            incremental_delay_picoseconds=delay, exp_z=_z_to_proto(gate, qubits[0])
        )

    if isinstance(gate, cirq.CZPowGate):
        if len(qubits) != 2:
            raise ValueError('Wrong number of qubits.')  # pragma: no cover
        return operations_pb2.Operation(
            incremental_delay_picoseconds=delay, exp_11=_cz_to_proto(gate, *qubits)
        )

    raise ValueError(f"Don't know how to serialize this gate: {gate!r}")


def _x_to_proto(gate: cirq.XPowGate, q: cirq.Qid) -> operations_pb2.ExpW:
    return operations_pb2.ExpW(
        target=_qubit_to_proto(q),
        axis_half_turns=_parameterized_value_to_proto(0),
        half_turns=_parameterized_value_to_proto(gate.exponent),
    )


def _y_to_proto(gate: cirq.YPowGate, q: cirq.Qid) -> operations_pb2.ExpW:
    return operations_pb2.ExpW(
        target=_qubit_to_proto(q),
        axis_half_turns=_parameterized_value_to_proto(0.5),
        half_turns=_parameterized_value_to_proto(gate.exponent),
    )


def _phased_x_to_proto(gate: cirq.PhasedXPowGate, q: cirq.Qid) -> operations_pb2.ExpW:
    return operations_pb2.ExpW(
        target=_qubit_to_proto(q),
        axis_half_turns=_parameterized_value_to_proto(gate.phase_exponent),
        half_turns=_parameterized_value_to_proto(gate.exponent),
    )


def _z_to_proto(gate: cirq.ZPowGate, q: cirq.Qid) -> operations_pb2.ExpZ:
    return operations_pb2.ExpZ(
        target=_qubit_to_proto(q), half_turns=_parameterized_value_to_proto(gate.exponent)
    )


def _cz_to_proto(gate: cirq.CZPowGate, p: cirq.Qid, q: cirq.Qid) -> operations_pb2.Exp11:
    return operations_pb2.Exp11(
        target1=_qubit_to_proto(p),
        target2=_qubit_to_proto(q),
        half_turns=_parameterized_value_to_proto(gate.exponent),
    )


def _qubit_to_proto(qubit):
    return operations_pb2.Qubit(row=qubit.row, col=qubit.col)


def _measure_to_proto(gate: cirq.MeasurementGate, qubits: Sequence[cirq.Qid]):
    if len(qubits) == 0:
        raise ValueError('Measurement gate on no qubits.')

    invert_mask = None
    if gate.invert_mask:
        invert_mask = gate.invert_mask + (False,) * (gate.num_qubits() - len(gate.invert_mask))

    if invert_mask and len(invert_mask) != len(qubits):
        raise ValueError(
            'Measurement gate had invert mask of length '
            'different than number of qubits it acts on.'
        )
    return operations_pb2.Measurement(
        targets=[_qubit_to_proto(q) for q in qubits],
        key=cirq.measurement_key_name(gate),
        invert_mask=invert_mask,
    )


def circuit_as_schedule_to_protos(circuit: cirq.Circuit) -> Iterator[operations_pb2.Operation]:
    """Convert a circuit into an iterable of protos.

    Args:
        circuit: The circuit to convert to a proto. Must contain only
            gates that can be cast to xmon gates.

    Yields:
        An Operation proto.
    """
    last_picos: Optional[int] = None
    time_picos = 0
    for op in circuit.all_operations():
        if last_picos is None:
            delay = time_picos
        else:
            delay = time_picos - last_picos
        op_proto = gate_to_proto(cast(cirq.Gate, op.gate), op.qubits, delay)
        time_picos += 1
        last_picos = time_picos
        yield op_proto


def circuit_from_schedule_from_protos(ops) -> cirq.Circuit:
    """Convert protos into a Circuit."""
    result = []
    for op in ops:
        xmon_op = xmon_op_from_proto(op)
        result.append(xmon_op)
    ret = cirq.Circuit(result)
    return ret


def pack_results(measurements: Sequence[Tuple[str, np.ndarray]]) -> bytes:
    """Pack measurement results into a byte string.

    Args:
        measurements: A sequence of tuples, one for each measurement, consisting
            of a string key and an array of boolean data. The data should be
            a 2-D array indexed by (repetition, qubit_index). All data for all
            measurements must have the same number of repetitions.

    Returns:
        Packed bytes, as described in the unpack_results docstring below.

    Raises:
        ValueError: If the measurement data do not have the compatible shapes.
    """
    if not measurements:
        return b''

    shapes = [(key, np.shape(data)) for key, data in measurements]
    if not all(len(shape) == 2 for _, shape in shapes):
        raise ValueError(f"Expected 2-D data: shapes={shapes}")

    reps = shapes[0][1][0]
    if not all(shape[0] == reps for _, shape in shapes):
        raise ValueError(f"Expected same reps for all keys: shapes={shapes}")

    bits = np.hstack([np.asarray(data, dtype=bool) for _, data in measurements])
    bits = bits.reshape(-1)

    # Pad length to multiple of 8 if needed.
    remainder = len(bits) % 8
    if remainder:
        bits = np.pad(bits, (0, 8 - remainder), 'constant')

    # Pack in little-endian bit order.
    bits = bits.reshape((-1, 8))[:, ::-1]
    byte_arr = np.packbits(bits, axis=1).reshape(-1)

    return byte_arr.tobytes()


def unpack_results(
    data: bytes, repetitions: int, key_sizes: Sequence[Tuple[str, int]]
) -> Dict[str, np.ndarray]:
    """Unpack data from a bitstring into individual measurement results.

    Args:
        data: Packed measurement results, in the form <rep0><rep1>...
            where each repetition is <key0_0>..<key0_{size0-1}><key1_0>...
            with bits packed in little-endian order in each byte.
        repetitions: number of repetitions.
        key_sizes: Keys and sizes of the measurements in the data.

    Returns:
        Dict mapping measurement key to a 2D array of boolean results. Each
        array has shape (repetitions, size) with size for that measurement.
    """
    bits_per_rep = sum(size for _, size in key_sizes)
    total_bits = repetitions * bits_per_rep

    byte_arr = np.frombuffer(data, dtype='uint8').reshape((len(data), 1))
    bits = np.unpackbits(byte_arr, axis=1)[:, ::-1].reshape(-1).astype(bool)
    bits = bits[:total_bits].reshape((repetitions, bits_per_rep))

    results = {}
    ofs = 0
    for key, size in key_sizes:
        results[key] = bits[:, ofs : ofs + size]
        ofs += size

    return results


def is_native_xmon_op(op: cirq.Operation) -> bool:
    """Check if the gate corresponding to an operation is a native xmon gate.

    Args:
        op: Input operation.

    Returns:
        True if the operation is native to the xmon, false otherwise.
    """
    return isinstance(op, cirq.GateOperation) and is_native_xmon_gate(op.gate)


def is_native_xmon_gate(gate: cirq.Gate) -> bool:
    """Check if a gate is a native xmon gate.

    Args:
        gate: Input gate.

    Returns:
        True if the gate is native to the xmon, false otherwise.
    """
    return isinstance(
        gate,
        (
            cirq.CZPowGate,
            cirq.MeasurementGate,
            cirq.PhasedXPowGate,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
        ),
    )


def xmon_op_from_proto(proto: operations_pb2.Operation) -> cirq.Operation:
    """Convert the proto to the corresponding operation.

    See protos in api/google/v1 for specification of the protos.

    Args:
        proto: Operation proto.

    Returns:
        The operation.

    Raises:
        ValueError: If the proto has an operation that is invalid.
    """
    param = _parameterized_value_from_proto
    qubit = _qubit_from_proto
    if proto.HasField('exp_w'):
        exp_w = proto.exp_w
        return cirq.PhasedXPowGate(
            exponent=param(exp_w.half_turns), phase_exponent=param(exp_w.axis_half_turns)
        ).on(qubit(exp_w.target))
    if proto.HasField('exp_z'):
        exp_z = proto.exp_z
        return cirq.Z(qubit(exp_z.target)) ** param(exp_z.half_turns)
    if proto.HasField('exp_11'):
        exp_11 = proto.exp_11
        return cirq.CZ(qubit(exp_11.target1), qubit(exp_11.target2)) ** param(exp_11.half_turns)
    if proto.HasField('measurement'):
        meas = proto.measurement
        return cirq.MeasurementGate(
            num_qubits=len(meas.targets), key=meas.key, invert_mask=tuple(meas.invert_mask)
        ).on(*[qubit(q) for q in meas.targets])

    raise ValueError(f'invalid operation: {proto}')


def _qubit_from_proto(proto: operations_pb2.Qubit):
    return cirq.GridQubit(row=proto.row, col=proto.col)


def _parameterized_value_from_proto(proto: operations_pb2.ParameterizedFloat) -> cirq.TParamVal:
    if proto.HasField('parameter_key'):
        return sympy.Symbol(proto.parameter_key)
    if proto.HasField('raw'):
        return proto.raw
    raise ValueError(
        'No value specified for parameterized float. '
        'Expected "raw" or "parameter_key" to be set. '
        f'proto: {proto!r}'
    )


def _parameterized_value_to_proto(param: cirq.TParamVal) -> operations_pb2.ParameterizedFloat:
    if isinstance(param, sympy.Symbol):
        return operations_pb2.ParameterizedFloat(parameter_key=str(param.free_symbols.pop()))
    else:
        return operations_pb2.ParameterizedFloat(raw=float(param))
