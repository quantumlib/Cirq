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
from typing import (Any, cast, Dict, Iterable, Optional, Sequence, Tuple,
                    TYPE_CHECKING, Iterator)
import numpy as np
import sympy

from cirq import devices, ops, protocols, value, circuits

if TYPE_CHECKING:
    import cirq


def _load_json_bool(b: Any):
    """Converts a json field to bool.  If already a bool, pass through."""
    if isinstance(b, bool):
        return b
    return json.loads(b)


def gate_to_proto_dict(gate: 'cirq.Gate',
                       qubits: Tuple['cirq.Qid', ...]) -> Dict:
    if isinstance(gate, ops.MeasurementGate):
        return _measure_to_proto_dict(gate, qubits)

    if isinstance(gate, ops.XPowGate):
        if len(qubits) != 1:
            # coverage: ignore
            raise ValueError('Wrong number of qubits.')
        return _x_to_proto_dict(gate, qubits[0])

    if isinstance(gate, ops.YPowGate):
        if len(qubits) != 1:
            # coverage: ignore
            raise ValueError('Wrong number of qubits.')
        return _y_to_proto_dict(gate, qubits[0])

    if isinstance(gate, ops.PhasedXPowGate):
        if len(qubits) != 1:
            # coverage: ignore
            raise ValueError('Wrong number of qubits.')
        return _phased_x_to_proto_dict(gate, qubits[0])

    if isinstance(gate, ops.ZPowGate):
        if len(qubits) != 1:
            # coverage: ignore
            raise ValueError('Wrong number of qubits.')
        return _z_to_proto_dict(gate, qubits[0])

    if isinstance(gate, ops.CZPowGate):
        if len(qubits) != 2:
            # coverage: ignore
            raise ValueError('Wrong number of qubits.')
        return _cz_to_proto_dict(gate, *qubits)

    raise ValueError("Don't know how to serialize this gate: {!r}".format(gate))


def _x_to_proto_dict(gate: 'cirq.XPowGate', q: 'cirq.Qid') -> Dict:
    exp_w = {
        'target': _qubit_to_proto_dict(q),
        'axis_half_turns': _parameterized_value_to_proto_dict(0),
        'half_turns': _parameterized_value_to_proto_dict(gate.exponent)
    }
    return {'exp_w': exp_w}


def _y_to_proto_dict(gate: 'cirq.YPowGate', q: 'cirq.Qid') -> Dict:
    exp_w = {
        'target': _qubit_to_proto_dict(q),
        'axis_half_turns': _parameterized_value_to_proto_dict(0.5),
        'half_turns': _parameterized_value_to_proto_dict(gate.exponent)
    }
    return {'exp_w': exp_w}


def _phased_x_to_proto_dict(gate: 'cirq.PhasedXPowGate', q: 'cirq.Qid') -> Dict:
    exp_w = {
        'target': _qubit_to_proto_dict(q),
        'axis_half_turns':
        _parameterized_value_to_proto_dict(gate.phase_exponent),
        'half_turns': _parameterized_value_to_proto_dict(gate.exponent)
    }
    return {'exp_w': exp_w}


def _z_to_proto_dict(gate: 'cirq.ZPowGate', q: 'cirq.Qid') -> Dict:
    exp_z = {
        'target': _qubit_to_proto_dict(q),
        'half_turns': _parameterized_value_to_proto_dict(gate.exponent),
    }
    return {'exp_z': exp_z}


def _cz_to_proto_dict(gate: 'cirq.CZPowGate', p: 'cirq.Qid',
                      q: 'cirq.Qid') -> Dict:
    exp_11 = {
        'target1': _qubit_to_proto_dict(p),
        'target2': _qubit_to_proto_dict(q),
        'half_turns': _parameterized_value_to_proto_dict(gate.exponent)
    }
    return {'exp_11': exp_11}


def _qubit_to_proto_dict(qubit):
    return {
        'row': qubit.row,
        'col': qubit.col,
    }


def _measure_to_proto_dict(gate: 'cirq.MeasurementGate',
                           qubits: Sequence['cirq.Qid']):
    if len(qubits) == 0:
        raise ValueError('Measurement gate on no qubits.')

    invert_mask = None
    if gate.invert_mask:
        invert_mask = gate.invert_mask + (False,) * (gate.num_qubits() -
                                                     len(gate.invert_mask))

    if invert_mask and len(invert_mask) != len(qubits):
        raise ValueError('Measurement gate had invert mask of length '
                         'different than number of qubits it acts on.')
    measurement = {
        'targets': [_qubit_to_proto_dict(q) for q in qubits],
        'key': protocols.measurement_key(gate),
    }
    if invert_mask:
        measurement['invert_mask'] = [json.dumps(x) for x in invert_mask]
    return {'measurement': measurement}


def circuit_as_schedule_to_proto_dicts(circuit: 'cirq.Circuit'
                                      ) -> Iterator[Dict]:
    """Convert a circuit into an iterable of proto dictionaries.

    Args:
        circuit: The circuit to convert to a proto dict. Must contain only
            gates that can be cast to xmon gates.

    Yields:
        A proto dictionary corresponding to an Operation proto.
    """
    last_picos: Optional[int] = None
    time_picos = 0
    for op in circuit.all_operations():
        proto = gate_to_proto_dict(cast(ops.Gate, op.gate), op.qubits)
        if last_picos is None:
            proto['incremental_delay_picoseconds'] = time_picos
        else:
            proto['incremental_delay_picoseconds'] = time_picos - last_picos
        time_picos += 1
        last_picos = time_picos
        yield proto


def circuit_from_schedule_from_proto_dicts(
        device: 'cirq.google.XmonDevice',
        ops: Iterable[Dict],
) -> 'cirq.Circuit':
    """Convert proto dictionaries into a Circuit for the given device."""
    result = []
    for op in ops:
        xmon_op = xmon_op_from_proto_dict(op)
        result.append(xmon_op)
    return circuits.Circuit(result, device=device)


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
        ValueError if the measurement data do not have the compatible shapes.
    """
    if not measurements:
        return b''

    shapes = [(key, np.shape(data)) for key, data in measurements]
    if not all(len(shape) == 2 for _, shape in shapes):
        raise ValueError("Expected 2-D data: shapes={}".format(shapes))

    reps = shapes[0][1][0]
    if not all(shape[0] == reps for _, shape in shapes):
        raise ValueError(
            "Expected same reps for all keys: shapes={}".format(shapes))

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


def unpack_results(data: bytes, repetitions: int,
                   key_sizes: Sequence[Tuple[str, int]]
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
        results[key] = bits[:, ofs:ofs + size]
        ofs += size

    return results


def is_native_xmon_op(op: 'cirq.Operation') -> bool:
    """Check if the gate corresponding to an operation is a native xmon gate.

    Args:
        op: Input operation.

    Returns:
        True if the operation is native to the xmon, false otherwise.
    """
    return (isinstance(op, ops.GateOperation) and is_native_xmon_gate(op.gate))


def is_native_xmon_gate(gate: 'cirq.Gate') -> bool:
    """Check if a gate is a native xmon gate.

    Args:
        gate: Input gate.

    Returns:
        True if the gate is native to the xmon, false otherwise.
    """
    return isinstance(gate,
                      (ops.CZPowGate, ops.MeasurementGate, ops.PhasedXPowGate,
                       ops.XPowGate, ops.YPowGate, ops.ZPowGate))


def xmon_op_from_proto_dict(proto_dict: Dict) -> 'cirq.Operation':
    """Convert the proto dictionary to the corresponding operation.

    See protos in api/google/v1 for specification of the protos.

    Args:
        proto_dict: Dictionary representing the proto. Keys are always
            strings, but values may be types correspond to a raw proto type
            or another dictionary (for messages).

    Returns:
        The operation.

    Raises:
        ValueError if the dictionary does not contain required values
        corresponding to the proto.
    """

    def raise_missing_fields(gate_name: str):
        raise ValueError('{} missing required fields: {}'.format(
            gate_name, proto_dict))

    param = _parameterized_value_from_proto_dict
    qubit = _qubit_from_proto_dict
    if 'exp_w' in proto_dict:
        exp_w = proto_dict['exp_w']
        if ('half_turns' not in exp_w or 'axis_half_turns' not in exp_w or
                'target' not in exp_w):
            raise_missing_fields('ExpW')
        return ops.PhasedXPowGate(
            exponent=param(exp_w['half_turns']),
            phase_exponent=param(exp_w['axis_half_turns']),
        ).on(qubit(exp_w['target']))
    if 'exp_z' in proto_dict:
        exp_z = proto_dict['exp_z']
        if 'half_turns' not in exp_z or 'target' not in exp_z:
            raise_missing_fields('ExpZ')
        return ops.Z(qubit(exp_z['target']))**param(exp_z['half_turns'])
    if 'exp_11' in proto_dict:
        exp_11 = proto_dict['exp_11']
        if ('half_turns' not in exp_11 or 'target1' not in exp_11 or
                'target2' not in exp_11):
            raise_missing_fields('Exp11')
        return ops.CZ(qubit(exp_11['target1']),
                      qubit(exp_11['target2']))**param(exp_11['half_turns'])
    if 'measurement' in proto_dict:
        meas = proto_dict['measurement']
        invert_mask = cast(Tuple[Any, ...], ())
        if 'invert_mask' in meas:
            invert_mask = tuple(_load_json_bool(x) for x in meas['invert_mask'])
        if 'key' not in meas or 'targets' not in meas:
            raise_missing_fields('Measurement')
        return ops.MeasurementGate(
            num_qubits=len(meas['targets']),
            key=meas['key'],
            invert_mask=invert_mask).on(*[qubit(q) for q in meas['targets']])

    raise ValueError('invalid operation: {}'.format(proto_dict))


def _qubit_from_proto_dict(proto_dict):
    """Proto dict must have 'row' and 'col' keys."""
    if 'row' not in proto_dict or 'col' not in proto_dict:
        raise ValueError(
            'Proto dict does not contain row or col: {}'.format(proto_dict))
    return devices.GridQubit(row=proto_dict['row'], col=proto_dict['col'])


def _parameterized_value_from_proto_dict(message: Dict) -> value.TParamVal:
    parameter_key = message.get('parameter_key', None)
    if parameter_key:
        return sympy.Symbol(parameter_key)
    if 'raw' in message:
        return message['raw']
    raise ValueError('No value specified for parameterized float. '
                     'Expected "raw" or "parameter_key" to be set. '
                     'message: {!r}'.format(message))


def _parameterized_value_to_proto_dict(param: value.TParamVal) -> Dict:
    out = {}  # type: Dict
    if isinstance(param, sympy.Symbol):
        out['parameter_key'] = str(param.free_symbols.pop())
    else:
        out['raw'] = float(param)
    return out
