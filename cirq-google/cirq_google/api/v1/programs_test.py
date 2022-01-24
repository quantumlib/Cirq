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
import numpy as np
import pytest
import sympy

import cirq
import cirq_google as cg
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2


def assert_proto_dict_convert(gate: cirq.Gate, proto: operations_pb2.Operation, *qubits: cirq.Qid):
    assert programs.gate_to_proto(gate, qubits, delay=0) == proto
    assert programs.xmon_op_from_proto(proto) == gate(*qubits)


def test_protobuf_round_trip():
    qubits = cirq.GridQubit.rect(1, 5)
    circuit = cirq.Circuit(
        [cirq.X(q) ** 0.5 for q in qubits],
        [
            cirq.CZ(q, q2)
            for q in [cirq.GridQubit(0, 0)]
            for q, q2 in zip(qubits, qubits)
            if q != q2
        ],
    )

    protos = list(programs.circuit_as_schedule_to_protos(circuit))
    s2 = programs.circuit_from_schedule_from_protos(protos)
    assert s2 == circuit


def test_protobuf_round_trip_device_deprecated():
    with cirq.testing.assert_deprecated('Foxtail', deadline='v0.15'):
        device = cg.Foxtail
    circuit = cirq.Circuit(
        [cirq.X(q) ** 0.5 for q in device.qubits],
        [cirq.CZ(q, q2) for q in [cirq.GridQubit(0, 0)] for q2 in device.neighbors_of(q)],
    )
    circuit._device = device

    protos = list(programs.circuit_as_schedule_to_protos(circuit))
    with cirq.testing.assert_deprecated(
        cirq.circuits.circuit._DEVICE_DEP_MESSAGE, deadline='v0.15'
    ):
        s2 = programs.circuit_from_schedule_from_protos(device, protos)
        assert s2 == circuit


def make_bytes(s: str) -> bytes:
    """Helper function to convert a string of digits into packed bytes.

    Ignores any characters other than 0 and 1, in particular whitespace. The
    bits are packed in little-endian order within each byte.
    """
    buf = []
    byte = 0
    idx = 0
    for c in s:
        if c == '0':
            pass
        elif c == '1':
            byte |= 1 << idx
        else:
            # coverage: ignore
            continue
        idx += 1
        if idx == 8:
            buf.append(byte)
            byte = 0
            idx = 0
    if idx:
        buf.append(byte)
    return bytearray(buf)


def test_pack_results():
    measurements = [
        (
            'a',
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                ]
            ),
        ),
        (
            'b',
            np.array(
                [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                ]
            ),
        ),
    ]
    data = programs.pack_results(measurements)
    expected = make_bytes(
        """
        000 00
        001 01
        010 10
        011 11
        100 00
        101 01
        110 10

        000 00 -- padding
    """
    )
    assert data == expected


def test_pack_results_no_measurements():
    assert programs.pack_results([]) == b''


def test_pack_results_incompatible_shapes():
    def bools(*shape):
        return np.zeros(shape, dtype=bool)

    with pytest.raises(ValueError):
        programs.pack_results([('a', bools(10))])

    with pytest.raises(ValueError):
        programs.pack_results([('a', bools(7, 3)), ('b', bools(8, 2))])


def test_unpack_results():
    data = make_bytes(
        """
        000 00
        001 01
        010 10
        011 11
        100 00
        101 01
        110 10
    """
    )
    assert len(data) == 5  # 35 data bits + 5 padding bits
    results = programs.unpack_results(data, 7, [('a', 3), ('b', 2)])
    assert 'a' in results
    assert results['a'].shape == (7, 3)
    assert results['a'].dtype == bool
    np.testing.assert_array_equal(
        results['a'],
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
        ],
    )

    assert 'b' in results
    assert results['b'].shape == (7, 2)
    assert results['b'].dtype == bool
    np.testing.assert_array_equal(
        results['b'],
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
            [1, 0],
        ],
    )


def test_single_qubit_measurement_proto_convert():
    gate = cirq.MeasurementGate(1, 'test')
    proto = operations_pb2.Operation(
        measurement=operations_pb2.Measurement(
            targets=[operations_pb2.Qubit(row=2, col=3)], key='test'
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))


def test_single_qubit_measurement_to_proto_convert_invert_mask():
    gate = cirq.MeasurementGate(1, 'test', invert_mask=(True,))
    proto = operations_pb2.Operation(
        measurement=operations_pb2.Measurement(
            targets=[operations_pb2.Qubit(row=2, col=3)], key='test', invert_mask=[True]
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))


def test_single_qubit_measurement_to_proto_pad_invert_mask():
    gate = cirq.MeasurementGate(2, 'test', invert_mask=(True,))
    proto = operations_pb2.Operation(
        measurement=operations_pb2.Measurement(
            targets=[operations_pb2.Qubit(row=2, col=3), operations_pb2.Qubit(row=2, col=4)],
            key='test',
            invert_mask=[True, False],
        )
    )
    assert (
        programs.gate_to_proto(gate, (cirq.GridQubit(2, 3), cirq.GridQubit(2, 4)), delay=0) == proto
    )


def test_multi_qubit_measurement_to_proto():
    gate = cirq.MeasurementGate(2, 'test')
    proto = operations_pb2.Operation(
        measurement=operations_pb2.Measurement(
            targets=[operations_pb2.Qubit(row=2, col=3), operations_pb2.Qubit(row=3, col=4)],
            key='test',
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))


def test_z_proto_convert():
    gate = cirq.Z ** sympy.Symbol('k')
    proto = operations_pb2.Operation(
        exp_z=operations_pb2.ExpZ(
            target=operations_pb2.Qubit(row=2, col=3),
            half_turns=operations_pb2.ParameterizedFloat(parameter_key='k'),
        )
    )

    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))
    gate = cirq.Z ** 0.5
    proto = operations_pb2.Operation(
        exp_z=operations_pb2.ExpZ(
            target=operations_pb2.Qubit(row=2, col=3),
            half_turns=operations_pb2.ParameterizedFloat(raw=0.5),
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))


def test_cz_proto_convert():
    gate = cirq.CZ ** sympy.Symbol('k')
    proto = operations_pb2.Operation(
        exp_11=operations_pb2.Exp11(
            target1=operations_pb2.Qubit(row=2, col=3),
            target2=operations_pb2.Qubit(row=3, col=4),
            half_turns=operations_pb2.ParameterizedFloat(parameter_key='k'),
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))

    gate = cirq.CZ ** 0.5
    proto = operations_pb2.Operation(
        exp_11=operations_pb2.Exp11(
            target1=operations_pb2.Qubit(row=2, col=3),
            target2=operations_pb2.Qubit(row=3, col=4),
            half_turns=operations_pb2.ParameterizedFloat(raw=0.5),
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))


def test_w_to_proto():
    gate = cirq.PhasedXPowGate(exponent=sympy.Symbol('k'), phase_exponent=1)
    proto = operations_pb2.Operation(
        exp_w=operations_pb2.ExpW(
            target=operations_pb2.Qubit(row=2, col=3),
            axis_half_turns=operations_pb2.ParameterizedFloat(raw=1),
            half_turns=operations_pb2.ParameterizedFloat(parameter_key='k'),
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))

    gate = cirq.PhasedXPowGate(exponent=0.5, phase_exponent=sympy.Symbol('j'))
    proto = operations_pb2.Operation(
        exp_w=operations_pb2.ExpW(
            target=operations_pb2.Qubit(row=2, col=3),
            axis_half_turns=operations_pb2.ParameterizedFloat(parameter_key='j'),
            half_turns=operations_pb2.ParameterizedFloat(raw=0.5),
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))

    gate = cirq.X ** 0.25
    proto = operations_pb2.Operation(
        exp_w=operations_pb2.ExpW(
            target=operations_pb2.Qubit(row=2, col=3),
            axis_half_turns=operations_pb2.ParameterizedFloat(raw=0.0),
            half_turns=operations_pb2.ParameterizedFloat(raw=0.25),
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))

    gate = cirq.Y ** 0.25
    proto = operations_pb2.Operation(
        exp_w=operations_pb2.ExpW(
            target=operations_pb2.Qubit(row=2, col=3),
            axis_half_turns=operations_pb2.ParameterizedFloat(raw=0.5),
            half_turns=operations_pb2.ParameterizedFloat(raw=0.25),
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))

    gate = cirq.PhasedXPowGate(exponent=0.5, phase_exponent=sympy.Symbol('j'))
    proto = operations_pb2.Operation(
        exp_w=operations_pb2.ExpW(
            target=operations_pb2.Qubit(row=2, col=3),
            axis_half_turns=operations_pb2.ParameterizedFloat(parameter_key='j'),
            half_turns=operations_pb2.ParameterizedFloat(raw=0.5),
        )
    )
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))


def test_unsupported_op():
    with pytest.raises(ValueError, match='invalid operation'):
        programs.xmon_op_from_proto(operations_pb2.Operation())
    with pytest.raises(ValueError, match='know how to serialize'):
        programs.gate_to_proto(
            cirq.CCZ, (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)), delay=0
        )


def test_invalid_to_proto_dict_qubit_number():
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        _ = programs.gate_to_proto(cirq.CZ ** 0.5, (cirq.GridQubit(2, 3),), delay=0)
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        programs.gate_to_proto(cirq.Z ** 0.5, (cirq.GridQubit(2, 3), cirq.GridQubit(3, 4)), delay=0)
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        programs.gate_to_proto(
            cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0),
            (cirq.GridQubit(2, 3), cirq.GridQubit(3, 4)),
            delay=0,
        )


def test_parameterized_value_from_proto():
    from_proto = programs._parameterized_value_from_proto

    m1 = operations_pb2.ParameterizedFloat(raw=5)
    assert from_proto(m1) == 5

    with pytest.raises(ValueError):
        from_proto(operations_pb2.ParameterizedFloat())

    m3 = operations_pb2.ParameterizedFloat(parameter_key='rr')
    assert from_proto(m3) == sympy.Symbol('rr')


def test_invalid_measurement_gate():
    with pytest.raises(ValueError, match='length'):
        _ = programs.gate_to_proto(
            cirq.MeasurementGate(3, 'test', invert_mask=(True,)),
            (cirq.GridQubit(2, 3), cirq.GridQubit(3, 4)),
            delay=0,
        )
    with pytest.raises(ValueError, match='no qubits'):
        _ = programs.gate_to_proto(cirq.MeasurementGate(1, 'test'), (), delay=0)


def test_is_supported():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(1, 0)
    assert programs.is_native_xmon_op(cirq.CZ(a, b))
    assert programs.is_native_xmon_op(cirq.X(a) ** 0.5)
    assert programs.is_native_xmon_op(cirq.Y(a) ** 0.5)
    assert programs.is_native_xmon_op(cirq.Z(a) ** 0.5)
    assert programs.is_native_xmon_op(cirq.PhasedXPowGate(phase_exponent=0.2).on(a) ** 0.5)
    assert programs.is_native_xmon_op(cirq.Z(a) ** 1)
    assert not programs.is_native_xmon_op(cirq.CCZ(a, b, c))
    assert not programs.is_native_xmon_op(cirq.SWAP(a, b))


def test_is_native_xmon_gate():
    assert programs.is_native_xmon_gate(cirq.CZ)
    assert programs.is_native_xmon_gate(cirq.X ** 0.5)
    assert programs.is_native_xmon_gate(cirq.Y ** 0.5)
    assert programs.is_native_xmon_gate(cirq.Z ** 0.5)
    assert programs.is_native_xmon_gate(cirq.PhasedXPowGate(phase_exponent=0.2) ** 0.5)
    assert programs.is_native_xmon_gate(cirq.Z ** 1)
    assert not programs.is_native_xmon_gate(cirq.CCZ)
    assert not programs.is_native_xmon_gate(cirq.SWAP)
