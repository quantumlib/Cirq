# Copyright 2026 The Cirq Developers
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

import cirq
import cirq_google as cg
from cirq_google.ops.leakage_iswap import LeakageISWAP


def test_leakage_iswap_properties():
    gate = LeakageISWAP()
    assert cirq.num_qubits(gate) == 2
    assert cirq.has_unitary(gate)
    assert gate.phase_matched is True

    gate_unmatched = LeakageISWAP(phase_matched=False)
    assert cirq.num_qubits(gate_unmatched) == 2
    assert cirq.has_unitary(gate_unmatched)
    assert gate_unmatched.phase_matched is False


def test_circuit_diagram():
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    op = LeakageISWAP()(q0, q1)
    circuit = cirq.Circuit(op)
    cirq.testing.assert_has_diagram(
        circuit,
        """
(0, 0): ───LiS───
           │
(0, 1): ───LiS───
""",
    )


def test_decomposition():
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    op = LeakageISWAP()(q0, q1)
    decomp = cirq.decompose_once(op)
    assert decomp == [cirq.I(q0), cirq.I(q1)]


def test_repr():
    gate_matched = LeakageISWAP(phase_matched=True)
    assert repr(gate_matched) == 'cirq_google.LeakageISWAP(phase_matched=True)'

    gate_unmatched = LeakageISWAP(phase_matched=False)
    assert repr(gate_unmatched) == 'cirq_google.LeakageISWAP(phase_matched=False)'


def test_value_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(LeakageISWAP(phase_matched=True), LeakageISWAP())
    eq.add_equality_group(LeakageISWAP(phase_matched=False))


def test_serialization_round_trip():
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    for phase_matched in [True, False]:
        gate = LeakageISWAP(phase_matched=phase_matched)
        op = gate(q0, q1)
        circuit = cirq.Circuit(op)

        # Serialize
        proto = cg.CIRCUIT_SERIALIZER.serialize(circuit)

        # Verify proto structure
        op_protos = [c.operation_value for c in proto.constants if c.HasField('operation_value')]
        assert len(op_protos) == 1
        op_proto = op_protos[0]

        assert op_proto.WhichOneof('gate_value') == 'internalgate'
        internal_gate_proto = op_proto.internalgate
        expected_name = "LeakageISWAPPhaseMatched" if phase_matched else "LeakageISWAPUnmatched"
        assert internal_gate_proto.name == expected_name
        assert internal_gate_proto.num_qubits == 2

        # Deserialize
        deserialized_circuit = cg.CIRCUIT_SERIALIZER.deserialize(proto)

        # Verify matching
        assert deserialized_circuit == circuit

        # Verify exact class
        deserialized_op = next(iter(deserialized_circuit.all_operations()))
        assert isinstance(deserialized_op.gate, LeakageISWAP)
        assert deserialized_op.gate.phase_matched == phase_matched
