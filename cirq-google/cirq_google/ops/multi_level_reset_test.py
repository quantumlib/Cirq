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
from cirq_google.ops.multi_level_reset import MultilevelResetViaResonator


def test_multi_level_reset_properties():
    gate = MultilevelResetViaResonator()
    assert cirq.num_qubits(gate) == 1
    assert gate.is_reset_gate()
    assert not cirq.has_unitary(gate)


def test_circuit_diagram():
    q = cirq.GridQubit(0, 0)
    op = MultilevelResetViaResonator()(q)
    circuit = cirq.Circuit(op)
    cirq.testing.assert_has_diagram(
        circuit,
        """
(0, 0): ───[R (ML)]───
""",
    )


def test_decomposition():
    q = cirq.GridQubit(0, 0)
    op = MultilevelResetViaResonator()(q)
    decomp = cirq.decompose(op)
    assert decomp == [cirq.ResetChannel()(q)]


def test_serialization_round_trip():
    q = cirq.GridQubit(0, 0)
    gate = MultilevelResetViaResonator()
    op = gate(q)
    circuit = cirq.Circuit(op)

    # Serialize
    proto = cg.CIRCUIT_SERIALIZER.serialize(circuit)

    # Verify proto structure
    op_protos = [c.operation_value for c in proto.constants if c.HasField('operation_value')]
    assert len(op_protos) == 1
    op_proto = op_protos[0]

    assert op_proto.WhichOneof('gate_value') == 'internalgate'
    internal_gate_proto = op_proto.internalgate
    assert internal_gate_proto.name == "MultilevelResetViaResonator"
    assert internal_gate_proto.module == "pyle.cirqtools.pyle_gates"
    assert internal_gate_proto.num_qubits == 1

    # Deserialize
    deserialized_circuit = cg.CIRCUIT_SERIALIZER.deserialize(proto)

    # Verify matching
    assert deserialized_circuit == circuit

    # Verify exact class
    deserialized_op = next(iter(deserialized_circuit.all_operations()))
    assert isinstance(deserialized_op.gate, MultilevelResetViaResonator)


def test_repr():
    gate = MultilevelResetViaResonator()
    assert repr(gate) == 'cirq_google.MultilevelResetViaResonator()'

    assert eval(repr(gate)) == gate
