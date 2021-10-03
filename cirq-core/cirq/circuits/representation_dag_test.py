# Copyright 2021 The Cirq Developers
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
import numpy as np


def test_circuit_dag_equivalence():
    circuit = cirq.Circuit(
        [
            cirq.H(cirq.LineQubit(0)),
            cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
            cirq.X(cirq.LineQubit(1)),
        ]
    )
    dag = cirq.CircuitDagRepresentation(circuit=circuit)
    new_circuit = dag.to_circuit()
    assert circuit == new_circuit


def test_dag_repr():
    circuit = cirq.Circuit(
        [
            cirq.H(cirq.LineQubit(0)),
            cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
            cirq.X(cirq.LineQubit(1)),
        ]
    )
    dag = cirq.CircuitDagRepresentation(circuit=circuit)
    repr_dag = repr(dag)
    new_dag: cirq.CircuitDagRepresentation = eval(repr_dag)
    assert circuit == new_dag.to_circuit()


def test_dag_json():
    circuit = cirq.Circuit(
        [
            cirq.H(cirq.LineQubit(0)),
            cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
            cirq.X(cirq.LineQubit(1)),
        ]
    )
    dag = cirq.CircuitDagRepresentation(circuit=circuit)
    repr_dag = cirq.to_json(dag)
    new_dag: cirq.CircuitDagRepresentation = cirq.read_json(json_text=repr_dag)
    assert circuit == new_dag.to_circuit()


def test_equality():
    circuit = cirq.Circuit(
        [
            cirq.H(cirq.LineQubit(0)),
            cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
            cirq.X(cirq.LineQubit(1)),
        ]
    )
    dag = cirq.CircuitDagRepresentation(circuit=circuit)
    repr_dag = cirq.to_json(dag)
    new_dag: cirq.CircuitDagRepresentation = cirq.read_json(json_text=repr_dag)
    assert dag == circuit
    assert dag == new_dag
    assert dag != np.zeros(shape=6)
