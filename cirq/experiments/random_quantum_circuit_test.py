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

import cirq


def test_random_quantum_circuit():
    # pylint: disable=line-too-long
    qubits = cirq.GridQubit.rect(3, 2)
    depth = 9
    circuit = cirq.experiments.random_quantum_circuit(qubits, depth, seed=1234)
    cirq.testing.assert_has_diagram(circuit,
                                    """
  (0, 0)        (0, 1)        (1, 0)        (1, 1)        (2, 0)        (2, 1)
  │             │             │             │             │             │
  PhX(0.25)^0.5 Y^0.5         X^0.5         X^0.5         X^0.5         Y^0.5
  │             │             │             │             │             │
  SYC───────────┼─────────────SYC           SYC───────────┼─────────────SYC
  │             │             │             │             │             │
  Y^0.5         PhX(0.25)^0.5 PhX(0.25)^0.5 PhX(0.25)^0.5 Y^0.5         X^0.5
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ │             SYC───────────┼─────────────SYC           │             │             │
│ │             │             SYC───────────┼─────────────SYC           │             │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  PhX(0.25)^0.5 X^0.5         X^0.5         X^0.5         X^0.5         Y^0.5
  │             │             │             │             │             │
  │             │             SYC───────────SYC           │             │
  │             │             │             │             │             │
  X^0.5         Y^0.5         Y^0.5         Y^0.5         PhX(0.25)^0.5 X^0.5
  │             │             │             │             │             │
  SYC───────────SYC           │             │             SYC───────────SYC
  │             │             │             │             │             │
  PhX(0.25)^0.5 PhX(0.25)^0.5 X^0.5         X^0.5         Y^0.5         Y^0.5
  │             │             │             │             │             │
  │             │             SYC───────────SYC           │             │
  │             │             │             │             │             │
  X^0.5         Y^0.5         Y^0.5         PhX(0.25)^0.5 X^0.5         X^0.5
  │             │             │             │             │             │
  SYC───────────SYC           │             │             SYC───────────SYC
  │             │             │             │             │             │
  Y^0.5         PhX(0.25)^0.5 PhX(0.25)^0.5 Y^0.5         Y^0.5         PhX(0.25)^0.5
  │             │             │             │             │             │
  SYC───────────┼─────────────SYC           SYC───────────┼─────────────SYC
  │             │             │             │             │             │
  PhX(0.25)^0.5 X^0.5         Y^0.5         X^0.5         PhX(0.25)^0.5 X^0.5
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ │             SYC───────────┼─────────────SYC           │             │             │
│ │             │             SYC───────────┼─────────────SYC           │             │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  Y^0.5         PhX(0.25)^0.5 PhX(0.25)^0.5 PhX(0.25)^0.5 X^0.5         PhX(0.25)^0.5
  │             │             │             │             │             │
  SYC───────────┼─────────────SYC           SYC───────────┼─────────────SYC
  │             │             │             │             │             │
  X^0.5         Y^0.5         Y^0.5         X^0.5         Y^0.5         Y^0.5
  │             │             │             │             │             │
""",
                                    transpose=True)

    qubits = cirq.GridQubit.rect(2, 3)
    depth = 9

    def two_qubit_op_factory(a, b, prng):
        z_exponents = [prng.uniform(0, 1) for _ in range(4)]
        yield cirq.Z(a)**z_exponents[0]
        yield cirq.Z(b)**z_exponents[1]
        yield cirq.google.SYC(a, b)
        yield cirq.Z(a)**z_exponents[2]
        yield cirq.Z(b)**z_exponents[3]

    circuit = cirq.experiments.random_quantum_circuit(
        qubits, depth, two_qubit_op_factory=two_qubit_op_factory, seed=1234)
    cirq.testing.assert_has_diagram(circuit,
                                    """
  (0, 0)        (0, 1)        (0, 2)        (1, 0)        (1, 1)        (1, 2)
  │             │             │             │             │             │
  PhX(0.25)^0.5 Y^0.5         X^0.5         X^0.5         X^0.5         Y^0.5
  │             │             │             │             │             │
  Z^0.78        │             Z^0.958       Z^(3/11)      │             Z^0.876
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ SYC───────────┼─────────────┼─────────────SYC           │             │             │
│ │             │             SYC───────────┼─────────────┼─────────────SYC           │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  Z^0.276       │             Z^(5/14)      Z^0.802       │             Z^0.501
  │             │             │             │             │             │
  Y^0.5         X^0.5         PhX(0.25)^0.5 PhX(0.25)^0.5 Y^0.5         X^0.5
  │             │             │             │             │             │
  │             Z^0.561       │             │             Z^0.503       │
  │             │             │             │             │             │
  │             SYC───────────┼─────────────┼─────────────SYC           │
  │             │             │             │             │             │
  │             Z^0.014       │             │             Z^0.773       │
  │             │             │             │             │             │
  X^0.5         PhX(0.25)^0.5 Y^0.5         Y^0.5         X^0.5         PhX(0.25)^0.5
  │             │             │             │             │             │
  │             Z^0.075       Z^0.369       Z^0.397       Z^0.789       │
  │             │             │             │             │             │
  │             SYC───────────SYC           SYC───────────SYC           │
  │             │             │             │             │             │
  │             Z^(14/15)     Z^0.651       Z^0.317       Z^0.568       │
  │             │             │             │             │             │
  Y^0.5         Y^0.5         X^0.5         X^0.5         PhX(0.25)^0.5 Y^0.5
  │             │             │             │             │             │
  Z^0.144       Z^0.704       │             │             Z^0.925       Z^0.442
  │             │             │             │             │             │
  SYC───────────SYC           │             │             SYC───────────SYC
  │             │             │             │             │             │
  Z^0.705       Z^0.219       │             │             Z^(10/11)     Z^0.06
  │             │             │             │             │             │
  PhX(0.25)^0.5 X^0.5         Y^0.5         Y^0.5         X^0.5         X^0.5
  │             │             │             │             │             │
  │             Z^0.595       Z^(8/15)      Z^0.33        Z^0.503       │
  │             │             │             │             │             │
  │             SYC───────────SYC           SYC───────────SYC           │
  │             │             │             │             │             │
  │             Z^0.043       Z^0.561       Z^(1/9)       Z^0.607       │
  │             │             │             │             │             │
  Y^0.5         Y^0.5         PhX(0.25)^0.5 X^0.5         Y^0.5         Y^0.5
  │             │             │             │             │             │
  Z^0.912       Z^0.791       │             │             Z^0.792       Z^(2/7)
  │             │             │             │             │             │
  SYC───────────SYC           │             │             SYC───────────SYC
  │             │             │             │             │             │
  Z^0.992       Z^0.959       │             │             Z^(5/8)       Z^0.478
  │             │             │             │             │             │
  PhX(0.25)^0.5 X^0.5         Y^0.5         Y^0.5         PhX(0.25)^0.5 X^0.5
  │             │             │             │             │             │
  Z^0.452       │             Z^0.739       Z^0.982       │             Z^0.587
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ SYC───────────┼─────────────┼─────────────SYC           │             │             │
│ │             │             SYC───────────┼─────────────┼─────────────SYC           │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  Z^0.124       │             Z^0.472       Z^0.119       │             Z^0.107
  │             │             │             │             │             │
  X^0.5         PhX(0.25)^0.5 X^0.5         PhX(0.25)^0.5 Y^0.5         Y^0.5
  │             │             │             │             │             │
  │             Z^0.536       │             │             Z^0.006       │
  │             │             │             │             │             │
  │             SYC───────────┼─────────────┼─────────────SYC           │
  │             │             │             │             │             │
  │             Z^0.301       │             │             Z^(7/16)      │
  │             │             │             │             │             │
  Y^0.5         X^0.5         PhX(0.25)^0.5 X^0.5         X^0.5         X^0.5
  │             │             │             │             │             │
  Z^0.706       │             Z^0.634       Z^0.15        │             Z^(7/16)
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ SYC───────────┼─────────────┼─────────────SYC           │             │             │
│ │             │             SYC───────────┼─────────────┼─────────────SYC           │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  Z^0.746       │             Z^(2/13)      Z^0.831       │             Z^0.568
  │             │             │             │             │             │
  PhX(0.25)^0.5 Y^0.5         X^0.5         PhX(0.25)^0.5 PhX(0.25)^0.5 Y^0.5
  │             │             │             │             │             │
""",
                                    transpose=True)

    qubits = cirq.GridQubit.rect(3, 2)
    depth = 9
    circuit = cirq.experiments.random_quantum_circuit(
        qubits, depth, pattern=cirq.experiments.GMON_EASY_PATTERN, seed=1234)
    cirq.testing.assert_has_diagram(circuit,
                                    """
  (0, 0)        (0, 1)        (1, 0)        (1, 1)        (2, 0)        (2, 1)
  │             │             │             │             │             │
  PhX(0.25)^0.5 Y^0.5         X^0.5         X^0.5         X^0.5         Y^0.5
  │             │             │             │             │             │
  SYC───────────SYC           SYC───────────SYC           SYC───────────SYC
  │             │             │             │             │             │
  Y^0.5         PhX(0.25)^0.5 PhX(0.25)^0.5 PhX(0.25)^0.5 Y^0.5         X^0.5
  │             │             │             │             │             │
  PhX(0.25)^0.5 X^0.5         X^0.5         X^0.5         X^0.5         Y^0.5
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ SYC───────────┼─────────────SYC           │             │             │             │
│ │             SYC───────────┼─────────────SYC           │             │             │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  X^0.5         Y^0.5         Y^0.5         Y^0.5         PhX(0.25)^0.5 X^0.5
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ │             │             SYC───────────┼─────────────SYC           │             │
│ │             │             │             SYC───────────┼─────────────SYC           │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  PhX(0.25)^0.5 PhX(0.25)^0.5 X^0.5         X^0.5         Y^0.5         Y^0.5
  │             │             │             │             │             │
  SYC───────────SYC           SYC───────────SYC           SYC───────────SYC
  │             │             │             │             │             │
  X^0.5         Y^0.5         Y^0.5         PhX(0.25)^0.5 X^0.5         X^0.5
  │             │             │             │             │             │
  Y^0.5         PhX(0.25)^0.5 PhX(0.25)^0.5 Y^0.5         Y^0.5         PhX(0.25)^0.5
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ SYC───────────┼─────────────SYC           │             │             │             │
│ │             SYC───────────┼─────────────SYC           │             │             │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  PhX(0.25)^0.5 X^0.5         Y^0.5         X^0.5         PhX(0.25)^0.5 X^0.5
  │             │             │             │             │             │
┌╴│             │             │             │             │             │            ╶┐
│ │             │             SYC───────────┼─────────────SYC           │             │
│ │             │             │             SYC───────────┼─────────────SYC           │
└╴│             │             │             │             │             │            ╶┘
  │             │             │             │             │             │
  Y^0.5         PhX(0.25)^0.5 PhX(0.25)^0.5 PhX(0.25)^0.5 X^0.5         PhX(0.25)^0.5
  │             │             │             │             │             │
  SYC───────────SYC           SYC───────────SYC           SYC───────────SYC
  │             │             │             │             │             │
  X^0.5         Y^0.5         Y^0.5         X^0.5         Y^0.5         Y^0.5
  │             │             │             │             │             │

""",
                                    transpose=True)
