# Copyright 2022 The Cirq Developers
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

import pytest
import cirq


def test_hardcoded_initial_mapper():
    input_map = {cirq.NamedQubit(str(i)): cirq.NamedQubit(str(-i)) for i in range(1, 6)}
    circuit = cirq.Circuit([cirq.H(cirq.NamedQubit(str(i))) for i in range(1, 6)])
    initial_mapper = cirq.HardCodedInitialMapper(input_map)

    assert input_map == initial_mapper.initial_mapping(circuit)
    assert str(initial_mapper) == f'cirq.HardCodedInitialMapper({input_map})'
    cirq.testing.assert_equivalent_repr(initial_mapper)

    circuit.append(cirq.H(cirq.NamedQubit(str(6))))
    with pytest.raises(
        ValueError, match="The qubits in circuit must be a subset of the keys in the mapping"
    ):
        initial_mapper.initial_mapping(circuit)
