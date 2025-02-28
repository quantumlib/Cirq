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

"""Methods related to optimizing and transforming PauliStrings."""

from cirq.contrib.paulistring.clifford_target_gateset import (
    CliffordTargetGateset as CliffordTargetGateset,
)

from cirq.contrib.paulistring.separate import (
    convert_and_separate_circuit as convert_and_separate_circuit,
    pauli_string_half as pauli_string_half,
    regular_half as regular_half,
)

from cirq.contrib.paulistring.pauli_string_dag import (
    pauli_string_dag_from_circuit as pauli_string_dag_from_circuit,
    pauli_string_reorder_pred as pauli_string_reorder_pred,
)

from cirq.contrib.paulistring.recombine import (
    move_pauli_strings_into_circuit as move_pauli_strings_into_circuit,
)

from cirq.contrib.paulistring.pauli_string_optimize import (
    pauli_string_optimized_circuit as pauli_string_optimized_circuit,
)

from cirq.contrib.paulistring.clifford_optimize import (
    clifford_optimized_circuit as clifford_optimized_circuit,
)

from cirq.contrib.paulistring.optimize import optimized_circuit as optimized_circuit
