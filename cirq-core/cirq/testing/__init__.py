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

"""Utilities for testing code."""

from cirq.testing.circuit_compare import (
    # pylint: disable=line-too-long
    assert_circuits_with_terminal_measurements_are_equivalent as assert_circuits_with_terminal_measurements_are_equivalent,
    assert_circuits_have_same_unitary_given_final_permutation as assert_circuits_have_same_unitary_given_final_permutation,
    assert_has_consistent_apply_channel as assert_has_consistent_apply_channel,
    assert_has_consistent_apply_unitary as assert_has_consistent_apply_unitary,
    assert_has_consistent_apply_unitary_for_various_exponents as assert_has_consistent_apply_unitary_for_various_exponents,
    assert_has_diagram as assert_has_diagram,
    assert_same_circuits as assert_same_circuits,
    highlight_text_differences as highlight_text_differences,
    assert_has_consistent_qid_shape as assert_has_consistent_qid_shape,
)

from cirq.testing.consistent_act_on import (
    # pylint: disable=line-too-long
    assert_all_implemented_act_on_effects_match_unitary as assert_all_implemented_act_on_effects_match_unitary,
)

from cirq.testing.consistent_channels import (
    assert_consistent_channel as assert_consistent_channel,
    assert_consistent_mixture as assert_consistent_mixture,
)

from cirq.testing.consistent_controlled_gate_op import (
    assert_controlled_and_controlled_by_identical as assert_controlled_and_controlled_by_identical,
    assert_controlled_unitary_consistent as assert_controlled_unitary_consistent,
)

from cirq.testing.consistent_decomposition import (
    assert_decompose_ends_at_default_gateset as assert_decompose_ends_at_default_gateset,
    assert_decompose_is_consistent_with_unitary as assert_decompose_is_consistent_with_unitary,
)

from cirq.testing.consistent_pauli_expansion import (
    # pylint: disable=line-too-long
    assert_pauli_expansion_is_consistent_with_unitary as assert_pauli_expansion_is_consistent_with_unitary,
)

from cirq.testing.consistent_phase_by import (
    assert_phase_by_is_consistent_with_unitary as assert_phase_by_is_consistent_with_unitary,
)

from cirq.testing.consistent_protocols import (
    # pylint: disable=line-too-long
    assert_eigengate_implements_consistent_protocols as assert_eigengate_implements_consistent_protocols,
    assert_has_consistent_trace_distance_bound as assert_has_consistent_trace_distance_bound,
    assert_implements_consistent_protocols as assert_implements_consistent_protocols,
    assert_commutes_magic_method_consistent_with_unitaries as assert_commutes_magic_method_consistent_with_unitaries,
)

from cirq.testing.consistent_qasm import (
    assert_qasm_is_consistent_with_unitary as assert_qasm_is_consistent_with_unitary,
)

from cirq.testing.consistent_resolve_parameters import (
    assert_consistent_resolve_parameters as assert_consistent_resolve_parameters,
)

from cirq.testing.consistent_specified_has_unitary import (
    assert_specifies_has_unitary_if_unitary as assert_specifies_has_unitary_if_unitary,
)

from cirq.testing.deprecation import assert_deprecated as assert_deprecated

from cirq.testing.devices import ValidatingTestDevice as ValidatingTestDevice

from cirq.testing.equals_tester import EqualsTester as EqualsTester

from cirq.testing.equivalent_basis_map import (
    assert_equivalent_computational_basis_map as assert_equivalent_computational_basis_map,
)

from cirq.testing.equivalent_repr_eval import assert_equivalent_repr as assert_equivalent_repr

from cirq.testing.gate_features import (
    SingleQubitGate as SingleQubitGate,
    TwoQubitGate as TwoQubitGate,
    ThreeQubitGate as ThreeQubitGate,
    DoesNotSupportSerializationGate as DoesNotSupportSerializationGate,
)

from cirq.testing.json import assert_json_roundtrip_works as assert_json_roundtrip_works

from cirq.testing.lin_alg_utils import (
    assert_allclose_up_to_global_phase as assert_allclose_up_to_global_phase,
    random_density_matrix as random_density_matrix,
    random_orthogonal as random_orthogonal,
    random_special_orthogonal as random_special_orthogonal,
    random_special_unitary as random_special_unitary,
    random_superposition as random_superposition,
    random_unitary as random_unitary,
)

from cirq.testing.logs import assert_logs as assert_logs

from cirq.testing.no_identifier_qubit import NoIdentifierQubit as NoIdentifierQubit

from cirq.testing.op_tree import assert_equivalent_op_tree as assert_equivalent_op_tree

from cirq.testing.order_tester import OrderTester as OrderTester

from cirq.testing.pytest_randomly_utils import (
    retry_once_with_later_random_values as retry_once_with_later_random_values,
)

from cirq.testing.random_circuit import (
    DEFAULT_GATE_DOMAIN as DEFAULT_GATE_DOMAIN,
    random_circuit as random_circuit,
    random_two_qubit_circuit_with_czs as random_two_qubit_circuit_with_czs,
)

from cirq.testing.repr_pretty_tester import (
    assert_repr_pretty as assert_repr_pretty,
    assert_repr_pretty_contains as assert_repr_pretty_contains,
    FakePrinter as FakePrinter,
)

from cirq.testing.routing_devices import (
    construct_grid_device as construct_grid_device,
    construct_ring_device as construct_ring_device,
    RoutingTestingDevice as RoutingTestingDevice,
)

from cirq.testing.sample_circuits import nonoptimal_toffoli_circuit as nonoptimal_toffoli_circuit

from cirq.testing.sample_gates import (
    PhaseUsingCleanAncilla as PhaseUsingCleanAncilla,
    PhaseUsingDirtyAncilla as PhaseUsingDirtyAncilla,
)

from cirq.testing.consistent_unitary import (
    assert_unitary_is_consistent as assert_unitary_is_consistent,
)
