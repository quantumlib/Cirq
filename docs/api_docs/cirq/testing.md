<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DEFAULT_GATE_DOMAIN"/>
</div>

# Module: cirq.testing

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Utilities for testing code.



## Modules

[`asynchronous`](../cirq/testing/asynchronous.md) module

[`circuit_compare`](../cirq/testing/circuit_compare.md) module

[`consistent_decomposition`](../cirq/testing/consistent_decomposition.md) module

[`consistent_pauli_expansion`](../cirq/testing/consistent_pauli_expansion.md) module

[`consistent_phase_by`](../cirq/testing/consistent_phase_by.md) module

[`consistent_protocols`](../cirq/testing/consistent_protocols.md) module

[`consistent_qasm`](../cirq/testing/consistent_qasm.md) module

[`consistent_specified_has_unitary`](../cirq/testing/consistent_specified_has_unitary.md) module

[`equals_tester`](../cirq/testing/equals_tester.md) module: A utility class for testing equality methods.

[`equivalent_repr_eval`](../cirq/testing/equivalent_repr_eval.md) module

[`json`](../cirq/testing/json.md) module

[`lin_alg_utils`](../cirq/testing/lin_alg_utils.md) module: A testing class with utilities for checking linear algebra.

[`logs`](../cirq/testing/logs.md) module: Helper for testing python logging statements.

[`order_tester`](../cirq/testing/order_tester.md) module: A utility class for testing ordering methods.

[`sample_circuits`](../cirq/testing/sample_circuits.md) module

## Classes

[`class EqualsTester`](../cirq/testing/EqualsTester.md): Tests equality against user-provided disjoint equivalence groups.

[`class OrderTester`](../cirq/testing/OrderTester.md): Tests ordering against user-provided disjoint ordered groups or items.

## Functions

[`assert_allclose_up_to_global_phase(...)`](../cirq/testing/assert_allclose_up_to_global_phase.md): Checks if a ~= b * exp(i t) for some t.

[`assert_circuits_with_terminal_measurements_are_equivalent(...)`](../cirq/testing/assert_circuits_with_terminal_measurements_are_equivalent.md): Determines if two circuits have equivalent effects.

[`assert_commutes_magic_method_consistent_with_unitaries(...)`](../cirq/testing/assert_commutes_magic_method_consistent_with_unitaries.md)

[`assert_decompose_is_consistent_with_unitary(...)`](../cirq/testing/assert_decompose_is_consistent_with_unitary.md): Uses `val._unitary_` to check `val._phase_by_`'s behavior.

[`assert_eigengate_implements_consistent_protocols(...)`](../cirq/testing/assert_eigengate_implements_consistent_protocols.md): Checks that an EigenGate subclass is internally consistent and has a

[`assert_equivalent_repr(...)`](../cirq/testing/assert_equivalent_repr.md): Checks that eval(repr(v)) == v.

[`assert_has_consistent_apply_unitary(...)`](../cirq/testing/assert_has_consistent_apply_unitary.md): Tests whether a value's _apply_unitary_ is correct.

[`assert_has_consistent_apply_unitary_for_various_exponents(...)`](../cirq/testing/assert_has_consistent_apply_unitary_for_various_exponents.md): Tests whether a value's _apply_unitary_ is correct.

[`assert_has_consistent_qid_shape(...)`](../cirq/testing/assert_has_consistent_qid_shape.md): Tests whether a value's `_qid_shape_` and `_num_qubits_` are correct and

[`assert_has_consistent_trace_distance_bound(...)`](../cirq/testing/assert_has_consistent_trace_distance_bound.md)

[`assert_has_diagram(...)`](../cirq/testing/assert_has_diagram.md): Determines if a given circuit has the desired text diagram.

[`assert_implements_consistent_protocols(...)`](../cirq/testing/assert_implements_consistent_protocols.md): Checks that a value is internally consistent and has a good __repr__.

[`assert_json_roundtrip_works(...)`](../cirq/testing/assert_json_roundtrip_works.md): Tests that the given object can serialized and de-serialized

[`assert_logs(...)`](../cirq/testing/assert_logs.md): A context manager for testing logging and warning events.

[`assert_pauli_expansion_is_consistent_with_unitary(...)`](../cirq/testing/assert_pauli_expansion_is_consistent_with_unitary.md): Checks Pauli expansion against unitary matrix.

[`assert_phase_by_is_consistent_with_unitary(...)`](../cirq/testing/assert_phase_by_is_consistent_with_unitary.md): Uses `val._unitary_` to check `val._phase_by_`'s behavior.

[`assert_qasm_is_consistent_with_unitary(...)`](../cirq/testing/assert_qasm_is_consistent_with_unitary.md): Uses `val._unitary_` to check `val._qasm_`'s behavior.

[`assert_same_circuits(...)`](../cirq/testing/assert_same_circuits.md): Asserts that two circuits are identical, with a descriptive error.

[`assert_specifies_has_unitary_if_unitary(...)`](../cirq/testing/assert_specifies_has_unitary_if_unitary.md): Checks that unitary values can be cheaply identifies as unitary.

[`asyncio_pending(...)`](../cirq/testing/asyncio_pending.md): Gives the given future a chance to complete, and determines if it didn't.

[`highlight_text_differences(...)`](../cirq/testing/highlight_text_differences.md)

[`nonoptimal_toffoli_circuit(...)`](../cirq/testing/nonoptimal_toffoli_circuit.md)

[`random_circuit(...)`](../cirq/testing/random_circuit.md): Generates a random circuit.

[`random_density_matrix(...)`](../cirq/testing/random_density_matrix.md): Returns a random density matrix distributed with Hilbert-Schmidt measure.

[`random_orthogonal(...)`](../cirq/testing/random_orthogonal.md): Returns a random orthogonal matrix distributed with Haar measure.

[`random_special_orthogonal(...)`](../cirq/testing/random_special_orthogonal.md): Returns a random special orthogonal matrix distributed with Haar measure.

[`random_special_unitary(...)`](../cirq/testing/random_special_unitary.md): Returns a random special unitary distributed with Haar measure.

[`random_superposition(...)`](../cirq/testing/random_superposition.md): Returns a random unit-length vector from the uniform distribution.

[`random_unitary(...)`](../cirq/testing/random_unitary.md): Returns a random unitary matrix distributed with Haar measure.

## Other Members

* `DEFAULT_GATE_DOMAIN` <a id="DEFAULT_GATE_DOMAIN"></a>
