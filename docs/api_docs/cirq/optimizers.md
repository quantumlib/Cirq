<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.optimizers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Circuit transformation utilities.



## Modules

[`controlled_gate_decomposition`](../cirq/optimizers/controlled_gate_decomposition.md) module

[`convert_to_cz_and_single_gates`](../cirq/optimizers/convert_to_cz_and_single_gates.md) module

[`cphase_to_fsim`](../cirq/optimizers/cphase_to_fsim.md) module

[`decompositions`](../cirq/optimizers/decompositions.md) module: Utility methods related to optimizing quantum circuits.

[`drop_empty_moments`](../cirq/optimizers/drop_empty_moments.md) module: An optimization pass that removes empty moments from a circuit.

[`drop_negligible`](../cirq/optimizers/drop_negligible.md) module: An optimization pass that removes operations with tiny effects.

[`eject_phased_paulis`](../cirq/optimizers/eject_phased_paulis.md) module: Pushes 180 degree rotations around axes in the XY plane later in the circuit.

[`eject_z`](../cirq/optimizers/eject_z.md) module: An optimization pass that pushes Z gates later and later in the circuit.

[`expand_composite`](../cirq/optimizers/expand_composite.md) module: An optimizer that expands composite operations via <a href="../cirq/protocols/decompose.md"><code>cirq.decompose</code></a>.

[`merge_interactions`](../cirq/optimizers/merge_interactions.md) module: An optimization pass that combines adjacent single-qubit rotations.

[`merge_single_qubit_gates`](../cirq/optimizers/merge_single_qubit_gates.md) module: An optimization pass that combines adjacent single-qubit rotations.

[`stratify`](../cirq/optimizers/stratify.md) module

[`synchronize_terminal_measurements`](../cirq/optimizers/synchronize_terminal_measurements.md) module: An optimization pass to put as many measurements possible at the end.

[`two_qubit_decompositions`](../cirq/optimizers/two_qubit_decompositions.md) module: Utility methods related to optimizing quantum circuits.

[`two_qubit_to_fsim`](../cirq/optimizers/two_qubit_to_fsim.md) module

## Classes

[`class ConvertToCzAndSingleGates`](../cirq/optimizers/ConvertToCzAndSingleGates.md): Attempts to convert strange multi-qubit gates into CZ and single qubit

[`class DropEmptyMoments`](../cirq/optimizers/DropEmptyMoments.md): Removes empty moments from a circuit.

[`class DropNegligible`](../cirq/optimizers/DropNegligible.md): An optimization pass that removes operations with tiny effects.

[`class EjectPhasedPaulis`](../cirq/optimizers/EjectPhasedPaulis.md): Pushes X, Y, and PhasedX gates towards the end of the circuit.

[`class EjectZ`](../cirq/optimizers/EjectZ.md): Pushes Z gates towards the end of the circuit.

[`class ExpandComposite`](../cirq/optimizers/ExpandComposite.md): An optimizer that expands composite operations via <a href="../cirq/protocols/decompose.md"><code>cirq.decompose</code></a>.

[`class MergeInteractions`](../cirq/optimizers/MergeInteractions.md): Combines series of adjacent one and two-qubit gates operating on a pair

[`class MergeSingleQubitGates`](../cirq/optimizers/MergeSingleQubitGates.md): Optimizes runs of adjacent unitary 1-qubit operations.

[`class SynchronizeTerminalMeasurements`](../cirq/optimizers/SynchronizeTerminalMeasurements.md): Move measurements to the end of the circuit.

## Functions

[`compute_cphase_exponents_for_fsim_decomposition(...)`](../cirq/optimizers/compute_cphase_exponents_for_fsim_decomposition.md): Returns intervals of CZPowGate exponents valid for FSim decomposition.

[`decompose_cphase_into_two_fsim(...)`](../cirq/optimizers/decompose_cphase_into_two_fsim.md): Decomposes CZPowGate into two FSimGates.

[`decompose_multi_controlled_rotation(...)`](../cirq/optimizers/decompose_multi_controlled_rotation.md): Implements action of multi-controlled unitary gate.

[`decompose_multi_controlled_x(...)`](../cirq/optimizers/decompose_multi_controlled_x.md): Implements action of multi-controlled Pauli X gate.

[`decompose_two_qubit_interaction_into_four_fsim_gates_via_b(...)`](../cirq/optimizers/decompose_two_qubit_interaction_into_four_fsim_gates_via_b.md): Decomposes operations into an FSimGate near theta=pi/2, phi=0.

[`is_negligible_turn(...)`](../cirq/optimizers/is_negligible_turn.md)

[`merge_single_qubit_gates_into_phased_x_z(...)`](../cirq/optimizers/merge_single_qubit_gates_into_phased_x_z.md): Canonicalizes runs of single-qubit rotations in a circuit.

[`merge_single_qubit_gates_into_phxz(...)`](../cirq/optimizers/merge_single_qubit_gates_into_phxz.md): Canonicalizes runs of single-qubit rotations in a circuit.

[`single_qubit_matrix_to_gates(...)`](../cirq/optimizers/single_qubit_matrix_to_gates.md): Implements a single-qubit operation with few gates.

[`single_qubit_matrix_to_pauli_rotations(...)`](../cirq/optimizers/single_qubit_matrix_to_pauli_rotations.md): Implements a single-qubit operation with few rotations.

[`single_qubit_matrix_to_phased_x_z(...)`](../cirq/optimizers/single_qubit_matrix_to_phased_x_z.md): Implements a single-qubit operation with a PhasedX and Z gate.

[`single_qubit_matrix_to_phxz(...)`](../cirq/optimizers/single_qubit_matrix_to_phxz.md): Implements a single-qubit operation with a PhasedXZ gate.

[`single_qubit_op_to_framed_phase_form(...)`](../cirq/optimizers/single_qubit_op_to_framed_phase_form.md): Decomposes a 2x2 unitary M into U^-1 * diag(1, r) * U * diag(g, g).

[`stratified_circuit(...)`](../cirq/optimizers/stratified_circuit.md): Repacks avoiding simultaneous operations with different classes.

[`two_qubit_matrix_to_operations(...)`](../cirq/optimizers/two_qubit_matrix_to_operations.md): Decomposes a two-qubit operation into Z/XY/CZ gates.

