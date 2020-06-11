<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.ops

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Types for representing and methods for manipulating circuit operation trees.



## Modules

[`arithmetic_operation`](../cirq/ops/arithmetic_operation.md) module: Helper class for implementing classical arithmetic operations.

[`clifford_gate`](../cirq/ops/clifford_gate.md) module

[`common_channels`](../cirq/ops/common_channels.md) module: Quantum channels that are commonly used in the literature.

[`common_gates`](../cirq/ops/common_gates.md) module: Quantum gates that are commonly used in the literature.

[`controlled_gate`](../cirq/ops/controlled_gate.md) module

[`controlled_operation`](../cirq/ops/controlled_operation.md) module

[`dense_pauli_string`](../cirq/ops/dense_pauli_string.md) module

[`eigen_gate`](../cirq/ops/eigen_gate.md) module

[`fourier_transform`](../cirq/ops/fourier_transform.md) module

[`fsim_gate`](../cirq/ops/fsim_gate.md) module: Defines the fermionic simulation gate family.

[`gate_features`](../cirq/ops/gate_features.md) module: Marker classes for indicating which additional features gates support.

[`gate_operation`](../cirq/ops/gate_operation.md) module: Basic types defining qubits, gates, and operations.

[`global_phase_op`](../cirq/ops/global_phase_op.md) module: A no-qubit global phase operation.

[`identity`](../cirq/ops/identity.md) module: IdentityGate.

[`linear_combinations`](../cirq/ops/linear_combinations.md) module

[`matrix_gates`](../cirq/ops/matrix_gates.md) module: Quantum gates defined by a matrix.

[`measure_util`](../cirq/ops/measure_util.md) module

[`measurement_gate`](../cirq/ops/measurement_gate.md) module

[`moment`](../cirq/ops/moment.md) module: A simplified time-slice of operations within a sequenced circuit.

[`named_qubit`](../cirq/ops/named_qubit.md) module

[`op_tree`](../cirq/ops/op_tree.md) module: A recursive type describing trees of operations, and utility methods for it.

[`parallel_gate_operation`](../cirq/ops/parallel_gate_operation.md) module

[`parity_gates`](../cirq/ops/parity_gates.md) module: Quantum gates that phase with respect to product-of-pauli observables.

[`pauli_gates`](../cirq/ops/pauli_gates.md) module

[`pauli_interaction_gate`](../cirq/ops/pauli_interaction_gate.md) module

[`pauli_string`](../cirq/ops/pauli_string.md) module

[`pauli_string_phasor`](../cirq/ops/pauli_string_phasor.md) module

[`pauli_string_raw_types`](../cirq/ops/pauli_string_raw_types.md) module

[`phased_iswap_gate`](../cirq/ops/phased_iswap_gate.md) module: ISWAPPowGate conjugated by tensor product Rz(phi) and Rz(-phi).

[`phased_x_gate`](../cirq/ops/phased_x_gate.md) module: An `XPowGate` conjugated by `ZPowGate`s.

[`phased_x_z_gate`](../cirq/ops/phased_x_z_gate.md) module

[`qubit_order`](../cirq/ops/qubit_order.md) module

[`qubit_order_or_list`](../cirq/ops/qubit_order_or_list.md) module: Any method taking a QubitOrder instance should also take raw qubit lists.

[`random_gate_channel`](../cirq/ops/random_gate_channel.md) module

[`raw_types`](../cirq/ops/raw_types.md) module: Basic types defining qubits, gates, and operations.

[`swap_gates`](../cirq/ops/swap_gates.md) module: SWAP and ISWAP gates.

[`tags`](../cirq/ops/tags.md) module: Canonical tags for the TaggedOperation class.

[`three_qubit_gates`](../cirq/ops/three_qubit_gates.md) module: Common quantum gates that target three qubits.

[`two_qubit_diagonal_gate`](../cirq/ops/two_qubit_diagonal_gate.md) module: Creates the gate instrance for a two qubit diagonal gate.

[`wait_gate`](../cirq/ops/wait_gate.md) module

## Classes

[`class AmplitudeDampingChannel`](../cirq/ops/AmplitudeDampingChannel.md): Dampen qubit amplitudes through dissipation.

[`class ArithmeticOperation`](../cirq/ops/ArithmeticOperation.md): A helper class for implementing reversible classical arithmetic.

[`class AsymmetricDepolarizingChannel`](../cirq/ops/AsymmetricDepolarizingChannel.md): A channel that depolarizes asymmetrically along different directions.

[`class BaseDensePauliString`](../cirq/ops/BaseDensePauliString.md): Parent class for `DensePauliString` and `MutableDensePauliString`.

[`class BitFlipChannel`](../cirq/ops/BitFlipChannel.md): Probabilistically flip a qubit from 1 to 0 state or vice versa.

[`class CCNotPowGate`](../cirq/ops/CCNotPowGate.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`class CCXPowGate`](../cirq/ops/CCNotPowGate.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`class CCZPowGate`](../cirq/ops/CCZPowGate.md): A doubly-controlled-Z that can be raised to a power.

[`class CNotPowGate`](../cirq/ops/CNotPowGate.md): A gate that applies a controlled power of an X gate.

[`class CSwapGate`](../cirq/ops/CSwapGate.md): A controlled swap gate. The Fredkin gate.

[`class CXPowGate`](../cirq/ops/CNotPowGate.md): A gate that applies a controlled power of an X gate.

[`class CZPowGate`](../cirq/ops/CZPowGate.md): A gate that applies a phase to the |11⟩ state of two qubits.

[`class ControlledGate`](../cirq/ops/ControlledGate.md): Augments existing gates to have one or more control qubits.

[`class ControlledOperation`](../cirq/ops/ControlledOperation.md): Augments existing operations to have one or more control qubits.

[`class DensePauliString`](../cirq/ops/DensePauliString.md): Parent class for `DensePauliString` and `MutableDensePauliString`.

[`class DepolarizingChannel`](../cirq/ops/DepolarizingChannel.md): A channel that depolarizes a qubit.

[`class EigenGate`](../cirq/ops/EigenGate.md): A gate with a known eigendecomposition.

[`class FSimGate`](../cirq/ops/FSimGate.md): Fermionic simulation gate family.

[`class Gate`](../cirq/ops/Gate.md): An operation type that can be applied to a collection of qubits.

[`class GateOperation`](../cirq/ops/GateOperation.md): An application of a gate to a sequence of qubits.

[`class GeneralizedAmplitudeDampingChannel`](../cirq/ops/GeneralizedAmplitudeDampingChannel.md): Dampen qubit amplitudes through non ideal dissipation.

[`class GlobalPhaseOperation`](../cirq/ops/GlobalPhaseOperation.md): An effect applied to a collection of qubits.

[`class HPowGate`](../cirq/ops/HPowGate.md): A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

[`class ISwapPowGate`](../cirq/ops/ISwapPowGate.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`class IdentityGate`](../cirq/ops/IdentityGate.md): A Gate that perform no operation on qubits.

[`class InterchangeableQubitsGate`](../cirq/ops/InterchangeableQubitsGate.md): Indicates operations should be equal under some qubit permutations.

[`class LinearCombinationOfGates`](../cirq/ops/LinearCombinationOfGates.md): Represents linear operator defined by a linear combination of gates.

[`class LinearCombinationOfOperations`](../cirq/ops/LinearCombinationOfOperations.md): Represents operator defined by linear combination of gate operations.

[`class MatrixGate`](../cirq/ops/MatrixGate.md): A unitary qubit or qudit gate defined entirely by its matrix.

[`class MeasurementGate`](../cirq/ops/MeasurementGate.md): A gate that measures qubits in the computational basis.

[`class Moment`](../cirq/ops/Moment.md): A time-slice of operations within a circuit.

[`class MutableDensePauliString`](../cirq/ops/MutableDensePauliString.md): Parent class for `DensePauliString` and `MutableDensePauliString`.

[`class NamedQubit`](../cirq/ops/NamedQubit.md): A qubit identified by name.

[`class Operation`](../cirq/ops/Operation.md): An effect applied to a collection of qubits.

[`class ParallelGateOperation`](../cirq/ops/ParallelGateOperation.md): An application of several copies of a gate to a group of qubits.

[`class Pauli`](../cirq/ops/Pauli.md): Represents the Pauli gates.

[`class PauliInteractionGate`](../cirq/ops/PauliInteractionGate.md): A CZ conjugated by arbitrary single qubit Cliffords.

[`class PauliString`](../cirq/ops/PauliString.md): An effect applied to a collection of qubits.

[`class PauliStringGateOperation`](../cirq/ops/PauliStringGateOperation.md): An effect applied to a collection of qubits.

[`class PauliStringPhasor`](../cirq/ops/PauliStringPhasor.md): An operation that phases the eigenstates of a Pauli string.

[`class PauliSum`](../cirq/ops/PauliSum.md): Represents operator defined by linear combination of PauliStrings.

[`class PauliTransform`](../cirq/ops/PauliTransform.md): PauliTransform(to, flip)

[`class PhaseDampingChannel`](../cirq/ops/PhaseDampingChannel.md): Dampen qubit phase.

[`class PhaseFlipChannel`](../cirq/ops/PhaseFlipChannel.md): Probabilistically flip the sign of the phase of a qubit.

[`class PhaseGradientGate`](../cirq/ops/PhaseGradientGate.md): Phases each state |k⟩ out of n by e^(2*pi*i*k/n*exponent).

[`class PhasedISwapPowGate`](../cirq/ops/PhasedISwapPowGate.md): Fractional ISWAP conjugated by Z rotations.

[`class PhasedXPowGate`](../cirq/ops/PhasedXPowGate.md): A gate equivalent to the circuit ───Z^-p───X^t───Z^p───.

[`class PhasedXZGate`](../cirq/ops/PhasedXZGate.md): A single qubit operation expressed as $Z^z Z^a X^x Z^{-a}$.

[`class Qid`](../cirq/ops/Qid.md): Identifies a quantum object such as a qubit, qudit, resonator, etc.

[`class QuantumFourierTransformGate`](../cirq/ops/QuantumFourierTransformGate.md): Switches from the computational basis to the frequency basis.

[`class QubitOrder`](../cirq/ops/QubitOrder.md): Defines the kronecker product order of qubits.

[`class RandomGateChannel`](../cirq/ops/RandomGateChannel.md): Applies a sub gate with some probability.

[`class ResetChannel`](../cirq/ops/ResetChannel.md): Reset a qubit back to its |0⟩ state.

[`class SingleQubitCliffordGate`](../cirq/ops/SingleQubitCliffordGate.md): Any single qubit Clifford rotation.

[`class SingleQubitGate`](../cirq/ops/SingleQubitGate.md): A gate that must be applied to exactly one qubit.

[`class SingleQubitPauliStringGateOperation`](../cirq/ops/SingleQubitPauliStringGateOperation.md): A Pauli operation applied to a qubit.

[`class SwapPowGate`](../cirq/ops/SwapPowGate.md): The SWAP gate, possibly raised to a power. Exchanges qubits.

[`class TaggedOperation`](../cirq/ops/TaggedOperation.md): A specific operation instance that has been identified with a set

[`class ThreeQubitDiagonalGate`](../cirq/ops/ThreeQubitDiagonalGate.md): A gate given by a diagonal 8x8 matrix.

[`class ThreeQubitGate`](../cirq/ops/ThreeQubitGate.md): A gate that must be applied to exactly three qubits.

[`class TwoQubitDiagonalGate`](../cirq/ops/TwoQubitDiagonalGate.md): A gate given by a diagonal 4\times 4 matrix.

[`class TwoQubitGate`](../cirq/ops/TwoQubitGate.md): A gate that must be applied to exactly two qubits.

[`class VirtualTag`](../cirq/ops/VirtualTag.md): A TaggedOperation tag indicating that the operation is virtual.

[`class WaitGate`](../cirq/ops/WaitGate.md): A single-qubit idle gate that represents waiting.

[`class XPowGate`](../cirq/ops/XPowGate.md): A gate that rotates around the X axis of the Bloch sphere.

[`class XXPowGate`](../cirq/ops/XXPowGate.md): The X-parity gate, possibly raised to a power.

[`class YPowGate`](../cirq/ops/YPowGate.md): A gate that rotates around the Y axis of the Bloch sphere.

[`class YYPowGate`](../cirq/ops/YYPowGate.md): The Y-parity gate, possibly raised to a power.

[`class ZPowGate`](../cirq/ops/ZPowGate.md): A gate that rotates around the Z axis of the Bloch sphere.

[`class ZZPowGate`](../cirq/ops/ZZPowGate.md): The Z-parity gate, possibly raised to a power.

## Functions

[`CCNOT(...)`](../cirq/ops/CCNOT.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`CCX(...)`](../cirq/ops/CCNOT.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`CCZ(...)`](../cirq/ops/CCZ.md): A doubly-controlled-Z that can be raised to a power.

[`CNOT(...)`](../cirq/ops/CNOT.md): A gate that applies a controlled power of an X gate.

[`CSWAP(...)`](../cirq/ops/CSWAP.md): A controlled swap gate. The Fredkin gate.

[`CX(...)`](../cirq/ops/CNOT.md): A gate that applies a controlled power of an X gate.

[`CZ(...)`](../cirq/ops/CZ.md): A gate that applies a phase to the |11⟩ state of two qubits.

[`FREDKIN(...)`](../cirq/ops/CSWAP.md): A controlled swap gate. The Fredkin gate.

[`H(...)`](../cirq/ops/H.md): A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

[`I(...)`](../cirq/ops/I.md): A Gate that perform no operation on qubits.

[`ISWAP(...)`](../cirq/ops/ISWAP.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`QFT(...)`](../cirq/ops/QFT.md): THIS FUNCTION IS DEPRECATED.

[`S(...)`](../cirq/ops/S.md): A gate that rotates around the Z axis of the Bloch sphere.

[`SWAP(...)`](../cirq/ops/SWAP.md): The SWAP gate, possibly raised to a power. Exchanges qubits.

[`T(...)`](../cirq/ops/T.md): A gate that rotates around the Z axis of the Bloch sphere.

[`TOFFOLI(...)`](../cirq/ops/CCNOT.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`X(...)`](../cirq/ops/X.md)

[`XX(...)`](../cirq/ops/XX.md): The X-parity gate, possibly raised to a power.

[`Y(...)`](../cirq/ops/Y.md)

[`YY(...)`](../cirq/ops/YY.md): The Y-parity gate, possibly raised to a power.

[`Z(...)`](../cirq/ops/Z.md)

[`ZZ(...)`](../cirq/ops/ZZ.md): The Z-parity gate, possibly raised to a power.

[`amplitude_damp(...)`](../cirq/ops/amplitude_damp.md): Returns an AmplitudeDampingChannel with the given probability gamma.

[`asymmetric_depolarize(...)`](../cirq/ops/asymmetric_depolarize.md): Returns a AsymmetricDepolarizingChannel with given parameter.

[`bit_flip(...)`](../cirq/ops/bit_flip.md): Construct a BitFlipChannel that flips a qubit state

[`depolarize(...)`](../cirq/ops/depolarize.md): Returns a DepolarizingChannel with given probability of error.

[`flatten_op_tree(...)`](../cirq/ops/flatten_op_tree.md): Performs an in-order iteration of the operations (leaves) in an OP_TREE.

[`flatten_to_ops(...)`](../cirq/ops/flatten_to_ops.md): Performs an in-order iteration of the operations (leaves) in an OP_TREE.

[`flatten_to_ops_or_moments(...)`](../cirq/ops/flatten_to_ops_or_moments.md): Performs an in-order iteration OP_TREE, yielding ops and moments.

[`freeze_op_tree(...)`](../cirq/ops/freeze_op_tree.md): Replaces all iterables in the OP_TREE with tuples.

[`generalized_amplitude_damp(...)`](../cirq/ops/generalized_amplitude_damp.md): Returns a GeneralizedAmplitudeDampingChannel with the given

[`givens(...)`](../cirq/ops/givens.md): Returns gate with matrix exp(-i angle_rads (Y⊗X - X⊗Y) / 2).

[`identity_each(...)`](../cirq/ops/identity_each.md): Returns a single IdentityGate applied to all the given qubits.

[`measure(...)`](../cirq/ops/measure.md): Returns a single MeasurementGate applied to all the given qubits.

[`measure_each(...)`](../cirq/ops/measure_each.md): Returns a list of operations individually measuring the given qubits.

[`phase_damp(...)`](../cirq/ops/phase_damp.md): Creates a PhaseDampingChannel with damping constant gamma.

[`phase_flip(...)`](../cirq/ops/phase_flip.md): Returns a PhaseFlipChannel that flips a qubit's phase with probability p

[`qft(...)`](../cirq/ops/qft.md): The quantum Fourier transform.

[`reset(...)`](../cirq/ops/reset.md): Returns a `ResetChannel` on the given qubit.

[`riswap(...)`](../cirq/ops/riswap.md): Returns gate with matrix exp(+i angle_rads (X⊗X + Y⊗Y) / 2).

[`rx(...)`](../cirq/ops/rx.md): Returns a gate with the matrix e^{-i X rads / 2}.

[`ry(...)`](../cirq/ops/ry.md): Returns a gate with the matrix e^{-i Y rads / 2}.

[`rz(...)`](../cirq/ops/rz.md): Returns a gate with the matrix e^{-i Z rads / 2}.

[`transform_op_tree(...)`](../cirq/ops/transform_op_tree.md): Maps transformation functions onto the nodes of an OP_TREE.

## Type Aliases

[`OP_TREE`](../cirq/ops/OP_TREE.md)

[`PAULI_GATE_LIKE`](../cirq/ops/PAULI_GATE_LIKE.md)

[`PAULI_STRING_LIKE`](../cirq/ops/PAULI_STRING_LIKE.md)

[`PauliSumLike`](../cirq/ops/PauliSumLike.md)

[`QubitOrderOrList`](../cirq/ops/QubitOrderOrList.md)

