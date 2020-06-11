<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DEFAULT_RESOLVERS"/>
</div>

# Module: cirq.protocols

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Modules

[`act_on_protocol`](../cirq/protocols/act_on_protocol.md) module: A protocol that wouldn't exist if python had __rimul__.

[`apply_channel_protocol`](../cirq/protocols/apply_channel_protocol.md) module: A protocol for implementing high performance channel evolutions.

[`apply_mixture_protocol`](../cirq/protocols/apply_mixture_protocol.md) module: A protocol for implementing high performance mixture evolutions.

[`apply_unitary_protocol`](../cirq/protocols/apply_unitary_protocol.md) module: A protocol for implementing high performance unitary left-multiplies.

[`approximate_equality_protocol`](../cirq/protocols/approximate_equality_protocol.md) module

[`circuit_diagram_info_protocol`](../cirq/protocols/circuit_diagram_info_protocol.md) module

[`commutes_protocol`](../cirq/protocols/commutes_protocol.md) module: Protocol for determining commutativity.

[`decompose_protocol`](../cirq/protocols/decompose_protocol.md) module

[`equal_up_to_global_phase_protocol`](../cirq/protocols/equal_up_to_global_phase_protocol.md) module

[`has_stabilizer_effect_protocol`](../cirq/protocols/has_stabilizer_effect_protocol.md) module

[`has_unitary_protocol`](../cirq/protocols/has_unitary_protocol.md) module

[`inverse_protocol`](../cirq/protocols/inverse_protocol.md) module

[`json_serialization`](../cirq/protocols/json_serialization.md) module

[`measurement_key_protocol`](../cirq/protocols/measurement_key_protocol.md) module: Protocol for object that have measurement keys.

[`mixture_protocol`](../cirq/protocols/mixture_protocol.md) module: Protocol for objects that are mixtures (probabilistic combinations).

[`mul_protocol`](../cirq/protocols/mul_protocol.md) module

[`pauli_expansion_protocol`](../cirq/protocols/pauli_expansion_protocol.md) module: Protocol for obtaining expansion of linear operators in Pauli basis.

[`phase_protocol`](../cirq/protocols/phase_protocol.md) module

[`pow_protocol`](../cirq/protocols/pow_protocol.md) module

[`qid_shape_protocol`](../cirq/protocols/qid_shape_protocol.md) module

[`unitary_protocol`](../cirq/protocols/unitary_protocol.md) module

## Classes

[`class ApplyChannelArgs`](../cirq/protocols/ApplyChannelArgs.md): Arguments for efficiently performing a channel.

[`class ApplyMixtureArgs`](../cirq/protocols/ApplyMixtureArgs.md): Arguments for performing a mixture of unitaries.

[`class ApplyUnitaryArgs`](../cirq/protocols/ApplyUnitaryArgs.md): Arguments for performing an efficient left-multiplication by a unitary.

[`class CircuitDiagramInfo`](../cirq/protocols/CircuitDiagramInfo.md): Describes how to draw an operation in a circuit diagram.

[`class CircuitDiagramInfoArgs`](../cirq/protocols/CircuitDiagramInfoArgs.md): A request for information on drawing an operation in a circuit diagram.

[`class QasmArgs`](../cirq/protocols/QasmArgs.md)

[`class QuilFormatter`](../cirq/protocols/QuilFormatter.md): A unique formatter to correctly output values to QUIL.

[`class SupportsActOn`](../cirq/protocols/SupportsActOn.md): An object that explicitly specifies how to act on simulator states.

[`class SupportsApplyChannel`](../cirq/protocols/SupportsApplyChannel.md): An object that can efficiently implement a channel.

[`class SupportsApplyMixture`](../cirq/protocols/SupportsApplyMixture.md): An object that can efficiently implement a mixture.

[`class SupportsApproximateEquality`](../cirq/protocols/SupportsApproximateEquality.md): Object which can be compared approximately.

[`class SupportsChannel`](../cirq/protocols/SupportsChannel.md): An object that may be describable as a quantum channel.

[`class SupportsCircuitDiagramInfo`](../cirq/protocols/SupportsCircuitDiagramInfo.md): A diagrammable operation on qubits.

[`class SupportsCommutes`](../cirq/protocols/SupportsCommutes.md): An object that can determine commutation relationships vs others.

[`class SupportsConsistentApplyUnitary`](../cirq/protocols/SupportsConsistentApplyUnitary.md): An object that can be efficiently left-multiplied into tensors.

[`class SupportsDecompose`](../cirq/protocols/SupportsDecompose.md): An object that can be decomposed into simpler operations.

[`class SupportsDecomposeWithQubits`](../cirq/protocols/SupportsDecomposeWithQubits.md): An object that can be decomposed into operations on given qubits.

[`class SupportsEqualUpToGlobalPhase`](../cirq/protocols/SupportsEqualUpToGlobalPhase.md): Object which can be compared for equality mod global phase.

[`class SupportsExplicitHasUnitary`](../cirq/protocols/SupportsExplicitHasUnitary.md): An object that explicitly specifies whether it has a unitary effect.

[`class SupportsExplicitNumQubits`](../cirq/protocols/SupportsExplicitNumQubits.md): A unitary, channel, mixture or other object that operates on a known

[`class SupportsExplicitQidShape`](../cirq/protocols/SupportsExplicitQidShape.md): A unitary, channel, mixture or other object that operates on a known

[`class SupportsJSON`](../cirq/protocols/SupportsJSON.md): An object that can be turned into JSON dictionaries.

[`class SupportsMeasurementKey`](../cirq/protocols/SupportsMeasurementKey.md): An object that is a measurement and has a measurement key or keys.

[`class SupportsMixture`](../cirq/protocols/SupportsMixture.md): An object that decomposes into a probability distribution of unitaries.

[`class SupportsParameterization`](../cirq/protocols/SupportsParameterization.md): An object that can be parameterized by Symbols and resolved

[`class SupportsPhase`](../cirq/protocols/SupportsPhase.md): An effect that can be phased around the Z axis of target qubits.

[`class SupportsQasm`](../cirq/protocols/SupportsQasm.md): An object that can be turned into QASM code.

[`class SupportsQasmWithArgs`](../cirq/protocols/SupportsQasmWithArgs.md): An object that can be turned into QASM code.

[`class SupportsQasmWithArgsAndQubits`](../cirq/protocols/SupportsQasmWithArgsAndQubits.md): An object that can be turned into QASM code if it knows its qubits.

[`class SupportsTraceDistanceBound`](../cirq/protocols/SupportsTraceDistanceBound.md): An effect with known bounds on how easy it is to detect.

[`class SupportsUnitary`](../cirq/protocols/SupportsUnitary.md): An object that may be describable by a unitary matrix.

## Functions

[`act_on(...)`](../cirq/protocols/act_on.md): Applies an action to a state argument.

[`apply_channel(...)`](../cirq/protocols/apply_channel.md): High performance evolution under a channel evolution.

[`apply_mixture(...)`](../cirq/protocols/apply_mixture.md): High performance evolution under a mixture of unitaries evolution.

[`apply_unitaries(...)`](../cirq/protocols/apply_unitaries.md): Apply a series of unitaries onto a state tensor.

[`apply_unitary(...)`](../cirq/protocols/apply_unitary.md): High performance left-multiplication of a unitary effect onto a tensor.

[`approx_eq(...)`](../cirq/protocols/approx_eq.md): Approximately compares two objects.

[`channel(...)`](../cirq/protocols/channel.md): Returns a list of matrices describing the channel for the given value.

[`circuit_diagram_info(...)`](../cirq/protocols/circuit_diagram_info.md): Requests information on drawing an operation in a circuit diagram.

[`commutes(...)`](../cirq/protocols/commutes.md): Determines whether two values commute.

[`decompose(...)`](../cirq/protocols/decompose.md): Recursively decomposes a value into <a href="../cirq/ops/Operation.md"><code>cirq.Operation</code></a>s meeting a criteria.

[`decompose_once(...)`](../cirq/protocols/decompose_once.md): Decomposes a value into operations, if possible.

[`decompose_once_with_qubits(...)`](../cirq/protocols/decompose_once_with_qubits.md): Decomposes a value into operations on the given qubits.

[`definitely_commutes(...)`](../cirq/protocols/definitely_commutes.md): Determines whether two values definitely commute.

[`equal_up_to_global_phase(...)`](../cirq/protocols/equal_up_to_global_phase.md): Determine whether two objects are equal up to global phase.

[`has_channel(...)`](../cirq/protocols/has_channel.md): Returns whether the value has a channel representation.

[`has_mixture(...)`](../cirq/protocols/has_mixture.md): Returns whether the value has a mixture representation.

[`has_mixture_channel(...)`](../cirq/protocols/has_mixture_channel.md): THIS FUNCTION IS DEPRECATED.

[`has_stabilizer_effect(...)`](../cirq/protocols/has_stabilizer_effect.md): Returns whether the input has a stabilizer effect.

[`has_unitary(...)`](../cirq/protocols/has_unitary.md): Determines whether the value has a unitary effect.

[`inverse(...)`](../cirq/protocols/inverse.md): Returns the inverse `val**-1` of the given value, if defined.

[`is_measurement(...)`](../cirq/protocols/is_measurement.md): Determines whether or not the given value is a measurement.

[`is_parameterized(...)`](../cirq/protocols/is_parameterized.md): Returns whether the object is parameterized with any Symbols.

[`json_serializable_dataclass(...)`](../cirq/protocols/json_serializable_dataclass.md): Create a dataclass that supports JSON serialization

[`measurement_key(...)`](../cirq/protocols/measurement_key.md): Get the single measurement key for the given value.

[`measurement_keys(...)`](../cirq/protocols/measurement_keys.md): Gets the measurement keys of measurements within the given value.

[`mixture(...)`](../cirq/protocols/mixture.md): Return a sequence of tuples representing a probabilistic unitary.

[`mixture_channel(...)`](../cirq/protocols/mixture_channel.md): THIS FUNCTION IS DEPRECATED.

[`mul(...)`](../cirq/protocols/mul.md): Returns lhs * rhs, or else a default if the operator is not implemented.

[`num_qubits(...)`](../cirq/protocols/num_qubits.md): Returns the number of qubits, qudits, or qids `val` operates on.

[`obj_to_dict_helper(...)`](../cirq/protocols/obj_to_dict_helper.md): Construct a dictionary containing attributes from obj

[`pauli_expansion(...)`](../cirq/protocols/pauli_expansion.md): Returns coefficients of the expansion of val in the Pauli basis.

[`phase_by(...)`](../cirq/protocols/phase_by.md): Returns a phased version of the effect.

[`pow(...)`](../cirq/protocols/pow.md): Returns `val**factor` of the given value, if defined.

[`qasm(...)`](../cirq/protocols/qasm.md): Returns QASM code for the given value, if possible.

[`qid_shape(...)`](../cirq/protocols/qid_shape.md): Returns a tuple describing the number of quantum levels of each

[`quil(...)`](../cirq/protocols/quil.md): Returns the QUIL code for the given value.

[`read_json(...)`](../cirq/protocols/read_json.md): Read a JSON file that optionally contains cirq objects.

[`resolve_parameters(...)`](../cirq/protocols/resolve_parameters.md): Resolves symbol parameters in the effect using the param resolver.

[`to_json(...)`](../cirq/protocols/to_json.md): Write a JSON file containing a representation of obj.

[`trace_distance_bound(...)`](../cirq/protocols/trace_distance_bound.md): Returns a maximum on the trace distance between this effect's input

[`trace_distance_from_angle_list(...)`](../cirq/protocols/trace_distance_from_angle_list.md): Given a list of arguments of the eigenvalues of a unitary matrix,

[`unitary(...)`](../cirq/protocols/unitary.md): Returns a unitary matrix describing the given value.

[`validate_mixture(...)`](../cirq/protocols/validate_mixture.md): Validates that the mixture's tuple are valid probabilities.

## Other Members

* `DEFAULT_RESOLVERS` <a id="DEFAULT_RESOLVERS"></a>
