<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="Bristlecone"/>
<meta itemprop="property" content="FSIM_GATESET"/>
<meta itemprop="property" content="Foxtail"/>
<meta itemprop="property" content="SQRT_ISWAP_GATESET"/>
<meta itemprop="property" content="SYC_GATESET"/>
<meta itemprop="property" content="Sycamore"/>
<meta itemprop="property" content="Sycamore23"/>
<meta itemprop="property" content="XMON"/>
</div>

# Module: cirq.google

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Modules

[`api`](../cirq/google/api.md) module: Helpers for converting to/from api data formats.

[`arg_func_langs`](../cirq/google/arg_func_langs.md) module

[`common_serializers`](../cirq/google/common_serializers.md) module: Common Serializers that can be used by APIs.

[`devices`](../cirq/google/devices.md) module

[`engine`](../cirq/google/engine.md) module: Client for running on Google's Quantum Engine.

[`gate_sets`](../cirq/google/gate_sets.md) module: Gate sets supported by Google's apis.

[`line`](../cirq/google/line.md) module

[`op_deserializer`](../cirq/google/op_deserializer.md) module

[`op_serializer`](../cirq/google/op_serializer.md) module

[`ops`](../cirq/google/ops.md) module

[`optimizers`](../cirq/google/optimizers.md) module: Package for optimizers and gate compilers related to Google-specific devices.

[`serializable_gate_set`](../cirq/google/serializable_gate_set.md) module: Support for serializing and deserializing cirq.google.api.v2 protos.

## Classes

[`class AnnealSequenceSearchStrategy`](../cirq/google/AnnealSequenceSearchStrategy.md): Linearized sequence search using simulated annealing method.

[`class Calibration`](../cirq/google/Calibration.md): A convenience wrapper for calibrations that acts like a dictionary.

[`class ConvertToSqrtIswapGates`](../cirq/google/ConvertToSqrtIswapGates.md): Attempts to convert gates into ISWAP**-0.5 gates.

[`class ConvertToSycamoreGates`](../cirq/google/ConvertToSycamoreGates.md): Attempts to convert non-native gates into SycamoreGates.

[`class ConvertToXmonGates`](../cirq/google/ConvertToXmonGates.md): Attempts to convert strange gates into XmonGates.

[`class DeserializingArg`](../cirq/google/DeserializingArg.md): Specification of the arguments to deserialize an argument to a gate.

[`class Engine`](../cirq/google/Engine.md): Runs programs via the Quantum Engine API.

[`class EngineJob`](../cirq/google/EngineJob.md): A job created via the Quantum Engine API.

[`class EngineProcessor`](../cirq/google/EngineProcessor.md): A processor available via the Quantum Engine API.

[`class EngineProgram`](../cirq/google/EngineProgram.md): A program created via the Quantum Engine API.

[`class EngineTimeSlot`](../cirq/google/EngineTimeSlot.md): A python wrapping of a Quantum Engine timeslot.

[`class GateOpDeserializer`](../cirq/google/GateOpDeserializer.md): Describes how to deserialize a proto to a given Gate type.

[`class GateOpSerializer`](../cirq/google/GateOpSerializer.md): Describes how to serialize a GateOperation for a given Gate type.

[`class GateTabulation`](../cirq/google/GateTabulation.md): A 2-qubit gate compiler based on precomputing/tabulating gate products.

[`class GreedySequenceSearchStrategy`](../cirq/google/GreedySequenceSearchStrategy.md): Greedy search method for linear sequence of qubits on a chip.

[`class LinePlacementStrategy`](../cirq/google/LinePlacementStrategy.md): Choice and options for the line placement calculation method.

[`class PhysicalZTag`](../cirq/google/PhysicalZTag.md): Class to add as a tag onto an Operation to denote a Physical Z operation.

[`class ProtoVersion`](../cirq/google/ProtoVersion.md): Protocol buffer version to use for requests to the quantum engine.

[`class QuantumEngineSampler`](../cirq/google/QuantumEngineSampler.md): A sampler that samples from processors managed by the Quantum Engine.

[`class SerializableDevice`](../cirq/google/SerializableDevice.md): Device object generated from a device specification proto.

[`class SerializableGateSet`](../cirq/google/SerializableGateSet.md): A class for serializing and deserializing programs and operations.

[`class SerializingArg`](../cirq/google/SerializingArg.md): Specification of the arguments for a Gate and its serialization.

[`class SycamoreGate`](../cirq/google/SycamoreGate.md): The Sycamore gate is a two-qubit gate equivalent to FSimGate(π/2, π/6).

[`class XmonDevice`](../cirq/google/XmonDevice.md): A device with qubits placed in a grid. Neighboring qubits can interact.

## Functions

[`SYC(...)`](../cirq/google/SYC.md): The Sycamore gate is a two-qubit gate equivalent to FSimGate(π/2, π/6).

[`engine_from_environment(...)`](../cirq/google/engine_from_environment.md): Returns an Engine instance configured using environment variables.

[`is_native_xmon_gate(...)`](../cirq/google/is_native_xmon_gate.md): Check if a gate is a native xmon gate.

[`is_native_xmon_op(...)`](../cirq/google/is_native_xmon_op.md): Check if the gate corresponding to an operation is a native xmon gate.

[`line_on_device(...)`](../cirq/google/line_on_device.md): Searches for linear sequence of qubits on device.

[`optimized_for_sycamore(...)`](../cirq/google/optimized_for_sycamore.md): Optimizes a circuit for Google devices.

[`optimized_for_xmon(...)`](../cirq/google/optimized_for_xmon.md)

[`pack_results(...)`](../cirq/google/pack_results.md): Pack measurement results into a byte string.

[`unpack_results(...)`](../cirq/google/unpack_results.md): Unpack data from a bitstring into individual measurement results.

## Other Members

* `Bristlecone` <a id="Bristlecone"></a>
* `FSIM_GATESET` <a id="FSIM_GATESET"></a>
* `Foxtail` <a id="Foxtail"></a>
* `SQRT_ISWAP_GATESET` <a id="SQRT_ISWAP_GATESET"></a>
* `SYC_GATESET` <a id="SYC_GATESET"></a>
* `Sycamore` <a id="Sycamore"></a>
* `Sycamore23` <a id="Sycamore23"></a>
* `XMON` <a id="XMON"></a>
