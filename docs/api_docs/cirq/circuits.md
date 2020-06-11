<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.circuits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Types and methods related to building and optimizing sequenced circuits.



## Modules

[`circuit`](../cirq/circuits/circuit.md) module: The circuit data structure.

[`circuit_dag`](../cirq/circuits/circuit_dag.md) module

[`insert_strategy`](../cirq/circuits/insert_strategy.md) module: Hard-coded options for adding multiple operations to a circuit.

[`optimization_pass`](../cirq/circuits/optimization_pass.md) module: Defines the OptimizationPass type.

[`qasm_output`](../cirq/circuits/qasm_output.md) module: Utility classes for representing QASM.

[`quil_output`](../cirq/circuits/quil_output.md) module

[`text_diagram_drawer`](../cirq/circuits/text_diagram_drawer.md) module

## Classes

[`class Circuit`](../cirq/circuits/Circuit.md): A mutable list of groups of operations to apply to some qubits.

[`class CircuitDag`](../cirq/circuits/CircuitDag.md): A representation of a Circuit as a directed acyclic graph.

[`class InsertStrategy`](../cirq/circuits/InsertStrategy.md): Indicates preferences on how to add multiple operations to a circuit.

[`class PointOptimizationSummary`](../cirq/circuits/PointOptimizationSummary.md): A description of a local optimization to perform.

[`class PointOptimizer`](../cirq/circuits/PointOptimizer.md): Makes circuit improvements focused on a specific location.

[`class QasmOutput`](../cirq/circuits/QasmOutput.md)

[`class QuilOutput`](../cirq/circuits/QuilOutput.md): An object for passing operations and qubits then outputting them to

[`class TextDiagramDrawer`](../cirq/circuits/TextDiagramDrawer.md): A utility class for creating simple text diagrams.

[`class Unique`](../cirq/circuits/Unique.md): A wrapper for a value that doesn't compare equal to other instances.

