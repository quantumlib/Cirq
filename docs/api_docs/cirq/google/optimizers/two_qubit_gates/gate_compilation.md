<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.two_qubit_gates.gate_compilation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.google.optimizers.two_qubit_gates.gate_compilation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/gate_compilation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Attempt to tabulate single qubit gates required to generate a target 2Q gate

with a product A k A.

## Classes

[`class GateTabulation`](../../../../cirq/google/GateTabulation.md): A 2-qubit gate compiler based on precomputing/tabulating gate products.

[`class TwoQubitGateCompilation`](../../../../cirq/google/optimizers/two_qubit_gates/gate_compilation/TwoQubitGateCompilation.md): Represents a compilation of a target 2-qubit with respect to a base gate.

## Functions

[`gate_product_tabulation(...)`](../../../../cirq/google/optimizers/gate_product_tabulation.md): Generate a GateTabulation for a base two qubit unitary.

[`reduce(...)`](../../../../cirq/google/optimizers/two_qubit_gates/gate_compilation/reduce.md): reduce(function, sequence[, initial]) -> value

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
