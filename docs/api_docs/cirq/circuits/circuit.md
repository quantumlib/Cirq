<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.circuit" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TIn"/>
<meta itemprop="property" content="TKey"/>
<meta itemprop="property" content="TOut"/>
<meta itemprop="property" content="TYPE_CHECKING"/>
<meta itemprop="property" content="T_DESIRED_GATE_TYPE"/>
</div>

# Module: cirq.circuits.circuit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The circuit data structure.


Circuits consist of a list of Moments, each Moment made up of a set of
Operations. Each Operation is a Gate that acts on some Qubits, for a given
Moment the Operations must all act on distinct Qubits.

## Classes

[`class Circuit`](../../cirq/circuits/Circuit.md): A mutable list of groups of operations to apply to some qubits.

## Other Members

* `TIn` <a id="TIn"></a>
* `TKey` <a id="TKey"></a>
* `TOut` <a id="TOut"></a>
* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
* `T_DESIRED_GATE_TYPE` <a id="T_DESIRED_GATE_TYPE"></a>
