<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.decompose_protocol" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="RaiseTypeErrorIfNotProvided"/>
<meta itemprop="property" content="TDefault"/>
<meta itemprop="property" content="TError"/>
<meta itemprop="property" content="TValue"/>
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.protocols.decompose_protocol

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/decompose_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Classes

[`class SupportsDecompose`](../../cirq/protocols/SupportsDecompose.md): An object that can be decomposed into simpler operations.

[`class SupportsDecomposeWithQubits`](../../cirq/protocols/SupportsDecomposeWithQubits.md): An object that can be decomposed into operations on given qubits.

## Functions

[`decompose(...)`](../../cirq/protocols/decompose.md): Recursively decomposes a value into <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a>s meeting a criteria.

[`decompose_once(...)`](../../cirq/protocols/decompose_once.md): Decomposes a value into operations, if possible.

[`decompose_once_with_qubits(...)`](../../cirq/protocols/decompose_once_with_qubits.md): Decomposes a value into operations on the given qubits.

## Type Aliases

[`DecomposeResult`](../../cirq/protocols/decompose_protocol/DecomposeResult.md)

[`OpDecomposer`](../../cirq/protocols/decompose_protocol/OpDecomposer.md)

## Other Members

* `RaiseTypeErrorIfNotProvided` <a id="RaiseTypeErrorIfNotProvided"></a>
* `TDefault` <a id="TDefault"></a>
* `TError` <a id="TError"></a>
* `TValue` <a id="TValue"></a>
* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
