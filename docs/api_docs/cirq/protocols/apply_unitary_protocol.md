<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.apply_unitary_protocol" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="RaiseTypeErrorIfNotProvided"/>
<meta itemprop="property" content="TDefault"/>
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.protocols.apply_unitary_protocol

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_unitary_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A protocol for implementing high performance unitary left-multiplies.



## Classes

[`class ApplyUnitaryArgs`](../../cirq/protocols/ApplyUnitaryArgs.md): Arguments for performing an efficient left-multiplication by a unitary.

[`class SupportsConsistentApplyUnitary`](../../cirq/protocols/SupportsConsistentApplyUnitary.md): An object that can be efficiently left-multiplied into tensors.

## Functions

[`apply_unitaries(...)`](../../cirq/protocols/apply_unitaries.md): Apply a series of unitaries onto a state tensor.

[`apply_unitary(...)`](../../cirq/protocols/apply_unitary.md): High performance left-multiplication of a unitary effect onto a tensor.

## Other Members

* `RaiseTypeErrorIfNotProvided` <a id="RaiseTypeErrorIfNotProvided"></a>
* `TDefault` <a id="TDefault"></a>
* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
