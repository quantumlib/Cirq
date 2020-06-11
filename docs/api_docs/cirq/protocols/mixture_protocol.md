<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.mixture_protocol" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="RaiseTypeErrorIfNotProvided"/>
</div>

# Module: cirq.protocols.mixture_protocol

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/mixture_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Protocol for objects that are mixtures (probabilistic combinations).



## Classes

[`class SupportsMixture`](../../cirq/protocols/SupportsMixture.md): An object that decomposes into a probability distribution of unitaries.

## Functions

[`has_mixture(...)`](../../cirq/protocols/has_mixture.md): Returns whether the value has a mixture representation.

[`has_mixture_channel(...)`](../../cirq/protocols/has_mixture_channel.md): THIS FUNCTION IS DEPRECATED.

[`mixture(...)`](../../cirq/protocols/mixture.md): Return a sequence of tuples representing a probabilistic unitary.

[`mixture_channel(...)`](../../cirq/protocols/mixture_channel.md): THIS FUNCTION IS DEPRECATED.

[`validate_mixture(...)`](../../cirq/protocols/validate_mixture.md): Validates that the mixture's tuple are valid probabilities.

## Other Members

* `RaiseTypeErrorIfNotProvided` <a id="RaiseTypeErrorIfNotProvided"></a>
