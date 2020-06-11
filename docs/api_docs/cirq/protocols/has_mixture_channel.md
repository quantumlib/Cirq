<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.has_mixture_channel" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.has_mixture_channel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/mixture_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



THIS FUNCTION IS DEPRECATED.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.has_mixture_channel`, `cirq.protocols.mixture_protocol.has_mixture_channel`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.has_mixture_channel(
    val: Any,
    *,
    allow_decompose: bool = True
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

IT WILL BE REMOVED IN `cirq v0.10.0`.

Use "cirq.has_mixture" instead.

Returns whether the value has a mixture representation.

    Args:
        val: The value to check.
        allow_decompose: Used by internal methods to stop redundant
            decompositions from being performed (e.g. there's no need to
            decompose an object to check if it is unitary as part of determining
            if the object is a quantum channel, when the quantum channel check
            will already be doing a more general decomposition check). Defaults
            to True. When false, the decomposition strategy for determining
            the result is skipped.

    Returns:
        If `val` has a `_has_mixture_` method and its result is not
        NotImplemented, that result is returned. Otherwise, if the value
        has a `_mixture_` method return True if that has a non-default value.
        Returns False if neither function exists.
    