<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.validate_normalized_state" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.validate_normalized_state

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
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
<p>`cirq.qis.states.validate_normalized_state`, `cirq.validate_normalized_state`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.validate_normalized_state(
    state_vector: np.ndarray,
    *,
    qid_shape: Tuple[int, ...] = np.complex64,
    dtype: Type[np.number] = 1e-07,
    atol: float = 1e-07
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

IT WILL BE REMOVED IN `cirq v0.10.0`.

Use <a href="../../cirq/qis/validate_normalized_state_vector.md"><code>cirq.validate_normalized_state_vector</code></a> instead.

Validates that the given state vector is a valid.