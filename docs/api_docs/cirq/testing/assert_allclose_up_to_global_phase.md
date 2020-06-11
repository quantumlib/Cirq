<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_allclose_up_to_global_phase" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_allclose_up_to_global_phase

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/lin_alg_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Checks if a ~= b * exp(i t) for some t.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.lin_alg_utils.assert_allclose_up_to_global_phase`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_allclose_up_to_global_phase(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    rtol: float = 1e-07,
    atol: float = True,
    equal_nan: bool = '',
    err_msg: Optional[str] = True,
    verbose: bool = True
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`actual`
</td>
<td>
A numpy array.
</td>
</tr><tr>
<td>
`desired`
</td>
<td>
Another numpy array.
</td>
</tr><tr>
<td>
`rtol`
</td>
<td>
Relative error tolerance.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute error tolerance.
</td>
</tr><tr>
<td>
`equal_nan`
</td>
<td>
Whether or not NaN entries should be considered equal to
other NaN entries.
</td>
</tr><tr>
<td>
`err_msg`
</td>
<td>
The error message to be printed in case of failure.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
If True, the conflicting values are appended to the error
message.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`AssertionError`
</td>
<td>
The matrices aren't nearly equal up to global phase.
</td>
</tr>
</table>

