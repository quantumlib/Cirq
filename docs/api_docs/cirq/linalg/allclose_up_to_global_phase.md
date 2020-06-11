<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.allclose_up_to_global_phase" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.allclose_up_to_global_phase

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/predicates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines if a ~= b * exp(i t) for some t.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.allclose_up_to_global_phase`, `cirq.linalg.predicates.allclose_up_to_global_phase`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.allclose_up_to_global_phase(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`a`
</td>
<td>
A numpy array.
</td>
</tr><tr>
<td>
`b`
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
</tr>
</table>

