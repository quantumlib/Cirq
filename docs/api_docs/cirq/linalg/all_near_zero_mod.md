<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.all_near_zero_mod" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.all_near_zero_mod

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/tolerance.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Checks if the tensor's elements are all near multiples of the period.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.all_near_zero_mod`, `cirq.linalg.tolerance.all_near_zero_mod`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.all_near_zero_mod(
    a: Union[float, complex, Iterable[float], np.ndarray],
    period: float,
    *,
    atol: float = 1e-08
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
Tensor of elements that could all be near multiples of the period.
</td>
</tr><tr>
<td>
`period`
</td>
<td>
The period, e.g. 2 pi when working in radians.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute tolerance.
</td>
</tr>
</table>

