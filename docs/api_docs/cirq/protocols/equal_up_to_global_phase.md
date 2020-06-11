<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.equal_up_to_global_phase" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.equal_up_to_global_phase

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/equal_up_to_global_phase_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determine whether two objects are equal up to global phase.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.equal_up_to_global_phase`, `cirq.protocols.equal_up_to_global_phase_protocol.equal_up_to_global_phase`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.equal_up_to_global_phase(
    val: Any,
    other: Any,
    *,
    atol: Union[int, float] = 1e-08
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

If `val` implements a `_equal_up_to_global_phase_` method then it is
invoked and takes precedence over all other checks:
 - For complex primitive type the magnitudes of the values are compared.
 - For `val` and `other` both iterable of the same length, consecutive
   elements are compared recursively. Types of `val` and `other` does not
   necessarily needs to match each other. They just need to be iterable and
   have the same structure.
 - For all other types, fall back to `_approx_eq_`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
Source object for approximate comparison.
</td>
</tr><tr>
<td>
`other`
</td>
<td>
Target object for approximate comparison.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
The minimum absolute tolerance. This places an upper bound on
the differences in *magnitudes* of two compared complex numbers.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if objects are approximately equal up to phase, False otherwise.
</td>
</tr>

</table>

