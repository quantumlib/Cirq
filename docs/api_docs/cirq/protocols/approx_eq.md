<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.approx_eq" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.approx_eq

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/approximate_equality_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Approximately compares two objects.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.approx_eq`, `cirq.protocols.approximate_equality_protocol.approx_eq`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.approx_eq(
    val: Any,
    other: Any,
    *,
    atol: Union[int, float] = 1e-08
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

If `val` implements SupportsApproxEquality protocol then it is invoked and
takes precedence over all other checks:
 - For primitive numeric types `int` and `float` approximate equality is
   delegated to math.isclose().
 - For complex primitive type the real and imaginary parts are treated
   independently and compared using math.isclose().
 - For `val` and `other` both iterable of the same length, consecutive
   elements are compared recursively. Types of `val` and `other` does not
   necessarily needs to match each other. They just need to be iterable and
   have the same structure.

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
The minimum absolute tolerance. See np.isclose() documentation for
details. Defaults to 1e-8 which matches np.isclose() default
absolute tolerance.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if objects are approximately equal, False otherwise.
</td>
</tr>

</table>

