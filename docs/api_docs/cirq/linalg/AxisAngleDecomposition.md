<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.AxisAngleDecomposition" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="canonicalize"/>
</div>

# cirq.linalg.AxisAngleDecomposition

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Represents a unitary operation as an axis, angle, and global phase.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.AxisAngleDecomposition`, `cirq.linalg.decompositions.AxisAngleDecomposition`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.AxisAngleDecomposition(
    *,
    angle: float,
    axis: Tuple[float, float, float],
    global_phase: Union[int, float, complex]
)
</code></pre>



<!-- Placeholder for "Used in" -->

The unitary $U$ is decomposed as follows:

    $$U = g e^{-i   heta/2 (xX + yY + zZ)}$$

where       heta is the rotation angle, (x, y, z) is a unit vector along the
rotation axis, and g is the global phase.

## Methods

<h3 id="canonicalize"><code>canonicalize</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>canonicalize(
    atol: float = 1e-08
) -> "AxisAngleDecomposition"
</code></pre>

Returns a standardized AxisAngleDecomposition with the same unitary.

Ensures the axis (x, y, z) satisfies x+y+z >= 0.
Ensures the angle theta satisfies -pi + atol < theta <= pi + atol.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`atol`
</td>
<td>
Absolute tolerance for errors in the representation and the
canonicalization. Determines how much larger a value needs to
be than pi before it wraps into the negative range (so that
approximation errors less than the tolerance do not cause sign
instabilities).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The canonicalized AxisAngleDecomposition.
</td>
</tr>

</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>






