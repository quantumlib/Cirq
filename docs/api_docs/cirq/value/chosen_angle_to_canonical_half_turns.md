<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.chosen_angle_to_canonical_half_turns" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.value.chosen_angle_to_canonical_half_turns

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/angle.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a canonicalized half_turns based on the given arguments.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.chosen_angle_to_canonical_half_turns`, `cirq.value.angle.chosen_angle_to_canonical_half_turns`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.chosen_angle_to_canonical_half_turns(
    half_turns: Optional[type_alias.TParamVal] = None,
    rads: Optional[float] = None,
    degs: Optional[float] = None,
    default: float = 1.0
) -> <a href="../../cirq/value/TParamVal.md"><code>cirq.value.TParamVal</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

At most one of half_turns, rads, degs must be specified. If none are
specified, the output defaults to half_turns=1.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`half_turns`
</td>
<td>
The number of half turns to rotate by.
</td>
</tr><tr>
<td>
`rads`
</td>
<td>
The number of radians to rotate by.
</td>
</tr><tr>
<td>
`degs`
</td>
<td>
The number of degrees to rotate by
</td>
</tr><tr>
<td>
`default`
</td>
<td>
The half turns angle to use if nothing else is specified.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A number of half turns.
</td>
</tr>

</table>

