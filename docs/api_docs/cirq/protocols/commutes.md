<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.commutes" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.commutes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/commutes_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines whether two values commute.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.commutes`, `cirq.protocols.commutes_protocol.commutes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.commutes(
    v1: Any,
    v2: Any,
    *,
    atol: Union[int, float] = 1e-08,
    default: <a href="../../cirq/protocols/commutes_protocol/TDefault.md"><code>cirq.protocols.commutes_protocol.TDefault</code></a> = cirq.protocols.commutes_protocol.RaiseTypeErrorIfNotProvided
) -> Union[bool, <a href="../../cirq/protocols/commutes_protocol/TDefault.md"><code>cirq.protocols.commutes_protocol.TDefault</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

This is determined by any one of the following techniques:

- Either value has a `_commutes_` method that returns 'True', 'False', or
    'None' (meaning indeterminate). If both methods either don't exist or
    return `NotImplemented` then another strategy is tried. `v1._commutes_`
    is tried before `v2._commutes_`.
- Both values are matrices. The return value is determined by checking if
    v1 @ v2 - v2 @ v1 is sufficiently close to zero.
- Both values are <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a> instances. If the operations apply to
    disjoint qubit sets then they commute. Otherwise, if they have unitary
    matrices, those matrices are checked for commutativity (while accounting
    for the fact that the operations may have different qubit orders or only
    partially overlap).

If none of these techniques succeeds, the commutativity is assumed to be
indeterminate.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`v1`
</td>
<td>
One of the values to check for commutativity. Can be a cirq object
such as an operation, or a numpy matrix.
</td>
</tr><tr>
<td>
`v2`
</td>
<td>
The other value to check for commutativity. Can be a cirq object
such as an operation, or a numpy matrix.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
A fallback value to return, instead of raising a ValueError, if
it is indeterminate whether or not the two values commute.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute error tolerance. If all entries in v1@v2 - v2@v1 have a
magnitude less than this tolerance, v1 and v2 can be reported as
commuting. Defaults to 1e-8.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`True`
</td>
<td>
`v1` and `v2` commute (or approximately commute).
</td>
</tr><tr>
<td>
`False`
</td>
<td>
`v1` and `v2` don't commute.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
The commutativity of `v1` and `v2` is indeterminate, or could
not be determined, and the `default` argument was specified.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
The commutativity of `v1` and `v2` is indeterminate, or could
not be determined, and the `default` argument was not specified.
</td>
</tr>
</table>

