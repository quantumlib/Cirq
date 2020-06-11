<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.act_on" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.act_on

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/act_on_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Applies an action to a state argument.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.act_on`, `cirq.protocols.act_on_protocol.act_on`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.act_on(
    action: Any,
    args: Any,
    *,
    allow_decompose: bool = True
)
</code></pre>



<!-- Placeholder for "Used in" -->

For example, the action may be a <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a> and the state argument may
represent the internal state of a state vector simulator (a
<a href="../../cirq/sim/ActOnStateVectorArgs.md"><code>cirq.ActOnStateVectorArgs</code></a>).

The action is applied by first checking if `action._act_on_` exists and
returns `True` (instead of `NotImplemented`) for the given object. Then
fallback strategies specified by the state argument via `_act_on_fallback_`
are attempted. If those also fail, the method fails with a `TypeError`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`action`
</td>
<td>
The action to apply to the state tensor. Typically a
<a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a>.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
A mutable state object that should be modified by the action. May
specify an `_act_on_fallback_` method to use in case the action
doesn't recognize it.
</td>
</tr><tr>
<td>
`allow_decompose`
</td>
<td>
Defaults to True. Forwarded into the
`_act_on_fallback_` method of `args`. Determines if decomposition
should be used or avoided when attempting to act `action` on `args`.
Used by internal methods to avoid redundant decompositions.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Nothing. Results are communicated by editing `args`.
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
Failed to act `action` on `args`.
</td>
</tr>
</table>

