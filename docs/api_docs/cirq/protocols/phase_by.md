<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.phase_by" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.phase_by

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/phase_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a phased version of the effect.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.phase_by`, `cirq.protocols.phase_protocol.phase_by`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.phase_by(
    val: Any,
    phase_turns: float,
    qubit_index: int,
    default: <a href="../../cirq/protocols/phase_protocol/TDefault.md"><code>cirq.protocols.phase_protocol.TDefault</code></a> = cirq.protocols.phase_protocol.RaiseTypeErrorIfNotProvided
)
</code></pre>



<!-- Placeholder for "Used in" -->

For example, an X gate phased by 90 degrees would be a Y gate.
This works by calling `val`'s _phase_by_ method and returning
the result.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value to describe with a unitary matrix.
</td>
</tr><tr>
<td>
`phase_turns`
</td>
<td>
The amount to phase the gate, in fractions of a whole
turn. Multiply by 2π to get radians.
</td>
</tr><tr>
<td>
`qubit_index`
</td>
<td>
The index of the target qubit the phasing applies to. For
operations this is the index of the qubit within the operation's
qubit list. For gates it's the index of the qubit within the tuple
of qubits taken by the gate's `on` method.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
The default value to return if `val` can't be phased. If not
specified, an error is raised when `val` can't be phased.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has a _phase_by_ method and its result is not NotImplemented,
that result is returned. Otherwise, the function will return the
default value provided or raise a TypeError if none was provided.
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
`val` doesn't have a _phase_by_ method (or that method returned
NotImplemented) and no `default` was specified.
</td>
</tr>
</table>

