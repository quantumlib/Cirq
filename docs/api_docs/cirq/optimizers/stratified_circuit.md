<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.stratified_circuit" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.stratified_circuit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/stratify.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Repacks avoiding simultaneous operations with different classes.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.optimizers.stratify.stratified_circuit`, `cirq.stratified_circuit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.stratified_circuit(
    circuit: "cirq.Circuit",
    *,
    categories: Iterable[Union['cirq.Gate', 'cirq.Operation', Type['cirq.Gate'], Type[
        'cirq.Operation'], Callable[['cirq.Operation'], bool]]]
) -> "cirq.Circuit"
</code></pre>



<!-- Placeholder for "Used in" -->

Sometimes, certain operations should not be done at the same time. For
example, the physical hardware may not be capable of doing certain
operations at the same time. Or it may have worse noise characteristics
when certain operations are done at the same time. In these cases, it
would be good to rearrange the circuit so that these operations always
occur in different moments.

(As a secondary effect, this may make the circuit easier to read.)

This methods takes a series of classifiers identifying categories of
operations and then ensures operations from each category only overlap
with operations from the same category. There is no guarantee that the
resulting circuit will be optimally packed under this constraint.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit whose operations should be re-arranged.
</td>
</tr><tr>
<td>
`categories`
</td>
<td>
A list of classifiers picking out certain operations.
There are several ways to specify a classifier. You can pass
in a gate instance (e.g. <a href="../../cirq/ops/X.md"><code>cirq.X</code></a>), a gate type (e.g.
<a href="../../cirq/ops/XPowGate.md"><code>cirq.XPowGate</code></a>), an operation instance (e.g.
`cirq.X(cirq.LineQubit(0))`), an operation type (e.g.
<a href="../../cirq/ops/GlobalPhaseOperation.md"><code>cirq.GlobalPhaseOperation</code></a>), or an arbitrary operation
predicate (e.g. `lambda op: len(op.qubits) == 2`).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A copy of the original circuit, but with re-arranged operations.
</td>
</tr>

</table>

