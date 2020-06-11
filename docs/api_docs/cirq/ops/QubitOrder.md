<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.QubitOrder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="as_qubit_order"/>
<meta itemprop="property" content="explicit"/>
<meta itemprop="property" content="map"/>
<meta itemprop="property" content="order_for"/>
<meta itemprop="property" content="sorted_by"/>
<meta itemprop="property" content="DEFAULT"/>
</div>

# cirq.ops.QubitOrder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/qubit_order.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Defines the kronecker product order of qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.QubitOrder`, `cirq.ops.qubit_order.QubitOrder`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.QubitOrder(
    explicit_func: Callable[[Iterable[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]], Tuple[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, ...]]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="as_qubit_order"><code>as_qubit_order</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/qubit_order.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>as_qubit_order(
    val: "qubit_order_or_list.QubitOrderOrList"
) -> "QubitOrder"
</code></pre>

Converts a value into a basis.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`val`
</td>
<td>
An iterable or a basis.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The basis implied by the value.
</td>
</tr>

</table>



<h3 id="explicit"><code>explicit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/qubit_order.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>explicit(
    fixed_qubits: Iterable[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>],
    fallback: Optional['QubitOrder'] = None
) -> "QubitOrder"
</code></pre>

A basis that contains exactly the given qubits in the given order.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fixed_qubits`
</td>
<td>
The qubits in basis order.
</td>
</tr><tr>
<td>
`fallback`
</td>
<td>
A fallback order to use for extra qubits not in the
fixed_qubits list. Extra qubits will always come after the
fixed_qubits, but will be ordered based on the fallback. If no
fallback is specified, a ValueError is raised when extra qubits
are specified.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Basis instance that forces the given qubits in the given order.
</td>
</tr>

</table>



<h3 id="map"><code>map</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/qubit_order.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map(
    internalize: Callable[[<a href="../../cirq/ops/qubit_order/TExternalQubit.md"><code>cirq.ops.qubit_order.TExternalQubit</code></a>], <a href="../../cirq/ops/qubit_order/TInternalQubit.md"><code>cirq.ops.qubit_order.TInternalQubit</code></a>],
    externalize: Callable[[<a href="../../cirq/ops/qubit_order/TInternalQubit.md"><code>cirq.ops.qubit_order.TInternalQubit</code></a>], <a href="../../cirq/ops/qubit_order/TExternalQubit.md"><code>cirq.ops.qubit_order.TExternalQubit</code></a>]
) -> "QubitOrder"
</code></pre>

Transforms the Basis so that it applies to wrapped qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`externalize`
</td>
<td>
Converts an internal qubit understood by the underlying
basis into an external qubit understood by the caller.
</td>
</tr><tr>
<td>
`internalize`
</td>
<td>
Converts an external qubit understood by the caller
into an internal qubit understood by the underlying basis.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A basis that transforms qubits understood by the caller into qubits
understood by an underlying basis, uses that to order the qubits,
then wraps the ordered qubits back up for the caller.
</td>
</tr>

</table>



<h3 id="order_for"><code>order_for</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/qubit_order.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>order_for(
    qubits: Iterable[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]
) -> Tuple[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, ...]
</code></pre>

Returns a qubit tuple ordered corresponding to the basis.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
Qubits that should be included in the basis. (Additional
qubits may be added into the output by the basis.)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of qubits in the same order that their single-qubit
matrices would be passed into `np.kron` when producing a matrix for
the entire system.
</td>
</tr>

</table>



<h3 id="sorted_by"><code>sorted_by</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/qubit_order.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>sorted_by(
    key: Callable[[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>], Any]
) -> "QubitOrder"
</code></pre>

A basis that orders qubits ascending based on a key function.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`key`
</td>
<td>
A function that takes a qubit and returns a key value. The
basis will be ordered ascending according to these key values.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A basis that orders qubits ascending based on a key function.
</td>
</tr>

</table>





## Class Variables

* `DEFAULT` <a id="DEFAULT"></a>
