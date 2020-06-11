<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.devices.LineQid" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__rsub__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="for_gate"/>
<meta itemprop="property" content="for_qid_shape"/>
<meta itemprop="property" content="is_adjacent"/>
<meta itemprop="property" content="neighbors"/>
<meta itemprop="property" content="range"/>
<meta itemprop="property" content="validate_dimension"/>
<meta itemprop="property" content="with_dimension"/>
</div>

# cirq.devices.LineQid

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A qid on a 1d lattice with nearest-neighbor connectivity.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.LineQid`, `cirq.devices.line_qubit.LineQid`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.devices.LineQid(
    x: int,
    dimension: int
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

`LineQid`s have a single attribute, and integer coordinate 'x', which
identifies the qids location on the line. `LineQid`s are ordered by
this integer.

One can construct new `LineQid`s by adding or subtracting integers:

    ```
    >>> cirq.LineQid(1, dimension=2) + 3
    cirq.LineQid(4, dimension=2)
    ```

    ```
    >>> cirq.LineQid(2, dimension=3) - 1
    cirq.LineQid(1, dimension=3)
    ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
The x coordinate.
</td>
</tr><tr>
<td>
`dimension`
</td>
<td>
The dimension of the qid, e.g. the number of quantum
levels.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dimension`
</td>
<td>
Returns the dimension or the number of quantum levels this qid has.
E.g. 2 for a qubit, 3 for a qutrit, etc.
</td>
</tr><tr>
<td>
`x`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="for_gate"><code>for_gate</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>for_gate(
    val: Any,
    start: int = 0,
    step: int = 1
) -> List['LineQid']
</code></pre>

Returns a range of line qids with the same qid shape as the gate.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`val`
</td>
<td>
Any value that supports the <a href="../../cirq/protocols/qid_shape.md"><code>cirq.qid_shape</code></a> protocol.  Usually
a gate.
</td>
</tr><tr>
<td>
`start`
</td>
<td>
The x coordinate of the first `LineQid`.
</td>
</tr><tr>
<td>
`step`
</td>
<td>
The amount to increment each x coordinate.
</td>
</tr>
</table>



<h3 id="for_qid_shape"><code>for_qid_shape</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>for_qid_shape(
    qid_shape: Sequence[int],
    start: int = 0,
    step: int = 1
) -> List['LineQid']
</code></pre>

Returns a range of line qids for each entry in `qid_shape` with
matching dimension.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qid_shape`
</td>
<td>
A sequence of dimensions for each `LineQid` to create.
</td>
</tr><tr>
<td>
`start`
</td>
<td>
The x coordinate of the first `LineQid`.
</td>
</tr><tr>
<td>
`step`
</td>
<td>
The amount to increment each x coordinate.
</td>
</tr>
</table>



<h3 id="is_adjacent"><code>is_adjacent</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_adjacent(
    other: "cirq.Qid"
) -> bool
</code></pre>

Determines if two qubits are adjacent line qubits.


<h3 id="neighbors"><code>neighbors</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>neighbors(
    qids: Optional[Iterable[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]] = None
) -> Set['_BaseLineQid']
</code></pre>

Returns qubits that are potential neighbors to this LineQubit


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
optional Iterable of qubits to constrain neighbors to.
</td>
</tr>
</table>



<h3 id="range"><code>range</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>range(
    *range_args,
    dimension: int
) -> List['LineQid']
</code></pre>

Returns a range of line qids.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*range_args`
</td>
<td>
Same arguments as python's built-in range method.
</td>
</tr><tr>
<td>
`dimension`
</td>
<td>
The dimension of the qid, e.g. the number of quantum
levels.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of line qids.
</td>
</tr>

</table>



<h3 id="validate_dimension"><code>validate_dimension</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>validate_dimension(
    dimension: int
) -> None
</code></pre>

Raises an exception if `dimension` is not positive.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
`dimension` is not positive.
</td>
</tr>
</table>



<h3 id="with_dimension"><code>with_dimension</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_dimension(
    dimension: int
) -> "LineQid"
</code></pre>

Returns a new qid with a different dimension.

Child classes can override.  Wraps the qubit object by default.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`dimension`
</td>
<td>
The new dimension or number of levels.
</td>
</tr>
</table>



<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other: int
) -> <a href="../../cirq/devices/line_qubit/TSelf.md"><code>cirq.devices.line_qubit.TSelf</code></a>
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__ge__"><code>__ge__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    other
)
</code></pre>

Return self>=value.


<h3 id="__gt__"><code>__gt__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    other
)
</code></pre>

Return self>value.


<h3 id="__le__"><code>__le__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    other
)
</code></pre>

Return self<=value.


<h3 id="__lt__"><code>__lt__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    other
)
</code></pre>

Return self<value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.


<h3 id="__neg__"><code>__neg__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__() -> <a href="../../cirq/devices/line_qubit/TSelf.md"><code>cirq.devices.line_qubit.TSelf</code></a>
</code></pre>




<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other: int
) -> <a href="../../cirq/devices/line_qubit/TSelf.md"><code>cirq.devices.line_qubit.TSelf</code></a>
</code></pre>




<h3 id="__rsub__"><code>__rsub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rsub__(
    other: int
) -> <a href="../../cirq/devices/line_qubit/TSelf.md"><code>cirq.devices.line_qubit.TSelf</code></a>
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/line_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other: int
) -> <a href="../../cirq/devices/line_qubit/TSelf.md"><code>cirq.devices.line_qubit.TSelf</code></a>
</code></pre>






