<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.pasqal.ThreeDGridQubit" />
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
<meta itemprop="property" content="cube"/>
<meta itemprop="property" content="distance"/>
<meta itemprop="property" content="is_adjacent"/>
<meta itemprop="property" content="neighbors"/>
<meta itemprop="property" content="parallelep"/>
<meta itemprop="property" content="rect"/>
<meta itemprop="property" content="square"/>
<meta itemprop="property" content="validate_dimension"/>
<meta itemprop="property" content="with_dimension"/>
</div>

# cirq.pasqal.ThreeDGridQubit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A qubit on a 3d lattice.

Inherits From: [`Qid`](../../cirq/ops/Qid.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.pasqal.pasqal_qubits.ThreeDGridQubit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.pasqal.ThreeDGridQubit(
    row: int,
    col: int,
    lay: int
)
</code></pre>



<!-- Placeholder for "Used in" -->

ThreeDGridQubits use row-column-layer ordering:

    ThreeDGridQubit(0, 0, 0) < ThreeDGridQubit(0, 0, 1)
    < ThreeDGridQubit(0, 1, 0)< ThreeDGridQubit(1, 0, 0)
    < ThreeDGridQubit(0, 1, 1)< ThreeDGridQubit(1, 0, 1)
    < ThreeDGridQubit(1, 1, 0)< ThreeDGridQubit(1, 1, 1)

New ThreeDGridQubit can be constructed by adding or subtracting tuples

    ```
    >>> cirq.pasqal.ThreeDGridQubit(2, 3, 4) + (3, 1, 6)
    pasqal.ThreeDGridQubit(5, 4, 10)
    ```

    ```
    >>> cirq.pasqal.ThreeDGridQubit(2, 3, 4) - (1, 2, 2)
    pasqal.ThreeDGridQubit(1, 1, 2)
    ```



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
</tr>
</table>



## Methods

<h3 id="cube"><code>cube</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>cube(
    diameter: int,
    top: int = 0,
    left: int = 0,
    upper: int = 0
) -> List['ThreeDGridQubit']
</code></pre>

Returns a cube of ThreeDGridQubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`diameter`
</td>
<td>
Length of a side of the square
</td>
</tr><tr>
<td>
`top`
</td>
<td>
Row number of the topmost row
</td>
</tr><tr>
<td>
`left`
</td>
<td>
Column number of the leftmost row
</td>
</tr><tr>
<td>
`upper`
</td>
<td>
Column number of the uppermost layer
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of ThreeDGridQubits filling in a square grid
</td>
</tr>

</table>



<h3 id="distance"><code>distance</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>distance(
    other: <a href="../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>
) -> float
</code></pre>

Returns the distance between two qubits in a 3D grid.


<h3 id="is_adjacent"><code>is_adjacent</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_adjacent(
    other: <a href="../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>
) -> bool
</code></pre>

Determines if two qubits are adjacent qubits.


<h3 id="neighbors"><code>neighbors</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>neighbors(
    qids: Optional[Iterable[<a href="../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]] = None
) -> Set['ThreeDGridQubit']
</code></pre>

Returns qubits that are potential neighbors to this ThreeDGridQubit


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qids`
</td>
<td>
optional Iterable of qubits to constrain neighbors to.
</td>
</tr>
</table>



<h3 id="parallelep"><code>parallelep</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>parallelep(
    rows: int,
    cols: int,
    lays: int,
    top: int = 0,
    left: int = 0,
    upper: int = 0
) -> List['ThreeDGridQubit']
</code></pre>

Returns a parallelepiped of ThreeDGridQubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`rows`
</td>
<td>
Number of rows in the rectangle
</td>
</tr><tr>
<td>
`cols`
</td>
<td>
Number of columns in the rectangle
</td>
</tr><tr>
<td>
`top`
</td>
<td>
Row number of the topmost row
</td>
</tr><tr>
<td>
`left`
</td>
<td>
Column number of the leftmost row
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of ThreeDGridQubits filling in a rectangular grid
</td>
</tr>

</table>



<h3 id="rect"><code>rect</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>rect(
    rows: int,
    cols: int,
    top: int = 0,
    left: int = 0
) -> List['ThreeDGridQubit']
</code></pre>

Returns a rectangle of ThreeDGridQubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`rows`
</td>
<td>
Number of rows in the rectangle
</td>
</tr><tr>
<td>
`cols`
</td>
<td>
Number of columns in the rectangle
</td>
</tr><tr>
<td>
`top`
</td>
<td>
Row number of the topmost row
</td>
</tr><tr>
<td>
`left`
</td>
<td>
Column number of the leftmost row
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of ThreeDGridQubits filling in a rectangular grid
</td>
</tr>

</table>



<h3 id="square"><code>square</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>square(
    diameter: int,
    top: int = 0,
    left: int = 0
) -> List['ThreeDGridQubit']
</code></pre>

Returns a square of ThreeDGridQubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`diameter`
</td>
<td>
Length of a side of the square
</td>
</tr><tr>
<td>
`top`
</td>
<td>
Row number of the topmost row
</td>
</tr><tr>
<td>
`left`
</td>
<td>
Column number of the leftmost row
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of ThreeDGridQubits filling in a square grid
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_dimension(
    dimension: int
) -> "Qid"
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other: Tuple[int, int, int]
) -> "ThreeDGridQubit"
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__() -> "ThreeDGridQubit"
</code></pre>




<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other: Tuple[int, int, int]
) -> "ThreeDGridQubit"
</code></pre>




<h3 id="__rsub__"><code>__rsub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rsub__(
    other: Tuple[int, int, int]
) -> "ThreeDGridQubit"
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_qubits.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other: Tuple[int, int, int]
) -> "ThreeDGridQubit"
</code></pre>






