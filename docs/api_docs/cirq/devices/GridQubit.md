<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.devices.GridQubit" />
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
<meta itemprop="property" content="from_diagram"/>
<meta itemprop="property" content="is_adjacent"/>
<meta itemprop="property" content="neighbors"/>
<meta itemprop="property" content="rect"/>
<meta itemprop="property" content="square"/>
<meta itemprop="property" content="validate_dimension"/>
<meta itemprop="property" content="with_dimension"/>
</div>

# cirq.devices.GridQubit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A qubit on a 2d square lattice.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.GridQubit`, `cirq.devices.grid_qubit.GridQubit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.devices.GridQubit(
    row: int,
    col: int
)
</code></pre>



<!-- Placeholder for "Used in" -->

GridQubits use row-major ordering:

    GridQubit(0, 0) < GridQubit(0, 1) < GridQubit(1, 0) < GridQubit(1, 1)

New GridQubits can be constructed by adding or subtracting tuples

    ```
    >>> cirq.GridQubit(2, 3) + (3, 1)
    cirq.GridQubit(5, 4)
    ```

    ```
    >>> cirq.GridQubit(2, 3) - (1, 2)
    cirq.GridQubit(1, 1)
    ```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`col`
</td>
<td>

</td>
</tr><tr>
<td>
`dimension`
</td>
<td>
Returns the dimension or the number of quantum levels this qid has.
E.g. 2 for a qubit, 3 for a qutrit, etc.
</td>
</tr><tr>
<td>
`row`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="from_diagram"><code>from_diagram</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_diagram(
    diagram: str
) -> List['GridQubit']
</code></pre>

Parse ASCII art device layout into info about qubits and
        connectivity. As an example, the below diagram will create a list of
        GridQubit in a pyramid structure.
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA

        You can use any character other than a hyphen to mark a qubit. As an
        example, the qubits for the Bristlecone device could be represented by
        the below diagram. This produces a diamond-shaped grid of qids, and
        qids with the same letter correspond to the same readout line.

        .....AB.....
        ....ABCD....
        ...ABCDEF...
        ..ABCDEFGH..
        .ABCDEFGHIJ.
        ABCDEFGHIJKL
        .CDEFGHIJKL.
        ..EFGHIJKL..
        ...GHIJKL...
        ....IJKL....
        .....KL.....

        Args:
            diagram: String representing the qubit layout. Each line represents
                a row. Alphanumeric characters are assigned as qid.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('
'), an error will
                be thrown. The top-left corner of the diagram will be have
                coordinate (0,0).

        Returns:
            A list of GridQubit corresponding to qubits in the provided diagram

        Raises:
            ValueError: If the input string contains an invalid character.
        

<h3 id="is_adjacent"><code>is_adjacent</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_adjacent(
    other: "cirq.Qid"
) -> bool
</code></pre>

Determines if two qubits are adjacent qubits.


<h3 id="neighbors"><code>neighbors</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>neighbors(
    qids: Optional[Iterable[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]] = None
) -> Set['_BaseGridQid']
</code></pre>

Returns qubits that are potential neighbors to this GridQid


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



<h3 id="rect"><code>rect</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>rect(
    rows: int,
    cols: int,
    top: int = 0,
    left: int = 0
) -> List['GridQubit']
</code></pre>

Returns a rectangle of GridQubits.


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
A list of GridQubits filling in a rectangular grid
</td>
</tr>

</table>



<h3 id="square"><code>square</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>square(
    diameter: int,
    top: int = 0,
    left: int = 0
) -> List['GridQubit']
</code></pre>

Returns a square of GridQubits.


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
A list of GridQubits filling in a square grid
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_dimension(
    dimension: int
) -> "GridQid"
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other: Tuple[int, int]
) -> "TSelf"
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__() -> "TSelf"
</code></pre>




<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other: Tuple[int, int]
) -> "TSelf"
</code></pre>




<h3 id="__rsub__"><code>__rsub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rsub__(
    other: Tuple[int, int]
) -> "TSelf"
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/grid_qubit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other: Tuple[int, int]
) -> "TSelf"
</code></pre>






