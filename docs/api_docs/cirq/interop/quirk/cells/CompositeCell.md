<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.interop.quirk.cells.CompositeCell" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="basis_change"/>
<meta itemprop="property" content="circuit"/>
<meta itemprop="property" content="controlled_by"/>
<meta itemprop="property" content="gate_count"/>
<meta itemprop="property" content="modify_column"/>
<meta itemprop="property" content="operations"/>
<meta itemprop="property" content="persistent_modifiers"/>
<meta itemprop="property" content="with_input"/>
<meta itemprop="property" content="with_line_qubits_mapped_to"/>
</div>

# cirq.interop.quirk.cells.CompositeCell

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/composite_cell.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A cell made up of a grid of sub-cells.

Inherits From: [`Cell`](../../../../cirq/interop/quirk/cells/Cell.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.interop.quirk.cells.composite_cell.CompositeCell`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.interop.quirk.cells.CompositeCell(
    height: int,
    sub_cell_cols_generator: Iterable[List[Optional[<a href="../../../../cirq/interop/quirk/cells/Cell.md"><code>cirq.interop.quirk.cells.Cell</code></a>]]],
    *,
    gate_count: int
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is used for custom circuit gates.

## Methods

<h3 id="basis_change"><code>basis_change</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>basis_change() -> "cirq.OP_TREE"
</code></pre>

Operations to conjugate a column with.

The main distinctions between operations performed during the body of a
column and operations performed during the basis change are:

1. Basis change operations are not affected by operation modifiers in
    the column. For example, adding a control into the same column will
    not affect the basis change.
2. Basis change operations happen twice, once when starting a column and
    a second time (but inverted) when ending a column.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.OP_TREE</code></a> of basis change operations.
</td>
</tr>

</table>



<h3 id="circuit"><code>circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/composite_cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>circuit() -> "cirq.Circuit"
</code></pre>




<h3 id="controlled_by"><code>controlled_by</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/composite_cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>controlled_by(
    qubit: "cirq.Qid"
) -> "CompositeCell"
</code></pre>

The same cell, but with an explicit control on its main operations.

Cells with effects that do not need to be controlled are permitted to
return themselves unmodified.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubit`
</td>
<td>
The control qubit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A modified cell with an additional control.
</td>
</tr>

</table>



<h3 id="gate_count"><code>gate_count</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/composite_cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gate_count() -> int
</code></pre>

Cheaply determines an upper bound on the resulting circuit size.

The upper bound may be larger than the actual count. For example, a
small circuit may nevertheless have involved a huge amount of rewriting
work to create. In such cases the `gate_count` is permitted to be large
to indicate the danger, despite the output being small.

This method exists in order to defend against billion laugh type
attacks. It is important that counting is fast and efficient even in
extremely adversarial conditions.

<h3 id="modify_column"><code>modify_column</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>modify_column(
    column: List[Optional['Cell']]
) -> None
</code></pre>

Applies this cell's modification to its column.

For example, a control cell will add a control qubit to other operations
in the column.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`column`
</td>
<td>
A mutable list of cells in the column, including empty
cells (with value `None`). This method is permitted to change
the items in the list, but must not change the length of the
list.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Nothing. The `column` argument is mutated in place.
</td>
</tr>

</table>



<h3 id="operations"><code>operations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/composite_cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>operations() -> "cirq.OP_TREE"
</code></pre>

Returns operations that implement the cell's main action.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.OP_TREE</code></a> of operations implementing the cell.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The cell is not ready for conversion into operations, e.g. it
may still have unspecified inputs.
</td>
</tr>
</table>



<h3 id="persistent_modifiers"><code>persistent_modifiers</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>persistent_modifiers() -> Dict[str, Callable[['Cell'], 'Cell']]
</code></pre>

Overridable modifications to apply to the rest of the circuit.

Persistent modifiers apply to all cells in the same column and also to
all cells in future columns (until a column overrides the modifier with
another one using the same key).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dictionary of keyed modifications. Each modifier lasts until a
later cell specifies a new modifier with the same key.
</td>
</tr>

</table>



<h3 id="with_input"><code>with_input</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/composite_cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_input(
    letter: str,
    register: Union[Sequence['cirq.Qid'], int]
) -> "CompositeCell"
</code></pre>

The same cell, but linked to an explicit input register or constant.

If the cell doesn't need the input, it is returned unchanged.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`letter`
</td>
<td>
The input variable name ('a', 'b', or 'r').
</td>
</tr><tr>
<td>
`register`
</td>
<td>
The list of qubits to use as the input, or else a
classical constant to use as the input.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The same cell, but with the specified input made explicit.
</td>
</tr>

</table>



<h3 id="with_line_qubits_mapped_to"><code>with_line_qubits_mapped_to</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/composite_cell.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_line_qubits_mapped_to(
    qubits: List['cirq.Qid']
) -> "Cell"
</code></pre>

Returns the same cell, but targeting different qubits.

It is assumed that the cell is currently targeting `LineQubit`
instances, where the x coordinate indicates the qubit to take from the
list.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The new qubits. The qubit at offset `x` will replace
<a href="../../../../cirq/devices/LineQubit.md"><code>cirq.LineQubit(x)</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The same cell, but with new qubits.
</td>
</tr>

</table>





