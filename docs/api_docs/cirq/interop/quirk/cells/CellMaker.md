<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.interop.quirk.cells.CellMaker" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# cirq.interop.quirk.cells.CellMaker

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/cell.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Turns Quirk identifiers into Cirq operations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.interop.quirk.cells.cell.CellMaker`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.interop.quirk.cells.CellMaker(
    _cls, identifier, size, maker
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`identifier`
</td>
<td>
A string that identifies the cell type, such as "X" or "QFT3".
</td>
</tr><tr>
<td>
`size`
</td>
<td>
The height of the operation. The number of qubits it covers.
</td>
</tr><tr>
<td>
`maker`
</td>
<td>
A function that takes a <a href="../../../../cirq/interop/quirk/cells/CellMakerArgs.md"><code>cirq.interop.quirk.cells.CellMakerArgs</code></a> and
returns either a <a href="../../../../cirq/ops/Operation.md"><code>cirq.Operation</code></a> or a <a href="../../../../cirq/interop/quirk/cells/Cell.md"><code>cirq.interop.quirk.cells.Cell</code></a>.
Returning a cell is more flexible, because cells can modify other cells
in the same column before producing operations, whereas returning an
operation is simple.
</td>
</tr>
</table>



