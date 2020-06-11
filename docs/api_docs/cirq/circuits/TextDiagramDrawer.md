<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.TextDiagramDrawer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="content_present"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="force_horizontal_padding_after"/>
<meta itemprop="property" content="force_vertical_padding_after"/>
<meta itemprop="property" content="grid_line"/>
<meta itemprop="property" content="height"/>
<meta itemprop="property" content="horizontal_line"/>
<meta itemprop="property" content="hstack"/>
<meta itemprop="property" content="insert_empty_columns"/>
<meta itemprop="property" content="insert_empty_rows"/>
<meta itemprop="property" content="render"/>
<meta itemprop="property" content="shift"/>
<meta itemprop="property" content="shifted"/>
<meta itemprop="property" content="superimpose"/>
<meta itemprop="property" content="superimposed"/>
<meta itemprop="property" content="transpose"/>
<meta itemprop="property" content="vertical_line"/>
<meta itemprop="property" content="vstack"/>
<meta itemprop="property" content="width"/>
<meta itemprop="property" content="write"/>
</div>

# cirq.circuits.TextDiagramDrawer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A utility class for creating simple text diagrams.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.TextDiagramDrawer`, `cirq.circuits.text_diagram_drawer.TextDiagramDrawer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.circuits.TextDiagramDrawer(
    entries: Optional[Mapping[Tuple[int, int], _DiagramText]] = None,
    horizontal_lines: Optional[Iterable[_HorizontalLine]] = None,
    vertical_lines: Optional[Iterable[_VerticalLine]] = None,
    horizontal_padding: Optional[Mapping[int, int]] = None,
    vertical_padding: Optional[Mapping[int, int]] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
    

## Methods

<h3 id="content_present"><code>content_present</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>content_present(
    x: int,
    y: int
) -> bool
</code></pre>

Determines if a line or printed text is at the given location.


<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy()
</code></pre>




<h3 id="force_horizontal_padding_after"><code>force_horizontal_padding_after</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>force_horizontal_padding_after(
    index: int,
    padding: Union[int, float]
) -> None
</code></pre>

Change the padding after the given column.


<h3 id="force_vertical_padding_after"><code>force_vertical_padding_after</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>force_vertical_padding_after(
    index: int,
    padding: Union[int, float]
) -> None
</code></pre>

Change the padding after the given row.


<h3 id="grid_line"><code>grid_line</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>grid_line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    emphasize: bool = False
)
</code></pre>

Adds a vertical or horizontal line from (x1, y1) to (x2, y2).

Horizontal line is selected on equality in the second coordinate and
vertical line is selected on equality in the first coordinate.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If line is neither horizontal nor vertical.
</td>
</tr>
</table>



<h3 id="height"><code>height</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>height() -> int
</code></pre>

Determines how many entry rows are in the diagram.


<h3 id="horizontal_line"><code>horizontal_line</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>horizontal_line(
    y: Union[int, float],
    x1: Union[int, float],
    x2: Union[int, float],
    emphasize: bool = False
) -> None
</code></pre>

Adds a line from (x1, y) to (x2, y).


<h3 id="hstack"><code>hstack</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>hstack(
    diagrams: Sequence['TextDiagramDrawer'],
    padding_resolver: Optional[Callable[[Sequence[Optional[int]]], int]] = None
)
</code></pre>

Horizontally stack text diagrams.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`diagrams`
</td>
<td>
The diagrams to stack, ordered from left to right.
</td>
</tr><tr>
<td>
`padding_resolver`
</td>
<td>
A function that takes a list of paddings
specified for a row and returns the padding to use in the
stacked diagram. Defaults to raising ValueError if the diagrams
to stack contain inconsistent padding in any row, including
if some specify a padding and others don't.
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
Inconsistent padding cannot be resolved.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The horizontally stacked diagram.
</td>
</tr>

</table>



<h3 id="insert_empty_columns"><code>insert_empty_columns</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>insert_empty_columns(
    x: int,
    amount: int = 1
) -> None
</code></pre>

Insert a number of columns after the given column.


<h3 id="insert_empty_rows"><code>insert_empty_rows</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>insert_empty_rows(
    y: int,
    amount: int = 1
) -> None
</code></pre>

Insert a number of rows after the given row.


<h3 id="render"><code>render</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>render(
    horizontal_spacing: int = 1,
    vertical_spacing: int = 1,
    crossing_char: str = None,
    use_unicode_characters: bool = True
) -> str
</code></pre>

Outputs text containing the diagram.


<h3 id="shift"><code>shift</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>shift(
    dx: int = 0,
    dy: int = 0
) -> "TextDiagramDrawer"
</code></pre>




<h3 id="shifted"><code>shifted</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>shifted(
    dx: int = 0,
    dy: int = 0
) -> "TextDiagramDrawer"
</code></pre>




<h3 id="superimpose"><code>superimpose</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>superimpose(
    other: "TextDiagramDrawer"
) -> "TextDiagramDrawer"
</code></pre>




<h3 id="superimposed"><code>superimposed</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>superimposed(
    other: "TextDiagramDrawer"
) -> "TextDiagramDrawer"
</code></pre>




<h3 id="transpose"><code>transpose</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>transpose() -> "TextDiagramDrawer"
</code></pre>

Returns the same diagram, but mirrored across its diagonal.


<h3 id="vertical_line"><code>vertical_line</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vertical_line(
    x: Union[int, float],
    y1: Union[int, float],
    y2: Union[int, float],
    emphasize: bool = False
) -> None
</code></pre>

Adds a line from (x, y1) to (x, y2).


<h3 id="vstack"><code>vstack</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>vstack(
    diagrams: Sequence['TextDiagramDrawer'],
    padding_resolver: Optional[Callable[[Sequence[Optional[int]]], int]] = None
)
</code></pre>

Vertically stack text diagrams.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`diagrams`
</td>
<td>
The diagrams to stack, ordered from bottom to top.
</td>
</tr><tr>
<td>
`padding_resolver`
</td>
<td>
A function that takes a list of paddings
specified for a column and returns the padding to use in the
stacked diagram. If None, defaults to raising ValueError if the
diagrams to stack contain inconsistent padding in any column,
including if some specify a padding and others don't.
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
Inconsistent padding cannot be resolved.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The vertically stacked diagram.
</td>
</tr>

</table>



<h3 id="width"><code>width</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>width() -> int
</code></pre>

Determines how many entry columns are in the diagram.


<h3 id="write"><code>write</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>write(
    x: int,
    y: int,
    text: str,
    transposed_text: Optional[str] = None
)
</code></pre>

Adds text to the given location.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
The column in which to write the text.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
The row in which to write the text.
</td>
</tr><tr>
<td>
`text`
</td>
<td>
The text to write at location (x, y).
</td>
</tr><tr>
<td>
`transposed_text`
</td>
<td>
Optional text to write instead, if the text
diagram is transposed.
</td>
</tr>
</table>



<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/text_diagram_drawer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>






