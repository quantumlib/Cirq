<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.vis.Heatmap" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="plot"/>
<meta itemprop="property" content="set_annotation_format"/>
<meta itemprop="property" content="set_annotation_map"/>
<meta itemprop="property" content="set_colorbar"/>
<meta itemprop="property" content="set_colormap"/>
<meta itemprop="property" content="set_url_map"/>
<meta itemprop="property" content="set_value_map"/>
<meta itemprop="property" content="unset_annotation"/>
<meta itemprop="property" content="unset_colorbar"/>
<meta itemprop="property" content="unset_url_map"/>
</div>

# cirq.vis.Heatmap

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Distribution of a value in 2D qubit lattice as a color map.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.Heatmap`, `cirq.vis.heatmap.Heatmap`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.vis.Heatmap(
    value_map: <a href="../../cirq/vis/heatmap/ValueMap.md"><code>cirq.vis.heatmap.ValueMap</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="plot"><code>plot</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>plot(
    ax: Optional[plt.Axes] = None,
    **pcolor_options
) -> Tuple[plt.Axes, mpl_collections.Collection, pd.DataFrame]
</code></pre>

Plots the heatmap on the given Axes.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`ax`
</td>
<td>
the Axes to plot on. If not given, a new figure is created,
plotted on, and shown.
</td>
</tr><tr>
<td>
`pcolor_options`
</td>
<td>
keyword arguments passed to ax.pcolor().
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A 3-tuple ``(ax, mesh, value_table)``. ``ax`` is the `plt.Axes` that
is plotted on. ``mesh`` is the collection of paths drawn and filled.
``value_table`` is the 2-D pandas DataFrame of values constructed
from the value_map.
</td>
</tr>

</table>



<h3 id="set_annotation_format"><code>set_annotation_format</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_annotation_format(
    annot_format: str,
    **text_options
) -> "Heatmap"
</code></pre>

Sets a format string to format values for each qubit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`annot_format`
</td>
<td>
the format string for formating values.
</td>
</tr><tr>
<td>
`text_options`
</td>
<td>
keyword arguments to matplotlib.text.Text().
</td>
</tr>
</table>



<h3 id="set_annotation_map"><code>set_annotation_map</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_annotation_map(
    annot_map: Mapping[<a href="../../cirq/vis/heatmap/QubitCoordinate.md"><code>cirq.vis.heatmap.QubitCoordinate</code></a>, str],
    **text_options
) -> "Heatmap"
</code></pre>

Sets the annotation text for each qubit.

Note that set_annotation_map() and set_annotation_format()
both sets the annotation map to be used. Whichever is called later wins.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`annot_map`
</td>
<td>
the texts to be drawn on each qubit cell.
</td>
</tr><tr>
<td>
`text_options`
</td>
<td>
keyword arguments passed to matplotlib.text.Text()
when drawing the annotation texts.
</td>
</tr>
</table>



<h3 id="set_colorbar"><code>set_colorbar</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_colorbar(
    position: str = 'right',
    size: str = '5%',
    pad: str = '2%',
    **colorbar_options
) -> "Heatmap"
</code></pre>

Sets location and style of colorbar.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`position`
</td>
<td>
colorbar position, one of 'left'|'right'|'top'|'bottom'.
</td>
</tr><tr>
<td>
`size`
</td>
<td>
a string ending in '%' to specify the width of the colorbar.
Nominally, '100%' means the same width as the heatmap.
</td>
</tr><tr>
<td>
`pad`
</td>
<td>
a string ending in '%' to specify the space between the
colorbar and the heatmap.
</td>
</tr><tr>
<td>
`colorbar_options`
</td>
<td>
keyword arguments passed to
matplotlib.Figure.colorbar().
</td>
</tr>
</table>



<h3 id="set_colormap"><code>set_colormap</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_colormap(
    colormap: Union[str, mpl.colors.Colormap] = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> "Heatmap"
</code></pre>

Sets the colormap.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`colormap`
</td>
<td>
either a colormap name or a Colormap instance.
</td>
</tr><tr>
<td>
`vmin`
</td>
<td>
the minimum value to map to the minimum color. Default is
the minimum value in value_map.
</td>
</tr><tr>
<td>
`vmax`
</td>
<td>
the maximum value to map to the maximum color. Default is
the maximum value in value_map.
</td>
</tr>
</table>



<h3 id="set_url_map"><code>set_url_map</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_url_map(
    url_map: Mapping[<a href="../../cirq/vis/heatmap/QubitCoordinate.md"><code>cirq.vis.heatmap.QubitCoordinate</code></a>, str]
) -> "Heatmap"
</code></pre>

Sets the URLs for each cell.


<h3 id="set_value_map"><code>set_value_map</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_value_map(
    value_map: <a href="../../cirq/vis/heatmap/ValueMap.md"><code>cirq.vis.heatmap.ValueMap</code></a>
) -> "Heatmap"
</code></pre>

Sets the values for each qubit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value_map`
</td>
<td>
the values for determining color for each cell.
</td>
</tr>
</table>



<h3 id="unset_annotation"><code>unset_annotation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unset_annotation() -> "Heatmap"
</code></pre>

Disables annotation. No texts are shown in cells.


<h3 id="unset_colorbar"><code>unset_colorbar</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unset_colorbar() -> "Heatmap"
</code></pre>

Disables colorbar. No colorbar is drawn.


<h3 id="unset_url_map"><code>unset_url_map</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unset_url_map() -> "Heatmap"
</code></pre>

Disables URL. No URLs are associated with cells.




