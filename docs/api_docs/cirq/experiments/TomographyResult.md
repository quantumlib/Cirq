<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.TomographyResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="plot"/>
</div>

# cirq.experiments.TomographyResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/qubit_characterizations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Results from a state tomography experiment.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.qubit_characterizations.TomographyResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.TomographyResult(
    density_matrix: np.ndarray
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
Returns an n^2 by n^2 complex matrix representing the density
matrix of the n-qubit system.
</td>
</tr>
</table>



## Methods

<h3 id="plot"><code>plot</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/qubit_characterizations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>plot(
    axes: Optional[List[plt.Axes]] = None,
    **plot_kwargs
) -> List[plt.Axes]
</code></pre>

Plots the real and imaginary parts of the density matrix as two
3D bar plots.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`axes`
</td>
<td>
a list of 2 `plt.Axes` instances. Note that they must be in
3d projections. If not given, a new figure is created with 2
axes and the plotted figure is shown.
</td>
</tr><tr>
<td>
`plot_kwargs`
</td>
<td>
the optional kwargs passed to bar3d.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
the list of `plt.Axes` being plotted on.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
ValueError if axes is a list with length != 2.
</td>
</tr>

</table>





