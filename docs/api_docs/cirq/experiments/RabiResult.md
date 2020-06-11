<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.RabiResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="plot"/>
</div>

# cirq.experiments.RabiResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/qubit_characterizations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Results from a Rabi oscillation experiment.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.qubit_characterizations.RabiResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.RabiResult(
    rabi_angles: Sequence[float],
    excited_state_probabilities: Sequence[float]
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
Returns a sequence of tuple pairs with the first item being a Rabi
angle and the second item being the corresponding excited state
probability.
</td>
</tr>
</table>



## Methods

<h3 id="plot"><code>plot</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/qubit_characterizations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>plot(
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes
</code></pre>

Plots excited state probability vs the Rabi angle (angle of rotation
around the x-axis).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`ax`
</td>
<td>
the plt.Axes to plot on. If not given, a new figure is created,
plotted on, and shown.
</td>
</tr><tr>
<td>
`**plot_kwargs`
</td>
<td>
Arguments to be passed to 'plt.Axes.plot'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The plt.Axes containing the plot.
</td>
</tr>

</table>





