<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.T1DecayResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="plot"/>
</div>

# cirq.experiments.T1DecayResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t1_decay_experiment.py">
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
<p>`cirq.experiments.t1_decay_experiment.T1DecayResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.T1DecayResult(
    data: pd.DataFrame
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
A data frame with delay_ns, false_count, true_count columns.
</td>
</tr>
</table>



## Methods

<h3 id="plot"><code>plot</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t1_decay_experiment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>plot(
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes
</code></pre>

Plots the excited state probability vs the amount of delay.


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



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t1_decay_experiment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t1_decay_experiment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
) -> bool
</code></pre>

Return self!=value.




