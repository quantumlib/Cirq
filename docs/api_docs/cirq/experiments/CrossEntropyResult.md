<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.CrossEntropyResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="depolarizing_model"/>
<meta itemprop="property" content="plot"/>
</div>

# cirq.experiments.CrossEntropyResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/cross_entropy_benchmarking.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Results from a cross-entropy benchmarking (XEB) experiment.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.CrossEntropyResult(
    data, repetitions
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
A sequence of NamedTuples, each of which contains two fields:
num_cycle: the circuit depth as the number of cycles, where
a cycle consists of a layer of single-qubit gates followed
by a layer of two-qubit gates.
xeb_fidelity: the XEB fidelity after the given cycle number.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of circuit repetitions used.
</td>
</tr>
</table>



## Methods

<h3 id="depolarizing_model"><code>depolarizing_model</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/cross_entropy_benchmarking.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>depolarizing_model() -> CrossEntropyDepolarizingModel
</code></pre>

Fit a depolarizing error model for a cycle.

Fits an exponential model f = S * p^d, where d is the number of cycles
and f is the cross entropy fidelity for that number of cycles,
using nonlinear least squares.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A DepolarizingModel object, which has attributes `coefficient`
representing the value S, `decay_constant` representing the value
p, and `covariance` representing the covariance in the estimation
of S and p in that order.
</td>
</tr>

</table>



<h3 id="plot"><code>plot</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/cross_entropy_benchmarking.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>plot(
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes
</code></pre>

Plots the average XEB fidelity vs the number of cycles.


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

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>






