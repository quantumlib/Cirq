<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.T2DecayResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="plot_bloch_vector"/>
<meta itemprop="property" content="plot_expectations"/>
</div>

# cirq.experiments.T2DecayResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t2_decay_experiment.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Results from a T2 decay experiment.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.t2_decay_experiment.T2DecayResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.T2DecayResult(
    x_basis_data: pd.DataFrame,
    y_basis_data: pd.DataFrame
)
</code></pre>



<!-- Placeholder for "Used in" -->

This object is a container for the measurement results in each basis
for each amount of delay.  These can be used to calculate Pauli
expectation values, length of the Bloch vector, and various fittings of
the data to calculate estimated T2 phase decay times.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`expectation_pauli_x`
</td>
<td>
A data frame with delay_ns, value columns.

This value contains the expectation of the Pauli X operator as
estimated by measurement outcomes.
</td>
</tr><tr>
<td>
`expectation_pauli_y`
</td>
<td>
A data frame with delay_ns, value columns.

This value contains the expectation of the Pauli X operator as
estimated by measurement outcomes.
</td>
</tr>
</table>



## Methods

<h3 id="plot_bloch_vector"><code>plot_bloch_vector</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t2_decay_experiment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>plot_bloch_vector(
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes
</code></pre>

Plots the estimated length of the Bloch vector versus time.

This plot estimates the Bloch Vector by squaring the Pauli expectation
value of X and adding it to the square of the Pauli expectation value of
Y.  This essentially projects the state into the XY plane.

Note that Z expectation is not considered, since T1 related amplitude
damping will generally push this value towards |0>
(expectation <Z> = -1) which will significantly distort the T2 numbers.

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



<h3 id="plot_expectations"><code>plot_expectations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t2_decay_experiment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>plot_expectations(
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes
</code></pre>

Plots the expectation values of Pauli operators versus delay time.


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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t2_decay_experiment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




