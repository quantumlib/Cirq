<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.t1_decay" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.t1_decay

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t1_decay_experiment.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Runs a t1 decay experiment.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.t1_decay_experiment.t1_decay`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.t1_decay(
    sampler: <a href="../../cirq/work/Sampler.md"><code>cirq.work.Sampler</code></a>,
    *,
    qubit: <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a> = None,
    num_points: int = 1000,
    max_delay: "cirq.DURATION_LIKE",
    min_delay: "cirq.DURATION_LIKE" = None,
    repetitions: int = 1000
) -> "cirq.experiments.T1DecayResult"
</code></pre>



<!-- Placeholder for "Used in" -->

Initializes a qubit into the |1⟩ state, waits for a variable amount of time,
and measures the qubit. Plots how often the |1⟩ state is observed for each
amount of waiting.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sampler`
</td>
<td>
The quantum engine or simulator to run the circuits.
</td>
</tr><tr>
<td>
`qubit`
</td>
<td>
The qubit under test.
</td>
</tr><tr>
<td>
`num_points`
</td>
<td>
The number of evenly spaced delays to test.
</td>
</tr><tr>
<td>
`max_delay`
</td>
<td>
The largest delay to test.
</td>
</tr><tr>
<td>
`min_delay`
</td>
<td>
The smallest delay to test. Defaults to no delay.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of repetitions of the circuit for each delay.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A T1DecayResult object that stores and can plot the data.
</td>
</tr>

</table>

