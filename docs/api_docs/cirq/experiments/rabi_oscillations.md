<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.rabi_oscillations" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.rabi_oscillations

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/qubit_characterizations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Runs a Rabi oscillation experiment.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.qubit_characterizations.rabi_oscillations`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.rabi_oscillations(
    sampler: <a href="../../cirq/work/Sampler.md"><code>cirq.work.Sampler</code></a>,
    qubit: <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>,
    max_angle: float = (2 * np.pi),
    *,
    repetitions: int = 1000,
    num_points: int = 200
) -> <a href="../../cirq/experiments/RabiResult.md"><code>cirq.experiments.RabiResult</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Rotates a qubit around the x-axis of the Bloch sphere by a sequence of Rabi
angles evenly spaced between 0 and max_angle. For each rotation, repeat
the circuit a number of times and measure the average probability of the
qubit being in the |1> state.

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
`max_angle`
</td>
<td>
The final Rabi angle in radians.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of repetitions of the circuit for each Rabi
angle.
</td>
</tr><tr>
<td>
`num_points`
</td>
<td>
The number of Rabi angles.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A RabiResult object that stores and plots the result.
</td>
</tr>

</table>

