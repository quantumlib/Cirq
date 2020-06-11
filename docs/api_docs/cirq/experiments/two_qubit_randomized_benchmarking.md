<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.two_qubit_randomized_benchmarking" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.two_qubit_randomized_benchmarking

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/qubit_characterizations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Clifford-based randomized benchmarking (RB) of two qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.qubit_characterizations.two_qubit_randomized_benchmarking`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.two_qubit_randomized_benchmarking(
    sampler: <a href="../../cirq/work/Sampler.md"><code>cirq.work.Sampler</code></a>,
    first_qubit: <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>,
    second_qubit: <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>,
    *,
    num_clifford_range: Sequence[int] = range(5, 50, 5),
    num_circuits: int = 20,
    repetitions: int = 1000
) -> <a href="../../cirq/experiments/RandomizedBenchMarkResult.md"><code>cirq.experiments.RandomizedBenchMarkResult</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

A total of num_circuits random circuits are generated, each of which
contains a fixed number of two-qubit Clifford gates plus one additional
Clifford that inverts the whole sequence and a measurement in the
z-basis. Each circuit is repeated a number of times and the average
|00> state population is determined from the measurement outcomes of all
of the circuits.

The above process is done for different circuit lengths specified by the
integers in num_clifford_range. For example, an integer 10 means the
random circuits will contain 10 Clifford gates each plus one inverting
Clifford. The user may use the result to extract an average gate fidelity,
by analyzing the change in the average |00> state population at different
circuit lengths. For actual experiments, one should choose
num_clifford_range such that a clear exponential decay is observed in the
results.

The two-qubit Cliffords here are decomposed into CZ gates plus single-qubit
x and y rotations. See Barends et al., Nature 508, 500 for details.

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
`first_qubit`
</td>
<td>
The first qubit under test.
</td>
</tr><tr>
<td>
`second_qubit`
</td>
<td>
The second qubit under test.
</td>
</tr><tr>
<td>
`num_clifford_range`
</td>
<td>
The different numbers of Cliffords in the RB study.
</td>
</tr><tr>
<td>
`num_circuits`
</td>
<td>
The number of random circuits generated for each
number of Cliffords.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of repetitions of each circuit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A RandomizedBenchMarkResult object that stores and plots the result.
</td>
</tr>

</table>

