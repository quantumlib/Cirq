<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.single_qubit_state_tomography" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.single_qubit_state_tomography

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/qubit_characterizations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Single-qubit state tomography.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.qubit_characterizations.single_qubit_state_tomography`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.single_qubit_state_tomography(
    sampler: <a href="../../cirq/work/Sampler.md"><code>cirq.work.Sampler</code></a>,
    qubit: <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>,
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    repetitions: int = 1000
) -> <a href="../../cirq/experiments/TomographyResult.md"><code>cirq.experiments.TomographyResult</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

The density matrix of the output state of a circuit is measured by first
doing projective measurements in the z-basis, which determine the
diagonal elements of the matrix. A X/2 or Y/2 rotation is then added before
the z-basis measurement, which determines the imaginary and real parts of
the off-diagonal matrix elements, respectively.

See Vandersypen and Chuang, Rev. Mod. Phys. 76, 1037 for details.

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
`circuit`
</td>
<td>
The circuit to execute on the qubit before tomography.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of measurements for each basis rotation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A TomographyResult object that stores and plots the density matrix.
</td>
</tr>

</table>

