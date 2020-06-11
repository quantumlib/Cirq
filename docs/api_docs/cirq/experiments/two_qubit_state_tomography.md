<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.two_qubit_state_tomography" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.two_qubit_state_tomography

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/qubit_characterizations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Two-qubit state tomography.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.qubit_characterizations.two_qubit_state_tomography`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.two_qubit_state_tomography(
    sampler: <a href="../../cirq/work/Sampler.md"><code>cirq.work.Sampler</code></a>,
    first_qubit: <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>,
    second_qubit: <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>,
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    repetitions: int = 1000
) -> <a href="../../cirq/experiments/TomographyResult.md"><code>cirq.experiments.TomographyResult</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

To measure the density matrix of the output state of a two-qubit circuit,
different combinations of I, X/2 and Y/2 operations are applied to the
two qubits before measurements in the z-basis to determine the state
probabilities P_00, P_01, P_10.

The density matrix rho is decomposed into an operator-sum representation
\sum_{i, j} c_ij * sigma_i \bigotimes sigma_j, where i, j = 0, 1, 2,
3 and sigma_0 = I, sigma_1 = sigma_x, sigma_2 = sigma_y, sigma_3 =
sigma_z are the single-qubit Identity and Pauli matrices.

Based on the measured probabilities probs and the transformations of the
measurement operator by different basis rotations, one can build an
overdetermined set of linear equations.

As an example, if the identity operation (I) is applied to both qubits,
the measurement operators are (I +/- sigma_z) \bigotimes (I +/- sigma_z).
The state probabilities P_00, P_01, P_10 thus obtained contribute to the
following linear equations (setting c_00 = 1):

c_03 + c_30 + c_33 = 4*P_00 - 1
-c_03 + c_30 - c_33 = 4*P_01 - 1
c_03 - c_30 - c_33 = 4*P_10 - 1

And if a Y/2 rotation is applied to the first qubit and a X/2 rotation
is applied to the second qubit before measurement, the measurement
operators are (I -/+ sigma_x) \bigotimes (I +/- sigma_y). The probabilities
obtained instead contribute to the following linear equations:

c_02 - c_10 - c_12 = 4*P_00 - 1
-c_02 - c_10 + c_12 = 4*P_01 - 1
c_02 + c_10 + c_12 = 4*P_10 - 1

Note that this set of equations has the same form as the first set under
the transformation c_03 <-> c_02, c_30 <-> -c_10 and c_33 <-> -c_12.

Since there are 9 possible combinations of rotations (each producing 3
independent probabilities) and a total of 15 unknown coefficients c_ij,
one can cast all the measurement results into a overdetermined set of
linear equations numpy.dot(mat, c) = probs. Here c is of length 15 and
contains all the c_ij's (except c_00 which is set to 1), and mat is a 27
by 15 matrix having three non-zero elements in each row that are either
1 or -1.

The least-square solution to the above set of linear equations is then
used to construct the density matrix rho.

See Vandersypen and Chuang, Rev. Mod. Phys. 76, 1037 for details and
Steffen et al, Science 313, 1423 for a related experiment.

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
`circuit`
</td>
<td>
The circuit to execute on the qubits before tomography.
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

