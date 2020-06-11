<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.StateTomographyExperiment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="fit_density_matrix"/>
</div>

# cirq.experiments.StateTomographyExperiment

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/n_qubit_tomography.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Experiment to conduct state tomography.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.n_qubit_tomography.StateTomographyExperiment`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.StateTomographyExperiment(
    qubits: Sequence['cirq.Qid'],
    prerotations: Optional[Sequence[Tuple[float, float]]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Generates data collection protocol for the state tomography experiment.
Does the fitting of generated data to determine the density matrix.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubits`
</td>
<td>
Qubits to do the tomography on.
</td>
</tr><tr>
<td>
`prerotations`
</td>
<td>
Tuples of (phase_exponent, exponent) parameters for
gates to apply to the qubits before measurement. The actual
rotation applied will be <a href="../../cirq/ops/PhasedXPowGate.md"><code>cirq.PhasedXPowGate</code></a> with the
specified values of phase_exponent and exponent. If None,
we use [(0, 0), (0, 0.5), (0.5, 0.5)], which corresponds
to rotation gates [I, X**0.5, Y**0.5].
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`rot_circuit`
</td>
<td>
Circuit with parameterized rotation gates to do before the
final measurements.
</td>
</tr><tr>
<td>
`rot_sweep`
</td>
<td>
The list of rotations on the qubits to perform before
measurement.
</td>
</tr><tr>
<td>
`mat`
</td>
<td>
Matrix of coefficients for the system.  Each row is one equation
corresponding to a rotation sequence and bit string outcome for
that rotation sequence.  Each column corresponds to the coefficient
on one term in the density matrix.
</td>
</tr><tr>
<td>
`num_qubits`
</td>
<td>
Number of qubits to do tomography on.
</td>
</tr>
</table>



## Methods

<h3 id="fit_density_matrix"><code>fit_density_matrix</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/n_qubit_tomography.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fit_density_matrix(
    counts: np.ndarray
) -> <a href="../../cirq/experiments/TomographyResult.md"><code>cirq.experiments.TomographyResult</code></a>
</code></pre>

Solves equation mat * rho = probs.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`counts`
</td>
<td>
A 2D array where each row contains measured counts
of all n-qubit bitstrings for the corresponding pre-rotations
in `rot_sweep`.  The order of the probabilities corresponds to
to `rot_sweep` and the order of the bit strings corresponds to
increasing integers up to 2**(num_qubits)-1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`TomographyResult` with density matrix corresponding to solution of
this system.
</td>
</tr>

</table>





