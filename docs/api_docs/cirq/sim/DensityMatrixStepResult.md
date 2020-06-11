<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.DensityMatrixStepResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="density_matrix"/>
<meta itemprop="property" content="sample"/>
<meta itemprop="property" content="sample_measurement_ops"/>
<meta itemprop="property" content="set_density_matrix"/>
</div>

# cirq.sim.DensityMatrixStepResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/density_matrix_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A single step in the simulation of the DensityMatrixSimulator.

Inherits From: [`StepResult`](../../cirq/sim/StepResult.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.DensityMatrixStepResult`, `cirq.sim.density_matrix_simulator.DensityMatrixStepResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.DensityMatrixStepResult(
    density_matrix: np.ndarray,
    measurements: Dict[str, np.ndarray],
    qubit_map: Dict[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, int],
    dtype: Type[np.number] = np.complex64
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`density_matrix`
</td>
<td>
The density matrix at this step. Can be mutated.
</td>
</tr><tr>
<td>
`measurements`
</td>
<td>
The measurements for this step of the simulation.
</td>
</tr><tr>
<td>
`qubit_map`
</td>
<td>
A map from qid to index used to define the
ordering of the basis in density_matrix.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The numpy dtype for the density matrix.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`qubit_map`
</td>
<td>
A map from the Qubits in the Circuit to the the index
of this qubit for a canonical ordering. This canonical ordering
is used to define the state vector (see the state_vector()
method).
</td>
</tr><tr>
<td>
`measurements`
</td>
<td>
A dictionary from measurement gate key to measurement
results, ordered by the qubits that the measurement operates on.
</td>
</tr>
</table>



## Methods

<h3 id="density_matrix"><code>density_matrix</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/density_matrix_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>density_matrix()
</code></pre>

Returns the density matrix at this step in the simulation.

The density matrix that is stored in this result is returned in the
computational basis with these basis states defined by the qubit_map.
In particular the value in the qubit_map is the index of the qubit,
and these are translated into binary vectors where the last qubit is
the 1s bit of the index, the second-to-last is the 2s bit of the index,
and so forth (i.e. big endian ordering). The density matrix is a
`2 ** num_qubits` square matrix, with rows and columns ordered by
the computational basis as just described.

#### Example:


* <b>`qubit_map`</b>: {QubitA: 0, QubitB: 1, QubitC: 2}
Then the returned density matrix will have (row and column) indices
mapped to qubit basis states like the following table

   |     | QubitA | QubitB | QubitC |
   | :-: | :----: | :----: | :----: |
   |  0  |   0    |   0    |   0    |
   |  1  |   0    |   0    |   1    |
   |  2  |   0    |   1    |   0    |
   |  3  |   0    |   1    |   1    |
   |  4  |   1    |   0    |   0    |
   |  5  |   1    |   0    |   1    |
   |  6  |   1    |   1    |   0    |
   |  7  |   1    |   1    |   1    |


<h3 id="sample"><code>sample</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/density_matrix_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sample(
    qubits: List[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>],
    repetitions: int = 1,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> np.ndarray
</code></pre>

Samples from the system at this point in the computation.

Note that this does not collapse the state vector.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The qubits to be sampled in an order that influence the
returned measurement results.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of samples to take.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A seed for the pseudorandom number generator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Measurement results with True corresponding to the ``|1⟩`` state.
The outer list is for repetitions, and the inner corresponds to
measurements ordered by the supplied qubits. These lists
are wrapped as an numpy ndarray.
</td>
</tr>

</table>



<h3 id="sample_measurement_ops"><code>sample_measurement_ops</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sample_measurement_ops(
    measurement_ops: List[<a href="../../cirq/ops/GateOperation.md"><code>cirq.ops.GateOperation</code></a>],
    repetitions: int = 1,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> Dict[str, np.ndarray]
</code></pre>

Samples from the system at this point in the computation.

Note that this does not collapse the state vector.

In contrast to `sample` which samples qubits, this takes a list of
<a href="../../cirq/ops/GateOperation.md"><code>cirq.GateOperation</code></a> instances whose gates are <a href="../../cirq/ops/MeasurementGate.md"><code>cirq.MeasurementGate</code></a>
instances and then returns a mapping from the key in the measurement
gate to the resulting bit strings. Different measurement operations must
not act on the same qubits.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`measurement_ops`
</td>
<td>
`GateOperation` instances whose gates are
`MeasurementGate` instances to be sampled form.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of samples to take.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A seed for the pseudorandom number generator.
</td>
</tr>
</table>


Returns: A dictionary from measurement gate key to measurement
    results. Measurement results are stored in a 2-dimensional
    numpy array, the first dimension corresponding to the repetition
    and the second to the actual boolean measurement results (ordered
    by the qubits being measured.)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the operation's gates are not `MeasurementGate`
instances or a qubit is acted upon multiple times by different
operations from `measurement_ops`.
</td>
</tr>
</table>



<h3 id="set_density_matrix"><code>set_density_matrix</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/density_matrix_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_density_matrix(
    density_matrix_repr: Union[int, np.ndarray]
)
</code></pre>

Set the density matrix to a new density matrix.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`density_matrix_repr`
</td>
<td>
If this is an int, the density matrix is set to
the computational basis state corresponding to this state. Otherwise
if this is a np.ndarray it is the full state, either a pure state
or the full density matrix.  If it is the pure state it must be the
correct size, be normalized (an L2 norm of 1), and be safely
castable to an appropriate dtype for the simulator.  If it is a
mixed state it must be correctly sized and positive semidefinite
with trace one.
</td>
</tr>
</table>





