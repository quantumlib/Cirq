<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.DensityMatrixTrialResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# cirq.sim.DensityMatrixTrialResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/density_matrix_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A `SimulationTrialResult` for `DensityMatrixSimulator` runs.

Inherits From: [`SimulationTrialResult`](../../cirq/sim/SimulationTrialResult.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.DensityMatrixTrialResult`, `cirq.sim.density_matrix_simulator.DensityMatrixTrialResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.DensityMatrixTrialResult(
    params: <a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a>,
    measurements: Dict[str, np.ndarray],
    final_simulator_state: <a href="../../cirq/sim/DensityMatrixSimulatorState.md"><code>cirq.sim.DensityMatrixSimulatorState</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

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




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`params`
</td>
<td>
A ParamResolver of settings used for this result.
</td>
</tr><tr>
<td>
`measurements`
</td>
<td>
A dictionary from measurement gate key to measurement
results. Measurement results are a numpy ndarray of actual boolean
measurement results (ordered by the qubits acted on by the
measurement gate.)
</td>
</tr><tr>
<td>
`final_simulator_state`
</td>
<td>
The final simulator state of the system after the
trial finishes.
</td>
</tr><tr>
<td>
`final_density_matrix`
</td>
<td>
The final density matrix of the system.
</td>
</tr><tr>
<td>
`qubit_map`
</td>
<td>
A map from Qid to index used to define the ordering of the basis in
the result.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>






