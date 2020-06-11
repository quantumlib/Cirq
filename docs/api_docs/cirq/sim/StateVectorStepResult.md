<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.StateVectorStepResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="sample"/>
<meta itemprop="property" content="sample_measurement_ops"/>
</div>

# cirq.sim.StateVectorStepResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Results of a step of a SimulatesIntermediateState.

Inherits From: [`StepResult`](../../cirq/sim/StepResult.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.StateVectorStepResult`, `cirq.sim.state_vector_simulator.StateVectorStepResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.StateVectorStepResult(
    measurements: Optional[Dict[str, List[int]]] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
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

<h3 id="sample"><code>sample</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
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





