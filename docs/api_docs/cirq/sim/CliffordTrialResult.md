<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.CliffordTrialResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# cirq.sim.CliffordTrialResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Results of a simulation by a SimulatesFinalState.

Inherits From: [`SimulationTrialResult`](../../cirq/sim/SimulationTrialResult.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CliffordTrialResult`, `cirq.sim.clifford.CliffordTrialResult`, `cirq.sim.clifford.clifford_simulator.CliffordTrialResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.CliffordTrialResult(
    params: <a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a>,
    measurements: Dict[str, np.ndarray],
    final_simulator_state: "CliffordState"
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Unlike TrialResult these results contain the final simulator_state of the
system. This simulator_state is dependent on the simulation implementation
and may be, for example, the state vector or the density matrix of the
system.



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






