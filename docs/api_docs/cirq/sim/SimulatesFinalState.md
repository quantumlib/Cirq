<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.SimulatesFinalState" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="simulate"/>
<meta itemprop="property" content="simulate_sweep"/>
</div>

# cirq.sim.SimulatesFinalState

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Simulator that allows access to the simulator's final state.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SimulatesFinalState`, `cirq.sim.simulator.SimulatesFinalState`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Implementors of this interface should implement the simulate_sweep
method. This simulator only returns the state of the quantum system
for the final step of a simulation. This simulator state may be a state
vector, the density matrix, or another representation, depending on the
implementation.  For simulators that also allow stepping through
a circuit see `SimulatesIntermediateState`.

## Methods

<h3 id="simulate"><code>simulate</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>simulate(
    program: "cirq.Circuit",
    param_resolver: "study.ParamResolverOrSimilarType" = None,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    initial_state: Any = None
) -> "SimulationTrialResult"
</code></pre>

Simulates the supplied Circuit.

This method returns a result which allows access to the entire
simulator's final state.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to simulate.
</td>
</tr><tr>
<td>
`param_resolver`
</td>
<td>
Parameters to run with the program.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines the canonical ordering of the qubits. This
is often used in specifying the initial state, i.e. the
ordering of the computational basis states.
</td>
</tr><tr>
<td>
`initial_state`
</td>
<td>
The initial state for the simulation. The  form of
this state depends on the simulation implementation.  See
documentation of the implementing class for details.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
SimulationTrialResults for the simulation. Includes the final state.
</td>
</tr>

</table>



<h3 id="simulate_sweep"><code>simulate_sweep</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>simulate_sweep(
    program: "cirq.Circuit",
    params: <a href="../../cirq/study/Sweepable.md"><code>cirq.study.Sweepable</code></a>,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    initial_state: Any = None
) -> List['SimulationTrialResult']
</code></pre>

Simulates the supplied Circuit.

This method returns a result which allows access to the entire final
simulator state. In contrast to simulate, this allows for sweeping
over different parameter values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to simulate.
</td>
</tr><tr>
<td>
`params`
</td>
<td>
Parameters to run with the program.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines the canonical ordering of the qubits. This
is often used in specifying the initial state, i.e. the
ordering of the computational basis states.
</td>
</tr><tr>
<td>
`initial_state`
</td>
<td>
The initial state for the simulation. The form of
this state depends on the simulation implementation.  See
documentation of the implementing class for details.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of SimulationTrialResults for this run, one for each
possible parameter resolver.
</td>
</tr>

</table>





