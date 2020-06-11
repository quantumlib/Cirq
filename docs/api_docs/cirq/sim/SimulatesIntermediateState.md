<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.SimulatesIntermediateState" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="simulate"/>
<meta itemprop="property" content="simulate_moment_steps"/>
<meta itemprop="property" content="simulate_sweep"/>
</div>

# cirq.sim.SimulatesIntermediateState

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A SimulatesFinalState that simulates a circuit by moments.

Inherits From: [`SimulatesFinalState`](../../cirq/sim/SimulatesFinalState.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SimulatesIntermediateState`, `cirq.sim.simulator.SimulatesIntermediateState`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Whereas a general SimulatesFinalState may return the entire simulator
state at the end of a circuit, a SimulatesIntermediateState can
simulate stepping through the moments of a circuit.

Implementors of this interface should implement the _simulator_iterator
method.

Note that state here refers to simulator state, which is not necessarily
a state vector.

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



<h3 id="simulate_moment_steps"><code>simulate_moment_steps</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>simulate_moment_steps(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    param_resolver: "study.ParamResolverOrSimilarType" = None,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    initial_state: Any = None
) -> Iterator
</code></pre>

Returns an iterator of StepResults for each moment simulated.

If the circuit being simulated is empty, a single step result should
be returned with the state being set to the initial state.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The Circuit to simulate.
</td>
</tr><tr>
<td>
`param_resolver`
</td>
<td>
A ParamResolver for determining values of Symbols.
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
this state depends on the simulation implementation. See
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
Iterator that steps through the simulation, simulating each
moment and returning a StepResult for each moment.
</td>
</tr>

</table>



<h3 id="simulate_sweep"><code>simulate_sweep</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>simulate_sweep(
    program: "cirq.Circuit",
    params: <a href="../../cirq/study/Sweepable.md"><code>cirq.study.Sweepable</code></a>,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    initial_state: Any = None
) -> List['SimulationTrialResult']
</code></pre>

Simulates the supplied Circuit.

This method returns a result which allows access to the entire
state vector. In contrast to simulate, this allows for sweeping
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
this state depends on the simulation implementation. See
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





