<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.SimulatesAmplitudes" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="compute_amplitudes"/>
<meta itemprop="property" content="compute_amplitudes_sweep"/>
</div>

# cirq.sim.SimulatesAmplitudes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Simulator that computes final amplitudes of given bitstrings.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SimulatesAmplitudes`, `cirq.sim.simulator.SimulatesAmplitudes`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Given a circuit and a list of bitstrings, computes the amplitudes
of the given bitstrings in the state obtained by applying the circuit
to the all zeros state. Implementors of this interface should implement
the compute_amplitudes_sweep method.

## Methods

<h3 id="compute_amplitudes"><code>compute_amplitudes</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_amplitudes(
    program: "cirq.Circuit",
    bitstrings: Sequence[int],
    param_resolver: "study.ParamResolverOrSimilarType" = None,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT
) -> Sequence[complex]
</code></pre>

Computes the desired amplitudes.

The initial state is assumed to be the all zeros state.

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
`bitstrings`
</td>
<td>
The bitstrings whose amplitudes are desired, input
as an integer array where each integer is formed from measured
qubit values according to `qubit_order` from most to least
significant qubit, i.e. in big-endian ordering.
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
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of amplitudes.
</td>
</tr>

</table>



<h3 id="compute_amplitudes_sweep"><code>compute_amplitudes_sweep</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>compute_amplitudes_sweep(
    program: "cirq.Circuit",
    bitstrings: Sequence[int],
    params: <a href="../../cirq/study/Sweepable.md"><code>cirq.study.Sweepable</code></a>,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT
) -> Sequence[Sequence[complex]]
</code></pre>

Computes the desired amplitudes.

The initial state is assumed to be the all zeros state.

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
`bitstrings`
</td>
<td>
The bitstrings whose amplitudes are desired, input
as an integer array where each integer is formed from measured
qubit values according to `qubit_order` from most to least
significant qubit, i.e. in big-endian ordering.
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
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of lists of amplitudes. The outer dimension indexes the
circuit parameters and the inner dimension indexes the bitstrings.
</td>
</tr>

</table>





