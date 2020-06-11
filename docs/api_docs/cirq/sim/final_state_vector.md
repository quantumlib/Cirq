<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.final_state_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.sim.final_state_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/mux.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the state vector resulting from acting operations on a state.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.final_state_vector`, `cirq.sim.mux.final_state_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.final_state_vector(
    program: "cirq.CIRCUIT_LIKE",
    *,
    initial_state: Union[int, Sequence[Union[int, float, complex]], np.ndarray] = 0,
    param_resolver: <a href="../../cirq/study/ParamResolverOrSimilarType.md"><code>cirq.study.ParamResolverOrSimilarType</code></a> = None,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    dtype: Type[np.number] = np.complex64,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> "np.ndarray"
</code></pre>



<!-- Placeholder for "Used in" -->

By default the input state is the computational basis zero state, in which
case the output is just the first column of the implied unitary matrix.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit, gate, operation, or tree of operations
to apply to the initial state in order to produce the result.
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
If an int, the state is set to the computational
basis state corresponding to this state. Otherwise  if this
is a np.ndarray it is the full initial state. In this case it
must be the correct size, be normalized (an L2 norm of 1), and
be safely castable to an appropriate dtype for the simulator.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The `numpy.dtype` used by the simulation. Typically one of
`numpy.complex64` or `numpy.complex128`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
The random seed to use for this simulator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The state vector resulting from applying the given unitary operations to
the desired initial state. Specifically, a numpy array containing the
the amplitudes in np.kron order, where the order of arguments to kron
is determined by the qubit order argument (which defaults to just
sorting the qubits that are present into an ascending order).
</td>
</tr>

</table>

