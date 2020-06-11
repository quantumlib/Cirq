<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.final_density_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.sim.final_density_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/mux.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the density matrix resulting from simulating the circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.final_density_matrix`, `cirq.sim.mux.final_density_matrix`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.final_density_matrix(
    program: "cirq.CIRCUIT_LIKE",
    *,
    noise: "cirq.NOISE_MODEL_LIKE" = None,
    initial_state: Union[int, Sequence[Union[int, float, complex]], np.ndarray] = 0,
    param_resolver: <a href="../../cirq/study/ParamResolverOrSimilarType.md"><code>cirq.study.ParamResolverOrSimilarType</code></a> = None,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    dtype: Type[np.number] = np.complex64,
    seed: Optional[Union[int, np.random.RandomState]] = None,
    ignore_measurement_results: bool = True
) -> "np.ndarray"
</code></pre>



<!-- Placeholder for "Used in" -->

Note that, unlike <a href="../../cirq/sim/final_state_vector.md"><code>cirq.final_state_vector</code></a>, terminal measurements
are not omitted. Instead, all measurements are treated as sources
of decoherence (i.e. measurements do not collapse, they dephase). See
ignore_measurement_results for details.

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
`noise`
</td>
<td>
Noise model to use while running the simulation.
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
</tr><tr>
<td>
`ignore_measurement_results`
</td>
<td>
Defaults to True. When True, the returned
density matrix is not conditioned on any measurement results.
For example, this effectively replaces computational basis
measurement with dephasing noise. The result density matrix in this
case should be unique. When False, the result will be conditioned on
sampled (but unreported) measurement results. In this case the
result may vary from call to call.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The density matrix for the state which results from applying the given
operations to the desired initial state.
</td>
</tr>

</table>

