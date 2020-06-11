<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.xeb_fidelity" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.xeb_fidelity

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/fidelity_estimation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Estimates XEB fidelity from one circuit using user-supplied estimator.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.fidelity_estimation.xeb_fidelity`, `cirq.xeb_fidelity`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.xeb_fidelity(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    bitstrings: Sequence[int],
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    amplitudes: Optional[Mapping[int, complex]] = None,
    estimator: Callable[[int, Sequence[float]], float] = cirq.experiments.linear_xeb_fidelity_from_probabilities
) -> float
</code></pre>



<!-- Placeholder for "Used in" -->

Fidelity quantifies the similarity of two quantum states. Here, we estimate
the fidelity between the theoretically predicted output state of circuit and
the state producted in its experimental realization. Note that we don't know
the latter state. Nevertheless, we can estimate the fidelity between the two
states from the knowledge of the bitstrings observed in the experiment.

In order to make the estimate more robust one should average the estimates
over many random circuits. The API supports per-circuit fidelity estimation
to enable users to examine the properties of estimate distribution over
many circuits.

See https://arxiv.org/abs/1608.00263 for more details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
Random quantum circuit which has been executed on quantum
processor under test.
</td>
</tr><tr>
<td>
`bitstrings`
</td>
<td>
Results of terminal all-qubit measurements performed after
each circuit execution as integer array where each integer is
formed from measured qubit values according to `qubit_order` from
most to least significant qubit, i.e. in the order consistent with
<a href="../../cirq/sim/final_state_vector.md"><code>cirq.final_state_vector</code></a>.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Qubit order used to construct bitstrings enumerating
qubits starting with the most sigificant qubit.
</td>
</tr><tr>
<td>
`amplitudes`
</td>
<td>
Optional mapping from bitstring to output amplitude.
If provided, simulation is skipped. Useful for large circuits
when an offline simulation had already been peformed.
</td>
</tr><tr>
<td>
`estimator`
</td>
<td>
Fidelity estimator to use, see above. Defaults to the
linear XEB fidelity estimator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Estimate of fidelity associated with an experimental realization of
circuit which yielded measurements in bitstrings.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
Circuit is inconsistent with qubit order or one of the
bitstrings is inconsistent with the number of qubits.
</td>
</tr>
</table>

