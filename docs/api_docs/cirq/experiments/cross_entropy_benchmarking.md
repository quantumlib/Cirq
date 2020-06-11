<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.cross_entropy_benchmarking" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.cross_entropy_benchmarking

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/cross_entropy_benchmarking.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Cross-entropy benchmarking (XEB) of multiple qubits.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.cross_entropy_benchmarking(
    sampler: <a href="../../cirq/work/Sampler.md"><code>cirq.work.Sampler</code></a>,
    qubits: Sequence[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>],
    *,
    benchmark_ops: Sequence[<a href="../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>] = None,
    num_circuits: int = 20,
    repetitions: int = 1000,
    cycles: Union[int, Iterable[int]] = range(2, 103, 10),
    scrambling_gates_per_cycle: List[List[<a href="../../cirq/ops/SingleQubitGate.md"><code>cirq.ops.SingleQubitGate</code></a>]] = None,
    simulator: <a href="../../cirq/sim/Simulator.md"><code>cirq.sim.Simulator</code></a> = None
) -> <a href="../../cirq/experiments/CrossEntropyResult.md"><code>cirq.experiments.CrossEntropyResult</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

A total of M random circuits are generated, each of which comprises N
layers where N = max('cycles') or 'cycles' if a single value is specified
for the 'cycles' parameter. Every layer contains randomly generated
single-qubit gates applied to each qubit, followed by a set of
user-defined benchmarking operations (e.g. a set of two-qubit gates).

Each circuit (circuit_m) from the M random circuits is further used to
generate a set of circuits {circuit_mn}, where circuit_mn is built from the
first n cycles of circuit_m. n spans all the values in 'cycles'.

For each fixed value n, the experiment performs the following:

1) Experimentally collect a number of bit-strings for each circuit_mn via
projective measurements in the z-basis.

2) Theoretically compute the expected bit-string probabilities
$P^{th, mn}_|...00>$,  $P^{th, mn}_|...01>$, $P^{th, mn}_|...10>$,
$P^{th, mn}_|...11>$ ... at the end of circuit_mn for all m and for all
possible bit-strings in the Hilbert space.

3) Compute an experimental XEB function for each circuit_mn:

$f_{mn}^{meas} = \langle D * P^{th, mn}_q - 1 \rangle$

where D is the number of states in the Hilbert space, $P^{th, mn}_q$ is the
theoretical probability of a bit-string q at the end of circuit_mn, and
$\langle \rangle$ corresponds to the ensemble average over all measured
bit-strings.

Then, take the average of $f_{mn}^{meas}$ over all circuit_mn with fixed
n to obtain:

$f_{n} ^ {meas} = (\sum_m f_{mn}^{meas}) / M$

4) Compute a theoretical XEB function for each circuit_mn:

$f_{mn}^{th} = D \sum_q (P^{th, mn}_q) ** 2 - 1$

where the summation goes over all possible bit-strings q in the Hilbert
space.

Similarly, we then average $f_m^{th}$ over all circuit_mn with fixed n to
obtain:

$f_{n} ^ {th} = (\sum_m f_{mn}^{th}) / M$

5) Calculate the XEB fidelity $\alpha_n$ at fixed n:

$\alpha_n = f_{n} ^ {meas} / f_{n} ^ {th}$

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sampler`
</td>
<td>
The quantum engine or simulator to run the circuits.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
The qubits included in the XEB experiment.
</td>
</tr><tr>
<td>
`benchmark_ops`
</td>
<td>
A sequence of ops.Moment containing gate operations
between specific qubits which are to be benchmarked for fidelity.
If more than one ops.Moment is specified, the random circuits
will rotate between the ops.Moment's. As an example,
if benchmark_ops = [Moment([ops.CZ(q0, q1), ops.CZ(q2, q3)]),
Moment([ops.CZ(q1, q2)]) where q0, q1, q2 and q3 are instances of
Qid (such as GridQubits), each random circuit will apply CZ gate
between q0 and q1 plus CZ between q2 and q3 for the first cycle,
CZ gate between q1 and q2 for the second cycle, CZ between q0 and
q1 and CZ between q2 and q3 for the third cycle and so on. If
None, the circuits will consist only of single-qubit gates.
</td>
</tr><tr>
<td>
`num_circuits`
</td>
<td>
The total number of random circuits to be used.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of measurements for each circuit to estimate
the bit-string probabilities.
</td>
</tr><tr>
<td>
`cycles`
</td>
<td>
The different numbers of circuit layers in the XEB study.
Could be a single or a collection of values.
</td>
</tr><tr>
<td>
`scrambling_gates_per_cycle`
</td>
<td>
If None (by default), the single-qubit
gates are chosen from X/2 ($\pi/2$ rotation around the X axis),
Y/2 ($\pi/2$ rotation around the Y axis) and (X + Y)/2 ($\pi/2$
rotation around an axis $\pi/4$ away from the X on the equator of
the Bloch sphere). Otherwise the single-qubit gates for each layer
are chosen from a list of possible choices (each choice is a list
of one or more single-qubit gates).
</td>
</tr><tr>
<td>
`simulator`
</td>
<td>
A simulator that calculates the bit-string probabilities
of the ideal circuit. By default, this is set to sim.Simulator().
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A CrossEntropyResult object that stores and plots the result.
</td>
</tr>

</table>

