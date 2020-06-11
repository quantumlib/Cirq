<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.random_circuit" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.random_circuit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/random_circuit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generates a random circuit.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.random_circuit(
    qubits: Union[Sequence[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>], int],
    n_moments: int,
    op_density: float,
    gate_domain: Optional[Dict[<a href="../../cirq/ops/Gate.md"><code>cirq.ops.Gate</code></a>, int]] = None,
    random_state: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubits`
</td>
<td>
If a sequence of qubits, then these are the qubits that
the circuit should act on. Because the qubits on which an
operation acts are chosen randomly, not all given qubits
may be acted upon. If an int, then this number of qubits will
be automatically generated, and the qubits will be
`cirq.NamedQubits` with names given by the integers in
`range(qubits)`.
</td>
</tr><tr>
<td>
`n_moments`
</td>
<td>
The number of moments in the generated circuit.
</td>
</tr><tr>
<td>
`op_density`
</td>
<td>
The probability that a gate is selected to operate on
randomly selected qubits. Note that this is not the expected number
of qubits that are acted on, since there are cases where the
number of qubits that a gate acts on does not evenly divide the
total number of qubits.
</td>
</tr><tr>
<td>
`gate_domain`
</td>
<td>
The set of gates to choose from, specified as a dictionary
where each key is a gate and the value of the key is the number of
qubits the gate acts on. If not provided, the default gate domain is
{X, Y, Z, H, S, T, CNOT, CZ, SWAP, ISWAP, CZPowGate()}. Only gates
which act on a number of qubits less than len(qubits) (or qubits if
provided as an int) are selected from the gate domain.
</td>
</tr><tr>
<td>
`random_state`
</td>
<td>
Random state or random state seed.
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
* op_density is not in (0, 1].
* gate_domain is empty.
* qubits is an int less than 1 or an empty sequence.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The randomly generated Circuit.
</td>
</tr>

</table>

