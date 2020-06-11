<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.build_entangling_layers" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.build_entangling_layers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/cross_entropy_benchmarking.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Builds a sequence of gates that entangle all pairs of qubits on a grid.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.build_entangling_layers(
    qubits: Sequence[<a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>],
    two_qubit_gate: <a href="../../cirq/ops/TwoQubitGate.md"><code>cirq.ops.TwoQubitGate</code></a>
) -> List[<a href="../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

The qubits are restricted to be physically on a square grid with distinct
row and column indices (not every node of the grid needs to have a
qubit). To entangle all pairs of qubits, a user-specified two-qubit gate
is applied between each and every pair of qubit that are next to each
other. In general, a total of four sets of parallel operations are needed to
perform all possible two-qubit gates. We proceed as follows:

The first layer applies two-qubit gates to qubits (i, j) and (i, j + 1)
where i is any integer and j is an even integer. The second layer
applies two-qubit gates to qubits (i, j) and (i + 1, j) where i is an even
integer and j is any integer. The third layer applies two-qubit gates
to qubits (i, j) and (i, j + 1) where i is any integer and j is an odd
integer. The fourth layer applies two-qubit gates to qubits (i, j) and
(i + 1, j) where i is an odd integer and j is any integer.

After the layers are built as above, any empty layer is ejected.:

             Cycle 1:                            Cycle 2:
    q00 ── q01    q02 ── q03            q00    q01    q02    q03
                                         |      |      |      |
    q10 ── q11    q12 ── q13            q10    q11    q12    q13

    q20 ── q21    q22 ── q23            q20    q21    q22    q23
                                         |      |      |      |
    q30 ── q31    q32 ── q33            q30    q31    q32    q33

              Cycle 3:                           Cycle 4:
    q00    q01 ── q02    q03            q00    q01    q02    q03

    q10    q11 ── q12    q13            q10    q11    q12    q13
                                         |      |      |      |
    q20    q21 ── q22    q23            q20    q21    q22    q23

    q30    q31 ── q32    q33            q30    q31    q32    q33

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The grid qubits included in the entangling operations.
</td>
</tr><tr>
<td>
`two_qubit_gate`
</td>
<td>
The two-qubit gate to be applied between all
neighboring pairs of qubits.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of ops.Moment, with a maximum length of 4. Each ops.Moment
includes two-qubit gates which can be performed at the same time.
</td>
</tr>

</table>

