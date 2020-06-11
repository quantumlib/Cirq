<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.WaveFunctionTrialResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="bloch_vector_of"/>
<meta itemprop="property" content="density_matrix_of"/>
<meta itemprop="property" content="dirac_notation"/>
<meta itemprop="property" content="state_vector"/>
</div>

# cirq.sim.WaveFunctionTrialResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Deprecated. Please use `StateVectorTrialResult` instead.

Inherits From: [`StateVectorTrialResult`](../../cirq/sim/StateVectorTrialResult.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.WaveFunctionTrialResult`, `cirq.sim.state_vector_simulator.WaveFunctionTrialResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.WaveFunctionTrialResult(
    params: <a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a>,
    measurements: Dict[str, np.ndarray],
    final_simulator_state: <a href="../../cirq/sim/StateVectorSimulatorState.md"><code>cirq.sim.StateVectorSimulatorState</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`final_state`
</td>
<td>
THIS FUNCTION IS DEPRECATED.

IT WILL BE REMOVED IN `cirq v0.10.0`.

Use final_state_vector instead.
</td>
</tr><tr>
<td>
`qubit_map`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="bloch_vector_of"><code>bloch_vector_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>bloch_vector_of(
    qubit: "cirq.Qid"
) -> np.ndarray
</code></pre>

Returns the bloch vector of a qubit in the state.

Calculates the bloch vector of the given qubit
in the state given by self.state_vector(), given that
self.state_vector() follows the standard Kronecker convention of
numpy.kron.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubit`
</td>
<td>
qubit who's bloch vector we want to find.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A length 3 numpy array representing the qubit's bloch vector.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the size of the state represents more than 25 qubits.
</td>
</tr><tr>
<td>
`IndexError`
</td>
<td>
if index is out of range for the number of qubits
corresponding to the state.
</td>
</tr>
</table>



<h3 id="density_matrix_of"><code>density_matrix_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>density_matrix_of(
    qubits: List[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>] = None
) -> np.ndarray
</code></pre>

Returns the density matrix of the state.

Calculate the density matrix for the system on the list, qubits.
Any qubits not in the list that are present in self.state_vector() will
be traced out. If qubits is None the full density matrix for
self.state_vector() is returned, given self.state_vector() follows
standard Kronecker convention of numpy.kron.

#### For example:


self.state_vector() = np.array([1/np.sqrt(2), 1/np.sqrt(2)],
    dtype=np.complex64)
qubits = None
gives us
    $$
    \rho = \begin{bmatrix}
                0.5 & 0.5 \\
                0.5 & 0.5
            \end{bmatrix}
    $$

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
list containing qubit IDs that you would like
to include in the density matrix (i.e.) qubits that WON'T
be traced out.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A numpy array representing the density matrix.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the size of the state represents more than 25 qubits.
</td>
</tr><tr>
<td>
`IndexError`
</td>
<td>
if the indices are out of range for the number of qubits
corresponding to the state.
</td>
</tr>
</table>



<h3 id="dirac_notation"><code>dirac_notation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dirac_notation(
    decimals: int = 2
) -> str
</code></pre>

Returns the state vector as a string in Dirac notation.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`decimals`
</td>
<td>
How many decimals to include in the pretty print.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A pretty string consisting of a sum of computational basis kets
and non-zero floats of the specified accuracy.
</td>
</tr>

</table>



<h3 id="state_vector"><code>state_vector</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>state_vector()
</code></pre>

Return the state vector at the end of the computation.

The state is returned in the computational basis with these basis
states defined by the qubit_map. In particular the value in the
qubit_map is the index of the qubit, and these are translated into
binary vectors where the last qubit is the 1s bit of the index, the
second-to-last is the 2s bit of the index, and so forth (i.e. big
endian ordering).

#### Example:


* <b>`qubit_map`</b>: {QubitA: 0, QubitB: 1, QubitC: 2}
Then the returned vector will have indices mapped to qubit basis
states like the following table

   |     | QubitA | QubitB | QubitC |
   | :-: | :----: | :----: | :----: |
   |  0  |   0    |   0    |   0    |
   |  1  |   0    |   0    |   1    |
   |  2  |   0    |   1    |   0    |
   |  3  |   0    |   1    |   1    |
   |  4  |   1    |   0    |   0    |
   |  5  |   1    |   0    |   1    |
   |  6  |   1    |   1    |   0    |
   |  7  |   1    |   1    |   1    |


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






