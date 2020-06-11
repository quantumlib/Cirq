<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.CliffordState" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="apply_single_qubit_unitary"/>
<meta itemprop="property" content="apply_unitary"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="destabilizers"/>
<meta itemprop="property" content="perform_measurement"/>
<meta itemprop="property" content="stabilizers"/>
<meta itemprop="property" content="state_vector"/>
<meta itemprop="property" content="to_numpy"/>
<meta itemprop="property" content="wave_function"/>
</div>

# cirq.sim.CliffordState

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A state of the Clifford simulation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CliffordState`, `cirq.sim.clifford.CliffordState`, `cirq.sim.clifford.clifford_simulator.CliffordState`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.CliffordState(
    qubit_map, initial_state=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

The state is stored using two complementary representations:
Anderson's tableaux form and Bravyi's CH-form.
The tableaux keeps track of the stabilizer operations, while the
CH-form allows access to the full state vector (including phase).

Gates and measurements are applied to each representation in O(n^2) time.

## Methods

<h3 id="apply_single_qubit_unitary"><code>apply_single_qubit_unitary</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_single_qubit_unitary(
    op: "cirq.Operation"
)
</code></pre>




<h3 id="apply_unitary"><code>apply_unitary</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_unitary(
    op: "cirq.Operation"
)
</code></pre>




<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy() -> "CliffordState"
</code></pre>




<h3 id="destabilizers"><code>destabilizers</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>destabilizers() -> List[<a href="../../cirq/ops/DensePauliString.md"><code>cirq.ops.DensePauliString</code></a>]
</code></pre>

Returns the destabilizer generators of the state. These
are n operators {S_1,S_2,...,S_n} such that along with the stabilizer
generators above generate the full Pauli group on n qubits.

<h3 id="perform_measurement"><code>perform_measurement</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>perform_measurement(
    qubits: Sequence[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>],
    prng: np.random.RandomState,
    collapse_state_vector=True
)
</code></pre>




<h3 id="stabilizers"><code>stabilizers</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stabilizers() -> List[<a href="../../cirq/ops/DensePauliString.md"><code>cirq.ops.DensePauliString</code></a>]
</code></pre>

Returns the stabilizer generators of the state. These
are n operators {S_1,S_2,...,S_n} such that S_i |psi> = |psi> 

<h3 id="state_vector"><code>state_vector</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>state_vector()
</code></pre>




<h3 id="to_numpy"><code>to_numpy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_numpy() -> np.ndarray
</code></pre>




<h3 id="wave_function"><code>wave_function</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>wave_function()
</code></pre>

THIS FUNCTION IS DEPRECATED.

IT WILL BE REMOVED IN `cirq v0.10.0`.

use state_vector instead

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






