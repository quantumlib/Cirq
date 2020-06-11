<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.CliffordTableau" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="destabilizers"/>
<meta itemprop="property" content="stabilizers"/>
</div>

# cirq.sim.CliffordTableau

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_tableau.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tableau representation of a stabilizer state

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CliffordTableau`, `cirq.sim.clifford.CliffordTableau`, `cirq.sim.clifford.clifford_tableau.CliffordTableau`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.CliffordTableau(
    num_qubits, initial_state=0
)
</code></pre>



<!-- Placeholder for "Used in" -->
(based on Aaronson and Gottesman 2006).

The tableau stores the stabilizer generators of
the state using three binary arrays: xs, zs, and rs.

Each row of the arrays represents a Pauli string, P, that is
an eigenoperator of the state vector with eigenvalue one: P|psi> = |psi>.

## Methods

<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_tableau.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy() -> "CliffordTableau"
</code></pre>




<h3 id="destabilizers"><code>destabilizers</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_tableau.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>destabilizers() -> List[<a href="../../cirq/ops/DensePauliString.md"><code>cirq.ops.DensePauliString</code></a>]
</code></pre>

Returns the destabilizer generators of the state. These
are n operators {S_1,S_2,...,S_n} such that along with the stabilizer
generators above generate the full Pauli group on n qubits.

<h3 id="stabilizers"><code>stabilizers</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_tableau.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stabilizers() -> List[<a href="../../cirq/ops/DensePauliString.md"><code>cirq.ops.DensePauliString</code></a>]
</code></pre>

Returns the stabilizer generators of the state. These
are n operators {S_1,S_2,...,S_n} such that S_i |psi> = |psi> 

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_tableau.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




