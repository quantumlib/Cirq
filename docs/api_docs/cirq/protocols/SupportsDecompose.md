<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.SupportsDecompose" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# cirq.protocols.SupportsDecompose

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/decompose_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An object that can be decomposed into simpler operations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SupportsDecompose`, `cirq.protocols.decompose_protocol.SupportsDecompose`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.SupportsDecompose(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

All decomposition methods should ultimately terminate on basic 1-qubit and
2-qubit gates included by default in Cirq. Cirq does not make any guarantees
about what the final gate set is. Currently, decompositions within Cirq
happen to converge towards the X, Y, Z, CZ, PhasedX, specified-matrix gates,
and others. This set will vary from release to release. Because of this
variability, it is important for consumers of decomposition to look for
generic properties of gates, such as "two qubit gate with a unitary matrix",
instead of specific gate types such as CZ gates (though a consumer is
of course free to handle CZ gates in a special way, and consumers can
give an `intercepting_decomposer` to <a href="../../cirq/protocols/decompose.md"><code>cirq.decompose</code></a> that attempts to
target a specific gate set).

For example, <a href="../../cirq/ops/CCNOT.md"><code>cirq.TOFFOLI</code></a> has a `_decompose_` method that returns a pair
of Hadamard gates surrounding a <a href="../../cirq/ops/CCZ.md"><code>cirq.CCZ</code></a>. Although <a href="../../cirq/ops/CCZ.md"><code>cirq.CCZ</code></a> is not a
1-qubit or 2-qubit operation, it specifies its own `_decompose_` method
that only returns 1-qubit or 2-qubit operations. This means that iteratively
decomposing <a href="../../cirq/ops/CCNOT.md"><code>cirq.TOFFOLI</code></a> terminates in 1-qubit and 2-qubit operations, and
so almost all decomposition-aware code will be able to handle <a href="../../cirq/ops/CCNOT.md"><code>cirq.TOFFOLI</code></a>
instances.

Callers are responsible for iteratively decomposing until they are given
operations that they understand. The <a href="../../cirq/protocols/decompose.md"><code>cirq.decompose</code></a> method is a simple way
to do this, because it has logic to recursively decompose until a given
`keep` predicate is satisfied.

Code implementing `_decompose_` MUST NOT create cycles, such as a gate A
decomposes into a gate B which decomposes back into gate A. This will result
in infinite loops when calling <a href="../../cirq/protocols/decompose.md"><code>cirq.decompose</code></a>.

It is permitted (though not recommended) for the chain of decompositions
resulting from an operation to hit a dead end before reaching 1-qubit or
2-qubit operations. When this happens, <a href="../../cirq/protocols/decompose.md"><code>cirq.decompose</code></a> will raise
a `TypeError` by default, but can be configured to ignore the issue or
raise a caller-provided error.

