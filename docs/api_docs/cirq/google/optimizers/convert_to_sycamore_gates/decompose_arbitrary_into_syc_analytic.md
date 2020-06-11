<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates.decompose_arbitrary_into_syc_analytic" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sycamore_gates.decompose_arbitrary_into_syc_analytic

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Synthesize an arbitrary 2 qubit operation to a sycamore operation using

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sycamore_gates.decompose_arbitrary_into_syc_analytic(
    qubit_a: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    qubit_b: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    op: <a href="../../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
the given Tabulation.

 Args:
        qubit_a: first qubit of the operation
        qubit_b: second qubit of the operation
        op: operation to decompose
        tabulation: A tabulation for the Sycamore gate.
    Returns:
        New operations iterable object
 