<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.QFT" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.QFT

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/fourier_transform.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



THIS FUNCTION IS DEPRECATED.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.QFT`, `cirq.ops.fourier_transform.QFT`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.QFT(
    *qubits,
    without_reverse: bool = False,
    inverse: bool = False
) -> "cirq.Operation"
</code></pre>



<!-- Placeholder for "Used in" -->

IT WILL BE REMOVED IN `cirq v0.10.0`.

Use cirq.qft instead.

The quantum Fourier transform.

    Transforms a qubit register from the computational basis to the frequency
    basis.

    The inverse quantum Fourier transform is `cirq.qft(*qubits)**-1` or
    equivalently `cirq.inverse(cirq.qft(*qubits))`.

    Args:
        qubits: The qubits to apply the qft to.
        without_reverse: When set, swap gates at the end of the qft are omitted.
            This reverses the qubit order relative to the standard qft effect,
            but makes the gate cheaper to apply.
        inverse: If set, the inverse qft is performed instead of the qft.
            Equivalent to calling <a href="../../cirq/protocols/inverse.md"><code>cirq.inverse</code></a> on the result, or raising it
            to the -1.

    Returns:
        A <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a> applying the qft to the given qubits.
    