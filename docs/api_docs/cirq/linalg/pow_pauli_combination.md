<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.pow_pauli_combination" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.pow_pauli_combination

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/operator_spaces.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes non-negative integer power of single-qubit Pauli combination.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.operator_spaces.pow_pauli_combination`, `cirq.pow_pauli_combination`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.pow_pauli_combination(
    ai: <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>,
    ax: <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>,
    ay: <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>,
    az: <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>,
    exponent: int
) -> Tuple[<a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>, <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>, <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>, <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

Returns scalar coefficients bi, bx, by, bz such that

    bi I + bx X + by Y + bz Z = (ai I + ax X + ay Y + az Z)^exponent

Correctness of the formulas below follows from the binomial expansion
and the fact that for any real or complex vector (ax, ay, az) and any
non-negative integer k:

     [ax X + ay Y + az Z]^(2k) = (ax^2 + ay^2 + az^2)^k I