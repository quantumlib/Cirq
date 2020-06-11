<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.two_qubit_gates.gate_compilation.reduce" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.two_qubit_gates.gate_compilation.reduce

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



reduce(function, sequence[, initial]) -> value

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.two_qubit_gates.gate_compilation.reduce()
</code></pre>



<!-- Placeholder for "Used in" -->

Apply a function of two arguments cumulatively to the items of a sequence,
from left to right, so as to reduce the sequence to a single value.
For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
of the sequence in the calculation, and serves as a default when the
sequence is empty.