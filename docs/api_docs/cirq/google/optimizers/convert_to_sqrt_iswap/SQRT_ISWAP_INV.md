<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sqrt_iswap.SQRT_ISWAP_INV" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sqrt_iswap.SQRT_ISWAP_INV

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sqrt_iswap.SQRT_ISWAP_INV(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

When exponent=1, swaps the two qubits and phases |01⟩ and |10⟩ by i. More
generally, this gate's matrix is defined as follows:

    ISWAP**t ≡ exp(+i π t (X⊗X + Y⊗Y) / 4)

which is given by the matrix:

    [[1, 0, 0, 0],
     [0, c, i·s, 0],
     [0, i·s, c, 0],
     [0, 0, 0, 1]]

where:

    c = cos(π·t/2)
    s = sin(π·t/2)

<a href="../../../../cirq/ops/ISWAP.md"><code>cirq.ISWAP</code></a>, the swap gate that applies i to the |01⟩ and |10⟩ states,
is an instance of this gate at exponent=1.

#### References:

"What is the matrix of the iSwap gate?"
https://quantumcomputing.stackexchange.com/questions/2594/
