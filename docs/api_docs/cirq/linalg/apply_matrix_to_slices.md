<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.apply_matrix_to_slices" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.apply_matrix_to_slices

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/transformations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Left-multiplies an NxN matrix onto N slices of a numpy array.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.apply_matrix_to_slices`, `cirq.linalg.transformations.apply_matrix_to_slices`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.apply_matrix_to_slices(
    target: np.ndarray,
    matrix: np.ndarray,
    slices: Sequence[_TSlice],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example:

The 4x4 matrix of a fractional SWAP gate can be expressed as

   [ 1       ]
   [   X**t  ]
   [       1 ]

Where X is the 2x2 Pauli X gate and t is the power of the swap with t=1
being a full swap. X**t is a power of the Pauli X gate's matrix.
Applying the fractional swap is equivalent to applying a fractional X
within the inner 2x2 subspace; the rest of the matrix is identity. This
can be expressed using `apply_matrix_to_slices` as follows:

    def fractional_swap(target):
        assert target.shape == (4,)
        return apply_matrix_to_slices(
            target=target,
            matrix=cirq.unitary(cirq.X**t),
            slices=[1, 2]
        )



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`target`
</td>
<td>
The input array with slices that need to be left-multiplied.
</td>
</tr><tr>
<td>
`matrix`
</td>
<td>
The linear operation to apply to the subspace defined by the
slices.
</td>
</tr><tr>
<td>
`slices`
</td>
<td>
The parts of the tensor that correspond to the "vector entries"
that the matrix should operate on. May be integers or complicated
multi-dimensional slices into a tensor. The slices must refer to
non-overlapping sections of the input all with the same shape.
</td>
</tr><tr>
<td>
`out`
</td>
<td>
Where to write the output. If not specified, a new numpy array is
created, with the same shape and dtype as the target, to store the
output.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The transformed array.
</td>
</tr>

</table>

