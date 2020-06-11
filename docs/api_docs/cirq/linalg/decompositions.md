<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.decompositions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="KAK_GAMMA"/>
<meta itemprop="property" content="KAK_MAGIC"/>
<meta itemprop="property" content="KAK_MAGIC_DAG"/>
<meta itemprop="property" content="MAGIC"/>
<meta itemprop="property" content="MAGIC_CONJ_T"/>
<meta itemprop="property" content="T"/>
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.linalg.decompositions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Utility methods for breaking matrices into useful pieces.



## Classes

[`class AxisAngleDecomposition`](../../cirq/linalg/AxisAngleDecomposition.md): Represents a unitary operation as an axis, angle, and global phase.

[`class KakDecomposition`](../../cirq/linalg/KakDecomposition.md): A convenient description of an arbitrary two-qubit operation.

## Functions

[`axis_angle(...)`](../../cirq/linalg/axis_angle.md): Decomposes a single-qubit unitary into axis, angle, and global phase.

[`deconstruct_single_qubit_matrix_into_angles(...)`](../../cirq/linalg/deconstruct_single_qubit_matrix_into_angles.md): Breaks down a 2x2 unitary into more useful ZYZ angle parameters.

[`kak_canonicalize_vector(...)`](../../cirq/linalg/kak_canonicalize_vector.md): Canonicalizes an XX/YY/ZZ interaction by swap/negate/shift-ing axes.

[`kak_decomposition(...)`](../../cirq/linalg/kak_decomposition.md): Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

[`kak_vector(...)`](../../cirq/linalg/kak_vector.md): Compute the KAK vectors of one or more two qubit unitaries.

[`kron_factor_4x4_to_2x2s(...)`](../../cirq/linalg/kron_factor_4x4_to_2x2s.md): Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

[`map_eigenvalues(...)`](../../cirq/linalg/map_eigenvalues.md): Applies a function to the eigenvalues of a matrix.

[`scatter_plot_normalized_kak_interaction_coefficients(...)`](../../cirq/linalg/scatter_plot_normalized_kak_interaction_coefficients.md): Plots the interaction coefficients of many two-qubit operations.

[`so4_to_magic_su2s(...)`](../../cirq/linalg/so4_to_magic_su2s.md): Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

[`unitary_eig(...)`](../../cirq/linalg/unitary_eig.md): Gives the guaranteed unitary eigendecomposition of a normal matrix.

## Other Members

* `KAK_GAMMA` <a id="KAK_GAMMA"></a>
* `KAK_MAGIC` <a id="KAK_MAGIC"></a>
* `KAK_MAGIC_DAG` <a id="KAK_MAGIC_DAG"></a>
* `MAGIC` <a id="MAGIC"></a>
* `MAGIC_CONJ_T` <a id="MAGIC_CONJ_T"></a>
* `T` <a id="T"></a>
* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
