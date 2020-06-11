<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.eye_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.eye_tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/states.py">
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
<p>`cirq.linalg.states.eye_tensor`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.eye_tensor(
    half_shape: Tuple[int, ...],
    *,
    dtype: Type[np.number]
) -> np.array
</code></pre>



<!-- Placeholder for "Used in" -->

IT WILL BE REMOVED IN `cirq v0.9`.

Use cirq.eye_tensor instead.

Returns an identity matrix reshaped into a tensor.

    Args:
        half_shape: A tuple representing the number of quantum levels of each
            qubit the returned matrix applies to.  `half_shape` is (2, 2, 2) for
            a three-qubit identity operation tensor.
        dtype: The numpy dtype of the new array.

    Returns:
        The created numpy array with shape `half_shape + half_shape`.
    