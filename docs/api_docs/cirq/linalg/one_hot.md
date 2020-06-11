<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.one_hot" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.one_hot

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
<p>`cirq.linalg.states.one_hot`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.one_hot(
    *,
    index: Union[None, int, Sequence[int]] = None,
    shape: Union[int, Sequence[int]] = 1,
    value: Any = 1,
    dtype: Type[np.number]
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

IT WILL BE REMOVED IN `cirq v0.9`.

Use cirq.one_hot instead.

Returns a numpy array with all 0s and a single non-zero entry(default 1).

    Args:
        index: The index that should store the `value` argument instead of 0.
            If not specified, defaults to the start of the array.
        shape: The shape of the array.
        value: The hot value to place at `index` in the result.
        dtype: The dtype of the array.

    Returns:
        The created numpy array.
    