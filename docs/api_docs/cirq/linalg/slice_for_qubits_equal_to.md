<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.slice_for_qubits_equal_to" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.slice_for_qubits_equal_to

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/predicates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns an index corresponding to a desired subset of an np.ndarray.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.predicates.slice_for_qubits_equal_to`, `cirq.slice_for_qubits_equal_to`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.slice_for_qubits_equal_to(
    target_qubit_axes: Sequence[int],
    little_endian_qureg_value: int = 0,
    *,
    big_endian_qureg_value: int = 0,
    num_qubits: Optional[int] = None,
    qid_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[Union[slice, int, 'ellipsis'], ...]
</code></pre>



<!-- Placeholder for "Used in" -->

It is assumed that the np.ndarray's shape is of the form (2, 2, 2, ..., 2).

#### Example:


```python
# A '4 qubit' tensor with values from 0 to 15.
r = np.array(range(16)).reshape((2,) * 4)

# We want to index into the subset where qubit #1 and qubit #3 are ON.
s = cirq.slice_for_qubits_equal_to([1, 3], 0b11)
print(s)
# (slice(None, None, None), 1, slice(None, None, None), 1, Ellipsis)

# Get that subset. It corresponds to numbers of the form 0b*1*1.
# where here '*' indicates any possible value.
print(r[s])
# [[ 5  7]
#  [13 15]]
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`target_qubit_axes`
</td>
<td>
The qubits that are specified by the index bits. All
other axes of the slice are unconstrained.
</td>
</tr><tr>
<td>
`little_endian_qureg_value`
</td>
<td>
An integer whose bits specify what value is
desired for of the target qubits. The integer is little endian
w.r.t. the target qubit axes, meaning the low bit of the integer
determines the desired value of the first targeted qubit, and so
forth with the k'th targeted qubit's value set to
bool(qureg_value & (1 << k)).
</td>
</tr><tr>
<td>
`big_endian_qureg_value`
</td>
<td>
Same as `little_endian_qureg_value` but big
endian w.r.t. to target qubit axes, meaning the low bit of the
integer dertemines the desired value of the last target qubit, and
so forth.  Specify exactly one of the `*_qureg_value` arguments.
</td>
</tr><tr>
<td>
`num_qubits`
</td>
<td>
If specified the slices will extend all the way up to
this number of qubits, otherwise if it is None, the final element
return will be Ellipsis. Optional and defaults to using Ellipsis.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
The qid shape of the state vector being sliced.  Specify this
instead of `num_qubits` when using qids with dimension != 2.  The
qureg value is interpreted to store digits with corresponding bases
packed into an int.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An index object that will slice out a mutable view of the desired subset
of a tensor.
</td>
</tr>

</table>

