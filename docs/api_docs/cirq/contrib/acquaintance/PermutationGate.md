<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.PermutationGate" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="controlled"/>
<meta itemprop="property" content="num_qubits"/>
<meta itemprop="property" content="on"/>
<meta itemprop="property" content="permutation"/>
<meta itemprop="property" content="update_mapping"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="validate_permutation"/>
<meta itemprop="property" content="with_probability"/>
<meta itemprop="property" content="wrap_in_linear_combination"/>
</div>

# cirq.contrib.acquaintance.PermutationGate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/permutation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A permutation gate indicates a change in the mapping from qubits to

Inherits From: [`Gate`](../../../cirq/ops/Gate.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.permutation.PermutationGate`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.PermutationGate(
    num_qubits: int,
    swap_gate: "cirq.Gate" = cirq.ops.SWAP
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
logical indices.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`swap_gate`
</td>
<td>
the gate that swaps the indices mapped to by a pair of
qubits (e.g. SWAP or fermionic swap).
</td>
</tr>
</table>



## Methods

<h3 id="controlled"><code>controlled</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>controlled(
    num_controls: int = None,
    control_values: Optional[Sequence[Union[int, Collection[int]]]] = None,
    control_qid_shape: Optional[Tuple[int, ...]] = None
) -> "Gate"
</code></pre>

Returns a controlled version of this gate. If no arguments are
specified, defaults to a single qubit control.

 num_controls: Total number of control qubits.
 control_values: For which control qubit values to apply the sub
     gate.  A sequence of length `num_controls` where each
     entry is an integer (or set of integers) corresponding to the
     qubit value (or set of possible values) where that control is
     enabled.  When all controls are enabled, the sub gate is
     applied.  If unspecified, control values default to 1.
 control_qid_shape: The qid shape of the controls.  A tuple of the
     expected dimension of each control qid.  Defaults to
     `(2,) * num_controls`.  Specify this argument when using qudits.

<h3 id="num_qubits"><code>num_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/permutation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_qubits() -> int
</code></pre>

The number of qubits this gate acts on.


<h3 id="on"><code>on</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on(
    *qubits
) -> "Operation"
</code></pre>

Returns an application of this gate to the given qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*qubits`
</td>
<td>
The collection of qubits to potentially apply the gate to.
</td>
</tr>
</table>



<h3 id="permutation"><code>permutation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/permutation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>permutation() -> Dict[int, int]
</code></pre>

permutation = {i: s[i]} indicates that the i-th element is mapped to
the s[i]-th element.

<h3 id="update_mapping"><code>update_mapping</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/permutation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_mapping(
    mapping: Dict[<a href="../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, <a href="../../../cirq/contrib/acquaintance/executor/LogicalIndex.md"><code>cirq.contrib.acquaintance.executor.LogicalIndex</code></a>],
    keys: Sequence['cirq.Qid']
) -> None
</code></pre>

Updates a mapping (in place) from qubits to logical indices.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`mapping`
</td>
<td>
The mapping to update.
</td>
</tr><tr>
<td>
`keys`
</td>
<td>
The qubits acted on by the gate.
</td>
</tr>
</table>



<h3 id="validate_args"><code>validate_args</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_args(
    qubits: Sequence['cirq.Qid']
) -> None
</code></pre>

Checks if this gate can be applied to the given qubits.

By default checks that:
- inputs are of type `Qid`
- len(qubits) == num_qubits()
- qubit_i.dimension == qid_shape[i] for all qubits

Child classes can override.  The child implementation should call
`super().validate_args(qubits)` then do custom checks.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The sequence of qubits to potentially apply the gate to.
</td>
</tr>
</table>



#### Throws:


* <b>`ValueError`</b>: The gate can't be applied to the qubits.


<h3 id="validate_permutation"><code>validate_permutation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/permutation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>validate_permutation(
    permutation: Dict[int, int],
    n_elements: int = None
) -> None
</code></pre>




<h3 id="with_probability"><code>with_probability</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_probability(
    probability: "cirq.TParamVal"
) -> "cirq.Gate"
</code></pre>




<h3 id="wrap_in_linear_combination"><code>wrap_in_linear_combination</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>wrap_in_linear_combination(
    coefficient: Union[complex, float, int] = 1
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other: Union['Gate', 'cirq.LinearCombinationOfGates']
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Call self as a function.


<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other: Union[complex, float, int]
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__neg__"><code>__neg__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__() -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    power
)
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    other: Union[complex, float, int]
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other: Union['Gate', 'cirq.LinearCombinationOfGates']
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__truediv__"><code>__truediv__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    other: Union[complex, float, int]
) -> "cirq.LinearCombinationOfGates"
</code></pre>






