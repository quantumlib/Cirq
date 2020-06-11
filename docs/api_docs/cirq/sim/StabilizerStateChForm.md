<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.StabilizerStateChForm" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="inner_product_of_state_and_x"/>
<meta itemprop="property" content="project_Z"/>
<meta itemprop="property" content="state_vector"/>
<meta itemprop="property" content="to_state_vector"/>
<meta itemprop="property" content="wave_function"/>
</div>

# cirq.sim.StabilizerStateChForm

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/stabilizer_state_ch_form.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A representation of stabilizer states using the CH form,

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.StabilizerStateChForm`, `cirq.sim.clifford.StabilizerStateChForm`, `cirq.sim.clifford.stabilizer_state_ch_form.StabilizerStateChForm`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.StabilizerStateChForm(
    num_qubits: int,
    initial_state: Union[int, np.ndarray] = 0
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

    $|\psi> = \omega U_C U_H |s>$

This representation keeps track of overall phase.

Reference: https://arxiv.org/abs/1808.00128

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_qubits`
</td>
<td>
The number of qubits in the system
</td>
</tr><tr>
<td>
`initial_state`
</td>
<td>
If an int, the state is set to the computational
basis state corresponding to this state.
If an np.ndarray it is the full initial state.
</td>
</tr>
</table>



## Methods

<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/stabilizer_state_ch_form.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy() -> "cirq.StabilizerStateChForm"
</code></pre>




<h3 id="inner_product_of_state_and_x"><code>inner_product_of_state_and_x</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/stabilizer_state_ch_form.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>inner_product_of_state_and_x(
    x: int
) -> Union[float, complex]
</code></pre>

Returns the amplitude of x'th element of
the state vector, i.e. <x|psi> 

<h3 id="project_Z"><code>project_Z</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/stabilizer_state_ch_form.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>project_Z(
    q, z
)
</code></pre>

Applies a Z projector on the q'th qubit.

Returns: a normalized state with Z_q |psi> = z |psi>

<h3 id="state_vector"><code>state_vector</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/stabilizer_state_ch_form.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>state_vector() -> np.ndarray
</code></pre>




<h3 id="to_state_vector"><code>to_state_vector</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/stabilizer_state_ch_form.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_state_vector() -> np.ndarray
</code></pre>




<h3 id="wave_function"><code>wave_function</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/stabilizer_state_ch_form.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>wave_function() -> np.ndarray
</code></pre>

THIS FUNCTION IS DEPRECATED.

IT WILL BE REMOVED IN `cirq v0.10.0`.

Use state_vector instead.

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>






