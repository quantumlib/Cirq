<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.QasmOutput" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="is_valid_qasm_id"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="valid_id_re"/>
</div>

# cirq.circuits.QasmOutput

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/qasm_output.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>





<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.QasmOutput`, `cirq.circuits.qasm_output.QasmOutput`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.circuits.QasmOutput(
    operations: "cirq.OP_TREE",
    qubits: Tuple['cirq.Qid', ...],
    header: str = '',
    precision: int = 10,
    version: str = '2.0'
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="is_valid_qasm_id"><code>is_valid_qasm_id</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/qasm_output.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_valid_qasm_id(
    id_str: str
) -> bool
</code></pre>

Test if id_str is a valid id in QASM grammar.


<h3 id="save"><code>save</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/qasm_output.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    path: Union[str, bytes, int]
) -> None
</code></pre>

Write QASM output to a file specified by path.




## Class Variables

* `valid_id_re` <a id="valid_id_re"></a>
