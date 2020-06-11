<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.json_serializable_dataclass" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.json_serializable_dataclass

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/json_serialization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create a dataclass that supports JSON serialization

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.json_serializable_dataclass`, `cirq.protocols.json_serialization.json_serializable_dataclass`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.json_serializable_dataclass(
    _cls: Optional[Type] = None,
    *,
    namespace: Optional[str] = None,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function defers to the ordinary ``dataclass`` decorator but appends
the ``_json_dict_`` protocol method which automatically determines
the appropriate fields from the dataclass.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`namespace`
</td>
<td>
An optional prefix to the value associated with the
key "cirq_type". The namespace name will be joined with the
class name via a dot (.)
init, repr, eq, order, unsafe_hash, frozen: Forwarded to the
``dataclass`` constructor.
</td>
</tr>
</table>

