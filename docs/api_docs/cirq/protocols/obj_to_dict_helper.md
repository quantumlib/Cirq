<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.obj_to_dict_helper" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.obj_to_dict_helper

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/json_serialization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Construct a dictionary containing attributes from obj

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.obj_to_dict_helper`, `cirq.protocols.json_serialization.obj_to_dict_helper`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.obj_to_dict_helper(
    obj: Any,
    attribute_names: Iterable[str],
    namespace: Optional[str] = None
) -> Dict[str, Any]
</code></pre>



<!-- Placeholder for "Used in" -->

This is useful as a helper function in objects implementing the
SupportsJSON protocol, particularly in the _json_dict_ method.

In addition to keys and values specified by `attribute_names`, the
returned dictionary has an additional key "cirq_type" whose value
is the string name of the type of `obj`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`obj`
</td>
<td>
A python object with attributes to be placed in the dictionary.
</td>
</tr><tr>
<td>
`attribute_names`
</td>
<td>
The names of attributes to serve as keys in the
resultant dictionary. The values will be the attribute values.
</td>
</tr><tr>
<td>
`namespace`
</td>
<td>
An optional prefix to the value associated with the
key "cirq_type". The namespace name will be joined with the
class name via a dot (.)
</td>
</tr>
</table>

