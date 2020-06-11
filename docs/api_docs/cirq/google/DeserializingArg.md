<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.DeserializingArg" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="required"/>
<meta itemprop="property" content="value_func"/>
</div>

# cirq.google.DeserializingArg

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/op_deserializer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Specification of the arguments to deserialize an argument to a gate.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.op_deserializer.DeserializingArg`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.DeserializingArg(
    serialized_name, constructor_arg_name, value_func=None, required=True
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`serialized_name`
</td>
<td>
The serialized name of the gate that is being
deserialized.
</td>
</tr><tr>
<td>
`constructor_arg_name`
</td>
<td>
The name of the argument in the constructor of
the gate corresponding to this serialized argument.
</td>
</tr><tr>
<td>
`value_func`
</td>
<td>
Sometimes a value from the serialized proto needs to
converted to an appropriate type or form. This function takes the
serialized value and returns the appropriate type. Defaults to
None.
</td>
</tr><tr>
<td>
`required`
</td>
<td>
Whether a value must be specified when constructing the
deserialized gate. Defaults to True.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>






## Class Variables

* `required = True` <a id="required"></a>
* `value_func = None` <a id="value_func"></a>
