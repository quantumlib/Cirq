<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.json_serialization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DEFAULT_RESOLVERS"/>
<meta itemprop="property" content="RESOLVER_CACHE"/>
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.protocols.json_serialization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/json_serialization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Classes

[`class CirqEncoder`](../../cirq/protocols/json_serialization/CirqEncoder.md): Extend json.JSONEncoder to support Cirq objects.

[`class SupportsJSON`](../../cirq/protocols/SupportsJSON.md): An object that can be turned into JSON dictionaries.

## Functions

[`json_serializable_dataclass(...)`](../../cirq/protocols/json_serializable_dataclass.md): Create a dataclass that supports JSON serialization

[`obj_to_dict_helper(...)`](../../cirq/protocols/obj_to_dict_helper.md): Construct a dictionary containing attributes from obj

[`read_json(...)`](../../cirq/protocols/read_json.md): Read a JSON file that optionally contains cirq objects.

[`to_json(...)`](../../cirq/protocols/to_json.md): Write a JSON file containing a representation of obj.

## Other Members

* `DEFAULT_RESOLVERS` <a id="DEFAULT_RESOLVERS"></a>
* `RESOLVER_CACHE` <a id="RESOLVER_CACHE"></a>
* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
