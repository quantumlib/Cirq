<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.SupportsDecomposeWithQubits" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# cirq.protocols.SupportsDecomposeWithQubits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/decompose_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An object that can be decomposed into operations on given qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SupportsDecomposeWithQubits`, `cirq.protocols.decompose_protocol.SupportsDecomposeWithQubits`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.SupportsDecomposeWithQubits(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returning `NotImplemented` or `None` means "not decomposable". Otherwise an
operation, list of operations, or generally anything meeting the `OP_TREE`
contract can be returned.

For example, a SWAP gate can be turned into three CNOTs. But in order to
describe those CNOTs one must be able to talk about "the target qubit" and
"the control qubit". This can only be done once the qubits-to-be-swapped are
known.

The main user of this protocol is `GateOperation`, which decomposes itself
by delegating to its gate. The qubits argument is needed because gates are
specified independently of target qubits and so must be told the relevant
qubits. A `GateOperation` implements `SupportsDecompose` as long as its gate
implements `SupportsDecomposeWithQubits`.

