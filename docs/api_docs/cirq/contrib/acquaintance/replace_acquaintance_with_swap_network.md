<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.replace_acquaintance_with_swap_network" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.replace_acquaintance_with_swap_network

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/mutation_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Replace every moment containing acquaintance gates (after

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.mutation_utils.replace_acquaintance_with_swap_network`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.replace_acquaintance_with_swap_network(
    circuit: "cirq.Circuit",
    qubit_order: Sequence['cirq.Qid'],
    acquaintance_size: Optional[int] = 0,
    swap_gate: "cirq.Gate" = cirq.ops.SWAP
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->
rectification) with a generalized swap network, with the partition
given by the acquaintance gates in that moment (and singletons for the
free qubits). Accounts for reversing effect of swap networks.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The acquaintance strategy.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
The qubits, in order, on which the replacing swap network
gate acts on.
</td>
</tr><tr>
<td>
`acquaintance_size`
</td>
<td>
The acquaintance size of the new swap network gate.
</td>
</tr><tr>
<td>
`swap_gate`
</td>
<td>
The gate used to swap logical indices.
</td>
</tr>
</table>


Returns: Whether or not the overall effect of the inserted swap network
    gates is to reverse the order of the qubits, i.e. the parity of the
    number of swap network gates inserted.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
circuit is not an acquaintance strategy.
</td>
</tr>
</table>

