<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.ApplyChannelArgs" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# cirq.protocols.ApplyChannelArgs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_channel_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Arguments for efficiently performing a channel.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ApplyChannelArgs`, `cirq.protocols.apply_channel_protocol.ApplyChannelArgs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.ApplyChannelArgs(
    target_tensor: np.ndarray,
    out_buffer: np.ndarray,
    auxiliary_buffer0: np.ndarray,
    auxiliary_buffer1: np.ndarray,
    left_axes: Iterable[int],
    right_axes: Iterable[int]
)
</code></pre>



<!-- Placeholder for "Used in" -->

A channel performs the mapping
    $$
    X \rightarrow \sum_i A_i X A_i^\dagger
    $$
for operators $A_i$ that satisfy the normalization condition
    $$
    \sum_i A_i^\dagger A_i = I.
    $$

The receiving object is expected to mutate `target_tensor` so that it
contains the density matrix after multiplication, and then return
`target_tensor`. Alternatively, if workspace is required,
the receiving object can overwrite `out_buffer` with the results
and return `out_buffer`. Or, if the receiving object is attempting to
be simple instead of fast, it can create an entirely new array and
return that.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`target_tensor`
</td>
<td>
The input tensor that needs to be left and right
multiplied and summed representing the effect of the channel.
The tensor will have the shape (2, 2, 2, ..., 2). It usually
corresponds to a multi-qubit density matrix, with the first
n indices corresponding to the rows of the density matrix and
the last n indices corresponding to the columns of the density
matrix.
</td>
</tr><tr>
<td>
`out_buffer`
</td>
<td>
Pre-allocated workspace with the same shape and
dtype as the target tensor. If buffers are used, the result
should end up in this buffer. It is the responsibility of
calling code to notice if the result is this buffer.
</td>
</tr><tr>
<td>
`auxiliary_buffer0`
</td>
<td>
Pre-allocated workspace with the same shape and
dtype as the target tensor.
</td>
</tr><tr>
<td>
`auxiliary_buffer1`
</td>
<td>
Pre-allocated workspace with the same shape
and dtype as the target tensor.
</td>
</tr><tr>
<td>
`left_axes`
</td>
<td>
Which axes to multiply the left action of the channel
upon.
</td>
</tr><tr>
<td>
`right_axes`
</td>
<td>
Which axes to multiply the right action of the channel
upon.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`target_tensor`
</td>
<td>
The input tensor that needs to be left and right
multiplied and summed, representing the effect of the channel.
The tensor will have the shape (2, 2, 2, ..., 2). It usually
corresponds to a multi-qubit density matrix, with the first
n indices corresponding to the rows of the density matrix and
the last n indices corresponding to the columns of the density
matrix.
</td>
</tr><tr>
<td>
`out_buffer`
</td>
<td>
Pre-allocated workspace with the same shape and
dtype as the target tensor. If buffers are used, the result should
end up in this buffer. It is the responsibility of calling code
to notice if the result is this buffer.
</td>
</tr><tr>
<td>
`auxiliary_buffer0`
</td>
<td>
Pre-allocated workspace with the same shape and dtype
as the target tensor.
</td>
</tr><tr>
<td>
`auxiliary_buffer1`
</td>
<td>
Pre-allocated workspace with the same shape
and dtype as the target tensor.
</td>
</tr><tr>
<td>
`left_axes`
</td>
<td>
Which axes to multiply the left action of the channel upon.
</td>
</tr><tr>
<td>
`right_axes`
</td>
<td>
Which axes to multiply the right action of the channel upon.
</td>
</tr>
</table>



