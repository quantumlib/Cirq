<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.asyncio_pending" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.asyncio_pending

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/asynchronous.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gives the given future a chance to complete, and determines if it didn't.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.asynchronous.asyncio_pending`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.asyncio_pending(
    future: Union[Awaitable, asyncio.Future, Coroutine],
    timeout: float = 0.001
) -> Awaitable[bool]
</code></pre>



<!-- Placeholder for "Used in" -->

This method is used in tests checking that a future actually depends on some
given event having happened. The test can assert, before the event, that the
future is still pending and then assert, after the event, that the future
has a result.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`future`
</td>
<td>
The future that may or may not be able to resolve when given
a bit of time.
</td>
</tr><tr>
<td>
`timeout`
</td>
<td>
The number of seconds to wait for the future. This should
generally be a small value (milliseconds) when expecting the future
to not resolve, and a large value (seconds) when expecting the
future to resolve.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if the future is still pending after the timeout elapses. False if
the future did complete (or fail) or was already completed (or already
failed).
</td>
</tr>

</table>



#### Examples:

>>> import asyncio
>>> import pytest
>>> @pytest.mark.asyncio
... async def test_completion_only_when_expected():
...     f = asyncio.Future()
...     assert await cirq.testing.asyncio_pending(f)
...     f.set_result(5)
...     assert await f == 5
