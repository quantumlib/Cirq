<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.work.work_pool.CompletionOrderedAsyncWorkPool" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="async_all_done"/>
<meta itemprop="property" content="include_work"/>
<meta itemprop="property" content="set_all_work_received_flag"/>
</div>

# cirq.work.work_pool.CompletionOrderedAsyncWorkPool

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/work_pool.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Ensures given work is executing, and exposes it in completion order.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.work.work_pool.CompletionOrderedAsyncWorkPool(
    loop: Optional[asyncio.AbstractEventLoop] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`num_active`
</td>
<td>
The amount of work in the pool that has not completed.
</td>
</tr><tr>
<td>
`num_uncollected`
</td>
<td>
The amount of work still in the pool that has completed.
</td>
</tr>
</table>



## Methods

<h3 id="async_all_done"><code>async_all_done</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/work_pool.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>async_all_done() -> Awaitable[None]
</code></pre>

An awaitable that completes once all work is completed.

Note: all work is completed only after the `set_all_work_received_flag`
method is called, and then all work still in the pool completes.

<h3 id="include_work"><code>include_work</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/work_pool.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>include_work(
    work: Union[Awaitable, asyncio.Future]
) -> None
</code></pre>

Adds asynchronous work into the pool and begins executing it.


<h3 id="set_all_work_received_flag"><code>set_all_work_received_flag</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/work_pool.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_all_work_received_flag() -> None
</code></pre>

Indicates to the work pool that no more work will be added.




