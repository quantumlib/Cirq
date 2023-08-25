# Access and Authentication

The Rigetti QCS API provides access to and descriptions of Rigetti quantum processors. The Rigetti QCS [online documentation](https://docs.rigetti.com) details:

* installing `cirq-rigetti`'s non-Python dependencies, namely the Quil [compiler and QVM](https://docs.rigetti.com/qcs/getting-started/installing-locally#install-the-compiler-and-qvm)
* invoking the Rigetti [QCS CLI](https://docs.rigetti.com/qcs/references/qcs-cli) to [configure credentials](https://docs.rigetti.com/qcs/guides/using-the-qcs-cli#configuring-credentials) within your local environment
* getting started with the Rigetti [JupyterLab IDE](https://docs.rigetti.com/qcs/getting-started/jupyterlab-ide), which comes with the aforementioned dependencies and credentials pre-installed

Note that you do not need Rigetti QCS credentials to execute on the Quil QVM, but you _will_ need them for execution on live Rigetti quantum processors.

With your environment setup, you will be able use the `cirq-rigetti` package as described in our [Getting Started Guide](./getting_started.ipynb).
