# Setting up access and authentication to Azure Quantum

[Azure Quantum](https://docs.microsoft.com/azure/quantum/overview-azure-quantum) is Microsoft's cloud service for running quantum computing programs or solving optimization problems and is currently in [Public Preview](https://cloudblogs.microsoft.com/quantum/2021/02/01/azure-quantum-preview/). You can try Azure Quantum for free today and submit quantum programs to Azure Quantum's partners and technologies. To send quantum programs with Cirq to IonQ or Honeywell via an Azure Quantum subscription, follow the simple steps below to set up access.

Note: There are two packages that can be used to access IonQ devices with `cirq`: `azure-quantum` or `cirq-ionq`. If you would like to access IonQ via a new or existing Azure subscription, use `azure-quantum` by following the steps in this tutorial. If you already have a preexisting IonQ API key through other means, follow the steps outlined in [Cirq with the IonQ API](/cirq/hardware/ionq/access.md) to use `cirq-ionq` instead.

## 1. Create an Azure Subscription

To work in Azure Quantum, you need an Azure subscription. If you don't have an Azure subscription, create a [free account](https://azure.microsoft.com/free/).

## 2. Create an Azure Quantum Workspace

Create an Azure Quantum Workspace and enable the hardware providers you would like to use (IonQ and/or Honeywell). For more information, see [Create an Azure Quantum workspace](https://docs.microsoft.com/azure/quantum/quickstart-microsoft-qc?pivots=platform-ionq#create-an-azure-quantum-workspace).

## 3. Install the `azure-quantum` Python client

To connect to your Workspace from Python and submit jobs with Cirq, install the `azure-quantum` client package with the optional `cirq` dependencies as follows:

```bash
pip install azure-quantum[cirq]
```

## 4. Next Steps

You're now all set up to get started submitting quantum circuits to Azure Quantum hardware providers with Cirq. To try it out, check out the tutorials below:

[Getting started with IonQ and Cirq on Azure Quantum](/cirq/hardware/azure-quantum/getting_started_ionq.ipynb)

[Getting started with Honeywell and Cirq on Azure Quantum](/cirq/hardware/azure-quantum/getting_started_honeywell.ipynb)
