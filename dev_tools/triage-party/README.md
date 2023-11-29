# Cirq Triage Party

See the [triaging process](../../docs/dev/triage.md) 

# Configuration 

For the dashboard you will have to edit the embedded YAML config in [configmap.yaml](kubernetes/02_deployment/configmap.yaml).
Please refer to the [triage party docs](https://github.com/google/triage-party/blob/main/docs/config.md) for configuration details.

## Secret 

Triage party needs a Github token - this is a one time setup (per cluster creation): 

```
kubectl create secret generic triage-party-github-token -n triage-party --from-file=token=$HOME/.github-token
```   

Where `$HOME/.github-token` is a file containing the token.

# Cloud Build based deployment 

On every push to main Triage Party is redeployed as defined by [cloudbuild-deploy.yaml](cloudbuild-deploy.yaml). 

# Deploying Triage Party manually

See [the cirq-infra documentation](../cirq-infra/README.md) for the required tools.

```
gcloud container clusters get-credentials cirq-infra --zone us-central1-a
skaffold run --force -f=dev_tools/triage-party/skaffold.yaml
```

