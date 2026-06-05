# Cirq Triage Party

[Triage Party](https://github.com/google/triage-party) is an issue and pull request triaging tool
for GitHub. For more information about the triaging process used in the Cirq project, please refer
to the [triage process documentation for Cirq](../../docs/dev/triage.md).

## Local testing

It is possible to run an instance of Triage Party on your local host. This is useful for testing
configuration changes.

1.  Generate a personal access token on GitHub with read-only permissions for `public_repo`. (Please
    refer to the GitHub documentation for [personal access tokens] for information on how to do
    that.) The token will be a string of characters that begins with the letters `gho_`.

2.  Store this token string in the environment variable `GITHUB_TOKEN`:

    ```shell
    export GITHUB_TOKEN="yourtokenhere"
    ```

3.  Run Triage Party locally in a container using either Podman or Docker. The script
    `./run-local.sh` makes this convenient; it pulls an image for Triage Party from a container
    registry and runs it with the appropriate Cirq configuration files. (Note: this produces
    _a lot_ of output.)

    ```shell
    ./run-local.sh
    ```

After some time and many lines of logs printed, if all went well, the output should end with a line
that looks more or less something like this:

```text
I0601 21:40:01.954553       1 updater.go:250] update cycle #1 took 7.005642265s
```

The exact numbers printed in the output are not important; what matters is that it did not end with
an error message and did not return to the shell, but rather paused with `update cycle #1` as the
final message. The Triage Party server will then be listening for web browser connections on port
8080 on your local host. It will produce more output as it detects that files have changed.

[personal access tokens]: https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line

## Cloud configuration & Setup

The deployment of Cirq's Triage Party instance on a Google Cloud server is managed using Kubernetes,
an open-source platform that automates the deployment, scaling, and management of containerized
applications. Specifically, we use the tools Kustomize and Skaffold.

### 1. Application Configuration

To customize the Triage Party dashboard (e.g., modifying collections, rules, and members), edit the
standalone YAML configuration file located at [kubernetes/02_deployment/config.yaml][config].
This file is read by the program `kustomize` to generate the `triage-party-config` ConfigMap for
Kubernetes.

Please refer to the official [Triage Party documentation] for details on the available configuration
options.

[Triage Party documentation]: https://github.com/google/triage-party/blob/main/docs/config.md

### 2. Static Resources & Theming (CSS & Favicon)

Extra resources like custom styling and the site's favicon are also managed automatically via
Kustomize:

-   **Custom CSS**: Overrides to the default Triage Party look and feel are defined in
    [kubernetes/02_deployment/custom.css][custom_css] as a ConfigMap.

-   **Favicon**: The favicon is bundled directly from the image file
    [kubernetes/02_deployment/cirq-icon-very-small.png][cirq_icon]. Kustomize reads this binary
    file and creates the `triage-party-favicon` ConfigMap automatically as defined in
    [kubernetes/kustomization.yaml][kustomization].

There is no need to manually run `kubectl create configmap` commands for these resources.

[cirq_icon]: kubernetes/02_deployment/cirq-icon-very-small.png
[config]: kubernetes/02_deployment/config.yaml
[custom_css]: kubernetes/02_deployment/custom.css
[kustomization]: kubernetes/kustomization.yaml

### 3. GitHub Token Secret (One-time Setup)

Triage Party requires a GitHub personal access token so that it can query the GitHub API to get data
about issues and pull requests. The token creation and assignment is a one-time setup. You must
create a Kubernetes Secret containing a GitHub token. Follow these steps:

1.  Obtain a new token from GitHub.

2.  Set the environment variable `GITHUB_TOKEN` to the token to be used.

3.  Run the following command in a terminal on your local host:

    ```shell
    # Authenticate to Google Cloud.
    gcloud auth login

    # Provision your underlying Application Default Credentials (ADC).
    gcloud auth application-default login

    # Authenticate your local terminal with the Kubernetes cluster running
    # on Google Kubernetes Engine (GKE) and save values in ~/.kube/config.
    gcloud container clusters get-credentials cirq-infra --zone us-central1-a

    # Finally, create the secret inside the cluster.
    kubectl create secret generic triage-party-github-token \
        -n triage-party --from-literal=token="${GITHUB_TOKEN}"
    ```

## Deployment

Google Cloud Build is Google Cloud Platform's (GCP) serverless CI/CD service. When code is pushed,
Cloud Build automatically imports the changes, executes the build files, compiles the necessary
containers, and safely applies them to the production environment without manual intervention.
Please refer to the [cirq-infra documentation](../cirq-infra/README.md) to find out more about the
software tools needed and how to install them.

### Google Cloud Build deployment

On every push to the `main` branch that involves files in `dev_tools/triage-party/`, the Cirq Triage
Party instance is automatically redeployed using the steps defined in the file
[cloudbuild-deploy.yaml](./cloudbuild-deploy.yaml). This is achieved thanks to a Cloud Build Trigger
defined in the `cirq-infra` project in GCP. Thus, when changes to the Triage Party configuration are
merged into the `main` branch on GitHub, the server instance will be updated automatically.

### Manual Deployment

It is possible to update the Triage Party deployment manually (e.g., for applying an immediate fix).
This can be done using the command-line tool `skaffold`. Execute these commands from the top level
of the Cirq repository:

```shell
# Authenticate to Google Cloud.
gcloud auth login

# Provision your underlying Application Default Credentials (ADC).
gcloud auth application-default login

# Authenticate your local terminal with the remote Kubernetes cluster running
# on Google Kubernetes Engine (GKE) and save values in ~/.kube/config.
gcloud container clusters get-credentials cirq-infra --zone us-central1-a

# Bundle the config, generate ConfigMaps, and deploy.
skaffold run --force -f=dev_tools/triage-party/skaffold.yaml
```
