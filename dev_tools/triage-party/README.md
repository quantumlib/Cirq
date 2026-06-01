# Cirq Triage Party

[Triage Party](https://github.com/google/triage-party) is an issue and pull request triaging tool for
GitHub. For more information about the triaging process used in the Cirq project, please refer to
the [triage process documentation for Cirq].

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

3.  Run Triage Party locally using Docker. The script `./run-local.sh` in this directory runs Triage
    Party in the Docker image built above. Note: this command produces _a lot_ of output as Triage
    Party pulls content from GitHub.
    ```shell
    ./run-local.sh
    ```

After some time and many lines of logs printed, if all went well, the output should end with a line that
looks more or less something like this:

```text
I0601 21:40:01.954553       1 updater.go:250] update cycle #1 took 7.005642265s
```

The exact numbers are not important; what matters is that it did not end with an error message and
did not return to the shell, but rather paused with `update cycle #1` as the final message. The
instance will be listening for web browser connections on the port 8080 on your local host.

## Cloud configuration & Setup

The deployment of Cirq's Triage Party instance on a Google Cloud server is managed using Kubernetes,
an open-source platform that automates the deployment, scaling, and management of containerized
applications. Specifically, we use the tools Kustomize and Skaffold.

### 1. Application Configuration (`config.yaml`)

To customize the dashboard (e.g., modifying collections, rules, and members), edit the standalone
YAML configuration file located at [kubernetes/02_deployment/config.yaml][config_yaml]. This file is
read by the program `kustomize` to generate the `triage-party-config` ConfigMap for Kubernetes. (A
ConfigMap is a Kubernetes resource used to store non-confidential configuration data as key-value
pairs.)

Please refer to the official [Triage Party documentation] for details on the available configuration
options.

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

### 3. GitHub Token Secret (One-time Setup)

Triage Party requires a GitHub personal access token to interact with the GitHub API. This is a
one-time setup per cluster. You must create a Kubernetes Secret containing a GitHub token.
(Kubernetes Secrets are designed to hold sensitive data—like passwords and tokens. They prevent
confidential information from being checked into version control or displayed in configuration
files.) Follow these steps:

1.  Set the environment variable `GITHUB_TOKEN` to the token to be used.

2.  Run the following command:

    ```shell
    kubectl create secret generic triage-party-github-token \
        -n triage-party --from-literal=token="${GITHUB_TOKEN}"
    ```

## Deployment

Google Cloud Build is Google Cloud Platform's (GCP) serverless CI/CD service. When code is pushed,
Cloud Build automatically imports the changes, executes the build files, compiles the necessary
containers, and safely applies them to the production environment without manual intervention.
Please refer to the [cirq-infra documentation] to find out more about the software tools needed and
how to install them.

### Google Cloud Build deployment

On every push to the `main` branch, the Cirq Triage Party instance is automatically redeployed as
defined in [cloudbuild-deploy.yaml][cloudbuild].

### Manual Deployment

If you need to deploy Triage Party manually (e.g., for applying an immediate fix), use `skaffold`.

```shell
gcloud container clusters get-credentials cirq-infra --zone us-central1-a
skaffold run --force -f=dev_tools/triage-party/skaffold.yaml
```

The `get-credentials` command securely authenticates your local terminal with the remote Kubernetes
cluster running on Google Kubernetes Engine (GKE). Once authenticated, your local deployment tools
(like `kubectl` and `skaffold`) gain the permission required to modify the cluster. The second
command above (running `skaffold`) will then correctly bundle your configuration, generate the
required ConfigMaps (including the favicon), and apply the manifests to the cluster.

[cirq-infra documentation]: ../cirq-infra/README.md
[cirq_icon]: kubernetes/02_deployment/cirq-icon-very-small.png
[cloudbuild]: cloudbuild-deploy.yaml
[config_yaml]: kubernetes/02_deployment/config.yaml
[custom_css]: kubernetes/02_deployment/custom.css
[kustomization]: kubernetes/kustomization.yaml
[personal access tokens]: https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line
[Triage Party documentation]: https://github.com/google/triage-party/blob/main/docs/config.md
[triage process documentation for Cirq]: ../../docs/dev/triage.md
