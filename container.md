## container.md

### Build container images config
```
#### Container image name used to run/build
APP_NAME=quantumlib/cirq

#### Used to build container images for specific release tag / branches
BUILD_IMAGE_TAG+=master
BUILD_IMAGE_TAG+=v0.7.0
BUILD_IMAGE_TAG+=v0.6.1
BUILD_IMAGE_TAG+=v0.6.0

#### Container engine (docker|podman) used by container_run.sh
APP_ENGINE=docker

#### Container image tag used by container_run.sh
APP_TAG=master
```

### Build container images using docker or podman
```
help                           Display this help
docker-build                   Build container images using docker
docker-build-nc                Build container images without cache using docker
docker-info-cirq               Display container cirq version using docker
docker-info-python             Display container python version using docker
podman-build                   Build container images using podman
podman-build-nc                Build container images without cache using podman
podman-info-cirq               Display container cirq version using podman
podman-info-python             Display container python version using podman
```

### Display cirq version for each container images
```
$ make docker-info-cirq
# INFO: container image quantumlib/cirq:master
cirq version 0.8.0.dev
# INFO: container image quantumlib/cirq:v0.7.0
^Bcirq version 0.8.0.dev
# INFO: container image quantumlib/cirq:v0.6.1
cirq version 0.6.1
# INFO: container image quantumlib/cirq:v0.6.0
cirq version 0.6.0

```

### Run or attach container images (docker | podman) 
```
#### if container is not running
$ ./container_run.sh
# INFO: Starting mrioux_quantumlib_cirq_master container
root@989aae24c1cf:/data#

#### if container is already running
$ ./container_run.sh
# INFO: Attaching to mrioux_quantumlib_cirq_master container already running
root@989aae24c1cf:/data#

```
