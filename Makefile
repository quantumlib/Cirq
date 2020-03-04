# import config.
# You can change the default config with `make cnf="config_special.env" build`
cnf ?= container_config.env
include $(cnf)
export $(shell sed 's/=.*//' $(cnf))

# HELP
# This will output the help for each task
.PHONY: help

help: ## Display this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

CONTAINER_BUILD=
CONTAINER_BUILD_OPT=
CONTAINER_BUILD_ARG=
CONTAINER_RUN=

_set_no_cache:
	$(eval CONTAINER_BUILD_OPT+=--no-cache)

_set_docker_cmd:
	$(eval CONTAINER_BUILD=docker build)
	$(eval CONTAINER_RUN=docker run)

_set_podman_cmd:
	$(eval CONTAINER_BUILD=podman build)
	$(eval CONTAINER_RUN=podman run)

_build:
	@for tag in $(BUILD_IMAGE_TAG); do \
	runcmd="$(CONTAINER_BUILD) --build-arg BUILD_CIRQ_TAG=$$tag $(CONTAINER_BUILD_OPT) -t $(APP_NAME):$$tag -f Dockerfile ."; \
	$$runcmd; \
	done

_info_python:
	@for tag in $(BUILD_IMAGE_TAG); do \
	echo "# INFO: container image $(APP_NAME):$$tag"; \
	runcmd="$(CONTAINER_RUN) -it $(APP_NAME):$$tag python --version"; \
	$$runcmd; \
	echo; \
	done

_info_cirq:
	@for tag in $(BUILD_IMAGE_TAG); do \
	echo "# INFO: container image $(APP_NAME):$$tag"; \
	$(CONTAINER_RUN) -it $(APP_NAME):$$tag python -c 'import cirq;print("cirq version %s" % cirq.__version__);' \
	echo; \
	done

info:
	@for tag in $(BUILD_IMAGE_TAG); do \
	echo $$tag; \
	done


# docker container
docker-build: _set_docker_cmd _build ## Build container images using docker

docker-build-nc: _set_no_cache _set_docker_cmd _build ## Build container images without cache using docker

docker-info-cirq: _set_docker_cmd _info_cirq ## Display container cirq version using docker

docker-info-python: _set_docker_cmd _info_python ## Display container python version using docker

# podman container
podman-build: _set_podman_cmd _build ## Build container images using podman

podman-build-nc: _set_no_cache _set_podman_cmd _build ## Build container images without cache using podman

podman-info-cirq: _set_podman_cmd _info_cirq ## Display container cirq version using podman

podman-info-python: _set_podman_cmd _info_python ## Display container python version using podman
