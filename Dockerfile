FROM python:3-slim

# ARG default version is master if no BUILD_CIRQ_TAG
ARG BUILD_CIRQ_TAG=master

# ARG default value is no editor 
ARG BUILD_CIRQ_EDITOR=""
#ARG BUILD_CIRQ_EDITOR="emacs vim nano"

LABEL maintainer="Cirq team"

LABEL description="Docker image python3 and cirq:$BUILD_CIRQ_TAG"

RUN apt update && \
    apt install -y make git $BUILD_CIRQ_EDITOR && \
    pip install git+https://github.com/quantumlib/Cirq.git@$BUILD_CIRQ_TAG

RUN git clone -n --depth=1 https://github.com/quantumlib/Cirq -b $BUILD_CIRQ_TAG && \
    (cd Cirq;git checkout HEAD -- examples;mv examples /) && \
    rm -rf Cirq

WORKDIR /examples

EXPOSE 8888
