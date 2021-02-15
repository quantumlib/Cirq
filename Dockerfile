FROM python:3.8-slim AS compile-image

# Install dependencies.
# rm -rf /var/lib/apt/lists/* cleans up apt cache. See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
     python3-pip \
     python3-tk \
     texlive-latex-base \
     latexmk \
     git \
     locales \
     && rm -rf /var/lib/apt/lists/*


# Configure UTF-8 encoding.
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 

# Make python3 default
RUN rm -f /usr/bin/python \
     && ln -s /usr/bin/python3 /usr/bin/python

# Create a virtual enironment to copy over into
# the final docker image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy current folder instead of cloning.
COPY ./ .
COPY requirements.txt .
COPY ./cirq/contrib/contrib-requirements.txt .
COPY ./dev_tools/conf/pip-list-dev-tools.txt .

RUN pip3 install -r requirements.txt -r contrib-requirements.txt -r pip-list-dev-tools.txt

# Install cirq
RUN pip3 install cirq

FROM python:3.8-slim AS build-image
COPY --from=compile-image /opt/venv /opt/venv

# Make sure scripts in .local are usable:
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /Cirq
EXPOSE 8888