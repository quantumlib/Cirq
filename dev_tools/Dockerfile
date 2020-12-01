FROM python:3.7-slim AS compile-image

# Install dependencies.
# rm -rf /var/lib/apt/lists/* cleans up apt cache. See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
     python3-pip \
     python3-tk \
     texlive-latex-base \
     latexmk \
     git \
     emacs \
     vim \
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

# Get a current copy of Cirq from Github.
RUN git clone https://github.com/quantumlib/Cirq

COPY requirements.txt .
COPY Cirq/cirq/contrib/contrib-requirements.txt .
COPY Cirq/dev_tools/conf/pip-list-dev-tools.txt .

RUN cat requirements.txt contrib-requirements.txt pip-list-dev-tools.txt >> dependencies.txt \
     && pip3 install --user -r dependencies

# Install the needed Python libraries.
RUN pip3 install --user .

FROM python:3.7-slim AS build-image
COPY --from=compile-image /root/.local /root/.local

WORKDIR /Cirq
EXPOSE 8888
