FROM alpine:latest

# Install dependencies.
RUN apk upgrade --update && \
    apk add --no-cache bash git emacs vim && \
    apk add libpng-dev freetype-dev openblas-dev libc-dev musl-dev && \
    apk add python3-dev python3-tkinter && \
    python3 -m pip install --upgrade pip && \
    pip3 install --no-cache --upgrade pip setuptools wheel

# Make python3 default
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

# Install Cirq library from Github.
RUN git clone https://github.com/quantumlib/Cirq
# Install the needed Python libraries.
RUN pip3 install -r Cirq/requirements.txt -r Cirq/cirq/contrib/contrib-requirements.txt -r Cirq/dev_tools/conf/pip-list-dev-tools.txt


# Configure UTF-8 encoding.
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 

WORKDIR /Cirq

EXPOSE 8888