FROM ubuntu

# Install dependencies.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
     python3-pip python3-tk texlive-latex-base latexmk git emacs vim locales

# Configure UTF-8 encoding.
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 

# Make python3 default
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

# Get a current copy of Cirq from Github.
RUN git clone https://github.com/quantumlib/Cirq

# Install the needed Python libraries.
RUN pip3 install -r Cirq/requirements.txt -r Cirq/cirq/contrib/contrib-requirements.txt -r Cirq/dev_tools/conf/pip-list-dev-tools.txt

WORKDIR /Cirq

EXPOSE 8888
