FROM ubuntu

# Install dependencies.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
     python python3-pip python3-tk texlive-latex-base latexmk git vim locales

# Configure UTF-8 encoding.
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 

# Install Cirq with the needed Python libraries.
RUN pip3 install numpy typing_extensions networkx sortedcontainers google-api-python-client cirq

EXPOSE 8888
