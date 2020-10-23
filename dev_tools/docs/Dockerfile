FROM nikolaik/python-nodejs:python3.7-nodejs14

RUN pip install -U git+https://github.com/tensorflow/docs && npm i -g markserv

RUN mkdir /cirq

WORKDIR /cirq

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . /cirq
RUN PYTHONPATH=/cirq python3 dev_tools/docs/build_api_docs.py --output_dir=docs/api_docs

WORKDIR /cirq/docs/api_docs
CMD ["markserv", "-b=false", "-p=8080", "-a=0.0.0.0"]