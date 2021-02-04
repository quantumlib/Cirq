# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# This is a Dockerfile to preview the api-docs generation
#
# Usage:
#     docker build -t cirq-api-docs -f dev_tools/docs/Dockerfile .
#     docker run -d --name=cirq-api-docs -p 8080:8080 cirq-api-docs
# To stop:
#     docker kill cirq-api-docs && docker rm cirq-api-docs
#
# You can then preview the site at http://localhost:8080
# Eventually this might become the basis of an automated preview service for each
# PR.
########################################################################################

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