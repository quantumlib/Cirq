FROM python:3-onbuild

RUN pip install --trusted-host pypi.python.org --no-cache-dir -r requirements.txt
