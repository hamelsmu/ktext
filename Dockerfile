FROM python:3.6-slim-stretch

RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install --reinstall build-essential -y
RUN apt install -y gcc g++

ENV CXXFLAGS="-std=c++11"
ENV CFLAGS="-std=c99"

COPY . ktext-lib/
WORKDIR ktext-lib

CMD /bin/bash integration_test.sh