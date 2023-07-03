FROM ubuntu:22.04
LABEL maintainer = dave.dong.gva@gmail.com
LABEL version = 1.0
LABEL environment = dev
ENV DEBIAN_FRONTEND = noninteractive AIRLFLOW_HOME = ~/airflow
EXPOSE 8080
EXPOSE 8793
RUN apt update
RUN ln -fs /usr/share/zoneinfo/America/New_York/etc/localtime && apt install -yq tzdata && dpkg-reconfigure --frontend noninteractive tzdata
RUN apt -yq install python3.10  && apt -yq install python3-pip && apt -yq install nano
RUN apt -yq install libpq-dev
RUN apt -yq install wget
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["tail", "-f", "/dev/null"]




