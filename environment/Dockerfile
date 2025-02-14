# hash:sha256:e13918673d414925f423e680f64e08cb50219c4d9d86e0257ee86e0876fbc7c5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.10.0 \
    pynwb==2.8.3 \
    scipy==1.15.1
