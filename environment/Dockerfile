# hash:sha256:4cb53cdcf30a81757df88cdc3dea6df245be591f19ff86c33aae44eb47c112f9
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    ipympl==0.9.6 \
    matplotlib==3.10.0 \
    pynwb==2.8.3 \
    scikit-learn==1.6.1 \
    scipy==1.15.1 \
    xarray==2025.1.2
