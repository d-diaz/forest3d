############################
# Common build args
############################
ARG CPU_BASE_IMAGE=ubuntu:22.04
ARG GPU_BASE_IMAGE=nvidia/cuda:12.9.1-runtime-ubuntu22.04

############################
# CPU target
############################
FROM ${CPU_BASE_IMAGE} AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-dev build-essential curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

COPY requirements-*.txt .
RUN uv pip install --system -r requirements-cpu.txt && rm requirements-*.txt

############################
# GPU target
############################
FROM ${GPU_BASE_IMAGE} AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-dev build-essential curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

COPY requirements-*.txt .
RUN uv pip install --system -r requirements-gpu.txt && rm requirements-*.txt