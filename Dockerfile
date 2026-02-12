FROM ghcr.io/prefix-dev/pixi:latest as base

ARG USER_NAME=morelia
ARG USER_UID=1000
ARG USER_GID=1000

# free existing UID/GID if they are occupied
RUN (id -u ${USER_UID} >/dev/null 2>&1 && userdel -r $(id -nu ${USER_UID}) || true) && \
    (getent group ${USER_GID} >/dev/null 2>&1 && groupdel $(getent group ${USER_GID} | cut -d: -f1) || true) && \
    groupadd -g ${USER_GID} ${USER_NAME} && \
    useradd -m -u ${USER_UID} -g ${USER_GID} ${USER_NAME}

RUN mkdir -p /workspace/forest3d && \
    chown -R "${USER_UID}:${USER_GID}" /workspace

USER ${USER_NAME}
WORKDIR /workspace/forest3d
COPY pyproject.toml pixi.lock* ./
RUN pixi install -e dev