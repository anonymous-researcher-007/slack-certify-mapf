# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# slack-certify-mapf reference image.
#
# Stage 1 (`builder`) installs C++ build tools, compiles the vendored MAPF
# solver sources in `third_party/`, and prepares a Python virtualenv with the
# package installed in editable mode.
#
# Stage 2 (`runtime`) is a slim image that copies just the virtualenv and the
# compiled solver binaries.
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        pkg-config \
        libboost-all-dev \
        libeigen3-dev \
        libgoogle-glog-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/slack-certify-mapf

# Install Python dependencies first so the layer caches across source edits.
COPY pyproject.toml requirements.txt requirements-dev.txt README.md LICENSE ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install -r requirements-dev.txt

# Now copy the rest of the repository and install the package.
COPY . .
RUN /opt/venv/bin/pip install -e ".[dev,viz]"

# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SLACKCERTIFY_HOME=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-program-options1.74.0 \
        libgoogle-glog0v5 \
        libgomp1 \
        ca-certificates \
        bash \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/slack-certify-mapf /opt/slack-certify-mapf

WORKDIR /workspace
VOLUME ["/workspace"]

LABEL org.opencontainers.image.title="slack-certify-mapf" \
      org.opencontainers.image.description="Slack-Certified One-Shot MAPF reference implementation." \
      org.opencontainers.image.source="https://github.com/ANONYMIZED/slack-certify-mapf-test" \
      org.opencontainers.image.licenses="MIT"

CMD ["bash"]
