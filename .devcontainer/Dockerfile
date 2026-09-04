FROM mcr.microsoft.com/devcontainers/rust:2-1-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ripgrep \
    vim \
    python3.13-venv \
    && rm -rf /var/lib/apt/lists/*