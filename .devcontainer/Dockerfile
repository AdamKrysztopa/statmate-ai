# Official Python base
ARG VARIANT="3.11"
FROM python:${VARIANT}

# Ensure non-interactive apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Install uv and necessary tools
# Added openssh-client (from workstation) for SSH agent/key-add
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    openssh-client \
    gnupg \
    dirmngr \
    lsb-release \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && pip install --no-cache-dir pyright ruff uvicorn \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME

# Accept optional build args
ARG OPENAI_API_KEY

# Set default shell
SHELL ["/bin/bash", "-c"]

# Ensure workspace directory exists and is owned by vscode user
RUN mkdir -p /workspace && chown -R $USERNAME:$USERNAME /workspace
WORKDIR /workspace

# Switch to vscode user
USER $USERNAME

# Create .ssh directory with proper permissions
RUN mkdir -p /home/$USERNAME/.ssh && chmod 700 /home/$USERNAME/.ssh

# Create virtual environment at workspace folder
RUN uv venv .venv

# Update PATH
ENV PATH="/workspace/.venv/bin:$PATH"

# Set PYTHONPATH for the workspace
ENV PYTHONPATH=/workspaces/statmate-ai

# Python dependencies will be installed via postCreateCommand to avoid
# needing a rebuild every time pyproject.toml changes.

# Leave final image as root to ensure devcontainer features can install correctly
USER root
