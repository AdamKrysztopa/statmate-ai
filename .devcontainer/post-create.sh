#!/bin/bash
set -e # Exit on any error

echo "--- Starting Post-Create Setup ---"

# 1. Setup Git
# These variables are passed from devcontainer.json's "containerEnv"
if [ -z "$PROJECT_GIT_USER_NAME" ] || [ -z "$PROJECT_GIT_USER_EMAIL" ]; then
    echo "Error: PROJECT_GIT_USER_NAME or PROJECT_GIT_USER_EMAIL is not set."
    echo "Please set these environment variables on your HOST machine."
    exit 1
else
    git config --global user.name "$PROJECT_GIT_USER_NAME"
    git config --global user.email "$PROJECT_GIT_USER_EMAIL"
    git config --global core.editor "code --wait --new-window"
    echo "✓ Git config set to: $PROJECT_GIT_USER_NAME <$PROJECT_GIT_USER_EMAIL>"
fi

# 2. Setup SSH
# This configures SSH to *always* use the mounted 'project_key' for github.com
echo "Setting up SSH to use project-specific key..."

# Set permissions for mounted key
if [ -f ~/.ssh/project_key ]; then
    chmod 600 ~/.ssh/project_key
    echo "✓ Set permissions for private key"
else
    echo "Warning: Mounted private key (~/.ssh/project_key) not found."
    echo "Please ensure 'PROJECT_SSH_KEY_PATH' is set correctly on your host."
fi
if [ -f ~/.ssh/project_key.pub ]; then
    chmod 644 ~/.ssh/project_key.pub
    echo "✓ Set permissions for public key"
fi

# Create SSH config to *only* use this key for GitHub
cat > ~/.ssh/config << 'EOF'
# GitHub - Project-Specific Key
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/project_key
    IdentitiesOnly yes
    AddKeysToAgent yes
EOF
chmod 600 ~/.ssh/config
echo "✓ Created SSH config for GitHub"

# Preload GitHub host key to avoid interactive prompt
ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null || true

# Start ssh-agent and add key
eval "$(ssh-agent -s)" > /dev/null 2>&1
if [ -f ~/.ssh/project_key ]; then
    ssh-add ~/.ssh/project_key 2>/dev/null || echo "Note: SSH key may require a passphrase on first use"
    echo "✓ Added project SSH key to agent"
fi

# 3. Install Python Dependencies
echo "Installing Python dependencies with uv..."
# Copy pyproject.toml from the workspace root (one level up from .devcontainer)
# The 'context: ..' in devcontainer.json makes this possible.
cp ../pyproject.toml ../uv.lock . 2>/dev/null || true
uv sync
echo "✓ Python dependencies installed."

# 4. Environment-Specific Logic
if [ "$DEV_ENV_TYPE" == "local" ]; then
    echo "✓ Running in 'local' environment."
    # Add any local-only setup steps here
elif [ "$DEV_ENV_TYPE" == "workstation" ]; then
    echo "✓ Running in 'workstation' environment."
    # Add any workstation-only setup steps here
else
    echo "Warning: DEV_ENV_TYPE is not set to 'local' or 'workstation'."
fi

echo "--- Post-Create Setup Complete ---"
