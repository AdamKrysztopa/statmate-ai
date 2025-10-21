#!/bin/bash
set -e

# Load environment from .env.devcontainer if it exists
ENV_FILE="/workspaces/statmate-ai/.env.devcontainer"
if [ -f "$ENV_FILE" ]; then
    echo "✓ Loading environment from .env.devcontainer"
    set -a
    source <(grep -v '^#' "$ENV_FILE" | grep -v '^$' | sed 's/\r$//')
    set +a
else
    echo "⚠ Warning: .env.devcontainer not found at $ENV_FILE"
    echo "  Please copy .devcontainer/env.devcontainer.example to .env.devcontainer"
    echo "  at the project root and fill in your values."
fi

if [ -z "$GIT_USER_NAME" ] || [ -z "$GIT_USER_EMAIL" ]; then
    echo "❌ GIT_USER_NAME or GIT_USER_EMAIL is not set."
    echo "   Please set them in .env.devcontainer file at project root."
    exit 1
fi

git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"
echo "✓ Git global config updated: $GIT_USER_NAME <$GIT_USER_EMAIL>"
