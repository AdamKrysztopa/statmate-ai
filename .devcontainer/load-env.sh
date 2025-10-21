#!/bin/bash
# Load environment variables from .env.devcontainer for the devcontainer

ENV_FILE="../.env.devcontainer"

if [ -f "$ENV_FILE" ]; then
    echo "✓ Loading environment from .env.devcontainer"
    # Export all variables from .env.devcontainer (ignore comments and empty lines)
    set -a
    source <(grep -v '^#' "$ENV_FILE" | grep -v '^$' | sed 's/\r$//')
    set +a
else
    echo "⚠ Warning: .env.devcontainer not found"
    echo "  Please copy .devcontainer/env.devcontainer.example to .env.devcontainer"
    echo "  and fill in your values."
    exit 1
fi

