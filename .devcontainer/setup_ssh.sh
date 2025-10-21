#!/bin/bash
set -e

echo "Setting up SSH for GitHub..."

# Get SSH key name from environment variable, default to id_rsa
SSH_KEY_NAME="${SSH_KEY_NAME:-id_rsa}"

# Create .ssh directory if it doesn't exist
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Set proper permissions for SSH keys
if [ -f ~/.ssh/$SSH_KEY_NAME ]; then
    chmod 600 ~/.ssh/$SSH_KEY_NAME
    echo "✓ Set permissions for private key: $SSH_KEY_NAME"
else
    echo "⚠ Warning: SSH private key not found at ~/.ssh/$SSH_KEY_NAME"
    echo "  SSH authentication may not work. Please ensure the SSH_KEY_NAME environment variable is set correctly."
    echo "  Continuing with setup..."
fi

if [ -f ~/.ssh/$SSH_KEY_NAME.pub ]; then
    chmod 644 ~/.ssh/$SSH_KEY_NAME.pub
    echo "✓ Set permissions for public key: $SSH_KEY_NAME.pub"
fi

# Create SSH config for GitHub
cat > ~/.ssh/config << EOF
# GitHub Configuration
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/$SSH_KEY_NAME
    IdentitiesOnly yes
    AddKeysToAgent yes
EOF

chmod 600 ~/.ssh/config
echo "✓ Created SSH config for GitHub (using key: $SSH_KEY_NAME)"

# Preload GitHub host key to avoid interactive prompt
ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null || true
echo "✓ Added GitHub host key to known_hosts"

# Start ssh-agent and add key
eval "$(ssh-agent -s)" > /dev/null 2>&1
if [ -f ~/.ssh/$SSH_KEY_NAME ]; then
    ssh-add ~/.ssh/$SSH_KEY_NAME 2>/dev/null || echo "  Note: SSH key may require a passphrase on first use"
    echo "✓ Added SSH key to agent"
fi

# Test GitHub connection
echo "Testing GitHub connection..."
if ssh -T -o StrictHostKeyChecking=accept-new -o BatchMode=yes git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✓ GitHub SSH connection successful!"
else
    echo "  Note: GitHub authentication test skipped or not yet configured"
    echo "  This is normal if you haven't added the SSH key to your GitHub account yet."
fi

echo "SSH setup complete!"

