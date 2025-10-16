#!/bin/bash
set -e

echo "Setting up SSH for GitHub..."

# Create .ssh directory if it doesn't exist
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Set proper permissions for SSH keys
if [ -f ~/.ssh/adam_private_gh ]; then
    chmod 600 ~/.ssh/adam_private_gh
    echo "✓ Set permissions for private key"
fi

if [ -f ~/.ssh/adam_private_gh.pub ]; then
    chmod 644 ~/.ssh/adam_private_gh.pub
    echo "✓ Set permissions for public key"
fi

# Create SSH config for GitHub
cat > ~/.ssh/config << 'EOF'
# GitHub - Personal Account
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/adam_private_gh
    IdentitiesOnly yes
    AddKeysToAgent yes
EOF

chmod 600 ~/.ssh/config
echo "✓ Created SSH config for GitHub"

# Start ssh-agent and add key
eval "$(ssh-agent -s)" > /dev/null 2>&1
if [ -f ~/.ssh/adam_private_gh ]; then
    ssh-add ~/.ssh/adam_private_gh 2>/dev/null || echo "Note: SSH key may require a passphrase on first use"
    echo "✓ Added SSH key to agent"
fi

# Test GitHub connection
echo "Testing GitHub connection..."
ssh -T git@github.com 2>&1 | grep -q "successfully authenticated" && echo "✓ GitHub SSH connection successful!" || echo "Note: You may need to authenticate on first push"

echo "SSH setup complete!"

