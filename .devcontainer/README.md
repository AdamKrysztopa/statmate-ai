# Unified DevContainer Configuration

This directory contains a **unified devcontainer configuration** that works both locally and in remote environments (like GitHub Codespaces or other dev machines).

## Key Features

✅ **No Hardcoded Secrets** - All user-specific data (Git name, email, SSH keys, API keys) are configured via environment variables

✅ **Configurable SSH Support** - Automatically mounts and configures SSH keys based on environment variables

✅ **Works Everywhere** - Same configuration works on your local machine, GitHub Codespaces, or any dev environment

✅ **Secure** - Secrets never committed to the repository; they stay in your local environment

## Quick Start

### 1. Set Environment Variables

Add these to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="your.email@example.com"
export SSH_KEY_NAME="id_rsa"  # Optional, defaults to 'id_rsa'
export OPENAI_API_KEY="sk-..."
```

See [env.template](./env.template) for a complete template.

### 2. Reload Shell

```bash
source ~/.bashrc  # or ~/.zshrc
```

### 3. Open in DevContainer

Open the project in VS Code and select "Reopen in Container" when prompted.

## Files Overview

| File | Purpose |
|------|---------|
| `devcontainer.json` | Main devcontainer configuration with SSH mounts and environment variables |
| `Dockerfile` | Container image with Python, Node.js, SSH, and development tools |
| `setup_git.sh` | Configures Git with user name and email from environment variables |
| `setup_ssh.sh` | Sets up SSH keys for GitHub authentication using configurable key names |
| `ENV_SETUP.md` | Detailed setup instructions and troubleshooting guide |
| `env.template` | Simple template for environment variables |
| `README.md` | This file |

## Architecture

### How It Works

1. **Environment Variables on Host** → Set on your local machine
2. **DevContainer Mounts** → SSH keys mounted read-only into container
3. **Setup Scripts** → Run during container creation to configure Git and SSH
4. **Result** → Fully configured dev environment without hardcoded secrets

### SSH Key Flow

```
Host: ~/.ssh/${SSH_KEY_NAME}
      ↓ (bind mount)
Container: /home/vscode/.ssh/${SSH_KEY_NAME}
      ↓ (setup_ssh.sh)
Configured SSH config for GitHub
```

### Configuration Flow

```
Host Environment Variables
      ↓
DevContainer Configuration (devcontainer.json)
      ↓
Container Environment Variables
      ↓
Setup Scripts (setup_git.sh, setup_ssh.sh)
      ↓
Fully Configured Container
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GIT_USER_NAME` | Yes | - | Your Git user name for commits |
| `GIT_USER_EMAIL` | Yes | - | Your Git email for commits |
| `SSH_KEY_NAME` | No | `id_rsa` | Name of SSH private key in ~/.ssh/ |
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for AI features |

## SSH Setup

If you don't have SSH keys:

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com" -f ~/.ssh/id_rsa

# Add to GitHub
cat ~/.ssh/id_rsa.pub
# Copy and add to GitHub Settings → SSH Keys
```

## Differences from Previous Configurations

### Before (Two Separate Configs)

- **fixes_local_dev_container**: Used environment variables but no SSH
- **fixes**: Had SSH but with hardcoded names and secrets

### After (Unified Config)

- ✅ Uses environment variables for everything
- ✅ SSH support with configurable key names  
- ✅ No hardcoded secrets anywhere
- ✅ Works in any environment
- ✅ Single configuration for all use cases

## Troubleshooting

See [ENV_SETUP.md](./ENV_SETUP.md) for detailed troubleshooting instructions.

### Quick Checks

```bash
# Verify environment variables
echo $GIT_USER_NAME $GIT_USER_EMAIL $SSH_KEY_NAME

# Check SSH keys exist
ls -la ~/.ssh/

# Test GitHub SSH connection (inside container)
ssh -T git@github.com
```

## Migration from Previous Branches

If you were using `fixes` or `fixes_local_dev_container`:

1. **Set environment variables** as described above
2. **Update SSH_KEY_NAME** if using custom key name (e.g., `adam_private_gh` → set `SSH_KEY_NAME=adam_private_gh`)
3. **Remove hardcoded values** from any local modifications
4. **Rebuild container** to apply changes

## Best Practices

1. ✅ **Do**: Store environment variables in your shell profile
2. ✅ **Do**: Use different SSH keys for different machines/purposes
3. ✅ **Do**: Keep secrets out of version control
4. ❌ **Don't**: Hardcode secrets in devcontainer files
5. ❌ **Don't**: Commit `.env` files with secrets
6. ❌ **Don't**: Share your API keys or SSH private keys

## Support

For issues or questions about this configuration, see:
- [ENV_SETUP.md](./ENV_SETUP.md) - Detailed setup guide
- [env.template](./env.template) - Environment variable template
- GitHub Issues - Report problems with the configuration

