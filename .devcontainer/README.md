# Unified DevContainer Configuration

This directory contains a **unified devcontainer configuration** that works both locally and in remote environments (like GitHub Codespaces or other dev machines).

## Key Features

✅ **No Hardcoded Secrets** - All user-specific data (Git name, email, SSH keys, API keys) are configured via environment variables

✅ **Configurable SSH Support** - Automatically mounts and configures SSH keys based on environment variables

✅ **Works Everywhere** - Same configuration works on your local machine, GitHub Codespaces, or any dev environment

✅ **Secure** - Secrets never committed to the repository; they stay in your local environment

## Quick Start

### 1. Create `.env.devcontainer` File

Copy the example file to the project root and fill in your values:

```bash
# From the project root (/workspaces/statmate-ai/)
cp .devcontainer/env.devcontainer.example .env.devcontainer
```

Then edit `.env.devcontainer` with your information:

```bash
nano .env.devcontainer
# or
code .env.devcontainer
```

Required values:
- `GIT_USER_NAME` - Your git user name (e.g., "Adam Krysztopa")
- `GIT_USER_EMAIL` - Your git email (e.g., "user@example.com")
- `SSH_KEY_NAME` - Your SSH private key filename (e.g., "id_rsa" or "adam_private_gh")
- `OPENAI_API_KEY` - Your OpenAI API key

See [env.devcontainer.example](./env.devcontainer.example) for a complete template.

### 2. Open in DevContainer

Open the project in VS Code and select "Reopen in Container" when prompted.

The setup scripts will automatically load your configuration from `.env.devcontainer`.

## Files Overview

| File                | Purpose                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `devcontainer.json` | Main devcontainer configuration with SSH mounts and environment variables |
| `Dockerfile`        | Container image with Python, Node.js, SSH, and development tools          |
| `setup_git.sh`      | Configures Git with user name and email from environment variables        |
| `setup_ssh.sh`      | Sets up SSH keys for GitHub authentication using configurable key names   |
| `ENV_SETUP.md`      | Detailed setup instructions and troubleshooting guide                     |
| `env.template`      | Simple template for environment variables                                 |
| `README.md`         | This file                                                                 |

## Architecture

### How It Works

1. **`.env.devcontainer` File** → Created at project root (git-ignored)
2. **DevContainer Mounts** → SSH directory mounted into container
3. **Setup Scripts** → Load `.env.devcontainer` and configure Git and SSH
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
.env.devcontainer File (at project root)
      ↓
DevContainer Starts (devcontainer.json)
      ↓
Setup Scripts Load .env.devcontainer
      ↓
setup_git.sh: Configure Git user/email
setup_ssh.sh: Configure SSH keys
      ↓
Fully Configured Container
```

## Environment Variables

These should be set in `.env.devcontainer` at the project root:

| Variable         | Required | Default  | Description                        |
| ---------------- | -------- | -------- | ---------------------------------- |
| `GIT_USER_NAME`  | Yes      | -        | Your Git user name for commits     |
| `GIT_USER_EMAIL` | Yes      | -        | Your Git email for commits         |
| `SSH_KEY_NAME`   | No       | `id_rsa` | Name of SSH private key in ~/.ssh/ |
| `OPENAI_API_KEY` | Yes      | -        | OpenAI API key for AI features     |

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

### Before (Hardcoded Config - tmp_.devcontainer)

- ❌ Hardcoded git user name and email in devcontainer.json
- ❌ Hardcoded git user name and email in setup_git.sh
- ❌ Hardcoded SSH key name (adam_private_gh) in setup_ssh.sh
- ❌ Secrets committed to repository

### After (Anonymous Config - Current)

- ✅ All configuration in `.env.devcontainer` (git-ignored)
- ✅ Setup scripts load configuration from `.env.devcontainer`
- ✅ SSH support with configurable key names  
- ✅ No hardcoded secrets anywhere
- ✅ Works in any environment
- ✅ True "anonymous vibe" - no personal info in repo

## Troubleshooting

See [ENV_SETUP.md](./ENV_SETUP.md) for detailed troubleshooting instructions.

### Quick Checks

```bash
# Check .env.devcontainer exists at project root
ls -la /workspaces/statmate-ai/.env.devcontainer

# Verify configuration values (inside container after startup)
cat /workspaces/statmate-ai/.env.devcontainer

# Check SSH keys exist
ls -la ~/.ssh/

# Test GitHub SSH connection (inside container)
ssh -T git@github.com
```

## Migration from Previous Branches

If you were using the hardcoded `tmp_.devcontainer` config:

1. **Create `.env.devcontainer`** at project root from the example
2. **Fill in your values** (GIT_USER_NAME, GIT_USER_EMAIL, SSH_KEY_NAME, OPENAI_API_KEY)
3. **Rebuild container** to apply changes
4. **Test git push** to verify SSH authentication works

The anonymous approach is now the default - no personal info in the repo!

## Best Practices

1. ✅ **Do**: Store configuration in `.env.devcontainer` at project root
2. ✅ **Do**: Use different SSH keys for different machines/purposes
3. ✅ **Do**: Keep secrets out of version control (`.env.devcontainer` is git-ignored)
4. ❌ **Don't**: Hardcode secrets in devcontainer files
5. ❌ **Don't**: Commit `.env.devcontainer` to git
6. ❌ **Don't**: Share your API keys or SSH private keys

## Support

For issues or questions about this configuration, see:
- [ENV_SETUP.md](./ENV_SETUP.md) - Detailed setup guide
- [env.template](./env.template) - Environment variable template
- GitHub Issues - Report problems with the configuration

