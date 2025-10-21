# DevContainer Environment Variables Setup

This document describes all environment variables required for the unified devcontainer configuration.

## Required Environment Variables

All environment variables should be set in your **LOCAL** environment (host machine), not inside the container.

### Git Configuration (Required)

```bash
# Your Git user name for commits
export GIT_USER_NAME="Your Name"

# Your Git email for commits
export GIT_USER_EMAIL="your.email@example.com"
```

### SSH Configuration (Optional, Recommended)

```bash
# Name of your SSH private key file (without path)
# Default: id_rsa (if not set)
# The key files must exist at ~/.ssh/${SSH_KEY_NAME} and ~/.ssh/${SSH_KEY_NAME}.pub
export SSH_KEY_NAME="id_rsa"
```

**Examples:**
- Default SSH key: `SSH_KEY_NAME=id_rsa`
- Custom GitHub key: `SSH_KEY_NAME=github_personal`
- Named key: `SSH_KEY_NAME=adam_private_gh`

### API Keys (Required)

```bash
# OpenAI API key for AI features
# Get from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-..."
```

## Setup Instructions

### Option 1: Shell Profile (Recommended)

Add these exports to your shell profile:

**For bash** (`~/.bashrc` or `~/.bash_profile`):
```bash
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="your.email@example.com"
export SSH_KEY_NAME="id_rsa"
export OPENAI_API_KEY="sk-..."
```

**For zsh** (`~/.zshrc`):
```bash
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="your.email@example.com"
export SSH_KEY_NAME="id_rsa"
export OPENAI_API_KEY="sk-..."
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Option 2: Project .env File

Create a `.env` file in your project root (make sure it's in `.gitignore`):

```bash
GIT_USER_NAME="Your Name"
GIT_USER_EMAIL="your.email@example.com"
SSH_KEY_NAME="id_rsa"
OPENAI_API_KEY="sk-..."
```

Then source it before opening VS Code:
```bash
source .env
code .
```

### Option 3: VS Code Settings (Not Recommended for Secrets)

You can set these in your VS Code `settings.json`, but **DO NOT** store secrets there:

```json
{
  "terminal.integrated.env.linux": {
    "GIT_USER_NAME": "Your Name",
    "GIT_USER_EMAIL": "your.email@example.com",
    "SSH_KEY_NAME": "id_rsa"
  }
}
```

## SSH Key Setup

If you don't have an SSH key yet:

### 1. Generate SSH Key

```bash
# Using Ed25519 (recommended)
ssh-keygen -t ed25519 -C "your.email@example.com" -f ~/.ssh/id_rsa

# Or using RSA
ssh-keygen -t rsa -b 4096 -C "your.email@example.com" -f ~/.ssh/id_rsa
```

### 2. Add SSH Key to GitHub

```bash
# Copy your public key
cat ~/.ssh/id_rsa.pub
```

Then:
1. Go to GitHub Settings → SSH and GPG keys → New SSH key
2. Paste the public key
3. Save

### 3. Set Environment Variable

```bash
export SSH_KEY_NAME="id_rsa"
```

## Verification

To verify your environment variables are set:

```bash
echo $GIT_USER_NAME
echo $GIT_USER_EMAIL
echo $SSH_KEY_NAME
echo $OPENAI_API_KEY  # This will show your API key
```

## How It Works

The devcontainer configuration uses these environment variables to:

1. **Mount SSH keys** into the container at runtime (read-only)
2. **Configure Git** with your name and email during container creation
3. **Setup SSH** for GitHub authentication automatically
4. **Provide API keys** to the application

All secrets remain on your host machine and are never committed to the repository!

## Troubleshooting

### SSH Key Not Found

If you see "SSH private key not found" during container creation:

1. Check the key exists: `ls -la ~/.ssh/`
2. Verify SSH_KEY_NAME matches your key filename
3. Ensure both private and public keys exist (`.pub` extension for public)

### Git Config Not Set

If Git commands fail with "Please tell me who you are":

1. Check environment variables: `echo $GIT_USER_NAME $GIT_USER_EMAIL`
2. Reload your shell: `source ~/.bashrc` or `source ~/.zshrc`
3. Rebuild the devcontainer

### GitHub SSH Connection Failed

If you can't push/pull with SSH:

1. Test connection: `ssh -T git@github.com`
2. Verify key is added to GitHub
3. Check SSH key permissions: `ls -la ~/.ssh/`
4. Ensure permissions are: `600` for private key, `644` for public key

## Platform-Specific Notes

### Linux/macOS
- Use `~/.bashrc` or `~/.zshrc`
- SSH keys location: `~/.ssh/`
- Use `$HOME` in paths

### Windows (WSL2)
- Use `~/.bashrc` in WSL
- SSH keys location: `~/.ssh/` (in WSL, not Windows)
- Environment variables must be set in WSL, not Windows

### Windows (Git Bash)
- Use `~/.bash_profile`
- SSH keys location: `~/.ssh/`
- May need to convert line endings (CRLF → LF)

