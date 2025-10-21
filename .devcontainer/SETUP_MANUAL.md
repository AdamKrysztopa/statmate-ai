# DevContainer Setup Manual - Complete Guide

## 🚀 Quick Start (TL;DR)

```bash
# 1. Copy the environment template
cp .devcontainer/env.devcontainer.example .env.devcontainer

# 2. Edit with your values
nano .env.devcontainer  # or use your preferred editor

# 3. Source it before opening VS Code
source .env.devcontainer
code .

# 4. In VS Code: Reopen in Container
```

---

## 📋 Table of Contents

1. [Understanding the Setup](#understanding-the-setup)
2. [Configuration Approaches](#configuration-approaches)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [SSH Configuration](#ssh-configuration)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

---

## Understanding the Setup

This devcontainer has **two configuration approaches**:

### Approach 1: origin/fixes Style (Current Default)
- **Use Case**: Secondary repo or specific SSH key needed
- **Method**: Mounts specific SSH keys into container
- **Requires**: `.env.devcontainer` file with `SSH_KEY_NAME`
- **Best For**: When working from different machines or repos

### Approach 2: origin/fixes_local_dev_container Style  
- **Use Case**: Primary machine, this is your main repo
- **Method**: Mounts entire `.ssh` directory
- **Requires**: `.env.devcontainer` without `SSH_KEY_NAME`
- **Best For**: Your main development machine

---

## Configuration Approaches

### 🔹 Approach 1: Specific SSH Key (Recommended for Most Users)

**When to use:**
- Working from a secondary repository
- Need specific SSH key for GitHub
- Want explicit control over which keys are used

**Setup:**

1. **Create `.env.devcontainer` in project root:**

```bash
# Git Configuration
GIT_USER_NAME=Adam Krysztopa
GIT_USER_EMAIL=krysztopa@gmail.com

# SSH Key Name (your private key filename in ~/.ssh/)
SSH_KEY_NAME=adam_private_gh

# OpenAI API Key
OPENAI_API_KEY=sk-your-actual-key-here
```

2. **Verify your SSH key exists:**

```bash
ls -la ~/.ssh/adam_private_gh
ls -la ~/.ssh/adam_private_gh.pub
```

3. **Make sure devcontainer.json has** (already configured):

```json
"mounts": [
    "source=${localEnv:HOME}/.ssh/${localEnv:SSH_KEY_NAME},target=/home/vscode/.ssh/${localEnv:SSH_KEY_NAME},type=bind,consistency=cached",
    "source=${localEnv:HOME}/.ssh/${localEnv:SSH_KEY_NAME}.pub,target=/home/vscode/.ssh/${localEnv:SSH_KEY_NAME}.pub,type=bind,consistency=cached"
]
```

---

### 🔹 Approach 2: Mount Entire .ssh Directory

**When to use:**
- This is your primary development machine
- This repo is your main working repo
- You have multiple keys and want them all available

**Setup:**

1. **Create `.env.devcontainer`:**

```bash
# Git Configuration  
GIT_USER_NAME=Your Name
GIT_USER_EMAIL=your.email@example.com

# OpenAI API Key
OPENAI_API_KEY=sk-your-actual-key-here

# NO SSH_KEY_NAME needed - will mount entire .ssh directory
```

2. **Update devcontainer.json mounts to:**

```json
"mounts": [
    "source=${localEnv:HOME}/.ssh,target=/home/vscode/.ssh,type=bind,consistency=cached"
]
```

3. **Update setup_ssh.sh** to use default key or auto-detect

---

## Step-by-Step Setup

### Step 1: Copy the Environment Template

```bash
cd /path/to/statmate-ai
cp .devcontainer/env.devcontainer.example .env.devcontainer
```

### Step 2: Edit `.env.devcontainer`

Open `.env.devcontainer` and fill in your actual values:

```bash
# Use your preferred editor
nano .env.devcontainer
# OR
code .env.devcontainer
# OR
vim .env.devcontainer
```

**Replace these values:**
- `GIT_USER_NAME` - Your actual name (for Git commits)
- `GIT_USER_EMAIL` - Your actual email (for Git commits)
- `SSH_KEY_NAME` - Your SSH private key filename (e.g., `id_rsa`, `github_personal`, etc.)
- `OPENAI_API_KEY` - Your actual OpenAI API key from https://platform.openai.com/api-keys

**Example:**
```bash
GIT_USER_NAME=John Doe
GIT_USER_EMAIL=john.doe@example.com
SSH_KEY_NAME=id_ed25519
OPENAI_API_KEY=sk-proj-abc123def456...
```

### Step 3: Source the Environment File

Before opening VS Code, source the environment file:

```bash
# In your terminal (outside VS Code)
cd /path/to/statmate-ai
source .env.devcontainer
```

**Why?** VS Code devcontainers read environment variables from your shell. By sourcing the file, these variables become available.

### Step 4: Open VS Code

```bash
code .
```

### Step 5: Reopen in Container

When VS Code opens:
1. You'll see a prompt: "Reopen in Container" - Click it
2. OR press `Ctrl+Shift+P` (Cmd+Shift+P on Mac)
3. Type: "Dev Containers: Reopen in Container"
4. Press Enter

The container will build (first time takes 5-10 minutes).

### Step 6: Verify Setup

Once inside the container, verify everything works:

```bash
# Check Git config
git config --global user.name
git config --global user.email

# Check SSH
ls -la ~/.ssh/
cat ~/.ssh/config

# Test GitHub connection
ssh -T git@github.com
# Should see: "Hi <username>! You've successfully authenticated..."

# Test a push
git push origin your-branch
```

---

## SSH Configuration

### Option A: You Already Have an SSH Key

1. **Find your key:**
```bash
ls -la ~/.ssh/
```

Look for files like: `id_rsa`, `id_ed25519`, `github_personal`, etc.

2. **Use that key name in `.env.devcontainer`:**
```bash
SSH_KEY_NAME=id_rsa  # or whatever your key is called
```

3. **Verify key is on GitHub:**
   - Go to https://github.com/settings/keys
   - Check if your public key is listed
   - If not, add it:
     ```bash
     cat ~/.ssh/id_rsa.pub
     # Copy the output and add to GitHub
     ```

### Option B: Generate a New SSH Key

1. **Generate the key:**
```bash
ssh-keygen -t ed25519 -C "your.email@example.com" -f ~/.ssh/github_key
```

Press Enter for no passphrase, or enter a passphrase (more secure).

2. **Add public key to GitHub:**
```bash
cat ~/.ssh/github_key.pub
```

Copy the output and add it to GitHub:
- Go to: https://github.com/settings/ssh/new
- Paste the public key
- Give it a title like "DevContainer Key"
- Click "Add SSH key"

3. **Update `.env.devcontainer`:**
```bash
SSH_KEY_NAME=github_key
```

4. **Test the connection:**
```bash
ssh -T git@github.com
```

---

## Troubleshooting

### ❌ Problem: "Permission denied (publickey)" when pushing

**Symptoms:**
```
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
```

**Solutions:**

1. **Verify SSH key is loaded:**
```bash
# Inside container
ls -la ~/.ssh/
cat ~/.ssh/config
```

2. **Check if key is on GitHub:**
   - Go to https://github.com/settings/keys
   - Verify your public key is there

3. **Test SSH connection:**
```bash
ssh -T git@github.com
```

4. **Check SSH_KEY_NAME matches:**
```bash
# Outside container
echo $SSH_KEY_NAME
ls -la ~/.ssh/$SSH_KEY_NAME
```

5. **Rebuild container:**
```
Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

---

### ❌ Problem: ".env.devcontainer not found"

**Symptoms:**
Container builds but Git config fails.

**Solution:**
```bash
# The file must be in PROJECT ROOT, not in .devcontainer/
cd /path/to/statmate-ai
cp .devcontainer/env.devcontainer.example .env.devcontainer
nano .env.devcontainer  # Fill in your values
```

---

### ❌ Problem: "GIT_USER_NAME is not set"

**Symptoms:**
```
GIT_USER_NAME or GIT_USER_EMAIL is not set. Please set them in your local environment.
```

**Solution:**
```bash
# 1. Make sure .env.devcontainer exists and has values
cat .env.devcontainer

# 2. Source it before opening VS Code
source .env.devcontainer

# 3. Verify variables are set
echo $GIT_USER_NAME
echo $GIT_USER_EMAIL

# 4. NOW open VS Code
code .
```

---

### ❌ Problem: Container builds but SSH still doesn't work

**Debug steps:**

1. **Inside container, check what's mounted:**
```bash
ls -la ~/.ssh/
```

2. **Check SSH config:**
```bash
cat ~/.ssh/config
```

Should show:
```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/your_key_name
    IdentitiesOnly yes
    AddKeysToAgent yes
```

3. **Check key permissions:**
```bash
ls -la ~/.ssh/
# Private key should be 600
# Public key should be 644
```

4. **Manual test:**
```bash
ssh -vT git@github.com
```

Look for lines showing which key it's trying to use.

---

### ❌ Problem: "Inappropriate ioctl for device" or key issues

**Solution:**
This usually means SSH agent issues. Inside container:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/your_key_name
ssh -T git@github.com
```

---

## FAQ

### Q: Do I need to source .env.devcontainer every time?

**A:** Yes, before opening VS Code. Better approach: Add it to your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
if [ -f ~/.env.statmate ]; then
    source ~/.env.statmate
fi
```

Then copy .env.devcontainer to ~/.env.statmate (outside repo).

### Q: Can I use my default id_rsa key?

**A:** Yes! Just set:
```bash
SSH_KEY_NAME=id_rsa
```

### Q: What if I have multiple SSH keys?

**A:** Use Approach 2 (mount entire .ssh directory) or specify which key to use with `SSH_KEY_NAME`.

### Q: Is .env.devcontainer committed to Git?

**A:** No! It's in .gitignore. Never commit it - it contains secrets!

### Q: Can I use this on Windows?

**A:** Yes, but:
- Use WSL2 (Windows Subsystem for Linux)
- SSH keys must be in WSL ~/.ssh/ (not Windows)
- Run all commands in WSL terminal

### Q: Can I use this in GitHub Codespaces?

**A:** Yes, but you'll need to add your SSH key to Codespaces secrets or use GitHub's built-in authentication.

### Q: Why not just use environment variables in my shell?

**A:** You can! That was the original approach. This `.env.devcontainer` file approach is simpler for users who don't want to modify their shell profiles.

---

## Complete Example Workflow

### Scenario: New Machine Setup

```bash
# 1. Clone repo
git clone git@github.com:yourusername/statmate-ai.git
cd statmate-ai

# 2. Check if you have SSH keys
ls -la ~/.ssh/

# 3. If no keys, generate one
ssh-keygen -t ed25519 -C "your@email.com" -f ~/.ssh/id_ed25519

# 4. Add public key to GitHub
cat ~/.ssh/id_ed25519.pub
# Copy and add to https://github.com/settings/ssh/new

# 5. Create .env.devcontainer
cp .devcontainer/env.devcontainer.example .env.devcontainer

# 6. Edit it
nano .env.devcontainer
# Set:
# GIT_USER_NAME=Your Name
# GIT_USER_EMAIL=your@email.com
# SSH_KEY_NAME=id_ed25519
# OPENAI_API_KEY=sk-...

# 7. Source it
source .env.devcontainer

# 8. Open VS Code
code .

# 9. Reopen in Container (Ctrl+Shift+P)

# 10. Verify inside container
git config --global user.name
ssh -T git@github.com
git push origin your-branch
```

---

## Summary of Files

| File | Location | Purpose |
|------|----------|---------|
| `env.devcontainer.example` | `.devcontainer/` | Template to copy |
| `.env.devcontainer` | Project root | Your actual config (gitignored) |
| `devcontainer.json` | `.devcontainer/` | Container configuration |
| `setup_git.sh` | `.devcontainer/` | Sets up Git with your name/email |
| `setup_ssh.sh` | `.devcontainer/` | Sets up SSH for GitHub |
| `Dockerfile` | `.devcontainer/` | Container image definition |

---

## Getting Help

1. **Check this manual** - Most issues are covered here
2. **Check ENV_SETUP.md** - More troubleshooting tips
3. **Test SSH outside container** - If `ssh -T git@github.com` fails outside, it won't work inside
4. **Rebuild container** - Often fixes issues
5. **Check GitHub Issues** - See if others had similar problems

---

**Last Updated:** October 2025  
**Tested With:** VS Code 1.x, Docker Desktop 4.x, WSL2 on Windows, macOS, Linux

