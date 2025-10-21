# Quick Start Guide - Get Running in 3 Minutes

## The Problem

Push doesn't work because SSH isn't configured properly? Let's fix it!

## The Solution - Use `.env.devcontainer`

The devcontainer setup scripts automatically load your configuration from `.env.devcontainer` at the project root. No need to source it manually!

---

## 3-Minute Setup

### 1. Copy the Template (10 seconds)

```bash
cd /workspaces/statmate-ai
cp .devcontainer/env.devcontainer.example .env.devcontainer
```

**Important:** The file MUST be at the project root (`/workspaces/statmate-ai/.env.devcontainer`), NOT inside `.devcontainer/`

### 2. Edit Your Values (2 minutes)

```bash
nano .env.devcontainer
# or
code .env.devcontainer
```

Change these lines to YOUR values:

```bash
GIT_USER_NAME=Your Name Here          # ← YOUR NAME HERE
GIT_USER_EMAIL=your.email@example.com # ← YOUR EMAIL HERE
SSH_KEY_NAME=id_rsa                   # ← YOUR SSH KEY NAME HERE
OPENAI_API_KEY=sk-your-key-here       # ← YOUR OPENAI KEY HERE
```

**Don't know your SSH key name?** Run:
```bash
ls -la ~/.ssh/
```

Look for files like `id_rsa`, `id_ed25519`, `adam_private_gh`, etc. (without `.pub`)

### 3. Reopen in Container (1 minute)

In VS Code:
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Dev Containers: Rebuild Container"
- Select it and wait for container to rebuild

The setup scripts will automatically load your `.env.devcontainer` configuration!

### 4. Test It Works (30 seconds)

Inside the container terminal:

```bash
# Test Git config
git config --global user.name
git config --global user.email

# Test SSH connection to GitHub
ssh -T git@github.com
# Should say: "Hi <username>! You've successfully authenticated"

# Test Git push (replace with your branch name)
git push
```

✅ **Done!** Push should work now.

---

## If Something Goes Wrong

### "Permission denied (publickey)"

1. **Check SSH key exists:**
   ```bash
   ls -la ~/.ssh/adam_private_gh  # Use YOUR key name
   ```

2. **Check if key is on GitHub:**
   - Go to: https://github.com/settings/keys
   - Is your public key listed? If not:
     ```bash
     cat ~/.ssh/adam_private_gh.pub
     ```
   - Copy output and add to GitHub

3. **Rebuild container:**
   `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"

### "GIT_USER_NAME not set"

The `.env.devcontainer` file doesn't exist or has incorrect values.

```bash
# Check if file exists at project root
ls -la /workspaces/statmate-ai/.env.devcontainer

# If not, create it:
cp .devcontainer/env.devcontainer.example .env.devcontainer
nano .env.devcontainer

# Then rebuild container
```

### ".env.devcontainer not found"

The file must be in the **project root**, not in `.devcontainer/`

```bash
# ✅ CORRECT location:
/workspaces/statmate-ai/.env.devcontainer

# ❌ WRONG location:
/workspaces/statmate-ai/.devcontainer/.env.devcontainer
```

---

## Full Manual

For complete documentation, see:
- **[SETUP_MANUAL.md](.devcontainer/SETUP_MANUAL.md)** - Complete guide
- **[ENV_SETUP.md](.devcontainer/ENV_SETUP.md)** - Environment setup details
- **[MIGRATION_GUIDE.md](.devcontainer/MIGRATION_GUIDE.md)** - Migration from old branches

---

## Summary

1. ✅ Copy `env.devcontainer.example` → `.env.devcontainer` (at project root)
2. ✅ Edit with your values (GIT_USER_NAME, GIT_USER_EMAIL, SSH_KEY_NAME, OPENAI_API_KEY)
3. ✅ Rebuild Container (Ctrl+Shift+P → "Dev Containers: Rebuild Container")
4. ✅ Test: `git push`

**Time: 3 minutes** ⏱️

The setup scripts automatically load `.env.devcontainer` - no manual sourcing needed!

---

**Still stuck?** Check [SETUP_MANUAL.md](./SETUP_MANUAL.md) for detailed troubleshooting!

