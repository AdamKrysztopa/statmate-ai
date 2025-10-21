# Quick Start Guide - Get Running in 5 Minutes

## The Problem

Push doesn't work because SSH isn't configured properly? Let's fix it!

## The Solution - Use `.env.devcontainer`

Instead of shell environment variables, we'll use a `.env.devcontainer` file that you source before opening VS Code.

---

## 5-Minute Setup

### 1. Copy the Template (10 seconds)

```bash
cd /path/to/statmate-ai
cp .devcontainer/env.devcontainer.example .env.devcontainer
```

### 2. Edit Your Values (2 minutes)

```bash
nano .env.devcontainer
```

Change these lines to YOUR values:

```bash
GIT_USER_NAME=Adam Krysztopa          # ← YOUR NAME HERE
GIT_USER_EMAIL=krysztopa@gmail.com    # ← YOUR EMAIL HERE
SSH_KEY_NAME=adam_private_gh          # ← YOUR SSH KEY NAME HERE
OPENAI_API_KEY=sk-your-key-here       # ← YOUR OPENAI KEY HERE
```

**Don't know your SSH key name?** Run:
```bash
ls -la ~/.ssh/
```

Look for files like `id_rsa`, `id_ed25519`, `github_key`, etc. (without `.pub`)

### 3. Source It (5 seconds)

```bash
source .env.devcontainer
```

### 4. Verify Variables Are Set (10 seconds)

```bash
echo $GIT_USER_NAME
echo $SSH_KEY_NAME
```

You should see your values printed.

### 5. Open VS Code (5 seconds)

```bash
code .
```

### 6. Reopen in Container (2 minutes)

- Click "Reopen in Container" when prompted
- OR press `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"

Wait for container to build...

### 7. Test It Works (30 seconds)

Inside the container terminal:

```bash
# Test Git config
git config --global user.name

# Test SSH
ssh -T git@github.com
# Should say: "Hi <username>! You've successfully authenticated"

# Test Push
git push origin unified-devcontainer
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

You forgot to source the .env file!

```bash
source .env.devcontainer
code .
```

### ".env.devcontainer not found"

The file must be in the **project root**, not in `.devcontainer/`

```bash
# Should be here:
/path/to/statmate-ai/.env.devcontainer

# NOT here:
/path/to/statmate-ai/.devcontainer/.env.devcontainer
```

---

## Make It Permanent (Optional)

Don't want to source `.env.devcontainer` every time?

### Option 1: Add to Shell Profile

```bash
# Add this to ~/.bashrc or ~/.zshrc:
if [ -f ~/Projects/statmate-ai/.env.devcontainer ]; then
    source ~/Projects/statmate-ai/.env.devcontainer
fi
```

Then:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Option 2: Create a Global Config

```bash
# Copy to your home directory
cp .env.devcontainer ~/.env.statmate

# Add to ~/.bashrc or ~/.zshrc:
if [ -f ~/.env.statmate ]; then
    source ~/.env.statmate
fi
```

---

## Two Configuration Approaches

### Current (Approach 1): Specific SSH Key

**Configuration:**
- In `.env.devcontainer`: `SSH_KEY_NAME=adam_private_gh`
- In `devcontainer.json`: Mounts specific key

**Use when:**
- Working from secondary repo
- Need specific SSH key for GitHub
- More control over keys

### Alternative (Approach 2): Mount Entire .ssh

**Configuration:**
- In `.env.devcontainer`: Remove or comment out `SSH_KEY_NAME`
- In `devcontainer.json`: Change mounts section (see commented instructions)

**Use when:**
- Primary development machine
- This is your main repo
- Want all your SSH keys available

**To switch to Approach 2:**

1. Edit `devcontainer.json`
2. Replace the `mounts` section with the commented alternative
3. Rebuild container

---

## Full Manual

For complete documentation, see:
- **[SETUP_MANUAL.md](.devcontainer/SETUP_MANUAL.md)** - Complete guide
- **[ENV_SETUP.md](.devcontainer/ENV_SETUP.md)** - Environment setup details
- **[MIGRATION_GUIDE.md](.devcontainer/MIGRATION_GUIDE.md)** - Migration from old branches

---

## Summary

1. ✅ Copy `env.devcontainer.example` → `.env.devcontainer`
2. ✅ Edit with your values
3. ✅ Source it: `source .env.devcontainer`
4. ✅ Open VS Code: `code .`
5. ✅ Reopen in Container
6. ✅ Test: `git push`

**Time: 5 minutes** ⏱️

---

**Still stuck?** Check [SETUP_MANUAL.md](./SETUP_MANUAL.md) for detailed troubleshooting!

