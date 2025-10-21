# Adam's DevContainer Configuration Guide

## ✅ What Was Done

I've set up your devcontainer with **TWO** configurations based on your existing branches:

### 1. **WORK Configuration** (origin/fixes) - ✅ CURRENTLY ACTIVE
- **File**: `.env.devcontainer` (already created!)
- **SSH Key**: `adam_private_gh`
- **Use When**: At work or secondary machine
- **Status**: ✅ SSH key found and verified

### 2. **HOME Configuration** (origin/fixes_local_dev_container)
- **File**: `.devcontainer/env.devcontainer.ADAM_HOME` (template ready)
- **SSH**: Mounts entire `.ssh` directory
- **Use When**: At home or primary machine
- **How to Switch**: See section below

---

## 🚀 Quick Start (Work Configuration)

Your `.env.devcontainer` file is **already created** with these settings from `origin/fixes`:

```bash
GIT_USER_NAME=Adam Krysztopa
GIT_USER_EMAIL=krysztopa@gmail.com
SSH_KEY_NAME=adam_private_gh
OPENAI_API_KEY=sk-your-actual-openai-api-key-here  # ← EDIT THIS!
```

### What You Need to Do Right Now:

```bash
# 1. Edit .env.devcontainer and add your OpenAI API key
nano .env.devcontainer
# Replace: sk-your-actual-openai-api-key-here with your real key

# 2. Source the file (makes variables available to VS Code)
source .env.devcontainer

# 3. Verify it worked
echo $GIT_USER_NAME
echo $SSH_KEY_NAME
# Should print: Adam Krysztopa adam_private_gh

# 4. Open VS Code FROM THIS TERMINAL
code .

# 5. In VS Code: Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# 6. Wait for build, then test push
git push origin unified-devcontainer
```

---

## 🏠 Switching to HOME Configuration

When you're at home and want to use the `origin/fixes_local_dev_container` style setup:

### Step 1: Use Home Template

```bash
# Replace current .env.devcontainer with home version
cp .devcontainer/env.devcontainer.ADAM_HOME .env.devcontainer

# Edit with your OpenAI API key
nano .env.devcontainer
```

### Step 2: Update devcontainer.json

Edit `.devcontainer/devcontainer.json`:

**Change FROM** (lines 21-24):
```json
"mounts": [
    "source=${localEnv:HOME}${localEnv:USERPROFILE}/.ssh/${localEnv:SSH_KEY_NAME},target=/home/vscode/.ssh/${localEnv:SSH_KEY_NAME},type=bind,consistency=cached",
    "source=${localEnv:HOME}${localEnv:USERPROFILE}/.ssh/${localEnv:SSH_KEY_NAME}.pub,target=/home/vscode/.ssh/${localEnv:SSH_KEY_NAME}.pub,type=bind,consistency=cached"
],
```

**Change TO**:
```json
"mounts": [
    "source=${localEnv:HOME}${localEnv:USERPROFILE}/.ssh,target=/home/vscode/.ssh,type=bind,consistency=cached"
],
```

### Step 3: Rebuild

```bash
# Source the file
source .env.devcontainer

# Open VS Code
code .

# Rebuild Container (required because devcontainer.json changed)
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

---

## 📋 Files Reference

| File                                       | Purpose                                           |
| ------------------------------------------ | ------------------------------------------------- |
| `.env.devcontainer`                        | Your active config (gitignored, contains secrets) |
| `.devcontainer/env.devcontainer.ADAM_WORK` | Work template (origin/fixes)                      |
| `.devcontainer/env.devcontainer.ADAM_HOME` | Home template (origin/fixes_local_dev_container)  |
| `.devcontainer/setup-env-work.sh`          | Auto-setup work config                            |
| `.devcontainer/setup-env-home.sh`          | Auto-setup home config                            |
| `.devcontainer/QUICKSTART.md`              | 5-minute quick start guide                        |
| `.devcontainer/SETUP_MANUAL.md`            | Complete manual (600+ lines)                      |

---

## 🔧 Quick Commands Reference

### Work Setup (Current)
```bash
bash .devcontainer/setup-env-work.sh
# Then edit .env.devcontainer with API key
source .env.devcontainer
code .
```

### Home Setup
```bash
bash .devcontainer/setup-env-home.sh
# Edit .env.devcontainer with API key
# Edit devcontainer.json mounts section
source .env.devcontainer
code .
```

### Test Everything Works
```bash
# Inside container:
git config --global user.name     # Should show: Adam Krysztopa
git config --global user.email    # Should show: krysztopa@gmail.com
ls -la ~/.ssh/                     # Should show your SSH keys
ssh -T git@github.com              # Should authenticate successfully
git push                           # Should work!
```

---

## 🐛 Troubleshooting

### Push Still Doesn't Work

**Debug steps:**

1. **Check variables are set:**
   ```bash
   echo $GIT_USER_NAME $GIT_USER_EMAIL $SSH_KEY_NAME
   ```

2. **Inside container, check SSH:**
   ```bash
   ls -la ~/.ssh/
   cat ~/.ssh/config
   ssh -T git@github.com
   ```

3. **If SSH test fails, check key on GitHub:**
   ```bash
   cat ~/.ssh/adam_private_gh.pub
   ```
   Go to https://github.com/settings/keys and verify this key is there.

4. **Rebuild container:**
   ```
   Ctrl+Shift+P → "Dev Containers: Rebuild Container"
   ```

### Forgot to Source .env.devcontainer

If you open VS Code without sourcing, variables won't be set:

```bash
# Exit VS Code
# In terminal:
source .env.devcontainer
code .
# Now reopen in container
```

### Need to Switch Between Work and Home Often?

Create shell aliases:

```bash
# Add to ~/.bashrc or ~/.zshrc:
alias statmate-work='cd ~/path/to/statmate-ai && source .env.devcontainer && code .'
alias statmate-home='cd ~/path/to/statmate-ai && source .env.devcontainer && code .'
```

---

## 📊 Configuration Comparison

| Feature               | Work (origin/fixes)            | Home (origin/fixes_local_dev) |
| --------------------- | ------------------------------ | ----------------------------- |
| **SSH Key**           | Specific key (adam_private_gh) | All keys in .ssh/             |
| **devcontainer.json** | Mounts specific keys           | Mounts entire .ssh dir        |
| **.env.devcontainer** | Includes SSH_KEY_NAME          | No SSH_KEY_NAME needed        |
| **Use Case**          | Secondary machine              | Primary machine               |
| **Current Status**    | ✅ Active                       | Template ready                |

---

## ✅ Current Status Summary

```
✅ Branch: unified-devcontainer
✅ .env.devcontainer: Created with work settings
✅ SSH Key: Found (adam_private_gh)
✅ Git Config: Adam Krysztopa / krysztopa@gmail.com
⚠️  TODO: Add your OpenAI API key to .env.devcontainer
⚠️  TODO: Source .env.devcontainer before opening VS Code
⚠️  TODO: Test push works
```

---

## 🎯 Next Steps

1. **Edit `.env.devcontainer`** - Add your OpenAI API key
2. **Source it** - `source .env.devcontainer`
3. **Open VS Code** - `code .`
4. **Reopen in Container** - Ctrl+Shift+P
5. **Test push** - `git push origin unified-devcontainer`

---

## 💡 Pro Tips

### Always Source Before Opening VS Code

VS Code devcontainers read environment variables from your shell. If you don't source `.env.devcontainer` first, variables won't be available.

### Make it Automatic

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Auto-source statmate-ai env when in that directory
cd() {
    builtin cd "$@"
    if [ -f ".env.devcontainer" ]; then
        source .env.devcontainer
        echo "✓ Loaded .env.devcontainer"
    fi
}
```

### Keep Secrets Separate

Never commit `.env.devcontainer`! It's already in `.gitignore`, but double-check:

```bash
git check-ignore .env.devcontainer
# Should output: .env.devcontainer
```

---

## 📞 Need Help?

1. **Quick Start**: See `.devcontainer/QUICKSTART.md`
2. **Full Manual**: See `.devcontainer/SETUP_MANUAL.md`
3. **Troubleshooting**: Both guides have extensive troubleshooting sections

---

**Created**: October 2025  
**Your Configuration**: origin/fixes (work) + origin/fixes_local_dev_container (home)  
**Status**: ✅ Ready to use (after adding OpenAI API key)

