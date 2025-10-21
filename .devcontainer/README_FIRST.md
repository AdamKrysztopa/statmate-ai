# 🎯 START HERE - Quick Reference

## ✅ What's Done

Your devcontainer is configured with **TWO setups**:

1. **WORK** (origin/fixes) - Currently active, using specific SSH key
2. **HOME** (origin/fixes_local_dev_container) - Ready when needed

## 🚀 Get Started NOW (30 seconds)

```bash
# 1. Add your OpenAI API key
nano .env.devcontainer
# Change: OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# 2. Source it
source .env.devcontainer

# 3. Verify
echo $GIT_USER_NAME $SSH_KEY_NAME

# 4. Open VS Code from this terminal
code .

# 5. Reopen in Container (Ctrl+Shift+P)
```

That's it! Push should work after container builds.

---

## 📖 Full Guides

- **Your Personal Guide**: [ADAM_GUIDE.md](./ADAM_GUIDE.md) ← **START HERE**
- **Quick Start**: [QUICKSTART.md](./QUICKSTART.md)
- **Complete Manual**: [SETUP_MANUAL.md](./SETUP_MANUAL.md)

---

## 🔄 Switch Between Work & Home

### At Work (Currently Active)
```bash
bash .devcontainer/setup-env-work.sh
# Edit .env.devcontainer with API key
source .env.devcontainer
code .
```

### At Home
```bash
bash .devcontainer/setup-env-home.sh
# Edit .env.devcontainer with API key
# Edit devcontainer.json mounts (see instructions in output)
source .env.devcontainer
code .
```

---

## 🆘 If Push Doesn't Work

```bash
# Inside container, run:
ssh -T git@github.com
# Should say: "Hi <username>! You've successfully authenticated"

# If that fails, check if your public key is on GitHub:
cat ~/.ssh/adam_private_gh.pub
# Add to: https://github.com/settings/keys
```

---

## 📁 Key Files

| File | What It Is |
|------|------------|
| `.env.devcontainer` | Your active config (has your settings) |
| `env.devcontainer.ADAM_WORK` | Work template |
| `env.devcontainer.ADAM_HOME` | Home template |
| `ADAM_GUIDE.md` | Your complete guide |

---

## ✅ Your Current Setup

```
Git User:  Adam Krysztopa
Git Email: krysztopa@gmail.com
SSH Key:   adam_private_gh ✅ (verified, exists)
API Key:   ⚠️ Needs to be added to .env.devcontainer
```

---

**Next**: Edit `.env.devcontainer`, add your API key, source it, and open VS Code!
