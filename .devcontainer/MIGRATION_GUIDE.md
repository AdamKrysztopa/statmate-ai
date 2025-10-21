# Migration Guide: Unified DevContainer Configuration

## Summary

Successfully unified the devcontainer configurations from `fixes_local_dev_container` and `fixes` branches into a single, configurable setup.

## What Changed

### ✅ Unified Configuration

| Aspect | Before (fixes) | Before (fixes_local_dev_container) | After (Unified) |
|--------|----------------|-------------------------------------|-----------------|
| Git config | ❌ Hardcoded | ✅ Environment variables | ✅ Environment variables |
| SSH keys | ❌ Hardcoded "adam_private_gh" | ❌ No SSH support | ✅ Configurable via SSH_KEY_NAME |
| Secrets | ❌ In devcontainer.json | ✅ Environment variables | ✅ Environment variables |
| Node.js | ✅ Installed | ❌ Not installed | ✅ Installed |
| SSH support | ✅ With hardcoded keys | ❌ None | ✅ With configurable keys |

### 🔧 Files Modified

1. **`.devcontainer/devcontainer.json`**
   - ✅ Removed all hardcoded values (Git name, email, SSH key names)
   - ✅ Added configurable SSH mounts using `${localEnv:SSH_KEY_NAME:id_rsa}`
   - ✅ Updated to use environment variables for all secrets
   - ✅ Added `context: ".."` for proper build context
   - ✅ Updated `postCreateCommand` to include SSH setup
   - ✅ Added `bradlc.vscode-tailwindcss` extension

2. **`.devcontainer/Dockerfile`**
   - ✅ Added SSH client support (`openssh-client`)
   - ✅ Added Node.js 20.x installation
   - ✅ Added GPG and other required tools
   - ✅ Added proper .ssh directory creation
   - ✅ Added npm dependency pre-installation
   - ✅ Added Python dependency pre-installation

3. **`.devcontainer/setup_git.sh`**
   - ✅ Already good - uses environment variables
   - ✅ No changes needed

4. **`.devcontainer/setup_ssh.sh`** (NEW)
   - ✅ Created new configurable SSH setup script
   - ✅ Uses `SSH_KEY_NAME` environment variable (defaults to `id_rsa`)
   - ✅ Sets proper permissions automatically
   - ✅ Creates SSH config for GitHub
   - ✅ Tests connection and provides helpful feedback

### 📄 New Documentation Files

1. **`.devcontainer/README.md`** - Overview of unified configuration
2. **`.devcontainer/ENV_SETUP.md`** - Detailed setup instructions
3. **`.devcontainer/env.template`** - Quick environment variable template
4. **`.devcontainer/MIGRATION_GUIDE.md`** - This file

## Environment Variables

All user-specific configuration is now via environment variables:

| Variable | Required | Default | Example |
|----------|----------|---------|---------|
| `GIT_USER_NAME` | ✅ Yes | - | `"John Doe"` |
| `GIT_USER_EMAIL` | ✅ Yes | - | `"john@example.com"` |
| `SSH_KEY_NAME` | ⚠️ Optional | `id_rsa` | `"id_rsa"` or `"adam_private_gh"` |
| `OPENAI_API_KEY` | ✅ Yes | - | `"sk-..."` |

## Migration Steps

### For Users of `fixes` Branch

If you were using the `fixes` branch with hardcoded "adam_private_gh":

1. **Set environment variables:**
   ```bash
   export GIT_USER_NAME="Adam Krysztopa"  # Or your name
   export GIT_USER_EMAIL="krysztopa@gmail.com"  # Or your email
   export SSH_KEY_NAME="adam_private_gh"  # Keep your existing key name
   export OPENAI_API_KEY="sk-..."
   ```

2. **Add to your shell profile** (`~/.bashrc` or `~/.zshrc`):
   ```bash
   echo 'export GIT_USER_NAME="Adam Krysztopa"' >> ~/.bashrc
   echo 'export GIT_USER_EMAIL="krysztopa@gmail.com"' >> ~/.bashrc
   echo 'export SSH_KEY_NAME="adam_private_gh"' >> ~/.bashrc
   echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Switch to unified branch:**
   ```bash
   git checkout unified-devcontainer
   ```

4. **Rebuild container:**
   - In VS Code: `Ctrl+Shift+P` → "Rebuild Container"

### For Users of `fixes_local_dev_container` Branch

If you were using the `fixes_local_dev_container` branch:

1. **Set SSH key name** (if using custom SSH key):
   ```bash
   export SSH_KEY_NAME="id_rsa"  # Or your SSH key name
   echo 'export SSH_KEY_NAME="id_rsa"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Switch to unified branch:**
   ```bash
   git checkout unified-devcontainer
   ```

3. **Rebuild container:**
   - In VS Code: `Ctrl+Shift+P` → "Rebuild Container"

### For New Users or Different Machines

1. **Set ALL required environment variables:**
   ```bash
   export GIT_USER_NAME="Your Name"
   export GIT_USER_EMAIL="your.email@example.com"
   export SSH_KEY_NAME="id_rsa"  # Optional, this is the default
   export OPENAI_API_KEY="sk-..."
   ```

2. **Make it permanent** in `~/.bashrc` or `~/.zshrc`

3. **Ensure SSH keys exist:**
   ```bash
   ls -la ~/.ssh/
   # Should show your key files
   ```

4. **Clone/open project and open in container**

## Benefits of Unified Configuration

### ✅ Security
- No secrets in repository
- No hardcoded user information
- SSH keys stay on host machine (read-only mount)

### ✅ Flexibility
- Works on any machine
- Easy to switch between different accounts/keys
- No need to modify files when changing machines

### ✅ Maintainability
- Single configuration for all use cases
- Changes apply everywhere automatically
- No more branch switching for different setups

### ✅ Documentation
- Clear setup instructions
- Environment variable templates
- Troubleshooting guide

## Verification Checklist

After migration, verify:

- [ ] Environment variables are set: `echo $GIT_USER_NAME $GIT_USER_EMAIL $SSH_KEY_NAME`
- [ ] SSH keys exist: `ls -la ~/.ssh/`
- [ ] Container builds successfully
- [ ] Git config is set: `git config --global user.name`
- [ ] SSH works: `ssh -T git@github.com`
- [ ] Can push/pull from GitHub

## Troubleshooting

### Environment Variables Not Found

**Problem:** Container fails to start with "GIT_USER_NAME not set"

**Solution:**
1. Check variables in terminal: `echo $GIT_USER_NAME`
2. If empty, add to shell profile: `export GIT_USER_NAME="Your Name"`
3. Reload shell: `source ~/.bashrc`
4. Restart VS Code
5. Rebuild container

### SSH Key Not Found

**Problem:** "SSH private key not found at ~/.ssh/..."

**Solution:**
1. Check key exists: `ls -la ~/.ssh/`
2. Verify `SSH_KEY_NAME` matches actual filename
3. If needed, generate new key:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com" -f ~/.ssh/id_rsa
   ```

### GitHub Authentication Failed

**Problem:** Can't push/pull to GitHub

**Solution:**
1. Verify SSH key is added to GitHub:
   - Copy public key: `cat ~/.ssh/id_rsa.pub`
   - Add to GitHub Settings → SSH Keys
2. Test connection: `ssh -T git@github.com`
3. Check SSH config inside container: `cat ~/.ssh/config`

## Next Steps

1. ✅ Review the changes in this guide
2. ✅ Set environment variables on your machine
3. ✅ Test the unified configuration
4. ✅ Optionally merge to main branch once verified
5. ✅ Update team documentation if needed

## Questions?

See:
- [README.md](.devcontainer/README.md) - Configuration overview
- [ENV_SETUP.md](.devcontainer/ENV_SETUP.md) - Detailed setup guide
- [env.template](.devcontainer/env.template) - Variable template

## Rollback

If you need to rollback to the previous configuration:

```bash
# To go back to fixes branch
git checkout fixes

# To go back to fixes_local_dev_container branch
git checkout fixes_local_dev_container
```

However, the unified configuration is recommended for long-term use.

