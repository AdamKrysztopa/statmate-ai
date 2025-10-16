# SSH Setup for GitHub in DevContainer

## What Was Configured

Your devcontainer has been configured to use your personal GitHub SSH key (`adam_private_gh`) for authentication.

### Files Modified/Created:

1. **`.devcontainer/devcontainer.json`**
   - Added mounts to bind your SSH keys from host to container
   - Updated `postCreateCommand` to run SSH setup script

2. **`.devcontainer/Dockerfile`**
   - Added `openssh-client` package
   - Created `.ssh` directory with proper permissions

3. **`.devcontainer/setup_ssh.sh`** (NEW)
   - Sets correct permissions on SSH keys (600 for private, 644 for public)
   - Creates SSH config to use `adam_private_gh` for github.com
   - Starts ssh-agent and adds the key
   - Tests GitHub connection

### SSH Configuration

The setup creates `~/.ssh/config` with:
```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/adam_private_gh
    IdentitiesOnly yes
    AddKeysToAgent yes
```

This ensures that all GitHub operations use your personal SSH key.

## How to Apply Changes

**IMPORTANT:** You need to rebuild the devcontainer for these changes to take effect.

### Option 1: Rebuild from Command Palette
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Dev Containers: Rebuild Container"
3. Press Enter

### Option 2: Rebuild from DevContainer Menu
1. Click on the green/blue icon in the bottom-left corner
2. Select "Rebuild Container"

## Verify Setup After Rebuild

After rebuilding, verify the setup:

```bash
# Check SSH keys exist
ls -la ~/.ssh/

# Check SSH config
cat ~/.ssh/config

# Test GitHub connection
ssh -T git@github.com
```

Expected output from GitHub test:
```
Hi AdamKrysztopa! You've successfully authenticated, but GitHub does not provide shell access.
```

## Troubleshooting

### Keys not found after rebuild
- Ensure the keys exist on your host machine at: `~/.ssh/adam_private_gh` and `~/.ssh/adam_private_gh.pub`
- Check that the key files have the correct permissions on your host

### Permission denied when pushing
```bash
# Manually add the key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/adam_private_gh

# Test connection
ssh -T git@github.com
```

### Still can't push
```bash
# Verify remote URL uses SSH
git remote -v

# Should show: git@github.com:AdamKrysztopa/statmate-ai.git
# If it shows https://, change it:
git remote set-url origin git@github.com:AdamKrysztopa/statmate-ai.git
```

## Notes

- The SSH keys are mounted from your host machine, not copied, so they stay secure
- Changes to keys on host are immediately reflected in the container
- The setup script runs automatically on container creation
- This configuration is container-only and doesn't affect your host machine

