#!/bin/bash
# Test script to verify devcontainer configuration

echo "========================================="
echo "DevContainer Configuration Test"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Test 1: Check if .env.devcontainer exists
echo "1. Checking for .env.devcontainer file..."
if [ -f "/workspaces/statmate-ai/.env.devcontainer" ]; then
    echo -e "   ${GREEN}✓${NC} .env.devcontainer exists at project root"
else
    echo -e "   ${RED}✗${NC} .env.devcontainer NOT FOUND at project root"
    echo "   Create it: cp .devcontainer/env.devcontainer.example .env.devcontainer"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Test 2: Check if .env.devcontainer has required variables
echo "2. Checking .env.devcontainer content..."
if [ -f "/workspaces/statmate-ai/.env.devcontainer" ]; then
    if grep -q "GIT_USER_NAME=" /workspaces/statmate-ai/.env.devcontainer; then
        GIT_NAME=$(grep "^GIT_USER_NAME=" /workspaces/statmate-ai/.env.devcontainer | cut -d'=' -f2-)
        if [ "$GIT_NAME" = "Your Name Here" ] || [ -z "$GIT_NAME" ]; then
            echo -e "   ${YELLOW}⚠${NC} GIT_USER_NAME not customized (still default value)"
            WARNINGS=$((WARNINGS + 1))
        else
            echo -e "   ${GREEN}✓${NC} GIT_USER_NAME is set: $GIT_NAME"
        fi
    else
        echo -e "   ${RED}✗${NC} GIT_USER_NAME not found in .env.devcontainer"
        ERRORS=$((ERRORS + 1))
    fi
    
    if grep -q "GIT_USER_EMAIL=" /workspaces/statmate-ai/.env.devcontainer; then
        GIT_EMAIL=$(grep "^GIT_USER_EMAIL=" /workspaces/statmate-ai/.env.devcontainer | cut -d'=' -f2-)
        if [ "$GIT_EMAIL" = "your.email@example.com" ] || [ -z "$GIT_EMAIL" ]; then
            echo -e "   ${YELLOW}⚠${NC} GIT_USER_EMAIL not customized (still default value)"
            WARNINGS=$((WARNINGS + 1))
        else
            echo -e "   ${GREEN}✓${NC} GIT_USER_EMAIL is set: $GIT_EMAIL"
        fi
    else
        echo -e "   ${RED}✗${NC} GIT_USER_EMAIL not found in .env.devcontainer"
        ERRORS=$((ERRORS + 1))
    fi
    
    if grep -q "SSH_KEY_NAME=" /workspaces/statmate-ai/.env.devcontainer; then
        SSH_KEY=$(grep "^SSH_KEY_NAME=" /workspaces/statmate-ai/.env.devcontainer | cut -d'=' -f2-)
        echo -e "   ${GREEN}✓${NC} SSH_KEY_NAME is set: $SSH_KEY"
    else
        echo -e "   ${RED}✗${NC} SSH_KEY_NAME not found in .env.devcontainer"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# Test 3: Check Git configuration
echo "3. Checking Git global configuration..."
GIT_CFG_NAME=$(git config --global user.name 2>/dev/null)
GIT_CFG_EMAIL=$(git config --global user.email 2>/dev/null)

if [ -n "$GIT_CFG_NAME" ]; then
    echo -e "   ${GREEN}✓${NC} Git user.name: $GIT_CFG_NAME"
else
    echo -e "   ${RED}✗${NC} Git user.name is NOT set"
    ERRORS=$((ERRORS + 1))
fi

if [ -n "$GIT_CFG_EMAIL" ]; then
    echo -e "   ${GREEN}✓${NC} Git user.email: $GIT_CFG_EMAIL"
else
    echo -e "   ${RED}✗${NC} Git user.email is NOT set"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Test 4: Check SSH keys
echo "4. Checking SSH keys..."
if [ -d ~/.ssh ]; then
    echo -e "   ${GREEN}✓${NC} .ssh directory exists"
    
    # Count SSH keys
    KEY_COUNT=$(ls -1 ~/.ssh/id_* ~/.ssh/*_private* 2>/dev/null | grep -v ".pub" | wc -l)
    if [ "$KEY_COUNT" -gt 0 ]; then
        echo -e "   ${GREEN}✓${NC} Found $KEY_COUNT SSH private key(s)"
        ls -1 ~/.ssh/id_* ~/.ssh/*_private* 2>/dev/null | grep -v ".pub" | sed 's|.*/||' | sed 's/^/     - /'
    else
        echo -e "   ${YELLOW}⚠${NC} No SSH private keys found in ~/.ssh/"
        WARNINGS=$((WARNINGS + 1))
    fi
    
    # Check SSH config
    if [ -f ~/.ssh/config ]; then
        echo -e "   ${GREEN}✓${NC} SSH config exists"
        if grep -q "github.com" ~/.ssh/config; then
            echo -e "   ${GREEN}✓${NC} GitHub configuration found in SSH config"
        else
            echo -e "   ${YELLOW}⚠${NC} GitHub not configured in SSH config"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo -e "   ${YELLOW}⚠${NC} SSH config not found (will be created on first use)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "   ${RED}✗${NC} .ssh directory does not exist"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Test 5: Test GitHub SSH connection
echo "5. Testing GitHub SSH connection..."
if ssh -T -o StrictHostKeyChecking=no -o BatchMode=yes git@github.com 2>&1 | grep -q "successfully authenticated"; then
    USERNAME=$(ssh -T -o StrictHostKeyChecking=no -o BatchMode=yes git@github.com 2>&1 | grep "successfully authenticated" | sed 's/.*Hi \(.*\)!.*/\1/')
    echo -e "   ${GREEN}✓${NC} GitHub SSH authentication successful!"
    echo "     GitHub username: $USERNAME"
else
    echo -e "   ${YELLOW}⚠${NC} GitHub SSH authentication test failed or skipped"
    echo "     This is normal if:"
    echo "     - SSH key is not yet added to GitHub"
    echo "     - SSH key requires a passphrase"
    echo "     - Network issues"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Your devcontainer is properly configured."
    echo "You should be able to git push successfully."
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Tests passed with $WARNINGS warning(s)${NC}"
    echo ""
    echo "Configuration is mostly correct but has some warnings."
    echo "Review the warnings above and fix if needed."
else
    echo -e "${RED}✗ Found $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before trying to git push."
    echo ""
    echo "Quick fix:"
    echo "  1. cp .devcontainer/env.devcontainer.example .env.devcontainer"
    echo "  2. nano .env.devcontainer  # Edit with your values"
    echo "  3. Rebuild container (Ctrl+Shift+P → Dev Containers: Rebuild Container)"
fi
echo ""

exit $ERRORS

