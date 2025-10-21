#!/bin/bash
# Quick setup script for HOME/PRIMARY machine configuration (origin/fixes_local_dev_container style)

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   Setting up DevContainer for HOME/PRIMARY machine                  ║"
echo "║   (origin/fixes_local_dev_container configuration)                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -d ".devcontainer" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   cd /path/to/statmate-ai && bash .devcontainer/setup-env-home.sh"
    exit 1
fi

# Step 1: Copy template
echo "📋 Step 1/5: Copying home configuration template..."
cp .devcontainer/env.devcontainer.ADAM_HOME .env.devcontainer
echo "✓ Created .env.devcontainer"
echo ""

# Step 2: Check SSH directory
echo "🔑 Step 2/5: Checking .ssh directory..."
if [ -d ~/.ssh ]; then
    echo "✓ .ssh directory found"
    echo "   Keys available:"
    ls -lh ~/.ssh/*.pub 2>/dev/null || echo "   No public keys found"
else
    echo "⚠️  Warning: ~/.ssh directory not found"
fi
echo ""

# Step 3: Remind about API key
echo "🔐 Step 3/5: OpenAI API Key"
echo "⚠️  IMPORTANT: Edit .env.devcontainer and replace:"
echo "   OPENAI_API_KEY=sk-your-actual-openai-api-key-here"
echo "   with your real OpenAI API key"
echo ""
echo "   Run: nano .env.devcontainer"
echo ""

# Step 4: Update devcontainer.json
echo "📝 Step 4/5: Update devcontainer.json"
echo "⚠️  IMPORTANT: You must update .devcontainer/devcontainer.json"
echo ""
echo "Change the 'mounts' section from:"
echo '  "mounts": ['
echo '      "source=${localEnv:HOME}/.ssh/${localEnv:SSH_KEY_NAME},...",'
echo '      "source=${localEnv:HOME}/.ssh/${localEnv:SSH_KEY_NAME}.pub,..."'
echo '  ],'
echo ""
echo "To:"
echo '  "mounts": ['
echo '      "source=${localEnv:HOME}/.ssh,target=/home/vscode/.ssh,type=bind,consistency=cached"'
echo '  ],'
echo ""
echo "(See the commented APPROACH 2 section in devcontainer.json)"
echo ""

# Step 5: Instructions to source
echo "📝 Step 5/5: Next steps:"
echo ""
echo "1. Edit .env.devcontainer with your OpenAI API key:"
echo "   nano .env.devcontainer"
echo ""
echo "2. Update .devcontainer/devcontainer.json mounts section (see above)"
echo ""
echo "3. Source the file:"
echo "   source .env.devcontainer"
echo ""
echo "4. Verify variables:"
echo "   echo \$GIT_USER_NAME"
echo ""
echo "5. Open VS Code:"
echo "   code ."
echo ""
echo "6. REBUILD Container (Ctrl+Shift+P → Rebuild Container)"
echo "   (Required because you changed devcontainer.json)"
echo ""
echo "7. Test push:"
echo "   git push origin unified-devcontainer"
echo ""
echo "✅ Setup template created! Follow the steps above."

