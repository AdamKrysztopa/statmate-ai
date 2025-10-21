#!/bin/bash
# Quick setup script for WORK/SECONDARY machine configuration (origin/fixes style)

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   Setting up DevContainer for WORK/SECONDARY machine                ║"
echo "║   (origin/fixes configuration)                                       ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -d ".devcontainer" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   cd /path/to/statmate-ai && bash .devcontainer/setup-env-work.sh"
    exit 1
fi

# Step 1: Copy template
echo "📋 Step 1/4: Copying work configuration template..."
cp .devcontainer/env.devcontainer.ADAM_WORK .env.devcontainer
echo "✓ Created .env.devcontainer"
echo ""

# Step 2: Verify SSH key exists
echo "🔑 Step 2/4: Checking SSH key..."
if [ -f ~/.ssh/adam_private_gh ] && [ -f ~/.ssh/adam_private_gh.pub ]; then
    echo "✓ SSH key found: ~/.ssh/adam_private_gh"
    ls -lh ~/.ssh/adam_private_gh*
else
    echo "⚠️  Warning: SSH key not found at ~/.ssh/adam_private_gh"
    echo "   You may need to:"
    echo "   1. Generate a new key: ssh-keygen -t ed25519 -C 'krysztopa@gmail.com' -f ~/.ssh/adam_private_gh"
    echo "   2. Or update SSH_KEY_NAME in .env.devcontainer to match your key"
fi
echo ""

# Step 3: Remind about API key
echo "🔐 Step 3/4: OpenAI API Key"
echo "⚠️  IMPORTANT: Edit .env.devcontainer and replace:"
echo "   OPENAI_API_KEY=sk-your-actual-openai-api-key-here"
echo "   with your real OpenAI API key"
echo ""
echo "   Run: nano .env.devcontainer"
echo ""

# Step 4: Instructions to source
echo "📝 Step 4/4: Next steps:"
echo ""
echo "1. Edit .env.devcontainer with your OpenAI API key:"
echo "   nano .env.devcontainer"
echo ""
echo "2. Source the file:"
echo "   source .env.devcontainer"
echo ""
echo "3. Verify variables:"
echo "   echo \$GIT_USER_NAME \$SSH_KEY_NAME"
echo ""
echo "4. Open VS Code:"
echo "   code ."
echo ""
echo "5. Reopen in Container (Ctrl+Shift+P)"
echo ""
echo "6. Test push:"
echo "   git push origin unified-devcontainer"
echo ""
echo "✅ Setup template created! Follow the steps above."

