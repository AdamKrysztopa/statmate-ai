# 🚀 StatmateAI Setup Guide

Quick guide to get StatmateAI up and running in minutes!

---

## 🎯 Choose Your Mode

StatmateAI has two modes:

- **🔧 DEV Mode** - For developers (uses `.env` file with your API keys)
- **🚀 PROD Mode** - For end users (users enter credentials via UI)

> **First time?** Start with DEV mode!

---

## 🔧 DEV Mode Setup (Recommended for First Time)

### Step 1: Install Dependencies

```bash
# Quick install
make install

# Or with pip
pip install -e .
```

### Step 2: Create `.env` File

```bash
# Create from template
make setup-env

# This copies .env.example to .env
```

### Step 3: Add Your API Keys

Edit `.env` and add **at least one** API key:

```bash
# Open in your editor
nano .env

# Or use your favorite editor
code .env
```

**Minimum required:**

```ini
ENVIRONMENT=development
OPENAI_API_KEY=sk-proj-...your-key-here...
```

**Full options:**

```ini
# Application Mode
ENVIRONMENT=development
DEBUG=true

# OpenAI (Most common)
OPENAI_API_KEY=sk-proj-...

# Anthropic (Optional)
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini (Optional)
GOOGLE_API_KEY=AI...

# Groq - Fast & Free tier (Optional)
GROQ_API_KEY=gsk_...

# Ollama - Local & FREE! (Optional)
OLLAMA_ENABLED=true
OLLAMA_DEFAULT_MODEL=deepseek-r1:8b
```

### Step 4: Initialize Database

```bash
make db-init db-seed
```

This creates the SQLite database and adds sample datasets.

### Step 5: Run the Application

```bash
# Start API (Terminal 1)
make api

# Start UI (Terminal 2)
make ui
```

Or use the all-in-one command:

```bash
make dev  # Starts API (you still need to run 'make ui' in another terminal)
```

### Step 6: Open in Browser

- **UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

---

## 🚀 PROD Mode Setup (For Deployment)

### Step 1: Install Dependencies

```bash
make install
```

### Step 2: Initialize Database

```bash
make db-init
```

### Step 3: Set Environment

```bash
# Create minimal .env (NO API keys!)
cat > .env << EOF
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
EOF
```

### Step 4: Run in Production Mode

```bash
make prod
```

### Step 5: Configure Credentials via UI

1. Open http://localhost:8501
2. You'll see the credential setup page
3. Enter your API keys in the web form
4. Click "Configure & Start"
5. Start analyzing! 🎉

---

## 🆓 Using Free Local Models (Ollama)

Want to use AI for **FREE** with **complete privacy**? Use Ollama!

### Install Ollama

```bash
# Visit https://ollama.ai and download for your OS

# Or on macOS:
brew install ollama

# On Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

### Pull DeepSeek-R1 (Reasoning Model)

```bash
# 8B model (4GB RAM)
ollama pull deepseek-r1:8b

# Or 70B model (48GB RAM, better quality)
ollama pull deepseek-r1:70b
```

### Start Ollama Server

```bash
ollama serve
```

### Configure StatmateAI

**For DEV mode** - Add to `.env`:
```ini
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_DEFAULT_MODEL=deepseek-r1:8b
```

**For PROD mode** - In the UI:
1. Go to "Ollama (Local)" tab
2. Check "Enable Ollama"
3. Enter model name: `deepseek-r1:8b`
4. Click "Configure & Start"

---

## 📋 Makefile Commands Cheat Sheet

```bash
# Quick Start
make quickstart       # First-time complete setup
make dev             # Run in DEV mode
make ui              # Run Streamlit UI only
make api             # Run FastAPI backend only

# Production
make prod            # Run in PROD mode
make prod-ui         # Run UI in PROD
make prod-api        # Run API in PROD

# Database
make db-init         # Initialize database
make db-seed         # Add sample data
make db-reset        # Reset database (⚠️ deletes all data)

# Setup
make install         # Install dependencies
make setup-env       # Create .env from template

# Utilities
make status          # Check system status
make clean           # Clean temp files
make help            # Show all commands
```

---

## 🔍 Verify Installation

```bash
# Check system status
make status

# Should show:
# ✓ .env file exists
# ✓ Database exists
# ✓ OpenAI (or other provider)
```

---

## 🌐 Where to Get API Keys

### OpenAI (GPT-4, GPT-4o)
- **Website:** https://platform.openai.com/api-keys
- **Cost:** Pay-as-you-go, ~$0.01/1K tokens
- **Free:** $5 credit for new accounts

### Anthropic (Claude)
- **Website:** https://console.anthropic.com/
- **Cost:** Pay-as-you-go, ~$0.003/1K tokens
- **Free:** Credits for new accounts

### Google (Gemini)
- **Website:** https://makersuite.google.com/app/apikey
- **Cost:** Free tier available
- **Free:** 60 requests/minute free tier

### Groq (Fast Inference)
- **Website:** https://console.groq.com/keys
- **Cost:** FREE (generous free tier)
- **Free:** Yes! Very fast inference

### Ollama (Local Models)
- **Website:** https://ollama.ai
- **Cost:** FREE (runs on your computer)
- **Free:** 100% free, private, unlimited

---

## ⚡ Quick Start (1 Minute)

```bash
# 1. Clone and enter directory
git clone <your-repo>
cd statmate-ai

# 2. One-command setup
make quickstart

# 3. Edit .env and add your API key
nano .env
# Add: OPENAI_API_KEY=sk-...

# 4. Start!
make dev     # Terminal 1
make ui      # Terminal 2

# 5. Open browser
# http://localhost:8501
```

---

## 🐛 Troubleshooting

### API not starting

```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
pkill -f "statmate/api"

# Try again
make api
```

### UI not connecting to API

```bash
# Make sure API is running
curl http://localhost:8000/health

# Should return: {"status": "healthy", ...}
```

### No models available

**DEV mode:**
```bash
# Check .env has API key
cat .env | grep API_KEY

# Should show: OPENAI_API_KEY=sk-...
```

**PROD mode:**
- Make sure you entered credentials in the UI
- Check API logs for errors

### Database errors

```bash
# Reset database
make db-reset

# This will delete all data and reinitialize
```

---

## 📚 Next Steps

After setup:

1. **Upload sample data** - Try the sample datasets in `data/uploads/`
2. **Run your first analysis** - Upload CSV/Excel file
3. **Explore model options** - Click "🤖 Model Config" button
4. **Read the docs** - Check `docs/` folder for detailed guides

---

## 🎓 Learning Resources

- **[DEV vs PROD Guide](docs/DEV_VS_PROD_GUIDE.md)** - Detailed mode comparison
- **[API Reference](docs/API_REFERENCE.md)** - API endpoints
- **[Quick Start](docs/QUICK_START.md)** - Usage examples
- **[Architecture](docs/ARCHITECTURE_PROPOSAL.md)** - System design

---

## 🆘 Getting Help

**Common issues:**
- Check `make status` for system health
- Review logs in `data/logs/`
- Ensure API is running before starting UI
- Verify API keys are valid

**Still stuck?**
- Check GitHub Issues
- Review documentation in `docs/`
- Run `make help` for all commands

---

## ✅ Checklist

Before first use:

- [ ] Installed dependencies (`make install`)
- [ ] Created `.env` file (`make setup-env`)
- [ ] Added at least one API key
- [ ] Initialized database (`make db-init`)
- [ ] Started API (`make api` or `make dev`)
- [ ] Started UI (`make ui`)
- [ ] Opened http://localhost:8501
- [ ] Saw green "✅ API Connected" banner

---

**🎉 You're ready to analyze! Happy StatMating! 📊**

