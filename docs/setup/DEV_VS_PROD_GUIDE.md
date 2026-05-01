# Development vs Production Mode Guide

## Overview

StatmateAI supports two distinct operating modes to handle AI model credentials securely:

- **Development Mode** 🔧 - API keys stored in `.env` file (never committed to git)
- **Production Mode** 🚀 - Users provide their own credentials through the UI

This guide explains how to use each mode and why this separation is important.

---

## 🔒 Security First

### Why Two Modes?

**The Problem:**
- API keys are sensitive credentials that should NEVER be committed to git
- Sharing `.env` files is insecure
- Production users should provide their own keys, not use shared credentials

**The Solution:**
- **DEV Mode**: Developers use their own API keys from `.env` (git-ignored)
- **PROD Mode**: End users enter credentials through a secure UI form

---

## 🔧 Development Mode

### What is it?

Development mode is for local development and testing. API keys are loaded from the `.env` file.

### When to use it?

- Local development on your machine
- Testing new features
- Running the app for personal use
- Contributing to the project

### Setup

#### 1. Create `.env` file

```bash
# Quick setup
make setup-env

# Or manually
cp .env.example .env
```

#### 2. Add your API keys to `.env`

```ini
# Required: At least one provider
OPENAI_API_KEY=sk-proj-...your-key-here...

# Optional: Additional providers
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...

# Optional: Local models (FREE!)
OLLAMA_ENABLED=true
OLLAMA_DEFAULT_MODEL=deepseek-r1:8b
```

#### 3. Set environment to development

```ini
ENVIRONMENT=development
DEBUG=true
```

#### 4. Run the application

```bash
# Easy way
make dev

# Or manual way
python statmate/api/main.py  # Terminal 1
streamlit run statmate/ui/app.py  # Terminal 2
```

### What happens in DEV mode?

```
┌─────────────────────────────────────┐
│  .env file (git-ignored)            │
│  ├─ OPENAI_API_KEY=sk-...           │
│  ├─ ANTHROPIC_API_KEY=sk-ant-...    │
│  └─ ENVIRONMENT=development          │
└─────────────┬───────────────────────┘
              │ (loaded at startup)
              ▼
┌─────────────────────────────────────┐
│  FastAPI Backend                    │
│  ├─ Loads credentials from .env     │
│  ├─ Initializes model factory       │
│  └─ Ready to analyze data           │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Streamlit UI                       │
│  ├─ Shows "DEV MODE" banner         │
│  ├─ No credential input required    │
│  └─ Ready to use                    │
└─────────────────────────────────────┘
```

### UI Indicators

When running in DEV mode, you'll see:
- 🔧 **DEV MODE** - Using credentials from .env file

---

## 🚀 Production Mode

### What is it?

Production mode is for deploying the app for end users. Users must provide their own API credentials through the UI.

### When to use it?

- Deploying to a server for multiple users
- Sharing the app with non-technical users
- Running in a cloud environment
- When each user should use their own API quota

### Setup

#### 1. Set environment to production

```bash
# Option 1: Environment variable
export ENVIRONMENT=production

# Option 2: In .env file (but don't include API keys!)
echo "ENVIRONMENT=production" > .env
echo "DEBUG=false" >> .env
```

#### 2. Do NOT include API keys in `.env`

```ini
# .env file for PROD (minimal)
ENVIRONMENT=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=postgresql://...  # Use PostgreSQL in prod
SECRET_KEY=your-secure-random-key
```

#### 3. Run in production mode

```bash
# Using Makefile
make prod

# Or manually
export ENVIRONMENT=production
python statmate/api/main.py &
streamlit run statmate/ui/app.py
```

### What happens in PROD mode?

```
┌─────────────────────────────────────┐
│  User opens Streamlit UI            │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Credential Setup Page               │
│  ┌───────────────────────────────┐  │
│  │ Please provide your API keys  │  │
│  │                               │  │
│  │ OpenAI: [.................]   │  │
│  │ Anthropic: [...............]  │  │
│  │ Google: [..................]  │  │
│  │ Groq: [...................]   │  │
│  │ Ollama: [☐] Enable           │  │
│  │                               │  │
│  │      [Configure & Start]      │  │
│  └───────────────────────────────┘  │
└─────────────┬───────────────────────┘
              │ (POST /api/v1/models/credentials)
              ▼
┌─────────────────────────────────────┐
│  FastAPI Backend                    │
│  ├─ Receives user credentials       │
│  ├─ Stores in memory (not disk!)    │
│  ├─ Reinitializes model factory     │
│  └─ Ready to analyze                │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Analysis Ready                     │
│  ├─ User can upload data            │
│  ├─ Run statistical tests           │
│  └─ View results                    │
└─────────────────────────────────────┘
```

### UI Flow

1. **First Visit** - User sees credential setup page
2. **Enter Credentials** - Fill in at least one API provider
3. **Configure** - Click "Configure & Start"
4. **Success** - Redirected to main analysis page
5. **Session** - Credentials stored in memory for session
6. **Change Keys** - Can reconfigure anytime via "⚙️ Change Keys" button

### UI Indicators

When running in PROD mode, you'll see:
- 🔐 **PROD MODE** - User credentials active
- ⚙️ **Change Keys** button to reconfigure

---

## 🔐 Security Considerations

### DEV Mode Security

✅ **DO:**
- Keep `.env` file local only (it's in `.gitignore`)
- Use personal API keys with limited quota
- Never commit `.env` to version control
- Rotate keys if exposed

❌ **DON'T:**
- Share your `.env` file
- Commit API keys to git
- Use production API keys in dev

### PROD Mode Security

✅ **DO:**
- Users provide their own API keys
- Keys stored only in memory (runtime)
- Keys never written to disk
- Each user uses their own quota
- Use HTTPS in production
- Implement authentication (future)

❌ **DON'T:**
- Store user API keys in database
- Log API keys
- Share keys between users
- Include keys in error messages

### Credential Storage

| Mode | Storage Location | Persistence | Security |
|------|-----------------|-------------|----------|
| **DEV** | `.env` file | Until manually changed | Git-ignored |
| **PROD** | Memory only | Session only | Never saved |

---

## 📋 Makefile Commands

### Quick Reference

```bash
# Development
make dev              # Run in DEV mode (uses .env)
make api              # Run only API (DEV)
make ui               # Run only UI

# Production
make prod             # Run in PROD mode (user credentials)
make prod-api         # Run only API (PROD)
make prod-ui          # Run only UI (PROD)

# Setup
make install          # Install dependencies
make setup-env        # Create .env from template
make db-init          # Initialize database
make db-seed          # Add sample data

# Utilities
make clean            # Clean temporary files
make status           # Check system status
make help             # Show all commands
```

### Detailed Commands

#### `make dev`

Runs in development mode:
- Checks for `.env` file
- Validates API keys are configured
- Initializes database if needed
- Starts FastAPI backend
- ⚠️ You need to start UI separately: `make ui`

#### `make prod`

Runs in production mode:
- Sets `ENVIRONMENT=production`
- Starts API without loading `.env` keys
- Starts UI which prompts for credentials
- Users enter keys through web interface

#### `make setup-env`

Creates `.env` file from `.env.example`:
```bash
make setup-env
# Edit .env and add your keys
nano .env
```

#### `make status`

Check current system status:
```bash
make status

# Output:
# System Status:
# Environment:
#   ✓ .env file exists
# Database:
#   ✓ Database exists
# API Keys Configured:
#   ✓ OpenAI
#   ✗ Anthropic
#   ✗ Google
#   ✗ Groq
#   ✓ Ollama (Local)
```

---

## 🆓 Using Free Local Models (Ollama)

Both DEV and PROD modes support **Ollama** for completely free, private, local AI models!

### Setup Ollama

```bash
# 1. Install Ollama
# Visit: https://ollama.ai

# 2. Pull a reasoning model
ollama pull deepseek-r1:8b

# 3. Start Ollama server
ollama serve
```

### Configure for DEV Mode

Add to `.env`:
```ini
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_DEFAULT_MODEL=deepseek-r1:8b
```

### Configure for PROD Mode

In the credential setup UI:
1. Go to "Ollama (Local)" tab
2. Check "Enable Ollama"
3. Set URL: `http://localhost:11434/v1`
4. Set Model: `deepseek-r1:8b`
5. Click "Configure & Start"

### Why Ollama?

- 🆓 **Completely FREE** - No API costs
- 🔒 **Private** - Data never leaves your machine
- ⚡ **Fast** - Local inference
- 🧠 **Powerful** - DeepSeek-R1 excellent for reasoning
- 📦 **Easy** - One command to install

---

## 🚦 Mode Comparison

| Feature | Development Mode | Production Mode |
|---------|------------------|-----------------|
| **API Keys Source** | `.env` file | User input (UI) |
| **Setup Complexity** | Easy (one-time) | Minimal (per user) |
| **Security** | Local only | Session-based |
| **Best For** | Developers | End users |
| **Credential Sharing** | No sharing | Each user owns keys |
| **Persistence** | File-based | Memory only |
| **Startup** | `make dev` | `make prod` |
| **Environment Var** | `ENVIRONMENT=development` | `ENVIRONMENT=production` |

---

## 🐛 Troubleshooting

### Issue: "No models configured" in DEV mode

**Solution:**
```bash
# Check if .env exists
cat .env | grep API_KEY

# Create if missing
make setup-env

# Add your key
echo "OPENAI_API_KEY=sk-..." >> .env

# Restart
make dev
```

### Issue: Stuck on credential page in PROD mode

**Solution:**
```bash
# Check environment
curl http://localhost:8000/api/v1/models/environment

# Should return: {"environment": "production", ...}

# If wrong, fix and restart:
export ENVIRONMENT=production
make prod-api
```

### Issue: Keys not working after entering in UI

**Possible causes:**
1. Invalid API key format
2. API key has no quota/credits
3. Network issues reaching AI provider

**Solution:**
```bash
# Check API logs
tail -f data/logs/*.log

# Test key manually
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_KEY"
```

### Issue: Can't switch from PROD to DEV mode

**Solution:**
```bash
# Stop all processes
pkill -f "statmate"

# Set environment
export ENVIRONMENT=development

# Restart
make dev
```

---

## 📚 Related Documentation

- [Quick Start Guide](QUICK_START.md) - First-time setup
- [Architecture Proposal](ARCHITECTURE_PROPOSAL.md) - System design
- [API Reference](API_REFERENCE.md) - API endpoints
- [Model Configuration](MODEL_CONFIGURATION.md) - AI model setup

---

## 🔄 Migration Between Modes

### DEV → PROD

```bash
# 1. Stop dev server
^C  # Ctrl+C

# 2. Remove API keys from .env (or create new .env)
cat > .env << EOF
ENVIRONMENT=production
DEBUG=false
EOF

# 3. Start in prod mode
make prod

# 4. Users enter credentials via UI
```

### PROD → DEV

```bash
# 1. Stop prod server
^C  # Ctrl+C

# 2. Add your API keys to .env
make setup-env
nano .env  # Add keys

# 3. Start in dev mode
make dev
```

---

## ✅ Best Practices

### For Developers

1. **Always use DEV mode** for local development
2. **Never commit `.env`** to git (already in `.gitignore`)
3. **Use `.env.example`** as template for team members
4. **Rotate keys** if accidentally exposed
5. **Use Ollama** for free local testing

### For Deployment

1. **Always use PROD mode** for hosted instances
2. **Never include API keys** in deployment configs
3. **Use HTTPS** in production
4. **Implement rate limiting** to prevent abuse
5. **Add authentication** for multi-user deployments
6. **Use PostgreSQL** instead of SQLite
7. **Monitor usage** and set quotas

### For End Users

1. **Use your own API keys** (don't share)
2. **Start with Ollama** (free and private)
3. **Monitor your usage** on provider dashboard
4. **Set spending limits** on API accounts
5. **Your keys = Your control** over costs

---

## 🎯 Quick Decision Guide

**Choose DEV mode if:**
- ✅ You're developing/testing locally
- ✅ You're the only user
- ✅ You want persistent configuration
- ✅ You trust your local environment

**Choose PROD mode if:**
- ✅ Deploying for multiple users
- ✅ Each user should use their own keys
- ✅ Running on a shared server
- ✅ You want session-based security

---

**Happy analyzing! 📊🚀**

