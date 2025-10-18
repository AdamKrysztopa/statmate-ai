# ⚡ Quick Reference Card

## 🚀 Start/Stop Commands

```bash
# Development Mode (uses .env API keys)
make dev              # Start API
make ui               # Start UI (in another terminal)
make kill             # Stop both API and UI

# Production Mode (users enter credentials via UI)
make prod             # Start both API and UI
make kill             # Stop everything

# First Time Setup
make quickstart       # Complete setup wizard
```

## 📁 Key Files

| File | Purpose | Commit to Git? |
|------|---------|----------------|
| `.env.example` | Template config | ✅ YES |
| `.env` | Your actual API keys | ❌ NO (git-ignored) |
| `Makefile` | Commands | ✅ YES |
| `database/statmate.db` | Local database | ❌ NO (git-ignored) |

## 🔐 Environment Modes

| Mode | API Keys From | Command | Use Case |
|------|--------------|---------|----------|
| **DEV** | `.env` file | `make dev` | Local development |
| **PROD** | User enters in UI | `make prod` | Deployment |

## 🔑 Get API Keys

| Provider | URL | Cost |
|----------|-----|------|
| OpenAI | platform.openai.com/api-keys | ~$0.01/1K tokens |
| Anthropic | console.anthropic.com | ~$0.003/1K tokens |
| Google | makersuite.google.com/app/apikey | Free tier |
| Groq | console.groq.com/keys | FREE |
| Ollama | ollama.ai | FREE (local) |

## 📋 Common Tasks

```bash
# Setup
make install          # Install dependencies
make setup-env        # Create .env file
make db-init          # Initialize database
make db-seed          # Add sample data

# Running
make dev              # DEV mode
make prod             # PROD mode
make api              # API only
make ui               # UI only
make kill             # Stop all processes

# Maintenance
make status           # Check configuration
make clean            # Clean temp files
make db-reset         # Reset database ⚠️
make help             # Show all commands
```

## 🆓 Free Options

**Ollama (Recommended for Free Usage)**
```bash
# Install
brew install ollama  # macOS
# or visit: https://ollama.ai

# Get model
ollama pull deepseek-r1:8b

# Start
ollama serve

# Configure in .env
OLLAMA_ENABLED=true
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| API not starting | `make status` → Check .env has keys |
| UI can't connect | Ensure API is running first |
| No models available | Add API key to .env or enter via UI |
| Port already in use | `make kill` then restart |
| Processes won't stop | `make kill` (force kills if needed) |

## 🌐 URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit UI | http://localhost:8501 | Main interface |
| API Docs | http://localhost:8000/docs | Interactive API docs |
| API Health | http://localhost:8000/health | Check API status |

## 📊 Workflow

1. **Upload** → CSV/Excel file
2. **Analyze** → Click "Run Stat Test"
3. **Results** → View p-values, summary, and details

## 🔒 Security Rules

✅ **DO:**
- Keep `.env` local (it's git-ignored)
- Use `make prod` for deployments
- Let users provide their own keys

❌ **DON'T:**
- Commit `.env` to git
- Share API keys
- Include keys in code

## 📚 Documentation

- `SETUP_GUIDE.md` - Getting started
- `docs/DEV_VS_PROD_GUIDE.md` - Detailed mode comparison
- `CREDENTIALS_SYSTEM.md` - Implementation details
- `README.md` - Full documentation

## ⌨️ One-Liners

```bash
# Complete first-time setup
make quickstart && nano .env && make dev

# Reset everything and start fresh
make clean-all && make db-init db-seed

# Check if everything is configured
make status

# Quick start DEV mode
cp .env.example .env && nano .env && make dev
```

---

**Quick tip:** Run `make help` anytime to see all available commands!

