# 🔐 Credentials System - Implementation Summary

## Overview

StatmateAI now supports two distinct credential management modes to ensure security and flexibility:

- **🔧 DEV Mode**: Uses `.env` file with developer's API keys (never committed to git)
- **🚀 PROD Mode**: Users provide their own credentials via secure UI

---

## ✅ What's Been Implemented

### 1. Security Infrastructure

- ✅ Updated `.gitignore` to exclude `.env` and sensitive files
- ✅ Created `.env.example` template (safe to commit)
- ✅ API keys never stored in database or logs
- ✅ Session-based credential management in PROD mode

### 2. Makefile Commands

Created convenient commands for both modes:

```bash
make dev          # Development mode (uses .env)
make prod         # Production mode (user credentials)
make setup-env    # Create .env from template
make status       # Check configuration status
```

### 3. API Endpoints

Added new endpoints for credential management:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/models/environment` | GET | Get current mode (dev/prod) |
| `/api/v1/models/credentials` | POST | Set user credentials (prod only) |
| `/api/v1/models/available` | GET | List available models |
| `/api/v1/models/current` | GET | Get current model config |

### 4. UI Components

#### `credentials.py`
New component handling:
- Mode detection (DEV vs PROD)
- Credential input form (PROD mode)
- Session management
- Security banners

#### Updated `app.py`
- Detects environment mode
- Shows credential setup page in PROD
- Displays mode banner
- "Change Keys" button in PROD

#### Updated `model_selector.py`
- Mode-aware help messages
- Different instructions for DEV vs PROD
- Context-sensitive error messages

### 5. Documentation

Created comprehensive guides:

- **`SETUP_GUIDE.md`** - Quick start for both modes
- **`docs/DEV_VS_PROD_GUIDE.md`** - Detailed comparison and best practices
- **`.env.example`** - Well-documented template
- **`Makefile`** - Self-documenting with help command

---

## 🔒 Security Features

### DEV Mode
- API keys in `.env` file (git-ignored)
- Local development only
- No network exposure of keys
- Per-developer credentials

### PROD Mode
- Users provide their own keys
- Stored in memory only (not persisted)
- Session-based (cleared on restart)
- No shared credentials
- Each user controls their own quota

---

## 🎯 Usage Examples

### For Developers (DEV Mode)

```bash
# 1. Setup
make setup-env
nano .env  # Add OPENAI_API_KEY=sk-...

# 2. Run
make dev    # Terminal 1
make ui     # Terminal 2

# 3. Use
# API loads keys from .env automatically
# UI shows "🔧 DEV MODE" banner
```

### For End Users (PROD Mode)

```bash
# 1. Admin deploys
export ENVIRONMENT=production
make prod

# 2. User visits http://localhost:8501
# 3. Sees credential setup page
# 4. Enters their own API keys
# 5. Clicks "Configure & Start"
# 6. Keys stored in memory for session
# 7. Can change via "⚙️ Change Keys" button
```

---

## 🏗️ Architecture

### DEV Mode Flow

```
.env file
  ↓ (read at startup)
settings.py
  ↓ (creates model config)
model_factory.py
  ↓ (initializes providers)
Ready to analyze
```

### PROD Mode Flow

```
User opens UI
  ↓
Credential form appears
  ↓ (user enters keys)
POST /api/v1/models/credentials
  ↓ (backend receives)
settings updated in memory
  ↓ (reinitialize)
model_factory reloaded
  ↓
Ready to analyze
```

---

## 📁 Files Modified/Created

### Created
- `.env.example` - Template configuration
- `Makefile` - Automation scripts
- `statmate/ui/components/credentials.py` - Credential UI
- `docs/DEV_VS_PROD_GUIDE.md` - Detailed guide
- `SETUP_GUIDE.md` - Quick start
- `CREDENTIALS_SYSTEM.md` - This file

### Modified
- `.gitignore` - Added `.env` and sensitive files
- `statmate/api/routes/models.py` - Added credential endpoints
- `statmate/ui/app.py` - Integrated credential system
- `statmate/ui/components/model_selector.py` - Mode-aware messages

---

## 🧪 Testing

### Test DEV Mode

```bash
# 1. Create .env with test key
echo "ENVIRONMENT=development" > .env
echo "OPENAI_API_KEY=sk-test-key" >> .env

# 2. Start
make dev

# 3. Check UI shows "DEV MODE" banner
curl http://localhost:8000/api/v1/models/environment
# Should return: {"environment": "development", ...}
```

### Test PROD Mode

```bash
# 1. Set production
export ENVIRONMENT=production

# 2. Start
make prod

# 3. Visit UI - should see credential form
# 4. Enter test credentials
# 5. Should redirect to main app
# 6. Check "PROD MODE" banner appears
```

---

## 🚦 Migration Path

### Existing Deployments

If you're already running StatmateAI:

**Option 1: Stay in DEV mode (Current behavior)**
```bash
# Nothing changes if ENVIRONMENT=development (default)
make dev
```

**Option 2: Switch to PROD mode**
```bash
# 1. Stop current instance
^C

# 2. Remove API keys from .env
nano .env  # Keep only ENVIRONMENT=production

# 3. Restart in PROD
make prod

# 4. Users enter credentials via UI
```

---

## 🔧 Configuration Options

### Environment Variables

```ini
# Required
ENVIRONMENT=development  # or 'production'

# DEV mode (in .env)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
OLLAMA_ENABLED=true

# PROD mode (via UI)
# Users enter above keys through web form
```

### Runtime Settings

```python
# Check mode programmatically
from config.settings import settings

if settings.ENVIRONMENT == 'production':
    # PROD behavior
else:
    # DEV behavior
```

---

## 🎨 UI Screenshots (Text Description)

### DEV Mode UI
```
┌──────────────────────────────────────┐
│ 📊 StatmateAI                        │
│ AI-driven statistical analysis...    │
│                                      │
│ ✅ API Connected (v0.1.0)  [🤖 Model Config] │
│ ℹ️  🔧 DEV MODE - Using .env API Keys │
├──────────────────────────────────────┤
│ [1️⃣ Upload] [2️⃣ Analyze] [3️⃣ Results] │
└──────────────────────────────────────┘
```

### PROD Mode UI (Before Credentials)
```
┌──────────────────────────────────────┐
│ 📊 StatmateAI                        │
│                                      │
│ ✅ API Connected                     │
│ ⚠️  🔐 PROD MODE - Credentials Required │
├──────────────────────────────────────┤
│ ### 🔐 API Credentials Setup         │
│ Please provide your AI model API... │
│                                      │
│ [OpenAI] [Anthropic] [Google] [...]│
│                                      │
│ OpenAI API Key: [................] │
│                                      │
│     [🚀 Configure & Start]           │
└──────────────────────────────────────┘
```

### PROD Mode UI (After Credentials)
```
┌──────────────────────────────────────┐
│ 📊 StatmateAI                        │
│                                      │
│ ✅ API Connected   [🤖 Model Config]  │
│ ✅ 🔐 PROD MODE - Active  [⚙️ Change Keys] │
├──────────────────────────────────────┤
│ [1️⃣ Upload] [2️⃣ Analyze] [3️⃣ Results] │
└──────────────────────────────────────┘
```

---

## 🔐 Security Checklist

Before deploying:

- [x] `.env` in `.gitignore`
- [x] `.env.example` has no real keys
- [x] PROD mode doesn't read `.env` keys
- [x] User keys stored in memory only
- [x] Keys never logged
- [x] Keys never saved to database
- [x] Session-based security
- [ ] TODO: Add HTTPS in production
- [ ] TODO: Add authentication layer
- [ ] TODO: Add rate limiting
- [ ] TODO: Add key encryption

---

## 📊 Benefits

### For Developers
✅ Fast local setup with `.env`
✅ Personal API keys and quotas
✅ No credential sharing needed
✅ Easy testing and development

### For Users
✅ Bring your own API keys
✅ Control your own costs
✅ Privacy (keys never shared)
✅ No account needed on our end

### For Deployers
✅ Secure by default
✅ No credential management burden
✅ Users self-service credentials
✅ Reduced liability

---

## 🚀 Future Enhancements

Potential improvements:

1. **Encrypted Storage** (Optional)
   - Store encrypted keys in database
   - User-specific passwords
   - Better for long-running sessions

2. **Authentication**
   - User accounts
   - Per-user credential storage
   - Usage tracking

3. **Key Management**
   - Multiple keys per provider
   - Rotation support
   - Quota monitoring

4. **Team Features**
   - Shared team credentials
   - Role-based access
   - Usage reporting

---

## 📞 Support

**Issues?**
- Check `make status` for configuration
- Review `docs/DEV_VS_PROD_GUIDE.md`
- Check API logs in `data/logs/`
- Verify API is running before UI

**Questions?**
- See `SETUP_GUIDE.md` for quick start
- Run `make help` for all commands
- Check `.env.example` for configuration options

---

## ✅ Summary

The credential system is now complete with:

1. ✅ Secure DEV mode (`.env` based)
2. ✅ Secure PROD mode (UI-based)
3. ✅ Easy Makefile commands
4. ✅ Comprehensive documentation
5. ✅ Mode-aware UI components
6. ✅ API endpoints for credentials
7. ✅ `.gitignore` protection
8. ✅ Template `.env.example`

**Result:** StatmateAI can now be safely developed locally AND deployed for end users without credential exposure risks! 🎉🔒

---

**Built with security in mind! 🛡️**

