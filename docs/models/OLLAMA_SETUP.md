# Ollama Setup Guide - Run AI Models Locally (FREE!)

## What is Ollama?

Ollama allows you to run powerful AI models **locally on your machine** - completely **free**, **private**, and with **no API keys required**. Perfect for StatmateAI statistical analysis!

### Benefits of Using Ollama

✅ **100% Free** - No API costs, no rate limits  
✅ **Private & Secure** - Your data never leaves your machine  
✅ **No Internet Required** - Works offline after initial model download  
✅ **Fast** - Local execution means low latency  
✅ **Reasoning Models** - DeepSeek-R1 competes with OpenAI's o1!  

---

## Step-by-Step Installation

### Step 1: Install Ollama

#### macOS
```bash
brew install ollama
```

**Alternative:** Download from [ollama.ai](https://ollama.ai) and install the `.dmg` file.

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
Download the installer from [ollama.ai](https://ollama.ai) and run it.

---

### Step 2: Start Ollama Service

#### macOS/Linux
```bash
# Start Ollama in the background
ollama serve &
```

**Or**, if you installed via Homebrew on macOS, it may start automatically as a service.

#### Windows
Ollama starts automatically after installation. Check the system tray for the Ollama icon.

---

### Step 3: Pull a Model

For **statistical analysis**, we recommend **reasoning models**:

#### Recommended: DeepSeek-R1 8B (Best for StatmateAI)
```bash
ollama pull deepseek-r1:8b
```
- **Size**: ~5 GB
- **RAM Required**: 8 GB
- **Quality**: Excellent reasoning capabilities (competes with GPT-4o)
- **Speed**: Fast on modern hardware

#### Alternative Models

**For more power** (if you have 16GB+ RAM):
```bash
ollama pull deepseek-r1:14b
```

**For general use** (no reasoning, but fast):
```bash
ollama pull llama3.1:8b
```

**List available models:**
```bash
ollama list
```

---

### Step 4: Test Your Model

Verify the model works:
```bash
ollama run deepseek-r1:8b "What is the p-value in a t-test?"
```

You should see a response about statistical significance testing.

---

### Step 5: Configure StatmateAI to Use Ollama

#### Option A: Edit `.env` File (Recommended)

1. **Open your `.env` file** in the project root:
   ```bash
   nano .env
   # or
   code .env
   ```

2. **Add/Update these lines:**
   ```bash
   # Enable Ollama
   OLLAMA_ENABLED=True
   OLLAMA_BASE_URL=http://localhost:11434/v1
   OLLAMA_DEFAULT_MODEL=deepseek-r1:8b
   
   # Set Ollama as default provider
   DEFAULT_MODEL_PROVIDER=ollama
   DEFAULT_MODEL_NAME=deepseek-r1:8b
   
   # Optional: Keep OpenAI as fallback
   # OPENAI_API_KEY=sk-your-key-here
   ```

3. **Save the file** and exit (Ctrl+X, then Y in nano)

#### Option B: Use PROD Mode (UI Configuration)

If you're running in **production mode** (`make prod`), you can configure Ollama through the UI:

1. Click **"⚙️ Change Keys"** in the app
2. Enable **"Use Ollama (Local)"**
3. Set model to **"deepseek-r1:8b"**
4. Click **"Configure & Start"**

---

### Step 6: Start StatmateAI

#### Development Mode (with .env)
```bash
# Stop any running processes first
make kill

# Start StatmateAI
make dev

# In another terminal, start the UI
make ui
```

#### Production Mode (UI credentials)
```bash
make kill
make prod
```

---

### Step 7: Verify It's Working

1. **Open your browser** to `http://localhost:8501`
2. **Check the banner** - you should see:
   - ✅ Environment: DEV
   - 🤖 Default Model: **deepseek-r1:8b**
   - 📍 Provider: **OLLAMA**
3. **Upload a dataset** or use a sample
4. **Run an analysis** - it should complete successfully!

---

## Troubleshooting

### Issue: "Connection refused" or "Cannot connect to Ollama"

**Solution:**
```bash
# Check if Ollama is running
pgrep ollama

# If not running, start it
ollama serve &

# Verify it's accessible
curl http://localhost:11434/api/version
```

---

### Issue: "Model not found"

**Solution:**
```bash
# List installed models
ollama list

# Pull the model if missing
ollama pull deepseek-r1:8b

# Verify the model name matches your .env
cat .env | grep OLLAMA_DEFAULT_MODEL
```

---

### Issue: "Out of memory" or slow performance

**Solution:**

Try a smaller model:
```bash
ollama pull deepseek-r1:8b  # Instead of 14b or 70b
```

Or use a non-reasoning model:
```bash
ollama pull llama3.1:8b
```

Update `.env`:
```bash
OLLAMA_DEFAULT_MODEL=llama3.1:8b
DEFAULT_MODEL_NAME=llama3.1:8b
```

---

### Issue: StatmateAI still using OpenAI

**Solution:**

1. **Check your `.env`** - make sure:
   ```bash
   DEFAULT_MODEL_PROVIDER=ollama  # Not "openai"
   DEFAULT_MODEL_NAME=deepseek-r1:8b
   ```

2. **Restart the app:**
   ```bash
   make kill
   make dev
   ```

3. **Verify in the UI** - check the banner shows "Provider: OLLAMA"

---

## Model Recommendations for Statistical Analysis

### Best Overall: DeepSeek-R1 Series
- **deepseek-r1:8b** - Perfect balance of speed and quality (Recommended!)
- **deepseek-r1:14b** - Higher quality, needs 16GB+ RAM
- **deepseek-r1:70b** - Best quality, needs 32GB+ RAM

### Fast & Efficient (No Reasoning)
- **llama3.1:8b** - Good general model
- **qwen2.5:7b** - Efficient, good at math
- **mistral:7b** - Fast and capable

### Compare Models
```bash
# Try different models
ollama run deepseek-r1:8b "Explain a t-test"
ollama run llama3.1:8b "Explain a t-test"
```

---

## Advanced Configuration

### Custom Ollama Host

If Ollama is running on a different machine or port:

```bash
# In .env
OLLAMA_BASE_URL=http://192.168.1.100:11434/v1
```

### Multiple Models

You can have multiple models installed and switch between them:

```bash
# Install multiple models
ollama pull deepseek-r1:8b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b

# List all installed models
ollama list
```

In the UI, you can select different models per analysis in the **"AI Model Selection"** section.

---

## Performance Tips

### GPU Acceleration
- Ollama automatically uses your GPU if available (NVIDIA/AMD/Apple Silicon)
- No configuration needed!

### RAM Requirements
- **8GB RAM**: Use 7B-8B models (deepseek-r1:8b, llama3.1:8b)
- **16GB RAM**: Use 13B-14B models (deepseek-r1:14b)
- **32GB+ RAM**: Use 70B models (deepseek-r1:70b)

### Speed Optimization
```bash
# Use smaller context window (faster inference)
# In .env:
MODEL_MAX_TOKENS=2048  # Instead of 4096+
```

---

## Uninstalling Ollama

If you want to remove Ollama:

### macOS
```bash
brew uninstall ollama
rm -rf ~/.ollama
```

### Linux
```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /usr/local/bin/ollama
sudo rm -rf /usr/share/ollama
rm -rf ~/.ollama
```

### Windows
Use "Add or Remove Programs" to uninstall Ollama.

---

## Comparison: Ollama vs Cloud Providers

| Feature | Ollama (Local) | OpenAI | Anthropic | Google |
|---------|----------------|--------|-----------|--------|
| **Cost** | FREE ⭐ | $$ | $$$ | $ |
| **Privacy** | 100% Private ⭐ | Cloud | Cloud | Cloud |
| **Speed** | Fast (local) ⭐ | Fast | Medium | Fast |
| **Internet** | Not required ⭐ | Required | Required | Required |
| **Setup** | One-time | API key | API key | API key |
| **Quality** | Excellent ⭐ | Excellent | Excellent | Excellent |

**Winner:** Ollama for most users! 🏆

---

## Getting Help

### Check Ollama Status
```bash
ollama list              # List installed models
ollama ps                # Show running models
curl http://localhost:11434/api/version  # Check service
```

### StatmateAI Logs
```bash
# Check API logs
tail -f data/logs/*.log

# Check model configuration
make dev | grep -i ollama
```

### Community Support
- Ollama GitHub: https://github.com/ollama/ollama
- StatmateAI Issues: [Your GitHub issues page]

---

## Quick Reference Card

```bash
# Installation
brew install ollama                    # macOS
curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Start Ollama
ollama serve &

# Get recommended model
ollama pull deepseek-r1:8b

# Configure StatmateAI (.env)
OLLAMA_ENABLED=True
OLLAMA_DEFAULT_MODEL=deepseek-r1:8b
DEFAULT_MODEL_PROVIDER=ollama
DEFAULT_MODEL_NAME=deepseek-r1:8b

# Run StatmateAI
make kill
make dev    # In terminal 1
make ui     # In terminal 2

# Test
curl http://localhost:11434/api/version
ollama list
```

---

## Next Steps

1. ✅ Install Ollama
2. ✅ Pull deepseek-r1:8b model
3. ✅ Configure `.env`
4. ✅ Start StatmateAI with `make dev`
5. ✅ Run your first analysis!

**Enjoy free, private, local AI-powered statistical analysis!** 🎉

---

*Last updated: 2025-10-17*

