#!/bin/bash
# Development startup script for StatmateAI

set -e

echo "🚀 Starting StatmateAI Development Environment"
echo "=============================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it with your API keys."
        echo "   Especially set OPENAI_API_KEY"
        exit 1
    else
        echo "❌ .env.example not found!"
        exit 1
    fi
fi

# Check OPENAI_API_KEY
if ! grep -q "OPENAI_API_KEY=.*[a-zA-Z0-9]" .env; then
    echo "⚠️  OPENAI_API_KEY not set in .env file"
    echo "   Please edit .env and add your OpenAI API key"
    exit 1
fi

# Initialize database if needed
if [ ! -f database/statmate.db ]; then
    echo "📦 Initializing database..."
    python scripts/init_db.py --seed
    echo "✅ Database initialized with sample data"
else
    echo "✅ Database already exists"
fi

# Create data directories if needed
mkdir -p data/uploads data/results data/logs
echo "✅ Data directories ready"

echo ""
echo "🎯 Starting FastAPI backend..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

# Start the API server
python statmate/api/main.py

