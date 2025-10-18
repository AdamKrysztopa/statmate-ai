#!/usr/bin/env python3
"""List all available models from configured providers.

This script queries each AI provider's API to show what models are actually available.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_openai_models():
    """List available OpenAI models."""
    print('\n' + '=' * 60)
    print('OpenAI Models')
    print('=' * 60)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('❌ OPENAI_API_KEY not set in .env')
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models = client.models.list()

        # Filter for chat models
        chat_models = [m for m in models.data if 'gpt' in m.id or 'o1' in m.id or 'o3' in m.id]
        chat_models = sorted(chat_models, key=lambda x: x.id)

        print(f'✅ Found {len(chat_models)} chat models:')
        for model in chat_models:
            print(f'  • {model.id}')

    except Exception as e:
        print(f'❌ Error: {e}')


def check_anthropic_models():
    """Show known Anthropic models (they don't have a list endpoint)."""
    print('\n' + '=' * 60)
    print('Anthropic Models (from documentation)')
    print('=' * 60)

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print('❌ ANTHROPIC_API_KEY not set in .env')
        return

    print('✅ Anthropic models (check docs.anthropic.com for latest):')
    known_models = [
        'claude-3-7-sonnet-20250219',
        'claude-3-5-sonnet-latest',
        'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-latest',
        'claude-3-5-haiku-20241022',
        'claude-3-opus-latest',
        'claude-3-opus-20240229',
    ]
    for model in known_models:
        print(f'  • {model}')

    print('\n💡 Verify at: https://docs.anthropic.com/en/docs/about-claude/models')


def check_google_models():
    """List available Google/Gemini models."""
    print('\n' + '=' * 60)
    print('Google (Gemini) Models')
    print('=' * 60)

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print('❌ GOOGLE_API_KEY not set in .env')
        return

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        models = genai.list_models()
        chat_models = [m for m in models if 'generateContent' in m.supported_generation_methods]

        print(f'✅ Found {len(chat_models)} models:')
        for model in chat_models:
            name = model.name.replace('models/', '')
            context = f'{model.input_token_limit:,}' if hasattr(model, 'input_token_limit') else 'N/A'
            print(f'  • {name} (context: {context} tokens)')

    except ImportError:
        print('❌ google-generativeai not installed')
        print('   Install: pip install google-generativeai')
    except Exception as e:
        print(f'❌ Error: {e}')


def check_groq_models():
    """List available Groq models."""
    print('\n' + '=' * 60)
    print('Groq Models')
    print('=' * 60)

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print('❌ GROQ_API_KEY not set in .env')
        return

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        models = client.models.list()

        print(f'✅ Found {len(models.data)} models:')
        for model in sorted(models.data, key=lambda x: x.id):
            print(f'  • {model.id}')

    except ImportError:
        print('❌ groq package not installed')
        print('   Install: pip install groq')
    except Exception as e:
        print(f'❌ Error: {e}')


def check_ollama_models():
    """List installed Ollama models."""
    print('\n' + '=' * 60)
    print('Ollama Models (Local)')
    print('=' * 60)

    enabled = os.getenv('OLLAMA_ENABLED', 'false').lower() == 'true'
    if not enabled:
        print('❌ OLLAMA_ENABLED not set to True in .env')
        return

    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print('✅ Installed models:')
                for line in lines[1:]:  # Skip header
                    parts = line.split()
                    if parts:
                        print(f'  • {parts[0]}')
            else:
                print('⚠️  No models installed')
                print('   Pull a model: ollama pull deepseek-r1:8b')
        else:
            print('❌ Ollama not running or not installed')
            print('   Install: brew install ollama')
            print('   Start: ollama serve')

    except FileNotFoundError:
        print('❌ Ollama not found')
        print('   Install: brew install ollama (macOS)')
        print('   Or visit: https://ollama.ai')
    except Exception as e:
        print(f'❌ Error: {e}')

    print('\n💡 Browse all available: https://ollama.ai/library')


def main():
    """Check all configured providers."""
    print('\n' + '🤖 ' * 20)
    print('StatmateAI - Available Models Checker')
    print('🤖 ' * 20)

    check_openai_models()
    check_anthropic_models()
    check_google_models()
    check_groq_models()
    check_ollama_models()

    print('\n' + '=' * 60)
    print('Done! ✅')
    print('=' * 60)
    print('\n💡 To add models to StatmateAI, edit:')
    print('   statmate/core/model_config.py')
    print('\n📖 Full guide:')
    print('   docs/ADDING_MODELS.md')
    print()


if __name__ == '__main__':
    main()

