"""Example: Using the Flexible Model System in StatMate AI.

This example demonstrates how to use different AI model providers
(OpenAI, Anthropic, Google, Groq, Ollama) with StatMate AI.
"""

import numpy as np

from statmate.agents import (
    StatTestDeps,
    create_agent_model_and_settings,
    pearson_agent,
    run_sync_agent,
)
from statmate.workflow.model_factory import create_model, get_default_factory


def example_1_use_default_model():
    """Example 1: Use the configured default model."""
    print('\n=== Example 1: Default Model ===')

    # Create model using default configuration
    model = create_model()
    print(f'Created model: {model}')


def example_2_specify_model():
    """Example 2: Specify a specific model."""
    print('\n=== Example 2: Specific Model ===')

    # Use GPT-4o
    model_gpt4o = create_model(model_name='gpt-4o')
    print(f'GPT-4o model: {model_gpt4o}')

    # Use Claude 3.7 Sonnet (if configured)
    try:
        model_claude = create_model(model_name='claude-3-7-sonnet-20250219', provider='anthropic')
        print(f'Claude model: {model_claude}')
    except Exception as e:
        print(f'Could not create Claude model: {e}')

    # Use Gemini 2.0 (if configured)
    try:
        model_gemini = create_model(model_name='gemini-2.0-flash-exp', provider='google')
        print(f'Gemini model: {model_gemini}')
    except Exception as e:
        print(f'Could not create Gemini model: {e}')


def example_3_local_ollama_model():
    """Example 3: Use local Ollama model."""
    print('\n=== Example 3: Local Ollama Model ===')

    try:
        # Use local DeepSeek-R1 (reasoning model!)
        model = create_model(model_name='deepseek-r1:8b', provider='ollama')
        print(f'Ollama DeepSeek-R1 model: {model}')
        print('✓ Running local REASONING model - no API costs, full privacy!')
        print('✓ DeepSeek-R1 competes with OpenAI o1 but runs FREE locally! 🔥')
    except Exception as e:
        print(f'Could not create Ollama model: {e}')
        print('To use Ollama with DeepSeek-R1:')
        print('1. Install from: https://ollama.ai')
        print('2. Run: ollama pull deepseek-r1:8b  (🔥 Recommended!)')
        print('   Or: ollama pull llama3.1:8b')
        print('3. Set OLLAMA_ENABLED=True in .env')


def example_4_model_for_tools():
    """Example 4: Get model for tool calling (ensures reasoning support)."""
    print('\n=== Example 4: Model for Tools ===')

    # This ensures we get a reasoning-capable model
    model = create_model(for_tools=True)
    print(f'Tool-capable model: {model}')


def example_5_list_available_models():
    """Example 5: List all available models."""
    print('\n=== Example 5: Available Models ===')

    factory = get_default_factory()

    # List all available models
    all_models = factory.list_available_models()
    if all_models:
        print('All available models:')
        for model_info in all_models:
            print(f'  - {model_info.display_name}: {model_info.description}')
    else:
        print('No models configured. Please set API keys in .env file.')

    # List only reasoning models
    reasoning_models = factory.list_available_models(for_tools=True)
    if reasoning_models:
        print('\nReasoning-capable models:')
        for model_info in reasoning_models:
            print(f'  - {model_info.display_name}')


def example_6_agent_with_flexible_model():
    """Example 6: Use flexible models in statistical agents."""
    print('\n=== Example 6: Agent with Flexible Model ===')

    # Generate sample data
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 0.8 * x + 0.2 * np.random.normal(0, 1, 100)

    try:
        # Method 1: Use helper functions (recommended for agents)
        model, settings = create_agent_model_and_settings(temperature=0.0, max_tokens=500)

        # Create agent
        agent = pearson_agent(model=model, model_settings=settings)

        # Run analysis
        deps = StatTestDeps(data=x, data_secondary=y, test_params={'alpha': 0.05})
        result = run_sync_agent(agent, user_prompt='', deps=deps)

        print('Pearson Correlation Analysis:')
        print(f'  Correlation: {result.statistical_test_result.statistics}')
        print(f'  P-value: {result.statistical_test_result.p_value}')
        print(f'  Result: {result.result}')
    except Exception as e:
        print(f'Could not run analysis: {e}')
        print('Please configure at least one AI provider in .env file.')


def example_7_compare_models():
    """Example 7: Compare different models on the same task."""
    print('\n=== Example 7: Compare Models ===')

    # Test data
    np.random.seed(42)
    x = np.random.normal(0, 1, 50)
    y = 0.9 * x + np.random.normal(0, 1, 50)

    models_to_test = [
        ('gpt-4o', 'openai'),
        ('gpt-4o-mini', 'openai'),
        ('claude-3-5-sonnet-20241022', 'anthropic'),
        ('gemini-2.0-flash-exp', 'google'),
        ('deepseek-r1:8b', 'ollama'),  # Reasoning model!
        ('llama3.1:8b', 'ollama'),
    ]

    for model_name, provider in models_to_test:
        try:
            print(f'\nTesting {model_name}...')
            model, settings = create_agent_model_and_settings(temperature=0.0)

            # Override with specific model
            model = create_model(model_name=model_name, provider=provider, for_tools=True)

            agent = pearson_agent(model=model, model_settings=settings)
            deps = StatTestDeps(data=x, data_secondary=y, test_params={'alpha': 0.05})
            result = run_sync_agent(agent, user_prompt='', deps=deps)

            print(f'✓ {model_name}: r={result.statistical_test_result.statistics:.3f}')
        except Exception as e:
            print(f'✗ {model_name}: {e}')


def example_8_override_api_key():
    """Example 8: Override API key per request."""
    print('\n=== Example 8: Override API Key ===')

    # This is useful for multi-tenant applications or testing
    try:
        model = create_model(model_name='gpt-4o', api_key='different-api-key-here')
        print('Created model with custom API key')
    except Exception as e:
        print(f'Error: {e}')


def main():
    """Run all examples."""
    print('=' * 70)
    print('StatMate AI - Flexible Model System Examples')
    print('=' * 70)

    example_1_use_default_model()
    example_2_specify_model()
    example_3_local_ollama_model()
    example_4_model_for_tools()
    example_5_list_available_models()
    example_6_agent_with_flexible_model()
    # example_7_compare_models()  # Uncomment to compare models (requires API keys)
    example_8_override_api_key()

    print('\n' + '=' * 70)
    print('Examples completed!')
    print('=' * 70)
    print('\nNext steps:')
    print('1. Configure your preferred providers in .env file')
    print('2. Set DEFAULT_MODEL_PROVIDER and DEFAULT_MODEL_NAME')
    print('3. Try running analyses with different models')
    print('4. Consider using Ollama for local, private, cost-free inference')
    print('\nFor more info, see: docs/MODEL_CONFIGURATION.md')


if __name__ == '__main__':
    main()
