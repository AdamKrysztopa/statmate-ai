"""Streamlit component for user credential management (PROD mode)."""

import httpx
import streamlit as st


def get_environment_mode(api_base_url: str) -> str:
    """Get current environment mode from API.

    Args:
        api_base_url: Base URL of the API.

    Returns:
        'development' or 'production'
    """
    try:
        response = httpx.get(f'{api_base_url}/models/environment', timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return data.get('environment', 'development')
    except Exception:
        return 'development'


def check_credentials_configured(api_base_url: str) -> bool:
    """Check if any credentials are configured.

    Args:
        api_base_url: Base URL of the API.

    Returns:
        True if credentials are configured, False otherwise.
    """
    try:
        response = httpx.get(f'{api_base_url}/models/available?for_tools=true', timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return len(data.get('models', [])) > 0
    except Exception:
        return False


def set_user_credentials(api_base_url: str, credentials: dict) -> tuple[bool, str]:
    """Send user credentials to API.

    Args:
        api_base_url: Base URL of the API.
        credentials: Dictionary with provider keys and values.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        response = httpx.post(
            f'{api_base_url}/models/credentials',
            json=credentials,
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        return True, data.get('message', 'Credentials configured successfully')
    except Exception as e:
        return False, str(e)


def render_credential_setup_page(api_base_url: str) -> bool:
    """Render credential setup page for PRODUCTION mode.

    Args:
        api_base_url: Base URL of the API.

    Returns:
        True if credentials are configured, False if user needs to set them up.
    """
    st.markdown('### 🔐 API Credentials Setup')
    st.markdown('Please provide your AI model API credentials to continue.')

    st.info(
        '**Why do I need this?**\n\n'
        'StatmateAI uses AI models (like GPT-4, Claude, Gemini) to analyze your data. '
        'You need to provide your own API keys. Your keys are stored securely in memory '
        'for this session only and are never saved to disk or shared.'
    )

    with st.expander('🆓 **Want to use FREE local models?**', expanded=False):
        st.markdown(
            """
            You can use **Ollama** to run AI models locally on your machine for FREE!

            **Quick Setup:**
            1. Install Ollama: [https://ollama.ai](https://ollama.ai)
            2. Run: `ollama pull deepseek-r1:8b`
            3. Start Ollama: `ollama serve`
            4. Check "Enable Ollama" below and click Configure

            **Benefits:**
            - 🆓 Completely free
            - 🔒 Private (data never leaves your machine)
            - ⚡ Fast inference
            - 🧠 DeepSeek-R1 is excellent for statistical reasoning!
            """
        )

    st.divider()

    # Tabs for different providers
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['OpenAI', 'Anthropic', 'Google', 'Groq', 'Ollama (Local)'])

    credentials = {}

    with tab1:
        st.markdown('**OpenAI (GPT-4, GPT-4o, etc.)**')
        st.caption('Get your API key at: https://platform.openai.com/api-keys')

        openai_key = st.text_input(
            'OpenAI API Key',
            type='password',
            placeholder='sk-...',
            help='Your OpenAI API key (starts with sk-)',
            key='openai_key',
        )
        if openai_key:
            credentials['openai_api_key'] = openai_key
            st.success('✓ OpenAI key entered')

    with tab2:
        st.markdown('**Anthropic (Claude models)**')
        st.caption('Get your API key at: https://console.anthropic.com/')

        anthropic_key = st.text_input(
            'Anthropic API Key',
            type='password',
            placeholder='sk-ant-...',
            help='Your Anthropic API key (starts with sk-ant-)',
            key='anthropic_key',
        )
        if anthropic_key:
            credentials['anthropic_api_key'] = anthropic_key
            st.success('✓ Anthropic key entered')

    with tab3:
        st.markdown('**Google (Gemini models)**')
        st.caption('Get your API key at: https://makersuite.google.com/app/apikey')

        google_key = st.text_input(
            'Google API Key',
            type='password',
            placeholder='AI...',
            help='Your Google API key',
            key='google_key',
        )
        if google_key:
            credentials['google_api_key'] = google_key
            st.success('✓ Google key entered')

    with tab4:
        st.markdown('**Groq (Fast inference - FREE tier available)**')
        st.caption('Get your API key at: https://console.groq.com/keys')

        groq_key = st.text_input(
            'Groq API Key',
            type='password',
            placeholder='gsk_...',
            help='Your Groq API key',
            key='groq_key',
        )
        if groq_key:
            credentials['groq_api_key'] = groq_key
            st.success('✓ Groq key entered')

    with tab5:
        st.markdown('**Ollama (Local models - Completely FREE!)**')
        st.caption('Run AI models on your own machine')

        ollama_enabled = st.checkbox(
            'Enable Ollama',
            help='Check this if you have Ollama running locally',
            key='ollama_enabled',
        )

        if ollama_enabled:
            ollama_url = st.text_input(
                'Ollama Base URL',
                value='http://localhost:11434/v1',
                help='URL where Ollama is running',
                key='ollama_url',
            )
            ollama_model = st.text_input(
                'Model Name',
                value='deepseek-r1:8b',
                help='Ollama model to use (e.g., deepseek-r1:8b, llama3.3:70b)',
                key='ollama_model',
            )
            credentials['ollama_enabled'] = 'true'
            credentials['ollama_base_url'] = ollama_url
            credentials['ollama_default_model'] = ollama_model
            st.success('✓ Ollama configuration entered')

    st.divider()

    # Configure button
    col1, col2 = st.columns([1, 3])

    with col1:
        configure_btn = st.button(
            '🚀 Configure & Start', type='primary', use_container_width=True, disabled=not credentials
        )

    with col2:
        if not credentials:
            st.warning('⚠️ Please enter at least one set of API credentials')
        else:
            st.info(f'✓ {len(credentials)} credential(s) entered')

    if configure_btn and credentials:
        with st.spinner('Configuring credentials...'):
            success, message = set_user_credentials(api_base_url, credentials)

            if success:
                st.success(f'✅ {message}')
                st.balloons()
                # Store in session state that credentials are configured
                st.session_state['credentials_configured'] = True
                st.rerun()
            else:
                st.error(f'❌ Failed to configure credentials: {message}')

    return False


def render_credentials_banner(api_base_url: str):
    """Render banner showing current credential status.

    Args:
        api_base_url: Base URL of the API.
    """
    env_mode = get_environment_mode(api_base_url)

    if env_mode == 'development':
        st.info('🔧 **DEV MODE** - Using credentials from .env file', icon='ℹ️')
    else:
        has_creds = check_credentials_configured(api_base_url)
        if has_creds:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success('🔐 **PROD MODE** - User credentials active', icon='✅')
            with col2:
                if st.button('⚙️ Change Keys', use_container_width=True):
                    st.session_state['show_credential_setup'] = True
                    st.rerun()
        else:
            st.error('🔐 **PROD MODE** - No credentials configured', icon='❌')


def require_credentials(api_base_url: str) -> bool:
    """Check if credentials are required and configured.

    Args:
        api_base_url: Base URL of the API.

    Returns:
        True if app can proceed, False if credential setup is needed.
    """
    env_mode = get_environment_mode(api_base_url)

    # DEV mode - credentials from .env
    if env_mode == 'development':
        return True

    # PROD mode - check if user has configured credentials
    if st.session_state.get('credentials_configured'):
        return True

    has_creds = check_credentials_configured(api_base_url)
    if has_creds:
        st.session_state['credentials_configured'] = True
        return True

    return False
