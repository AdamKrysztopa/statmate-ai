"""Main Streamlit application for StatmateAI."""

import time

import httpx
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title='StatmateAI',
    page_icon='📊',
    layout='wide',
)

# API Configuration
API_BASE_URL = 'http://localhost:8000/api/v1'


# Helper functions
def upload_dataset(file, description=None):
    """Upload a dataset to the API."""
    files = {'file': (file.name, file, file.type)}
    data = {}
    if description:
        data['description'] = description

    response = httpx.post(f'{API_BASE_URL}/datasets/upload', files=files, data=data, timeout=30.0)
    response.raise_for_status()
    return response.json()


def get_dataset_preview(dataset_id, num_rows=10):
    """Get dataset preview."""
    response = httpx.get(f'{API_BASE_URL}/datasets/{dataset_id}/preview?num_rows={num_rows}', timeout=10.0)
    response.raise_for_status()
    return response.json()


def run_analysis(dataset_id, selected_columns=None, model_name=None, provider=None):
    """Run statistical analysis."""
    payload = {'dataset_id': dataset_id}
    if selected_columns:
        payload['selected_columns'] = selected_columns
    if model_name:
        payload['model_name'] = model_name
    if provider:
        payload['provider'] = provider

    response = httpx.post(f'{API_BASE_URL}/analysis/run', json=payload, timeout=10.0)
    response.raise_for_status()
    return response.json()


def get_analysis_status(analysis_id):
    """Get analysis status."""
    response = httpx.get(f'{API_BASE_URL}/analysis/{analysis_id}', timeout=10.0)
    response.raise_for_status()
    return response.json()


def get_analysis_results(analysis_id):
    """Get analysis results."""
    response = httpx.get(f'{API_BASE_URL}/analysis/{analysis_id}/results', timeout=10.0)
    response.raise_for_status()
    return response.json()


def list_datasets():
    """List all datasets."""
    response = httpx.get(f'{API_BASE_URL}/datasets/', timeout=10.0)
    response.raise_for_status()
    return response.json()


# Main App
st.title('📊 StatmateAI')
st.markdown('*AI-driven statistical analysis for clinical and observational research*')

# Check API health
try:
    health = httpx.get('http://localhost:8000/health', timeout=5.0).json()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f'✅ API Connected (v{health["version"]})')
    with col2:
        if st.button('🤖 Model Config', use_container_width=True):
            st.session_state['show_model_page'] = True
            st.rerun()
except Exception as e:
    st.error(f'❌ API not available: {e}')
    st.stop()

st.divider()

# Show model configuration page if requested
if st.session_state.get('show_model_page'):
    from statmate.ui.components.model_selector import render_model_info_page

    if st.button('← Back to Analysis', use_container_width=True):
        st.session_state['show_model_page'] = False
        st.rerun()

    st.divider()
    render_model_info_page(API_BASE_URL)
    st.stop()

# Initialize session state
if 'current_dataset_id' not in st.session_state:
    st.session_state.current_dataset_id = None
if 'current_analysis_id' not in st.session_state:
    st.session_state.current_analysis_id = None

# Sidebar: Existing datasets
with st.sidebar:
    st.header('📁 Existing Datasets')

    # Show current selection
    if st.session_state.current_dataset_id:
        st.success('✓ Selected')
        if st.button('Clear Selection', use_container_width=True):
            st.session_state.current_dataset_id = None
            st.session_state.current_analysis_id = None
            st.rerun()
        st.divider()

    try:
        datasets = list_datasets()
        if datasets:
            for ds in datasets[:5]:  # Show last 5
                is_selected = st.session_state.current_dataset_id == ds['id']
                button_label = f'{"✓ " if is_selected else ""}{ds["original_filename"]}'
                button_type = 'primary' if is_selected else 'secondary'

                if st.button(button_label, key=f'ds_{ds["id"]}', use_container_width=True, type=button_type):
                    st.session_state.current_dataset_id = ds['id']
                    st.success(f'Loaded: {ds["original_filename"]}')
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.info('No datasets yet. Upload one below!')
    except Exception as e:
        st.error(f'Error loading datasets: {e}')

# Main content tabs
tab1, tab2, tab3 = st.tabs(['1️⃣ Upload Data', '2️⃣ Run Analysis', '3️⃣ View Results'])

# Tab 1: Upload
with tab1:
    st.header('Upload Dataset')

    uploaded_file = st.file_uploader(
        'Choose a CSV or Excel file',
        type=['csv', 'xlsx', 'xls'],
        help='Upload your statistical dataset',
    )

    description = st.text_area('Description (optional)', placeholder='Brief description of your dataset...')

    if st.button('📤 Upload Dataset', type='primary', disabled=uploaded_file is None):
        with st.spinner('Uploading...'):
            try:
                result = upload_dataset(uploaded_file, description)
                st.session_state.current_dataset_id = result['dataset_id']
                st.success(f'✅ Uploaded successfully! Dataset ID: {result["dataset_id"]}')
                st.balloons()
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f'Upload failed: {e}')

# Tab 2: Run Analysis
with tab2:
    st.header('Run Statistical Analysis')

    if not st.session_state.current_dataset_id:
        st.info('👈 **Select a dataset from the sidebar** or upload a new one in Tab 1.')
        st.markdown("""
        ### Available Sample Datasets:
        - 📄 **patient_blood_pressure.csv** - Paired data (80 rows)
        - 📄 **smoking_exercise_study.csv** - Categorical data (200 rows)
        
        Click on a dataset in the sidebar to get started!
        """)
    else:
        dataset_id = st.session_state.current_dataset_id
        st.info(f'🎯 Working with dataset ID: `{dataset_id[:8]}...`')

        # Load preview
        try:
            preview = get_dataset_preview(dataset_id, num_rows=10)

            st.subheader(f'Dataset: {preview["original_filename"]}')
            st.caption(f'📊 {preview["row_count"]} rows × {len(preview["column_names"])} columns')

            # Show preview
            with st.expander('👁️ Preview Data', expanded=True):
                df = pd.DataFrame(preview['preview_data'])
                st.dataframe(df, use_container_width=True)

            # Column selection
            st.subheader('Select Columns for Analysis')
            selected_columns = st.multiselect(
                'Choose columns (leave empty for all)',
                options=preview['column_names'],
                default=None,
                help='Select specific columns or leave empty to analyze all columns',
            )

            # Run analysis
            st.divider()
            col1, col2 = st.columns([1, 3])

            with col1:
                if st.button('🚀 Run Stat Test', type='primary', use_container_width=True):
                    with st.spinner('Starting analysis...'):
                        try:
                            result = run_analysis(
                                dataset_id,
                                selected_columns if selected_columns else None,
                            )
                            st.session_state.current_analysis_id = result['id']
                            st.success(f'✅ Analysis started! ID: {result["id"]}')
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f'Failed to start analysis: {e}')

            with col2:
                st.info('💡 Analysis may take 30-60 seconds depending on dataset size.')

        except Exception as e:
            st.error(f'Error loading dataset: {e}')

# Tab 3: View Results
with tab3:
    st.header('Analysis Results')

    if not st.session_state.current_analysis_id:
        st.warning('⚠️ No analysis running. Start one in Tab 2.')
    else:
        analysis_id = st.session_state.current_analysis_id

        # Status check
        try:
            status = get_analysis_status(analysis_id)

            status_emoji = {
                'pending': '⏳',
                'running': '⚙️',
                'completed': '✅',
                'failed': '❌',
            }

            st.subheader(f'{status_emoji.get(status["status"], "❓")} Status: {status["status"].upper()}')

            if status['status'] == 'running':
                st.info('🔄 Analysis is running... Refresh to check status.')
                if st.button('🔄 Refresh Status'):
                    st.rerun()

            elif status['status'] == 'completed':
                # Get results
                try:
                    results = get_analysis_results(analysis_id)

                    st.success('✅ Analysis Complete!')

                    # Display model used
                    if results.get('model_name') or results.get('provider'):
                        col1, col2 = st.columns(2)
                        with col1:
                            if results.get('model_name'):
                                st.metric('🤖 Model Used', results['model_name'])
                        with col2:
                            if results.get('provider'):
                                st.metric('Provider', results['provider'].upper())
                        st.divider()

                    # Summary
                    if results.get('summary'):
                        st.subheader('📝 Summary')
                        st.write(results['summary'])

                    # P-values
                    if results.get('probabilities'):
                        st.subheader('📊 Statistical Tests')
                        p_vals = results['probabilities']

                        for test_name, p_value in p_vals.items():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f'**{test_name}**')
                            with col2:
                                st.write(f'p = {p_value:.4f}')
                            with col3:
                                if p_value < 0.05:
                                    st.success('Significant')
                                else:
                                    st.info('Not significant')

                    # Detailed results
                    if results.get('results_detail'):
                        with st.expander('📋 Detailed Results'):
                            st.json(results['results_detail'])

                    # Log
                    if results.get('log_available'):
                        with st.expander('📜 Execution Log'):
                            try:
                                log_response = httpx.get(
                                    f'{API_BASE_URL}/analysis/{analysis_id}/log',
                                    timeout=10.0,
                                )
                                log_response.raise_for_status()
                                log_data = log_response.json()
                                st.code(log_data['log_content'], language='text')
                            except Exception as e:
                                st.error(f'Could not load log: {e}')

                except Exception as e:
                    st.error(f'Error loading results: {e}')

            elif status['status'] == 'failed':
                st.error('❌ Analysis failed')
                if status.get('message'):
                    st.code(status['message'])

            elif status['status'] == 'pending':
                st.info('⏳ Analysis is queued...')
                if st.button('🔄 Refresh Status'):
                    st.rerun()

        except Exception as e:
            st.error(f'Error checking status: {e}')

# Footer
st.divider()
st.caption('StatmateAI v0.1.0 | Powered by FastAPI + LangGraph + OpenAI')
