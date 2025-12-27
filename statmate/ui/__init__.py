"""Streamlit UI for StatmateAI.

Interactive web interface for statistical analysis workflow.

This package contains the Streamlit-based frontend that provides:
- File upload (CSV/Excel)
- Data preview and column selection
- Real-time analysis execution
- Results visualization
- Task management

Architecture:

    Browser
        ↓
    Streamlit UI (app.py)
        ↓ HTTP (HTTPX)
    FastAPI Backend (localhost:8000)

Main File:
    app.py - Single-file Streamlit application

Usage:
    streamlit run statmate/ui/app.py --server.port 8501

    # Then open browser to:
    http://localhost:8501

Features:

    Tab 1 - Upload Data:
        - Drag-and-drop file upload
        - CSV and Excel support
        - Optional description
        - Automatic conversion to Parquet

    Tab 2 - Run Analysis:
        - Dataset preview (first 10 rows)
        - Interactive data table
        - Column selection (multi-select)
        - Run analysis button
        - Progress feedback

    Tab 3 - View Results:
        - Analysis status tracking
        - Statistical summary
        - P-values table with significance
        - Detailed JSON results
        - Execution log viewer

    Sidebar:
        - List of existing datasets
        - Visual selection feedback
        - Clear selection button

State Management:
    Uses Streamlit session state to persist:
    - current_dataset_id: Selected dataset
    - current_analysis_id: Running/completed analysis

API Communication:
    All backend communication via HTTPX:
    - Synchronous HTTP calls
    - Error handling and timeouts
    - JSON request/response
    - Multipart file uploads

Configuration:
    API_BASE_URL = 'http://localhost:8000/api/v1'

    Customize in app.py if API runs on different host/port

User Flow:
    1. Upload dataset or select from sidebar
    2. Preview data and select columns
    3. Click "Run Stat Test"
    4. Switch to Results tab
    5. View status, refresh as needed
    6. See summary, p-values, and logs when complete

Design:
    - Clean, minimal interface
    - Color-coded feedback (green=success, red=error, blue=info)
    - Visual selection state
    - Real-time updates via button clicks
    - Expandable sections for details

Dependencies:
    - Streamlit: UI framework
    - HTTPX: HTTP client
    - Pandas: Data display

Note:
    This is the V1 frontend. See ARCHITECTURE_PROPOSAL.md for plans
    for V2 (React/TypeScript) and mobile apps (React Native/Flutter).
"""

__version__ = '0.1.0'
