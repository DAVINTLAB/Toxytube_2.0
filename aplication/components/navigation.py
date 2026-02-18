"""
Navigation component for Toxicytube application
Provides consistent navigation sidebar
"""
import streamlit as st


def is_configuration_complete():
    """
    Validates if all required configurations are complete.
    Returns a tuple (is_complete: bool, missing_items: list)
    
    Required configurations:
    1. Dataset loaded (not None)
    2. Text column selected (not None)
    3. Output file name defined (not empty)
    4. Output format selected (not None)
    """
    if 'globalData' not in st.session_state:
        return False, ['Dataset not loaded', 'Text column not selected', 'Output file name not defined', 'Output format not selected']

    global_data = st.session_state.globalData
    missing = []

    # Check dataset
    if global_data.get('dataset') is None:
        missing.append('Dataset not loaded')

    # Check text column
    if global_data.get('textColumn') is None:
        missing.append('Text column not selected')

    # Check output file name
    if not global_data.get('outputFileName'):
        missing.append('Output file name not defined')

    # Check output format
    if global_data.get('outputFormat') is None:
        missing.append('Output format not selected')

    return len(missing) == 0, missing


def get_configuration_status():
    """
    Returns a status dict with detailed configuration state.
    Useful for displaying status in UI components.
    """
    if 'globalData' not in st.session_state:
        return {
            'complete': False,
            'dataset_loaded': False,
            'text_column_set': False,
            'output_name_set': False,
            'output_format_set': False,
            'dataset_name': None,
            'text_column': None,
            'output_name': None,
            'output_format': None
        }

    gd = st.session_state.globalData

    dataset_loaded = gd.get('dataset') is not None
    text_column_set = gd.get('textColumn') is not None
    output_name_set = bool(gd.get('outputFileName'))
    output_format_set = gd.get('outputFormat') is not None

    return {
        'complete': dataset_loaded and text_column_set and output_name_set and output_format_set,
        'dataset_loaded': dataset_loaded,
        'text_column_set': text_column_set,
        'output_name_set': output_name_set,
        'output_format_set': output_format_set,
        'dataset_name': gd.get('originalFileName'),
        'text_column': gd.get('textColumn'),
        'output_name': gd.get('outputFileName'),
        'output_format': gd.get('outputFormat')
    }


def render_navigation(current_page):
    """
    Renders the navigation sidebar
    
    Args:
        current_page: String identifier for current page
                     Options: 'home', 'dataset', 'youtube', 'classification', 'detoxify', 'llm', 'analysis'
    """
    # Hide default Streamlit navigation
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # Navigation buttons
    if st.sidebar.button(
        "🏠 Home",
        use_container_width=True,
        type="primary" if current_page == 'home' else "secondary"
    ):
        if current_page != 'home':
            st.switch_page("Home.py")

    # Data Collect Section
    st.sidebar.markdown("### Data Collect")

    if st.sidebar.button(
        "🎥 YouTube Comments",
        use_container_width=True,
        type="primary" if current_page == 'youtube' else "secondary"
    ):
        if current_page != 'youtube':
            st.switch_page("pages/1_YouTube_Comments.py")

    # Data Load Section
    st.sidebar.markdown("### Data Load")

    if st.sidebar.button(
        "📂 Dataset Upload",
        use_container_width=True,
        type="primary" if current_page == 'dataset' else "secondary"
    ):
        if current_page != 'dataset':
            st.switch_page("pages/0_Dataset_Upload.py")

    # Show configuration status (reactive check)
    config_status = get_configuration_status()
    if config_status['complete']:
        st.sidebar.markdown("<p style='text-align: center; font-size: 12px; color: #0f9960;'>✅ Dataset loaded</p>", unsafe_allow_html=True)
    elif config_status['dataset_loaded']:
        st.sidebar.markdown("<p style='text-align: center; font-size: 12px; color: #ffa500;'>⚠️ Configuration pending</p>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<p style='text-align: center; font-size: 12px; color: #ffa500;'>⚠️ No dataset loaded</p>", unsafe_allow_html=True)

    # Data Classification Section
    st.sidebar.markdown("### Data Classification")

    if st.sidebar.button(
        "🤖 BERT Classifier",
        use_container_width=True,
        type="primary" if current_page == 'classification' else "secondary"
    ):
        if current_page != 'classification':
            st.switch_page("pages/2_Bert_Classifier.py")

    if st.sidebar.button(
        "🛡️ Detoxify Classifier",
        use_container_width=True,
        type="primary" if current_page == 'detoxify' else "secondary"
    ):
        if current_page != 'detoxify':
            st.switch_page("pages/3_Detoxify_Classifier.py")

    if st.sidebar.button(
        "🧠 LLM Classifier",
        use_container_width=True,
        type="primary" if current_page == 'llm' else "secondary"
    ):
        if current_page != 'llm':
            st.switch_page("pages/4_LLM_Classifier.py")

    # Data Visualization Section
    st.sidebar.markdown("### Data Visualization")

    if st.sidebar.button(
        "📈 Data Analysis",
        use_container_width=True,
        type="primary" if current_page == 'analysis' else "secondary"
    ):
        if current_page != 'analysis':
            st.switch_page("pages/5_Classified_Data_Analysis.py")
