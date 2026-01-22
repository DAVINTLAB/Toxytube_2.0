"""
Navigation component for Toxicytube application
Provides consistent navigation sidebar
"""
import streamlit as st


def render_navigation(current_page):
    """
    Renders the navigation sidebar
    
    Args:
        current_page: String identifier for current page
                     Options: 'home', 'classification', 'youtube', 'analysis', 'detoxify', 'llm'
    """
    # Hide default Streamlit navigation
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")

    # Navigation buttons
    if st.sidebar.button(
        "🏠 Home",
        use_container_width=True,
        type="primary" if current_page == 'home' else "secondary"
    ):
        if current_page != 'home':
            st.switch_page("Home.py")

    if st.sidebar.button(
        "🎥 YouTube Comments",
        use_container_width=True,
        type="primary" if current_page == 'youtube' else "secondary"
    ):
        if current_page != 'youtube':
            st.switch_page("pages/1_YouTube_Comments.py")

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

    if st.sidebar.button(
        "📊 Data Analysis",
        use_container_width=True,
        type="primary" if current_page == 'analysis' else "secondary"
    ):
        if current_page != 'analysis':
            st.switch_page("pages/5_Classified_Data_Analysis.py")
    
    # Show global dataset status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Dataset Status")
    
    if 'globalData' in st.session_state and st.session_state.globalData.get('datasetLoaded'):
        st.sidebar.success(f"✅ Loaded: {st.session_state.globalData.get('originalFileName', 'Unknown')}")
        st.sidebar.info(f"📊 {len(st.session_state.globalData.get('dataset', [])):,} rows")
    else:
        st.sidebar.warning("⚠️ No dataset loaded")
