"""
Navigation component for Toxytube application
Provides consistent navigation sidebar with confirmation dialog
"""
import streamlit as st


def render_navigation(current_page):
    """
    Renders the navigation sidebar with confirmation dialog
    
    Args:
        current_page: String identifier for current page
                     Options: 'home', 'classification', 'youtube', 'analysis'
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

    # Initialize confirmation state
    if 'confirm_navigation' not in st.session_state:
        st.session_state.confirm_navigation = False
    if 'target_page' not in st.session_state:
        st.session_state.target_page = None

    # Navigation buttons
    if st.sidebar.button(
        "🏠 Home",
        use_container_width=True,
        type="primary" if current_page == 'home' else "secondary"
    ):
        if current_page != 'home':
            st.session_state.target_page = "Home.py"
            st.session_state.confirm_navigation = True
            st.rerun()

    if st.sidebar.button(
        "🎥 YouTube Comments",
        use_container_width=True,
        type="primary" if current_page == 'youtube' else "secondary"
    ):
        if current_page != 'youtube':
            st.session_state.target_page = "pages/1_YouTube_Comments.py"
            st.session_state.confirm_navigation = True
            st.rerun()

    if st.sidebar.button(
        "🤖 BERT Classifier",
        use_container_width=True,
        type="primary" if current_page == 'classification' else "secondary"
    ):
        if current_page != 'classification':
            st.session_state.target_page = "pages/2_Bert_Classifier.py"
            st.session_state.confirm_navigation = True
            st.rerun()

    if st.sidebar.button(
        "📊 Data Analysis",
        use_container_width=True,
        type="primary" if current_page == 'analysis' else "secondary"
    ):
        if current_page != 'analysis':
            st.session_state.target_page = "pages/3_Classified_Data_Analysis.py"
            st.session_state.confirm_navigation = True
            st.rerun()

    if st.sidebar.button(
        "🛡️ Detoxify Classifier",
        use_container_width=True,
        type="primary" if current_page == 'detoxify' else "secondary"
    ):
        if current_page != 'detoxify':
            st.session_state.target_page = "pages/4_Detoxify_Classifier.py"
            st.session_state.confirm_navigation = True
            st.rerun()

    if st.sidebar.button(
        "🧠 LLM Classifier",
        use_container_width=True,
        type="primary" if current_page == 'llm' else "secondary"
    ):
        if current_page != 'llm':
            st.session_state.target_page = "pages/5_LLM_Classifier.py"
            st.session_state.confirm_navigation = True
            st.rerun()

    # Show confirmation dialog
    if st.session_state.confirm_navigation and st.session_state.target_page:
        with st.sidebar:
            st.markdown("---")
            st.warning("⚠️ **Warning!**")
            st.markdown("Are you sure you want to switch screens?")
            st.markdown("**Note:** All filled data will be lost.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes", use_container_width=True):
                    target = st.session_state.target_page
                    st.session_state.confirm_navigation = False
                    st.session_state.target_page = None
                    st.switch_page(target)
            with col2:
                if st.button("No", use_container_width=True):
                    st.session_state.confirm_navigation = False
                    st.session_state.target_page = None
                    st.rerun()
