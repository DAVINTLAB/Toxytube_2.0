import streamlit as st
from components.navigation import render_navigation

# Configuração da página
st.set_page_config(
    page_title="Toxytube",
    page_icon="🤖",
    layout="wide",
)

# Render navigation sidebar
render_navigation('home')

# =============================================================================
# Main Content
# =============================================================================

# Hero Section
st.markdown("## 🚀 Welcome to Toxytube")
st.markdown("""
A complete and intuitive platform to perform content moderation studies without writing code.

### ✨ Main Features:

- **🤖 BERT Classifier**: Classify text using any BERT-based model (BERT, RoBERTa, DistilBERT, etc.) for sentiment analysis, toxicity detection, and document classification
- **🎥 Comment Collection**: Collect and analyze YouTube comments via API
- **📊 Classified Data Analysis**: Visualize distributions and temporal trends of already classified datasets
""")

# Footer
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔗 Useful Links")
    st.markdown("- [Hugging Face Models](https://huggingface.co/models)")
    st.markdown("- [Dataset Examples](https://huggingface.co/datasets)")
    #st.markdown("- [Application Repository](https://github.com)")
