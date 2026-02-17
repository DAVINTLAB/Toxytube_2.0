import streamlit as st
import pandas as pd
import os
from datetime import datetime
from components.navigation import render_navigation

# =============================================================================
# Initialize Global Session State - Shared Dataset
# =============================================================================

if 'globalData' not in st.session_state:
    st.session_state.globalData = {
        'dataset': None,                    # Global dataset shared across all pages
        'textColumn': None,                 # Column containing text for classification
        'outputDirectory': os.path.expanduser('~/Downloads'),  # Output directory
        'outputFileName': '',               # Output file name
        'outputFormat': None,              # File format (csv, xlsx, json, parquet)
        'datasetLoaded': False,             # Dataset loading status
        'originalFileName': ''              # Original uploaded file name
    }

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Toxicytube - Home",
    page_icon="🤖",
    layout="wide",
)

# Render navigation sidebar
render_navigation('home')

# =============================================================================
# Main Content
# =============================================================================

# Hero Section
st.markdown("## 🚀 Welcome to Toxicytube")
st.markdown("""
A complete and intuitive platform to perform content moderation studies without writing code.

### ✨ Main Features:

- **📁 Dataset Upload**: Upload and configure your dataset (CSV, Excel, JSON, Parquet)
- **🎥 YouTube Comments**: Collect and analyze YouTube comments via API
- **🤖 BERT Classifier**: Classify text using any BERT-based model (BERT, RoBERTa, DistilBERT, etc.)
- **🛡️ Detoxify Classifier**: Advanced toxicity detection using the Detoxify library
- **🧠 LLM Classifier**: Leverage Large Language Models for sophisticated text classification
- **📊 Classified Data Analysis**: Visualize distributions and temporal trends of classified datasets
""")

st.markdown("---")

col1 = st.columns(1)

with col1[0]:
    st.markdown("### 🔗 Useful Links")
    st.markdown("- [Hugging Face Models](https://huggingface.co/models)")
    st.markdown("- [Dataset Examples](https://huggingface.co/datasets)")


