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
        'outputFormat': 'csv',              # File format (csv, xlsx, json)
        'datasetLoaded': False,             # Dataset loading status
        'originalFileName': ''              # Original uploaded file name
    }

# Configuração da página
st.set_page_config(
    page_title="Toxicytube",
    page_icon="🤖",
    layout="wide",
)

# Render navigation sidebar
render_navigation('home')

# =============================================================================
# Helper Functions
# =============================================================================

def save_global_dataset():
    """Save the global dataset to the configured output path"""
    if st.session_state.globalData['dataset'] is None:
        return False, "No dataset loaded"

    try:
        output_dir = st.session_state.globalData['outputDirectory']
        output_name = st.session_state.globalData['outputFileName']
        output_format = st.session_state.globalData['outputFormat']

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Build full path
        full_path = os.path.join(output_dir, f"{output_name}.{output_format}")

        # Save based on format
        df = st.session_state.globalData['dataset']

        if output_format == 'csv':
            df.to_csv(full_path, index=False)
        elif output_format == 'xlsx':
            df.to_excel(full_path, index=False)
        elif output_format == 'json':
            df.to_json(full_path, orient='records', indent=2)
        elif output_format == 'parquet':
            df.to_parquet(full_path, index=False)

        return True, full_path
    except Exception as e:
        return False, str(e)

# =============================================================================
# Main Content
# =============================================================================

# Hero Section
st.markdown("## 🚀 Welcome to Toxicytube")
st.markdown("""
A complete and intuitive platform to perform content moderation studies without writing code.

### ✨ Main Features:

- **🎥 YouTube Comments**: Collect and analyze YouTube comments via API
- **🤖 BERT Classifier**: Classify text using any BERT-based model (BERT, RoBERTa, DistilBERT, etc.) for sentiment analysis, toxicity detection, and document classification
- **🛡️ Detoxify Classifier**: Advanced toxicity detection using the Detoxify library with pre-trained models
- **🧠 LLM Classifier**: Leverage Large Language Models for sophisticated text classification tasks
- **📊 Classified Data Analysis**: Visualize distributions and temporal trends of already classified datasets
""")

st.markdown("---")

# =============================================================================
# Global Dataset Configuration
# =============================================================================

st.markdown("## 📁 Global Dataset Configuration")
st.markdown("Upload a dataset that will be used across all classification pages. Configure the output settings for saving results.")

with st.container(border=True):
    st.markdown("### 📤 Dataset Upload")

    uploadedFile = st.file_uploader(
        "Select a dataset file:",
        type=['csv', 'json', 'xlsx', 'xls', 'parquet'],
        help="Supported formats: CSV, Excel (XLSX/XLS), JSON, Parquet"
    )

    if uploadedFile is not None:
        try:
            # Determine file type and load accordingly
            file_extension = uploadedFile.name.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(uploadedFile)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploadedFile)
            elif file_extension == 'json':
                df = pd.read_json(uploadedFile)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploadedFile)
            else:
                st.error(f"❌ Unsupported file format: {file_extension}")
                df = None

            if df is not None:
                st.session_state.globalData['dataset'] = df
                st.session_state.globalData['datasetLoaded'] = True
                st.session_state.globalData['originalFileName'] = uploadedFile.name

                # Set default output filename based on uploaded file
                base_name = os.path.splitext(uploadedFile.name)[0]
                if not st.session_state.globalData['outputFileName']:
                    st.session_state.globalData['outputFileName'] = f"{base_name}_processed"

                st.success(f"✅ Dataset loaded successfully: **{uploadedFile.name}**")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# Show dataset info and configuration if loaded
if st.session_state.globalData['datasetLoaded'] and st.session_state.globalData['dataset'] is not None:
    dataset = st.session_state.globalData['dataset']

    # Dataset statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Total Rows", f"{len(dataset):,}")
    with col2:
        st.metric("📋 Total Columns", len(dataset.columns))
    with col3:
        st.metric("📄 Source File", st.session_state.globalData['originalFileName'])

    # Preview
    st.markdown("**Dataset Preview (first 10 rows):**")
    st.dataframe(dataset.head(10), use_container_width=True)

    st.markdown("---")

    # Text Column Selection
    with st.container(border=True):
        st.markdown("### 📝 Text Column Selection")
        st.markdown("Select the column containing text for classification operations.")

        # Get text columns (object type)
        textColumns = [col for col in dataset.columns if dataset[col].dtype == 'object']

        if textColumns:
            # Add placeholder
            placeholder = "-- Select a column --"
            options = [placeholder] + textColumns

            # Get current selection
            current = st.session_state.globalData.get('textColumn')
            default_idx = options.index(current) if current and current in textColumns else 0

            selectedColumn = st.selectbox(
                "Text column for classification:",
                options=options,
                index=default_idx,
                help="This column will be used as input for all classifiers"
            )

            if selectedColumn != placeholder:
                st.session_state.globalData['textColumn'] = selectedColumn
                st.success(f"✅ Text column selected: **{selectedColumn}**")
            else:
                st.session_state.globalData['textColumn'] = None
                st.warning("⚠️ Please select a text column to enable classification.")
        else:
            st.warning("⚠️ No text columns found in the dataset.")

    st.markdown("---")

    # Output Configuration
    with st.container(border=True):
        st.markdown("### 💾 Output Configuration")
        st.markdown("Configure where and how the processed dataset will be saved after each operation.")

        col1, col2 = st.columns(2)

        with col1:
            outputFileName = st.text_input(
                "Output file name (without extension):",
                value=st.session_state.globalData.get('outputFileName', ''),
                placeholder="processed_dataset",
                help="Name of the output file"
            )
            st.session_state.globalData['outputFileName'] = outputFileName

        with col2:
            outputFormat = st.selectbox(
                "Output format:",
                options=['csv', 'xlsx', 'json', 'parquet'],
                index=['csv', 'xlsx', 'json', 'parquet'].index(
                    st.session_state.globalData.get('outputFormat', 'csv')
                ),
                help="Format for saving the dataset"
            )
            st.session_state.globalData['outputFormat'] = outputFormat

        outputDirectory = st.text_input(
            "Output directory:",
            value=st.session_state.globalData.get('outputDirectory', os.path.expanduser('~/Downloads')),
            placeholder="/path/to/output/directory",
            help="Directory where the file will be saved"
        )
        st.session_state.globalData['outputDirectory'] = outputDirectory

        # Show full path preview
        if outputFileName and outputDirectory:
            full_path = os.path.join(outputDirectory, f"{outputFileName}.{outputFormat}")
            st.info(f"📁 **Output path:** `{full_path}`")

        # Check if configuration is complete
        config_complete = (
            st.session_state.globalData['textColumn'] is not None and
            st.session_state.globalData['outputFileName'] != '' and
            st.session_state.globalData['outputDirectory'] != ''
        )

        if config_complete:
            st.success("✅ **Configuration complete!** You can now use the classifiers.")
        else:
            st.warning("⚠️ Complete all configuration fields to enable classifiers.")

else:
    st.info("📤 **Upload a dataset to get started.** The dataset will be shared across all classification pages.")

# Footer
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔗 Useful Links")
    st.markdown("- [Hugging Face Models](https://huggingface.co/models)")
    st.markdown("- [Dataset Examples](https://huggingface.co/datasets)")
    #st.markdown("- [Application Repository](https://github.com)")
