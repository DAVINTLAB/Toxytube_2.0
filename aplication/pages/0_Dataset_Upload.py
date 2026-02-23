"""
Dataset Upload and Configuration Page
Centralized page for managing the global dataset used across all tools
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from io import BytesIO
from components.navigation import render_navigation, is_configuration_complete, get_configuration_status

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
        'originalFileName': '',             # Original uploaded file name
        'uploadStatus': None,               # Upload status for messages
        'uploadMessage': ''                 # Upload message details
    }

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Dataset Upload - Toxicytube",
    page_icon="📁",
    layout="wide",
)

# Render navigation sidebar
render_navigation('dataset')

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

st.markdown("# 📁 Dataset Upload & Configuration")
st.markdown("Upload and configure the global dataset that will be used across all classification and analysis tools.")

st.markdown("---")

# =============================================================================
# Dataset Upload Section
# =============================================================================

with st.container(border=True):
    st.markdown("### 📤 Dataset Upload")
    st.markdown("Upload a dataset file to be used throughout the application. Supported formats include CSV, Excel, JSON, and Parquet.")

    uploadedFile = st.file_uploader(
        "Select a dataset file:",
        type=['csv', 'json', 'xlsx', 'xls', 'parquet'],
        help="Supported formats: CSV, Excel (XLSX/XLS), JSON, Parquet"
    )

    if uploadedFile is not None:
        try:
            # Determine file type and load accordingly
            file_extension = uploadedFile.name.split('.')[-1].lower()

            with st.spinner(f"Loading {file_extension.upper()} file..."):
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
                st.session_state.globalData['uploadStatus'] = 'success'
                st.session_state.globalData['uploadMessage'] = f"Configuration complete!"

                # Set default output filename based on uploaded file
                base_name = os.path.splitext(uploadedFile.name)[0]
                st.session_state.globalData['outputFileName'] = f"{base_name}_processed"

        except Exception as e:
            st.session_state.globalData['uploadStatus'] = 'error'
            st.session_state.globalData['uploadMessage'] = f"Error loading file: {str(e)}"

    # Show dataset info and preview if loaded
    if st.session_state.globalData['dataset'] is not None:
        dataset = st.session_state.globalData['dataset']

        st.markdown("---")

        # Custom CSS for smaller metrics
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 20px;
            }
            [data-testid="stMetricLabel"] {
                font-size: 14px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📊 Rows", f"{len(dataset):,}")
        with col2:
            st.metric("📋 Columns", len(dataset.columns))
        with col3:
            # Calculate dataset size in memory
            size_bytes = dataset.memory_usage(deep=True).sum()
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024**2:
                size_str = f"{size_bytes/1024:.2f} KB"
            elif size_bytes < 1024**3:
                size_str = f"{size_bytes/(1024**2):.2f} MB"
            else:
                size_str = f"{size_bytes/(1024**3):.2f} GB"
            st.metric("💾 Memory Size", size_str)
        with col4:
            st.metric("📄 Source File", st.session_state.globalData['originalFileName'])

        # Preview
        st.markdown("**Dataset Preview (first 10 rows):**")
        st.dataframe(dataset.head(10), use_container_width=True)

        # Small download button for the preview dataset (respects configured name and format)
        try:
            buffer = BytesIO()
            out_name = st.session_state.globalData.get('outputFileName') or st.session_state.globalData.get('originalFileName') or 'dataset'
            out_format = (st.session_state.globalData.get('outputFormat') or 'csv').lower()

            full_df = st.session_state.globalData.get('dataset')
            if full_df is None:
                full_df = dataset

            if out_format == 'csv':
                full_df.to_csv(buffer, index=False)
                mime = 'text/csv'
            elif out_format == 'xlsx':
                full_df.to_excel(buffer, index=False, engine='openpyxl')
                mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif out_format == 'json':
                buffer.write(full_df.to_json(orient='records', indent=2).encode('utf-8'))
                mime = 'application/json'
            elif out_format == 'parquet':
                full_df.to_parquet(buffer, index=False)
                mime = 'application/octet-stream'
            else:
                dataset.to_csv(buffer, index=False)
                mime = 'text/csv'

            data = buffer.getvalue()
            base = out_name.rsplit('.', 1)[0]
            st.download_button(
                label="📥 Download Dataset",
                data=data,
                file_name=f"{base}.{out_format}",
                mime=mime,
                use_container_width=False
            )
        except Exception:
            pass


        # Show upload status messages
        if st.session_state.globalData.get('uploadStatus') == 'success':
            st.success(f"✅ {st.session_state.globalData.get('uploadMessage', 'Upload successful')}")
        elif st.session_state.globalData.get('uploadStatus') == 'error':
            st.error(f"❌ {st.session_state.globalData.get('uploadMessage', 'Upload failed')}")

st.markdown("")

# =============================================================================
# Dataset Configuration (Only if loaded)
# =============================================================================

if st.session_state.globalData['dataset'] is not None:
    dataset = st.session_state.globalData['dataset']

    # Text Column Selection
    with st.container(border=True):
        st.markdown("### 📝 Text Column Selection")
        st.markdown("Select the column containing text data that will be used for classification operations.")

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

                # Show sample from selected column
                st.markdown("**Sample values from selected column:**")
                sample_values = dataset[selectedColumn].dropna().head(3).tolist()
                for i, value in enumerate(sample_values, 1):
                    st.text(f"{i}. {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")

                st.success(f"✅ Configuration complete.")
            else:
                st.session_state.globalData['textColumn'] = None
                st.warning("⚠️ Please select a text column.")
        else:
            st.warning("⚠️ No text columns found in the dataset. Make sure your dataset contains text data.")

    st.markdown("")

    # Output Configuration
    with st.container(border=True):
        st.markdown("### 💾 Output Configuration")
        st.markdown("Configure the name and format for saving the processed dataset. Files will be saved to your Downloads folder.")

        col1, col2 = st.columns([3, 1])

        with col1:
            outputFileName = st.text_input(
                "Output file name (without extension):",
                value=st.session_state.globalData.get('outputFileName', ''),
                placeholder="processed_dataset",
                help="Name of the output file (extension will be added automatically)"
            )
            st.session_state.globalData['outputFileName'] = outputFileName

        with col2:
            # Add placeholder for format selection
            format_placeholder = "-- Select a format --"
            format_options = [format_placeholder, 'csv', 'xlsx', 'json', 'parquet']

            # Get current format or use placeholder
            current_format = st.session_state.globalData.get('outputFormat', format_placeholder)
            if current_format and current_format != format_placeholder:
                default_idx = format_options.index(current_format)
            else:
                default_idx = 0

            outputFormat = st.selectbox(
                "Output format:",
                options=format_options,
                index=default_idx,
                help="Format for saving the dataset"
            )

            if outputFormat != format_placeholder:
                st.session_state.globalData['outputFormat'] = outputFormat
            else:
                st.session_state.globalData['outputFormat'] = None

        # Fixed output directory
        st.session_state.globalData['outputDirectory'] = os.path.expanduser('~/Downloads')
        outputDirectory = st.session_state.globalData['outputDirectory']

        # Show full path preview
        if outputFileName and st.session_state.globalData.get('outputFormat'):
            outputFormat = st.session_state.globalData['outputFormat']
            full_path = os.path.join(outputDirectory, f"{outputFileName}.{outputFormat}")

        # Check if configuration is complete
        config_complete = (
            st.session_state.globalData['textColumn'] is not None and
            st.session_state.globalData['outputFileName'] != '' and
            st.session_state.globalData.get('outputFormat') is not None
        )

        if config_complete:
            st.success("✅ Configuration complete.")
        else:
            st.warning("⚠️ Define a name and format for the output file to complete the configuration.")

    st.markdown("")

    # Final Status Container - Use centralized validation
    config_status = get_configuration_status()

    if config_status['complete']:
        with st.container(border=True):
            st.markdown("### Configuration Complete!")
            st.markdown("""
            All settings have been configured successfully. You can now proceed to:
            
            - **🤖 BERT Classifier**: Classify text using BERT-based models
            - **🛡️ Detoxify Classifier**: Detect toxic content in text
            - **🧠 LLM Classifier**: Use Large Language Models for classification
            - **📊 Classified Data Analysis**: Analyze and visualize classification results
            
            Use the navigation sidebar to access these tools.
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Text Column", config_status['text_column'])
            with col2:
                st.metric("📁 Output Name", config_status['output_name'])
            with col3:
                st.metric("📄 Output Format", config_status['output_format'].upper() if config_status['output_format'] else 'N/A')