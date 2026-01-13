"""
Detoxify Classification Page
Implements the complete text toxicity classification workflow using Detoxify library
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime

from components.navigation import render_navigation
from components.detoxify_classifier import (
    get_available_models,
    get_available_device,
    load_detoxify_model,
    unload_detoxify_model,
    classify_single_text,
    classify_texts,
    create_results_dataframe,
    save_classification_results
)

# =============================================================================
# Initialize Session State - Classification Process Data
# =============================================================================

# Dictionary with all data needed for the Detoxify classification process
if 'detoxifyData' not in st.session_state:
    st.session_state.detoxifyData = {
        'dataset': None,                    # Dataset loaded by user
        'textColumn': None,                 # Column selected for classification
        'modelName': None,                  # Selected Detoxify model name
        'modelLoaded': False,               # Model loading status
        'model': None,                      # Loaded model
        'modelInfo': {},                    # Model information
        'threshold': 0.5,                   # Classification threshold
        'outputDirectory': os.path.expanduser('~/Downloads'),  # Output directory
        'outputFileName': f"detoxify_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Output file name
        'outputFormat': 'csv',              # File format (csv, xlsx, json)
        'classificationResults': None,      # Classification results
        'isExecuting': False,               # Execution status
        'fileSaved': False                  # File saved status
    }

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(page_title="Detoxify Classifier - Toxytube", page_icon="🛡️", layout="wide")

# Render navigation sidebar
render_navigation('detoxify')

# =============================================================================
# Main Content
# =============================================================================

# Page header
st.markdown("# 🛡️ Detoxify Classifier")
st.markdown("Classify text toxicity using the Detoxify library. Detect toxic content, insults, threats, obscenity, and more.")

st.markdown("---")

# =============================================================================
# Step 1: Dataset Selection
# =============================================================================

# Check if step 1 is complete
step1_complete = (
    st.session_state.detoxifyData['dataset'] is not None and
    st.session_state.detoxifyData['textColumn'] is not None
)

with st.container(border=True):
    st.markdown("### 📁 Step 1: Dataset Selection")
    st.markdown("Upload a file for toxicity classification using Detoxify.")

    uploadedFile = st.file_uploader("Select a file:", type=['csv', 'json', 'xlsx', 'xls'], key="detoxify_uploader")

    if uploadedFile is not None:
        try:
            # Load dataset based on file type
            file_extension = uploadedFile.name.split('.')[-1].lower()

            if file_extension == 'csv':
                dataset = pd.read_csv(uploadedFile)
            elif file_extension in ['xlsx', 'xls']:
                dataset = pd.read_excel(uploadedFile)
            elif file_extension == 'json':
                dataset = pd.read_json(uploadedFile)
            else:
                st.error(f"❌ File format not supported: {file_extension}")
                dataset = None

            if dataset is not None:
                st.session_state.detoxifyData['dataset'] = dataset

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    # Preview the dataset if loaded
    if st.session_state.detoxifyData['dataset'] is not None:
        st.markdown("#### Dataset Preview")

        dataset = st.session_state.detoxifyData['dataset']

        # General information
        column1, column2, column3 = st.columns(3)

        with column1:
            st.metric("Rows", len(dataset))
        with column2:
            st.metric("Columns", len(dataset.columns))
        with column3:
            st.metric("Memory", f"{dataset.memory_usage(deep=True).sum() / 1048576:.2f} MB")

        # Preview the data
        st.markdown("**First 10 rows:**")
        st.dataframe(dataset.head(10), use_container_width=True)

        # Text column selection
        textColumns = [col for col in dataset.columns if dataset[col].dtype == 'object']

        # Add placeholder option
        placeholder = "-- Select a column --"
        options_with_placeholder = [placeholder] + textColumns

        # Get current selection or default to placeholder
        current_selection = st.session_state.detoxifyData.get('textColumn')
        if current_selection and current_selection in textColumns:
            default_index = options_with_placeholder.index(current_selection)
        else:
            default_index = 0

        selectedTextColumn = st.selectbox(
            "Select the column containing text for classification (must be a text/string column):",
            options=options_with_placeholder,
            index=default_index,
            help="This column will be used as input for the Detoxify model",
            key="detoxify_text_column"
        )

        # Only update if a valid column is selected (not placeholder)
        if selectedTextColumn != placeholder:
            st.session_state.detoxifyData['textColumn'] = selectedTextColumn
        else:
            st.session_state.detoxifyData['textColumn'] = None

    # Step 1 completion indicator
    if step1_complete:
        st.success("✅ **Step 1 completed!** Dataset loaded and text column selected.")

st.markdown("")

# =============================================================================
# Step 2: Model Selection
# =============================================================================

# Check if step 2 is complete
step2_complete = st.session_state.detoxifyData['modelLoaded']

with st.container(border=True):
    st.markdown("### 🛡️ Step 2: Detoxify Model Selection")
    st.markdown("Choose a Detoxify model for toxicity classification. Each model has different capabilities and label sets.")

    # Get available models
    available_models = get_available_models()

    # Model selection
    st.markdown("#### Available Models")

    # Display model options
    model_options = list(available_models.keys())
    model_labels = [f"{available_models[m]['name']} - {available_models[m]['description']}" for m in model_options]

    # Get current selection
    current_model = st.session_state.detoxifyData.get('modelName')
    if current_model and current_model in model_options:
        default_model_index = model_options.index(current_model)
    else:
        default_model_index = 0

    selected_model = st.selectbox(
        "Select a Detoxify model:",
        options=model_options,
        index=default_model_index,
        format_func=lambda x: f"{available_models[x]['name']} - {available_models[x]['description']}",
        key="detoxify_model_select"
    )

    st.session_state.detoxifyData['modelName'] = selected_model

    # Show selected model info
    if selected_model:
        model_info = available_models[selected_model]

        st.markdown("**Model Labels:**")
        labels_display = ", ".join([f"`{label}`" for label in model_info['labels']])
        st.markdown(labels_display)

    st.markdown("---")

    # Model loading controls
    col1, col2 = st.columns(2)

    with col1:
        # Button always visible, disabled if model already loaded
        if st.button(
            "📥 Load Model",
            use_container_width=True,
            disabled=st.session_state.detoxifyData['modelLoaded'],
            key="load_detoxify_model"
        ):
            with st.spinner(f"Loading Detoxify model '{selected_model}'..."):
                model, model_info, error = load_detoxify_model(selected_model)

                if error:
                    st.error(f"❌ Error loading model: {error}")
                else:
                    st.session_state.detoxifyData['model'] = model
                    st.session_state.detoxifyData['modelInfo'] = model_info
                    st.session_state.detoxifyData['modelLoaded'] = True
                    st.success(f"✅ Model '{model_info['model_display_name']}' loaded successfully!")
                    st.rerun()

    with col2:
        # Button always visible, disabled if model not loaded
        if st.button(
            "🗑️ Remove Model from Memory",
            use_container_width=True,
            disabled=not st.session_state.detoxifyData['modelLoaded'],
            key="unload_detoxify_model"
        ):
            unload_detoxify_model(st.session_state.detoxifyData['model'])
            st.session_state.detoxifyData['model'] = None
            st.session_state.detoxifyData['modelLoaded'] = False
            st.session_state.detoxifyData['modelInfo'] = {}
            st.success("✅ Model removed from memory!")
            st.rerun()

    # Show hardware notification if model loaded
    if st.session_state.detoxifyData['modelLoaded']:
        model_info = st.session_state.detoxifyData['modelInfo']
        device_name = model_info.get('device_name', 'Unknown')
        device_type = model_info.get('device_type', 'info')

        if device_type == 'success':
            st.success(f"**Allocated Hardware:** {device_name}")
        elif device_type == 'warning':
            st.warning(f"**Allocated Hardware:** {device_name}")
        else:
            st.info(f"**Allocated Hardware:** {device_name}")

    # Show model information if loaded
    if st.session_state.detoxifyData['modelLoaded']:
        st.markdown("#### ℹ️ Model Information")

        model_info = st.session_state.detoxifyData['modelInfo']

        # Create columns for model info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model", model_info.get('model_display_name', 'N/A'))
        with col2:
            st.metric("Number of Labels", model_info.get('num_labels', 'N/A'))
        with col3:
            st.metric("Device", model_info.get('device', 'N/A').upper())

        # Show labels
        if model_info.get('labels'):
            st.markdown("**Detection Labels:**")
            labels_df = pd.DataFrame([
                {'Label': label, 'Description': label.replace('_', ' ').title()}
                for label in model_info['labels']
            ])
            st.dataframe(labels_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Test model section
        st.markdown("#### 🧪 Test Model")
        st.markdown("Enter text to test the model's toxicity detection.")

        test_text = st.text_area(
            "Test text:",
            placeholder="Enter sample text to classify here...",
            height=100,
            key="detoxify_test_text"
        )

        if st.button("🔍 Classify Test Text", use_container_width=True, key="test_detoxify"):
            if test_text.strip():
                with st.spinner("Classifying..."):
                    result, error = classify_single_text(
                        test_text,
                        st.session_state.detoxifyData['model']
                    )

                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.markdown("**Classification Results:**")

                        # Create a dataframe for display
                        results_df = pd.DataFrame([
                            {'Label': label, 'Score': f"{score:.4f}", 'Percentage': f"{score*100:.2f}%"}
                            for label, score in result.items()
                        ])
                        results_df = results_df.sort_values('Score', ascending=False)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)

                        # Show dominant label
                        max_label = max(result.items(), key=lambda x: x[1])
                        if max_label[1] >= 0.5:
                            st.warning(f"⚠️ **Dominant Label:** {max_label[0]} ({max_label[1]*100:.2f}%)")
                        else:
                            st.success(f"✅ **Dominant Label:** {max_label[0]} ({max_label[1]*100:.2f}%) - Below threshold")
            else:
                st.warning("⚠️ Please enter text to test.")

    # Step 2 completion indicator
    if step2_complete:
        st.success("✅ **Step 2 completed!** Model loaded and ready to use.")

st.markdown("")

# =============================================================================
# Step 3: Output Configuration
# =============================================================================

# Check if step 3 is complete
step3_complete = (
    st.session_state.detoxifyData['outputDirectory'] != '' and
    st.session_state.detoxifyData['outputFileName'] != ''
)

with st.container(border=True):
    st.markdown("### ⚙️ Step 3: Output Configuration")
    st.markdown("Configure where and how to save the classification results.")

    col1, col2 = st.columns(2)

    with col1:
        output_format = st.selectbox(
            "File format:",
            options=['csv', 'xlsx', 'json'],
            index=['csv', 'xlsx', 'json'].index(st.session_state.detoxifyData.get('outputFormat', 'csv')),
            key="detoxify_output_format"
        )
        st.session_state.detoxifyData['outputFormat'] = output_format

    with col2:
        output_filename = st.text_input(
            "File name:",
            value=st.session_state.detoxifyData.get('outputFileName', f"detoxify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            help="Do not include file extension",
            key="detoxify_output_filename"
        )
        st.session_state.detoxifyData['outputFileName'] = output_filename

    output_directory = st.text_input(
        "Output directory:",
        value=st.session_state.detoxifyData.get('outputDirectory', os.path.expanduser("~/Downloads")),
        help="Full path to the directory where the file will be saved",
        key="detoxify_output_directory"
    )
    st.session_state.detoxifyData['outputDirectory'] = output_directory

    # Threshold configuration
    st.markdown("---")
    st.markdown("**Classification Threshold:**")
    threshold = st.slider(
        "Threshold for positive classification:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.detoxifyData.get('threshold', 0.5),
        step=0.05,
        help="Scores above this threshold will be considered positive for that label",
        key="detoxify_threshold"
    )
    st.session_state.detoxifyData['threshold'] = threshold

    # Show preview of full path
    if output_filename and output_directory:
        full_path_preview = os.path.join(output_directory, f"{output_filename}.{output_format}")
        st.info(f"📁 File will be saved as: `{full_path_preview}`")
        step3_complete = True

    # Step 3 completion indicator
    if step3_complete:
        st.success("✅ **Step 3 completed!** Output configuration ready.")

st.markdown("")

# =============================================================================
# Step 4: Detoxify Classification
# =============================================================================

# Check if step 4 is complete
step4_complete = st.session_state.detoxifyData['classificationResults'] is not None

# Check if ready for classification
can_classify = (
    st.session_state.detoxifyData['dataset'] is not None and
    st.session_state.detoxifyData['textColumn'] is not None and
    st.session_state.detoxifyData['modelLoaded']
)

with st.container(border=True):
    st.markdown("### 🚀 Step 4: Toxicity Classification")

    if can_classify:
        st.markdown("✅ Everything ready for classification!")

        # Show dataset info
        dataset = st.session_state.detoxifyData['dataset']
        text_column = st.session_state.detoxifyData['textColumn']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Texts", len(dataset))
        with col2:
            st.metric("Text Column", text_column)
        with col3:
            st.metric("Model", st.session_state.detoxifyData['modelInfo'].get('model_display_name', 'N/A'))

        # Classification button
        if st.button("🚀 Start Classification", use_container_width=True, type="primary", key="start_detoxify"):
            st.session_state.detoxifyData['isExecuting'] = True

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Preparing data...")

                # Get all texts
                all_texts = dataset[text_column].astype(str).tolist()
                total_texts = len(all_texts)

                status_text.text(f"Classifying {total_texts} texts...")

                # Progress callback
                def update_progress(current, total):
                    progress = current / total * 0.7  # 70% for classification
                    progress_bar.progress(progress)
                    status_text.text(f"Classifying... {current}/{total} texts")

                # Classify
                model = st.session_state.detoxifyData['model']
                classification_results, error = classify_texts(
                    all_texts,
                    model,
                    batch_size=32,
                    progress_callback=update_progress
                )

                if error:
                    st.error(f"❌ Error during classification: {error}")
                else:
                    progress_bar.progress(0.8)
                    status_text.text("Organizing results...")

                    # Create results dataframe
                    labels = st.session_state.detoxifyData['modelInfo']['labels']
                    threshold = st.session_state.detoxifyData['threshold']
                    results_df = create_results_dataframe(
                        dataset,
                        text_column,
                        classification_results,
                        labels,
                        threshold
                    )

                    # Save results
                    st.session_state.detoxifyData['classificationResults'] = results_df

                    progress_bar.progress(0.9)
                    status_text.text("Saving results...")

                    # Auto-save file
                    output_format = st.session_state.detoxifyData['outputFormat']
                    output_filename = st.session_state.detoxifyData['outputFileName']
                    output_directory = st.session_state.detoxifyData['outputDirectory']

                    filename = f"{output_filename}.{output_format}"
                    full_path = os.path.join(output_directory, filename)

                    save_result = save_classification_results(results_df, full_path, output_format)

                    if save_result['success']:
                        st.session_state.detoxifyData['fileSaved'] = True
                        progress_bar.progress(1.0)
                        status_text.text("Classification completed!")
                        st.success(f"✅ Classification completed! File saved at: `{full_path}`")
                    else:
                        st.warning(f"⚠️ Classification completed but error saving file: {save_result.get('error', 'Unknown error')}")

                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
            finally:
                st.session_state.detoxifyData['isExecuting'] = False

    else:
        missing = []
        if st.session_state.detoxifyData['dataset'] is None:
            missing.append("📁 Dataset")
        if st.session_state.detoxifyData['textColumn'] is None:
            missing.append("📝 Text column")
        if not st.session_state.detoxifyData['modelLoaded']:
            missing.append("🛡️ Loaded model")

        st.info("⚠️ Complete the previous steps before classification.")
        st.markdown("**Missing requirements:**")
        for item in missing:
            st.markdown(f"- {item}")

    # Step 4 completion indicator
    if step4_complete:
        st.success("✅ **Step 4 completed!** Classification performed successfully.")

        # Results Preview
        st.markdown("---")
        st.markdown("#### 📊 Results Preview")

        results_df = st.session_state.detoxifyData['classificationResults']

        # Show toxicity distribution
        st.markdown("**Toxicity Distribution (is_toxic_any):**")
        toxic_dist = results_df['is_toxic_any'].value_counts()
        toxic_dist.index = toxic_dist.index.map({True: 'Toxic', False: 'Not Toxic'})
        st.bar_chart(toxic_dist)

        # Show dominant label distribution
        st.markdown("**Dominant Label Distribution:**")
        label_dist = results_df['dominant_label'].value_counts()
        st.bar_chart(label_dist)

        # Show preview
        st.markdown("**Classified Dataset Preview:**")
        st.dataframe(results_df.head(20), use_container_width=True)

        # Show file location if saved
        if st.session_state.detoxifyData.get('fileSaved', False):
            output_format = st.session_state.detoxifyData['outputFormat']
            output_filename = st.session_state.detoxifyData['outputFileName']
            output_directory = st.session_state.detoxifyData['outputDirectory']
            full_path = os.path.join(output_directory, f"{output_filename}.{output_format}")

            st.success(f"💾 File saved at: `{full_path}`")
        else:
            st.info("⏳ File will be saved automatically after classification.")
