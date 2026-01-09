"""
Text Classification Page
Implements the complete text classification workflow
"""
import streamlit as st
import pandas as pd
import os
import pathlib
from datetime import datetime
import sys

from components.navigation import render_navigation
from components.text_classifier import (
    get_available_device,
    is_bert_classifier,
    load_model,
    unload_model,
    classify_single_text,
    classify_texts,
    calculate_text_lengths,
    create_results_dataframe,
    save_classification_results
)

# =============================================================================
# Initialize Session State - Classification Process Data
# =============================================================================

# Dictionary with all data needed for the text classification process
if 'classificationData' not in st.session_state:
    st.session_state.classificationData = {
        'dataset': None,                    # Dataset loaded by user
        'textColumn': None,                 # Column selected for classification
        'modelId': None,                    # Hugging Face model ID
        'modelLoaded': False,               # Model loading status
        'model': None,                      # Loaded model
        'tokenizer': None,                  # Model tokenizer
        'modelInfo': {},                    # Model information
        'outputDirectory': os.path.expanduser('~/Downloads'),  # Output directory
        'outputFileName': f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Output file name
        'outputFormat': 'csv',              # File format (csv, xlsx, json)
        'classificationResults': None,      # Classification results
        'isExecuting': False,               # Execution status
        'fileSaved': False                  # File saved status
    }

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(page_title="BERT Classifier - Toxytube", page_icon="🤖", layout="wide")

# Render navigation sidebar
render_navigation('classification')



# =============================================================================
# Main Content
# =============================================================================

# Page header
st.markdown("# 🤖 BERT Classifier")
st.markdown("Classify text using BERT-based models from Hugging Face in a simple and intuitive way. Compatible with any BERT architecture model.")

st.markdown("---")

# =============================================================================
# Step 1: Dataset Selection
# =============================================================================

# Check if step 1 is complete
step1_complete = (
    st.session_state.classificationData['dataset'] is not None and
    st.session_state.classificationData['textColumn'] is not None
)

with st.container(border=True):
    st.markdown("### 📁 Step 1: Dataset Selection")
    st.markdown("Upload a file for BERT-based text classification.")

    uploadedFile = st.file_uploader("Select a file:", type=['csv', 'json', 'xlsx', 'xls'])

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
                st.session_state.classificationData['dataset'] = dataset

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    # Preview the dataset if loaded
    if st.session_state.classificationData['dataset'] is not None:
        st.markdown("#### Dataset Preview")

        dataset = st.session_state.classificationData['dataset']

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
        current_selection = st.session_state.classificationData.get('textColumn')
        if current_selection and current_selection in textColumns:
            default_index = options_with_placeholder.index(current_selection)
        else:
            default_index = 0

        selectedTextColumn = st.selectbox(
            "Select the column containing text for classification (must be a text/string column):",
            options=options_with_placeholder,
            index=default_index,
            help="This column will be used as input for the BERT model"
        )

        # Only update if a valid column is selected (not placeholder)
        if selectedTextColumn != placeholder:
            st.session_state.classificationData['textColumn'] = selectedTextColumn
        else:
            st.session_state.classificationData['textColumn'] = None

    # Step 1 completion indicator
    if step1_complete:
        st.success("✅ **Step 1 completed!** Dataset loaded and text column selected.")

st.markdown("")

# =============================================================================
# Step 2: Model Selection
# =============================================================================

# Check if step 2 is complete
step2_complete = st.session_state.classificationData['modelLoaded']

with st.container(border=True):
    st.markdown("### 🤖 Step 2: BERT Model Selection")
    st.markdown("Choose any BERT-based classifier model from Hugging Face. Works with BERT, RoBERTa, DistilBERT, ALBERT, and other BERT architectures.")

    # Model ID input
    col1, col2 = st.columns([3, 1])

    with col1:
        model_id_input = st.text_input(
            "Hugging Face BERT Model ID:",
            value=st.session_state.classificationData.get('modelId', ''),
            placeholder="Ex: neuralmind/bert-base-portuguese-cased",
            help="Enter the complete model ID of any BERT-based classifier (user/model-name)"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Check Model", use_container_width=True):
            if model_id_input:
                with st.spinner("Checking model..."):
                    is_valid, config = is_bert_classifier(model_id_input)

                    if is_valid:
                        st.success(f"✅ Valid BERT classifier model!")
                        st.session_state.classificationData['modelId'] = model_id_input
                    else:
                        st.error("❌ This model is not a valid BERT classifier. Please choose another model.")
            else:
                st.warning("⚠️ Please enter a model ID.")

    # Model loading controls
    if st.session_state.classificationData.get('modelId'):
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Button always visible, disabled if model already loaded
            if st.button(
                "📥 Download and Load Model",
                use_container_width=True,
                disabled=st.session_state.classificationData['modelLoaded']
            ):
                with st.spinner(f"Downloading model {st.session_state.classificationData['modelId']}..."):
                    model, tokenizer, model_info, error = load_model(st.session_state.classificationData['modelId'])

                    if error is None:
                        st.session_state.classificationData['model'] = model
                        st.session_state.classificationData['tokenizer'] = tokenizer
                        st.session_state.classificationData['modelInfo'] = model_info
                        st.session_state.classificationData['modelLoaded'] = True
                        st.rerun()
                    else:
                        st.error(f"❌ Error loading model: {error}")

        with col2:
            # Button always visible, disabled if model not loaded
            if st.button(
                "🗑️ Remove Model from Memory",
                use_container_width=True,
                disabled=not st.session_state.classificationData['modelLoaded']
            ):
                unload_model(
                    st.session_state.classificationData['model'],
                    st.session_state.classificationData['tokenizer']
                )
                st.session_state.classificationData['model'] = None
                st.session_state.classificationData['tokenizer'] = None
                st.session_state.classificationData['modelLoaded'] = False
                st.session_state.classificationData['modelInfo'] = {}
                st.success("✅ Model removed from memory!")
                st.rerun()

        # Show hardware notification if model loaded
        if st.session_state.classificationData['modelLoaded']:
            model_info = st.session_state.classificationData['modelInfo']
            device_name = model_info.get('device_name', 'Unknown')
            device_type = model_info.get('device_type', 'info')

            if device_type == 'success':
                st.success(f"**Allocated Hardware:** {device_name}")
            elif device_type == 'warning':
                st.warning(f"**Allocated Hardware:** {device_name}")
            else:
                st.info(f"**Allocated Hardware:** {device_name}")

    # Show model information if loaded
    if st.session_state.classificationData['modelLoaded']:
        st.markdown("#### ℹ️ Model Information")

        model_info = st.session_state.classificationData['modelInfo']

        # Create columns for model info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model Type", model_info.get('model_type', 'N/A').upper())
            st.metric("Number of Labels", model_info.get('num_labels', 'N/A'))
            st.metric("Device", model_info.get('device', 'N/A'))

        with col2:
            st.metric("Max Length", model_info.get('max_position_embeddings', 'N/A'))
            st.metric("Vocabulary Size", f"{model_info.get('vocab_size', 0):,}")
            st.metric("Hidden Layers", model_info.get('num_hidden_layers', 'N/A'))

        with col3:
            st.metric("Hidden Size", model_info.get('hidden_size', 'N/A'))
            st.metric("Attention Heads", model_info.get('num_attention_heads', 'N/A'))

        # Show labels
        if model_info.get('labels'):
            st.markdown("**Model Labels:**")
            labels_df = pd.DataFrame([
                {'ID': k, 'Label': v}
                for k, v in model_info['labels'].items()
            ])
            st.dataframe(labels_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Test model section
        st.markdown("#### 🧪 Test Model")
        st.markdown("Enter text to test the model's classification.")

        test_text = st.text_area(
            "Test text:",
            placeholder="Enter sample text to classify here...",
            height=100
        )

        if st.button("🔍 Classify Test Text", use_container_width=True):
            if test_text.strip():
                with st.spinner("Classifying..."):
                    try:
                        model = st.session_state.classificationData['model']
                        tokenizer = st.session_state.classificationData['tokenizer']
                        max_length = model_info['max_position_embeddings']
                        labels = model_info['labels']

                        prediction, probabilities = classify_single_text(
                            test_text,
                            model,
                            tokenizer,
                            max_length
                        )

                        st.markdown("**Result:**")
                        st.success(f"**Predicted Class:** {labels[prediction]}")

                        st.markdown("**Probabilities by Class:**")
                        prob_df = pd.DataFrame([
                            {'Class': labels[i], 'Probability': f"{prob:.4f}", 'Percentage': f"{prob*100:.2f}%"}
                            for i, prob in enumerate(probabilities)
                        ]).sort_values('Probability', ascending=False)

                        st.dataframe(prob_df, use_container_width=True, hide_index=True)

                        # Bar chart
                        st.bar_chart(prob_df.set_index('Class')['Probability'].astype(float))

                    except Exception as e:
                        st.error(f"❌ Classification error: {str(e)}")
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
    st.session_state.classificationData['outputDirectory'] != '' and
    st.session_state.classificationData['outputFileName'] != ''
)

with st.container(border=True):
    st.markdown("### ⚙️ Step 3: Output Configuration")
    st.markdown("Configure where and how to save the classification results.")

    col1, col2 = st.columns(2)

    with col1:
        output_format = st.selectbox(
            "File format:",
            options=['csv', 'xlsx', 'json'],
            index=['csv', 'xlsx', 'json'].index(st.session_state.classificationData.get('outputFormat', 'csv'))
        )
        st.session_state.classificationData['outputFormat'] = output_format

    with col2:
        output_filename = st.text_input(
            "File name:",
            value=st.session_state.classificationData.get('outputFileName', f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            help="Do not include file extension"
        )
        st.session_state.classificationData['outputFileName'] = output_filename

    output_directory = st.text_input(
        "Output directory:",
        value=st.session_state.classificationData.get('outputDirectory', os.path.expanduser("~/Downloads")),
        help="Full path to the directory where the file will be saved"
    )
    st.session_state.classificationData['outputDirectory'] = output_directory

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
# Step 4: BERT Classification
# =============================================================================

# Check if step 4 is complete
step4_complete = st.session_state.classificationData['classificationResults'] is not None

# Check if ready for classification
can_classify = (
    st.session_state.classificationData['dataset'] is not None and
    st.session_state.classificationData['textColumn'] is not None and
    st.session_state.classificationData['modelLoaded']
)

with st.container(border=True):
    st.markdown("### 🚀 Step 4: BERT Classification")

    if can_classify:
        st.markdown("✅ Everything ready for classification!")

        # Check text lengths
        dataset = st.session_state.classificationData['dataset']
        text_column = st.session_state.classificationData['textColumn']
        tokenizer = st.session_state.classificationData['tokenizer']
        max_length = st.session_state.classificationData['modelInfo']['max_position_embeddings']

        # Calculate text lengths
        texts = dataset[text_column].astype(str).tolist()
        length_stats = calculate_text_lengths(texts, tokenizer, sample_size=100)
        avg_length = length_stats['avg_length']
        max_text_length = length_stats['max_length']

        # Show text length info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Length (tokens)", f"{avg_length:.0f}")
        with col2:
            st.metric("Maximum Length (tokens)", max_text_length)
        with col3:
            color = "green" if max_text_length <= max_length else "red"
            st.metric("Model Limit", max_length)

        if max_text_length > max_length:
            st.warning(f"⚠️ Some texts exceed the model's maximum length ({max_length} tokens). They will be truncated.")

        # Classification button
        if st.button("🚀 Start Classification", use_container_width=True, type="primary"):
            st.session_state.classificationData['isExecuting'] = True

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
                    progress = 0.2 + (current / total * 0.6)  # 20% to 80%
                    progress_bar.progress(progress)

                # Classify
                model = st.session_state.classificationData['model']
                predictions, probabilities = classify_texts(
                    all_texts,
                    model,
                    tokenizer,
                    max_length,
                    batch_size=8,
                    progress_callback=update_progress
                )

                progress_bar.progress(0.8)
                status_text.text("Organizing results...")

                # Create results dataframe
                labels = st.session_state.classificationData['modelInfo']['labels']
                results_df = create_results_dataframe(
                    dataset,
                    text_column,
                    predictions,
                    probabilities,
                    labels
                )

                # Save results
                st.session_state.classificationData['classificationResults'] = results_df

                progress_bar.progress(0.9)
                status_text.text("Saving results...")

                # Auto-save file
                output_format = st.session_state.classificationData['outputFormat']
                output_filename = st.session_state.classificationData['outputFileName']
                output_directory = st.session_state.classificationData['outputDirectory']

                filename = f"{output_filename}.{output_format}"
                full_path = os.path.join(output_directory, filename)

                save_result = save_classification_results(results_df, full_path, output_format)

                if save_result['success']:
                    st.session_state.classificationData['fileSaved'] = True
                    progress_bar.progress(1.0)
                    status_text.text("✅ Classification completed and file saved!")
                    st.success(f"✅ {total_texts} texts classified and saved at: {save_result['path']}")
                else:
                    progress_bar.progress(1.0)
                    status_text.text("✅ Classification completed!")
                    st.warning(f"✅ {total_texts} texts classified, but error saving file: {save_result['error']}")

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
            finally:
                st.session_state.classificationData['isExecuting'] = False

    else:
        missing = []
        if st.session_state.classificationData['dataset'] is None:
            missing.append("📁 Dataset")
        if st.session_state.classificationData['textColumn'] is None:
            missing.append("📝 Text column")
        if not st.session_state.classificationData['modelLoaded']:
            missing.append("🤖 Loaded model")

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

        results_df = st.session_state.classificationData['classificationResults']

        # Show distribution
        st.markdown("**Label Distribution:**")
        label_dist = results_df['predicted_label'].value_counts()
        st.bar_chart(label_dist)

        # Show preview
        st.markdown("**Classified Dataset Preview:**")
        st.dataframe(results_df.head(20), use_container_width=True)

        # Show file location if saved
        if st.session_state.classificationData.get('fileSaved', False):
            output_format = st.session_state.classificationData['outputFormat']
            output_filename = st.session_state.classificationData['outputFileName']
            output_directory = st.session_state.classificationData['outputDirectory']
            full_path = os.path.join(output_directory, f"{output_filename}.{output_format}")

            st.success(f"💾 File saved at: `{full_path}`")
        else:
            st.info("⏳ File will be saved automatically after classification.")

