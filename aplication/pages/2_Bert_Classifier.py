"""
Text Classification Page
Implements the complete text classification workflow using global dataset
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
        'modelId': None,                    # Hugging Face model ID
        'modelLoaded': False,               # Model loading status
        'model': None,                      # Loaded model
        'tokenizer': None,                  # Model tokenizer
        'modelInfo': {},                    # Model information
        'classificationResults': None,      # Classification results
        'isExecuting': False,               # Execution status
    }

# Initialize global data if not exists
if 'globalData' not in st.session_state:
    st.session_state.globalData = {
        'dataset': None,
        'textColumn': None,
        'outputDirectory': os.path.expanduser('~/Downloads'),
        'outputFileName': '',
        'outputFormat': 'csv',
        'datasetLoaded': False,
        'originalFileName': ''
    }

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
# Page Configuration
# =============================================================================

st.set_page_config(page_title="BERT Classifier - Toxicytube", page_icon="🤖", layout="wide")

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
# Global Dataset Preview
# =============================================================================

with st.container(border=True):
    st.markdown("### 📁 Global Dataset Preview")
    
    if st.session_state.globalData['datasetLoaded'] and st.session_state.globalData['dataset'] is not None:
        dataset = st.session_state.globalData['dataset']
        text_column = st.session_state.globalData['textColumn']
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Rows", f"{len(dataset):,}")
        with col2:
            st.metric("📋 Columns", len(dataset.columns))
        with col3:
            st.metric("📝 Text Column", text_column if text_column else "Not selected")
        with col4:
            st.metric("📄 Source", st.session_state.globalData['originalFileName'])
        
        # Preview
        st.markdown("**Dataset Preview (first 5 rows):**")
        st.dataframe(dataset.head(5), use_container_width=True)
        
        # Check if configuration is complete
        if text_column is None:
            st.warning("⚠️ Please select a text column in the Home page to enable classification.")
    else:
        st.warning("⚠️ **No dataset loaded.** Please upload a dataset in the Home page first.")
        st.info("👈 Go to **Home** to upload and configure your dataset.")

st.markdown("")

# =============================================================================
# Step 1: Model Selection
# =============================================================================

# Check if step 1 is complete
step1_complete = st.session_state.classificationData['modelLoaded']

with st.container(border=True):
    st.markdown("### 🤖 Step 1: BERT Model Selection")
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

    # Step 1 completion indicator
    if step1_complete:
        st.success("✅ **Step 1 completed!** Model loaded and ready to use.")

st.markdown("")

# =============================================================================
# Step 2: BERT Classification
# =============================================================================

# Check if step 2 is complete
step2_complete = st.session_state.classificationData['classificationResults'] is not None

# Check if ready for classification
can_classify = (
    st.session_state.globalData['dataset'] is not None and
    st.session_state.globalData['textColumn'] is not None and
    st.session_state.classificationData['modelLoaded']
)

with st.container(border=True):
    st.markdown("### 🚀 Step 2: BERT Classification")

    if can_classify:
        st.markdown("✅ Everything ready for classification!")

        # Check text lengths
        dataset = st.session_state.globalData['dataset']
        text_column = st.session_state.globalData['textColumn']
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

                # Update global dataset with classification results
                st.session_state.globalData['dataset'] = results_df
                st.session_state.classificationData['classificationResults'] = results_df

                progress_bar.progress(0.9)
                status_text.text("Saving results...")

                # Auto-save global dataset
                success, result = save_global_dataset()

                if success:
                    progress_bar.progress(1.0)
                    status_text.text("✅ Classification completed and file saved!")
                    st.success(f"✅ {total_texts} texts classified and saved at: {result}")
                else:
                    progress_bar.progress(1.0)
                    status_text.text("✅ Classification completed!")
                    st.warning(f"✅ {total_texts} texts classified, but error saving file: {result}")

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
            finally:
                st.session_state.classificationData['isExecuting'] = False

    else:
        missing = []
        if st.session_state.globalData['dataset'] is None:
            missing.append("📁 Dataset (upload in Home)")
        if st.session_state.globalData['textColumn'] is None:
            missing.append("📝 Text column (select in Home)")
        if not st.session_state.classificationData['modelLoaded']:
            missing.append("🤖 Loaded model")

        st.info("⚠️ Complete the previous steps before classification.")
        st.markdown("**Missing requirements:**")
        for item in missing:
            st.markdown(f"- {item}")

    # Step 2 completion indicator
    if step2_complete:
        st.success("✅ **Step 2 completed!** Classification performed successfully.")

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

        # Show file location
        output_format = st.session_state.globalData['outputFormat']
        output_filename = st.session_state.globalData['outputFileName']
        output_directory = st.session_state.globalData['outputDirectory']
        full_path = os.path.join(output_directory, f"{output_filename}.{output_format}")
        st.success(f"💾 File saved at: `{full_path}`")
