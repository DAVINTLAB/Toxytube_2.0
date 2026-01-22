"""
Detoxify Classification Page
Implements the complete text toxicity classification workflow using global dataset
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
        'modelName': None,                  # Selected Detoxify model name
        'modelLoaded': False,               # Model loading status
        'model': None,                      # Loaded model
        'modelInfo': {},                    # Model information
        'threshold': 0.5,                   # Classification threshold
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

st.set_page_config(page_title="Detoxify Classifier - Toxicytube", page_icon="🛡️", layout="wide")

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
step1_complete = st.session_state.detoxifyData['modelLoaded']

with st.container(border=True):
    st.markdown("### 🛡️ Step 1: Detoxify Model Selection")
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

    # Step 1 completion indicator
    if step1_complete:
        st.success("✅ **Step 1 completed!** Model loaded and ready to use.")

st.markdown("")

# =============================================================================
# Step 2: Threshold Configuration
# =============================================================================

with st.container(border=True):
    st.markdown("### ⚙️ Step 2: Classification Threshold")
    st.markdown("Configure the threshold for positive classification.")

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

    st.info(f"📊 Current threshold: **{threshold:.2f}** - Scores ≥ {threshold:.2f} will be marked as positive.")

st.markdown("")

# =============================================================================
# Step 3: Detoxify Classification
# =============================================================================

# Check if step 3 is complete
step3_complete = st.session_state.detoxifyData['classificationResults'] is not None

# Check if ready for classification
can_classify = (
    st.session_state.globalData['dataset'] is not None and
    st.session_state.globalData['textColumn'] is not None and
    st.session_state.detoxifyData['modelLoaded']
)

with st.container(border=True):
    st.markdown("### 🚀 Step 3: Toxicity Classification")

    if can_classify:
        st.markdown("✅ Everything ready for classification!")

        # Show dataset info
        dataset = st.session_state.globalData['dataset']
        text_column = st.session_state.globalData['textColumn']

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

                    # Update global dataset with classification results
                    st.session_state.globalData['dataset'] = results_df
                    st.session_state.detoxifyData['classificationResults'] = results_df

                    progress_bar.progress(0.9)
                    status_text.text("Saving results...")

                    # Auto-save global dataset
                    success, result = save_global_dataset()

                    if success:
                        progress_bar.progress(1.0)
                        status_text.text("Classification completed!")
                        st.success(f"✅ Classification completed! File saved at: `{result}`")
                    else:
                        st.warning(f"⚠️ Classification completed but error saving file: {result}")

                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
            finally:
                st.session_state.detoxifyData['isExecuting'] = False

    else:
        missing = []
        if st.session_state.globalData['dataset'] is None:
            missing.append("📁 Dataset (upload in Home)")
        if st.session_state.globalData['textColumn'] is None:
            missing.append("📝 Text column (select in Home)")
        if not st.session_state.detoxifyData['modelLoaded']:
            missing.append("🛡️ Loaded model")

        st.info("⚠️ Complete the previous steps before classification.")
        st.markdown("**Missing requirements:**")
        for item in missing:
            st.markdown(f"- {item}")

    # Step 3 completion indicator
    if step3_complete:
        st.success("✅ **Step 3 completed!** Classification performed successfully.")

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

        # Show file location
        output_format = st.session_state.globalData['outputFormat']
        output_filename = st.session_state.globalData['outputFileName']
        output_directory = st.session_state.globalData['outputDirectory']
        full_path = os.path.join(output_directory, f"{output_filename}.{output_format}")
        st.success(f"💾 File saved at: `{full_path}`")
