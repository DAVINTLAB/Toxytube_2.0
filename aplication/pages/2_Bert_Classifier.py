"""
Text Classification Page
Implements the complete text classification workflow using global dataset
"""
import streamlit as st
import pandas as pd
import os
import pathlib
import traceback
from datetime import datetime
from io import BytesIO
import sys

from components.navigation import render_navigation, is_configuration_complete, get_configuration_status
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
        'outputFormat': None,
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
    st.markdown("### 📁 Dataset Preview")

    config_status = get_configuration_status()
    if config_status['complete']:
        dataset = st.session_state.globalData['dataset']
        text_column = st.session_state.globalData['textColumn']

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
        col1, col2, col3, spacer, col4, col5 = st.columns([1, 1, 1, 0.5, 1.5, 1.5])

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
            st.metric("💾 Size", size_str)

        with col4:
            st.metric("📄 Source", st.session_state.globalData['originalFileName'])
        with col5:
            st.metric("📝 Text Column", text_column if text_column else "Not selected")



        # Preview
        st.markdown("**Dataset Preview (first 5 rows):**")
        st.dataframe(dataset.head(5), use_container_width=True)

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

        # Check if configuration is complete
        if text_column is None:
            st.warning("⚠️ Please select a text column in the Home page to enable classification.")
    else:
        st.warning("⚠️ **No dataset loaded.** Please upload a dataset in the Home page first.")

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
        # Default model id (prefill) unless user/session already has one
        default_model_id = st.session_state.classificationData.get('modelId') or "cardiffnlp/twitter-xlm-roberta-base-sentiment"

        model_id_input = st.text_input(
            "Hugging Face BERT Model ID:",
            value=default_model_id,
            placeholder="Ex: neuralmind/bert-base-portuguese-cased",
            help="Enter the complete model ID of any BERT-based classifier (user/model-name)"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        check_model_clicked = st.button("🔍 Check Model", use_container_width=True)

    # Model validation result (full width)
    if check_model_clicked:
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
                st.info(f"💡 **Allocated Hardware:** {device_name}")

    # Show model information if loaded
    if st.session_state.classificationData['modelLoaded']:
        st.markdown("#### ℹ️ Model Information")

        model_info = st.session_state.classificationData['modelInfo']

        # Create columns for model info - all in one line
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

        with col1:
            st.metric("Model Type", model_info.get('model_type', 'N/A').upper())
        with col2:
            st.metric("Num Labels", model_info.get('num_labels', 'N/A'))
        with col3:
            st.metric("Device", model_info.get('device', 'N/A'))
        with col4:
            st.metric("Max Length", model_info.get('max_position_embeddings', 'N/A'))
        with col5:
            st.metric("Vocab Size", f"{model_info.get('vocab_size', 0):,}")
        with col6:
            st.metric("Hidden Layers", model_info.get('num_hidden_layers', 'N/A'))
        with col7:
            st.metric("Hidden Size", model_info.get('hidden_size', 'N/A'))
        with col8:
            st.metric("Attn Heads", model_info.get('num_attention_heads', 'N/A'))

        # Show labels
        if model_info.get('labels'):
            st.markdown("**Model Labels:**")
            labels_df = pd.DataFrame([
                {'ID': k, 'Label': v}
                for k, v in model_info['labels'].items()
            ])
            st.dataframe(labels_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Label customization section
        with st.expander("🏷️ Customize Label Names (Optional)", expanded=False):
            st.markdown("Edit the label names according to the model documentation. This will be used in the classification results.")

            original_labels = model_info.get('labels', {})

            # Initialize custom labels in session state if not exists
            if 'customLabels' not in st.session_state.classificationData:
                st.session_state.classificationData['customLabels'] = original_labels.copy()

            custom_labels = st.session_state.classificationData['customLabels']

            st.markdown("**Current Labels:**")

            # Create input fields for each label
            updated_labels = {}
            for label_id, label_name in sorted(original_labels.items()):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.text_input(
                        f"ID {label_id}",
                        value=str(label_id),
                        disabled=True,
                        key=f"label_id_{label_id}"
                    )
                with col2:
                    new_name = st.text_input(
                        f"Label Name",
                        value=custom_labels.get(label_id, label_name),
                        key=f"label_name_{label_id}",
                        placeholder=f"Original: {label_name}"
                    )
                    updated_labels[label_id] = new_name

            # Update button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Custom Labels", use_container_width=True):
                    st.session_state.classificationData['customLabels'] = updated_labels
                    st.success("✅ Custom labels saved!")
                    st.rerun()

            with col2:
                if st.button("🔄 Reset to Original Labels", use_container_width=True):
                    st.session_state.classificationData['customLabels'] = original_labels.copy()
                    st.success("✅ Labels reset to original!")
                    st.rerun()


        # Test model section
        with st.expander("🧪 Test Model", expanded=False):
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

                            # Use custom labels if available, otherwise use original labels
                            if 'customLabels' in st.session_state.classificationData:
                                labels = st.session_state.classificationData['customLabels']
                            else:
                                labels = model_info['labels']

                            prediction, probabilities = classify_single_text(
                                test_text,
                                model,
                                tokenizer,
                                max_length
                            )

                            st.markdown("**Result:**")
                            st.info(f"💡 **Predicted Class:** {labels[prediction]}")

                            st.markdown("**Probabilities by Class:**")
                            prob_df = pd.DataFrame([
                                {'Class': labels[i], 'Probability': f"{prob:.4f}", 'Percentage': f"{prob*100:.2f}%"}
                                for i, prob in enumerate(probabilities)
                            ]).sort_values('Probability', ascending=False)

                            st.dataframe(prob_df, use_container_width=True, hide_index=True)

                            # Pie chart
                            import plotly.express as px

                            # Prepare data for pie chart
                            pie_data = pd.DataFrame([
                                {'Class': labels[i], 'Probability': prob}
                                for i, prob in enumerate(probabilities)
                            ])

                            fig = px.pie(
                                pie_data,
                                values='Probability',
                                names='Class',
                                title='Probability Distribution',
                                hole=0.3  # Makes it a donut chart for better aesthetics
                            )

                            fig.update_traces(
                                textposition='inside',
                                textinfo='percent+label',
                                hovertemplate='<b>%{label}</b><br>Probability: %{value:.4f}<br>Percentage: %{percent}<extra></extra>'
                            )

                            st.plotly_chart(fig, use_container_width=True)

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

        st.markdown("")

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

                # Progress callback: barra 0→1 conforme textos processados; contagem na tela
                def update_progress(current, total):
                    progress_bar.progress(current / total if total else 0)
                    status_text.text(f"Classifying... {current}/{total} texts")

                update_progress(0, total_texts)

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

                progress_bar.progress(1.0)
                status_text.text("Organizing results...")

                # Use custom labels if available, otherwise use original labels
                if 'customLabels' in st.session_state.classificationData:
                    labels = st.session_state.classificationData['customLabels']
                else:
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
                with st.expander("🔍 Traceback completo (clique para expandir)", expanded=True):
                    st.code(traceback.format_exc(), language="text")
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

        st.info("💡 Complete the previous steps before classification.")
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

        # Show threshold-based distribution chart
        st.markdown("**Threshold-based Class Distribution:**")
        st.markdown("This chart shows how many comments would be classified as each class at different probability thresholds.")

        import plotly.graph_objects as go
        import numpy as np

        # Get probability columns
        prob_columns = [col for col in results_df.columns if col.startswith('bert_prob_')]

        # Generate thresholds from 0.0 to 1.0
        thresholds = np.linspace(0.0, 1.0, 101)

        fig = go.Figure()

        # Add a line for each class
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']

        for idx, col in enumerate(prob_columns):
            class_name = col.replace('bert_prob_', '')
            counts_at_threshold = []

            for threshold in thresholds:
                count = (results_df[col] >= threshold).sum()
                counts_at_threshold.append(count)

            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=thresholds,
                y=counts_at_threshold,
                mode='lines',
                name=class_name,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{class_name}</b><br>Threshold: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>'
            ))

        fig.update_layout(
            title='Comments per Class at Different Thresholds',
            xaxis_title='Probability Threshold',
            yaxis_title='Number of Comments',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show preview
        st.markdown("**Classified Dataset Preview:**")
        st.dataframe(results_df.head(20), use_container_width=True)

        # Download button
        output_format = st.session_state.globalData['outputFormat']
        output_filename = st.session_state.globalData['outputFileName']

        st.markdown("---")
        st.markdown("#### 💾 Download Classified Dataset")

        # Convert DataFrame to bytes based on format
        if output_format == 'csv':
            file_data = results_df.to_csv(index=False).encode('utf-8')
            mime_type = 'text/csv'
        elif output_format == 'xlsx':
            buffer = BytesIO()
            results_df.to_excel(buffer, index=False, engine='openpyxl')
            file_data = buffer.getvalue()
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif output_format == 'json':
            file_data = results_df.to_json(orient='records', indent=2).encode('utf-8')
            mime_type = 'application/json'
        elif output_format == 'parquet':
            buffer = BytesIO()
            results_df.to_parquet(buffer, index=False)
            file_data = buffer.getvalue()
            mime_type = 'application/octet-stream'
        else:
            file_data = results_df.to_csv(index=False).encode('utf-8')
            mime_type = 'text/csv'
            output_format = 'csv'

        st.download_button(
            label=f"📥 Download {output_filename}.{output_format}",
            data=file_data,
            file_name=f"{output_filename}.{output_format}",
            mime=mime_type,
            use_container_width=True,
            type="primary"
        )
