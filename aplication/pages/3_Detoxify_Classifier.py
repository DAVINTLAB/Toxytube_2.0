"""
Detoxify Classification Page
Implements the complete text toxicity classification workflow using global dataset
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime

from components.navigation import render_navigation, is_configuration_complete, get_configuration_status
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
            st.info(f"💡 **Allocated Hardware:** {device_name}")
        elif device_type == 'warning':
            st.info(f"💡 **Allocated Hardware:** {device_name}")
        else:
            st.info(f"💡 **Allocated Hardware:** {device_name}")

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
        with st.expander("🧪 Test Model", expanded=False):
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
                            # Show dominant label with info notification
                            max_label = max(result.items(), key=lambda x: x[1])
                            st.info(f"💡 **Dominant Label:** {max_label[0]} ({max_label[1]*100:.2f}%)")

                            st.markdown("**Scores by Label:**")

                            # Create a dataframe for display
                            results_df = pd.DataFrame([
                                {'Label': label, 'Score': f"{score:.4f}", 'Percentage': f"{score*100:.2f}%"}
                                for label, score in result.items()
                            ])
                            results_df = results_df.sort_values('Score', ascending=False)
                            st.dataframe(results_df, use_container_width=True, hide_index=True)

                            # Pie chart
                            import plotly.express as px

                            # Prepare data for pie chart
                            pie_data = pd.DataFrame([
                                {'Label': label, 'Score': score}
                                for label, score in result.items()
                            ])

                            fig = px.pie(
                                pie_data,
                                values='Score',
                                names='Label',
                                title='Toxicity Score Distribution',
                                hole=0.3  # Makes it a donut chart for better aesthetics
                            )

                            fig.update_traces(
                                textposition='inside',
                                textinfo='percent+label',
                                hovertemplate='<b>%{label}</b><br>Score: %{value:.4f}<br>Percentage: %{percent}<extra></extra>'
                            )

                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Please enter text to test.")

    # Step 1 completion indicator
    if step1_complete:
        st.success("✅ **Step 1 completed!** Model loaded and ready to use.")

st.markdown("")

# =============================================================================
# Step 2: Detoxify Classification
# =============================================================================

# Check if step 2 is complete
step2_complete = st.session_state.detoxifyData['classificationResults'] is not None

# Check if ready for classification
can_classify = (
    st.session_state.globalData['dataset'] is not None and
    st.session_state.globalData['textColumn'] is not None and
    st.session_state.detoxifyData['modelLoaded']
)

with st.container(border=True):
    st.markdown("### 🚀 Step 2: Toxicity Classification")

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
                    results_df = create_results_dataframe(
                        dataset,
                        text_column,
                        classification_results
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

        results_df = st.session_state.detoxifyData['classificationResults']

        # Show threshold-based distribution chart
        st.markdown("**Threshold-based Class Distribution:**")
        st.markdown("This chart shows how many comments would be classified as each class at different probability thresholds.")

        import plotly.graph_objects as go
        import numpy as np

        # Get probability columns
        prob_columns = [col for col in results_df.columns if col.startswith('detoxify_prob_')]

        # Generate thresholds from 0.0 to 1.0
        thresholds = np.linspace(0.0, 1.0, 101)

        fig = go.Figure()

        # Add a line for each class
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']

        for idx, col in enumerate(prob_columns):
            class_name = col.replace('detoxify_prob_', '')
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

        # Show file location
        output_format = st.session_state.globalData['outputFormat']
        output_filename = st.session_state.globalData['outputFileName']
        output_directory = st.session_state.globalData['outputDirectory']
        full_path = os.path.join(output_directory, f"{output_filename}.{output_format}")
        st.info(f"💡 File saved at: `{full_path}`")
