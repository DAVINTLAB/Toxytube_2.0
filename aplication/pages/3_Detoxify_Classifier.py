"""
Detoxify Classification Page
Implements the complete text toxicity classification workflow using global dataset
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from io import BytesIO

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
        'lastLikesColumn': None,            # Last selected likes column
        'lastTopN': 100,                    # Last selected top N value
        'classificationTime': None,         # Total classification time
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

        st.markdown("")

        # Classification options: all or top-N by likes
        classify_all = st.checkbox("Classify all comments", value=True, key="detoxify_classify_all")

        likes_column = None
        top_n = None
        if not classify_all:
            st.markdown("Select the likes column and how many top comments to classify:")
            
            # Determine default index for likes column
            last_likes_col = st.session_state.detoxifyData.get('lastLikesColumn')
            likes_column_options = list(dataset.columns)
            if last_likes_col and last_likes_col in likes_column_options:
                likes_default_index = likes_column_options.index(last_likes_col)
            else:
                likes_default_index = 0
            
            # Allow user to select likes column
            likes_column = st.selectbox(
                "Likes column:",
                options=likes_column_options,
                index=likes_default_index,
                help="Select the column that contains the number of likes for each comment.",
                key="detoxify_likes_column_select"
            )

            max_n = len(dataset)
            last_top_n = st.session_state.detoxifyData.get('lastTopN', 100)
            default_n = min(last_top_n, max_n)
            top_n = st.number_input(
                "Number of top comments to classify (N):",
                min_value=1,
                max_value=max_n,
                value=default_n,
                step=1,
                key="detoxify_top_n"
            )

            st.info(f"Only the top {top_n} comments by `{likes_column}` will be classified.")

        # Classification button
        if st.button("🚀 Start Classification", use_container_width=True, type="primary", key="start_detoxify"):
            st.session_state.detoxifyData['isExecuting'] = True

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()

            try:
                status_text.text("Preparing data...")

                # Decide dataset to classify
                if classify_all:
                    working_df = dataset.copy()
                    original_indices = working_df.index.tolist()
                else:
                    # Ensure likes_column selected
                    if likes_column is None:
                        st.error("⚠️ Please select a likes column to proceed.")
                        st.session_state.detoxifyData['isExecuting'] = False
                        raise Exception("Likes column not selected")

                    # Build top-N dataframe preserving original indices
                    try:
                        top_n_int = int(top_n)
                    except Exception:
                        top_n_int = min(100, len(dataset))

                    working_df = dataset.sort_values(by=likes_column, ascending=False).head(top_n_int)
                    original_indices = working_df.index.tolist()

                # Get texts to classify
                all_texts = working_df[text_column].astype(str).tolist()
                total_texts = len(all_texts)

                status_text.text(f"Classifying {total_texts} texts...")

                # Time tracking variables
                import time
                start_time = time.time()

                # Progress callback with time estimation
                def update_progress(current, total):
                    progress = current / total * 0.7  # 70% for classification
                    progress_bar.progress(progress)
                    status_text.text(f"Classifying... {current}/{total} texts")
                    
                    # Calculate time estimates
                    if current > 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_text = elapsed_time / current
                        remaining_texts = total - current
                        estimated_remaining = avg_time_per_text * remaining_texts
                        
                        # Format time
                        def format_time(seconds):
                            if seconds < 60:
                                return f"{int(seconds)}s"
                            elif seconds < 3600:
                                mins = int(seconds // 60)
                                secs = int(seconds % 60)
                                return f"{mins}m {secs}s"
                            else:
                                hours = int(seconds // 3600)
                                mins = int((seconds % 3600) // 60)
                                return f"{hours}h {mins}m"
                        
                        time_text.text(f"⏱️ Elapsed: {format_time(elapsed_time)} | Avg: {avg_time_per_text:.2f}s/text | Estimated remaining: {format_time(estimated_remaining)}")

                # Classify
                model = st.session_state.detoxifyData['model']
                classification_results, error = classify_texts(
                    all_texts,
                    model,
                    batch_size=16,
                    progress_callback=update_progress
                )

                if error:
                    st.error(f"❌ Error during classification: {error}")
                else:
                    progress_bar.progress(0.8)
                    status_text.text("Organizing results...")

                    # If we classified the entire dataset, simply build results df
                    if classify_all:
                        results_df = create_results_dataframe(
                            dataset,
                            text_column,
                            classification_results
                        )
                    else:
                        # Create dataframe for just the top-N results
                        temp_results_df = create_results_dataframe(
                            working_df,
                            text_column,
                            classification_results
                        )

                        # Initialize results as a copy of the original dataset
                        results_df = dataset.copy()

                        # Ensure classification columns exist and default to 'not_classified'
                        # Get probability columns from temp_results_df
                        prob_columns = [col for col in temp_results_df.columns if col.startswith('detoxify_prob_')]
                        for col in prob_columns:
                            results_df[col] = 'not_classified'

                        # Map the classified top-N back to original indices
                        for col in prob_columns:
                            results_df.loc[original_indices, col] = temp_results_df[col].values

                    # Update global dataset with classification results
                    st.session_state.globalData['dataset'] = results_df
                    st.session_state.detoxifyData['classificationResults'] = results_df
                    
                    # Save classification settings and time
                    total_time = time.time() - start_time
                    st.session_state.detoxifyData['classificationTime'] = total_time
                    if not classify_all and likes_column:
                        st.session_state.detoxifyData['lastLikesColumn'] = likes_column
                        st.session_state.detoxifyData['lastTopN'] = top_n

                    progress_bar.progress(1.0)
                    status_text.text("✅ Classification completed!")
                    st.success(f"✅ Classification completed successfully!")

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
        
        # Show classification time if available
        classification_time = st.session_state.detoxifyData.get('classificationTime')
        if classification_time:
            if classification_time < 60:
                time_str = f"{classification_time:.1f}s"
            elif classification_time < 3600:
                mins = int(classification_time // 60)
                secs = int(classification_time % 60)
                time_str = f"{mins}m {secs}s"
            else:
                hours = int(classification_time // 3600)
                mins = int((classification_time % 3600) // 60)
                time_str = f"{hours}h {mins}m"
            st.info(f"⏱️ **Classification completed in:** {time_str}")

        # Results Preview
        st.markdown("---")
        st.markdown("#### 📊 Results Preview")

        results_df = st.session_state.detoxifyData['classificationResults']

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
