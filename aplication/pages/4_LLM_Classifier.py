"""
LLM Classification Page
Implements text classification workflow using LiteLLM and DSPy with global dataset
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from io import BytesIO

from components.navigation import render_navigation, is_configuration_complete, get_configuration_status
from components.llm_classifier import (
    get_available_models,
    get_providers,
    get_models_by_provider,
    set_api_key,
    validate_api_key,
    test_model_connection,
    classify_single_text_llm,
    classify_texts_llm,
    create_results_dataframe,
    save_classification_results,
    cleanup
)

# =============================================================================
# Initialize Session State - Classification Process Data
# =============================================================================

# Dictionary with all data needed for the LLM classification process
if 'llmData' not in st.session_state:
    st.session_state.llmData = {
        'selectedProvider': None,           # Selected LLM provider
        'selectedModel': None,              # Selected LLM model ID
        'apiKey': '',                        # API key
        'apiKeyValidated': False,           # API key validation status
        'promptInstructions': '',           # Classification instructions
        'promptLabels': '',                 # Optional: comma-separated labels
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

st.set_page_config(page_title="LLM Classifier - Toxicytube", page_icon="🧠", layout="wide")

# Render navigation sidebar
render_navigation('llm')

# =============================================================================
# Main Content
# =============================================================================

# Page header
st.markdown("# 🧠 LLM Classifier")
st.markdown("Classify text using Large Language Models via LiteLLM and DSPy. Build custom prompts and classify text with any major LLM provider.")

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
# Step 1: Model and API Key Configuration
# =============================================================================

# Check if step 1 is complete
step1_complete = (
    st.session_state.llmData['selectedModel'] is not None and
    st.session_state.llmData['apiKey'] != '' and
    st.session_state.llmData['apiKeyValidated']
)

with st.container(border=True):
    st.markdown("### 🤖 Step 1: Model and API Key Configuration")
    st.markdown("Select an LLM provider and model, then configure your API key.")

    # Get available models
    available_models = get_available_models()
    providers = get_providers()

    col1, col2 = st.columns(2)

    with col1:
        # Provider selection
        current_provider = st.session_state.llmData.get('selectedProvider')
        if current_provider and current_provider in providers:
            provider_index = providers.index(current_provider)
        else:
            provider_index = 0

        selected_provider = st.selectbox(
            "Select Provider:",
            options=providers,
            index=provider_index,
            key="llm_provider_select"
        )
        st.session_state.llmData['selectedProvider'] = selected_provider

    with col2:
        # Model selection (filtered by provider)
        provider_models = get_models_by_provider(selected_provider)
        model_ids = list(provider_models.keys())

        if model_ids:
            current_model = st.session_state.llmData.get('selectedModel')
            if current_model and current_model in model_ids:
                model_index = model_ids.index(current_model)
            else:
                model_index = 0

            selected_model = st.selectbox(
                "Select Model:",
                options=model_ids,
                index=model_index,
                format_func=lambda x: f"{available_models[x]['name']}",
                key="llm_model_select"
            )
            st.session_state.llmData['selectedModel'] = selected_model
        else:
            st.warning("⚠️ No models available for this provider.")
            selected_model = None

    # API Key input

    api_key_input = st.text_input(
        f"Enter your {selected_provider} API Key:",
        value=st.session_state.llmData.get('apiKey', ''),
        type="password",
        help=f"Your API key for {selected_provider}. This is stored only in session memory.",
        key="llm_api_key"
    )
    st.session_state.llmData['apiKey'] = api_key_input

    # Reset validation if API key changes
    if api_key_input != st.session_state.llmData.get('_last_api_key', ''):
        st.session_state.llmData['apiKeyValidated'] = False
        st.session_state.llmData['_last_api_key'] = api_key_input

    if st.button("🔍 Validate API Key", use_container_width=True, key="validate_api_key"):
        if api_key_input:
            with st.spinner("Validating API key..."):
                is_valid, message = validate_api_key(selected_model, api_key_input)
                if is_valid:
                    st.session_state.llmData['apiKeyValidated'] = True
                    st.rerun()
                else:
                    st.session_state.llmData['apiKeyValidated'] = False
                    st.error(message)
        else:
            st.warning("⚠️ Please enter an API key.")

    if st.session_state.llmData['apiKeyValidated']:
        st.success("✅ API Key Validated")


        # Test model section
        with st.expander("🧪 Test Model Connection", expanded=False):
            st.markdown("Send a test message to verify the model is working correctly.")

            test_message = st.text_area(
                "Test message:",
                placeholder="Enter a message to test the model...\nExample: Hello, can you classify this text as positive or negative?",
                height=100,
                key="llm_test_message"
            )

            if st.button("🚀 Send Test Message", use_container_width=True, key="test_llm_connection"):
                if test_message.strip() and api_key_input:
                    with st.spinner("Sending message to model..."):
                        response, model_returned, error = test_model_connection(selected_model, api_key_input, test_message)

                        if error:
                            st.error(f"❌ {error}")
                        else:
                            st.markdown("**Response:**")
                            st.markdown(f"> {response}")
                            st.markdown(f"**Model (from response):** {model_returned if model_returned else 'Unknown'}")
                elif not api_key_input:
                    st.warning("⚠️ Please enter an API key first.")
                else:
                    st.warning("⚠️ Please enter a test message.")

    # Step 1 completion indicator
    if step1_complete:
        st.success("✅ **Step 1 completed!** Model configured and API key validated.")


# =============================================================================
# Step 2: Prompt Configuration (DSPy Style)
# =============================================================================

# Check if step 2 is complete
step2_complete = st.session_state.llmData['promptInstructions'] != ''

with st.container(border=True):
    st.markdown("### 📝 Step 2: Prompt Configuration")

    st.markdown("#### Classification Instructions")
    st.markdown("Define the instructions for the LLM classifier. Be specific about what the model should classify and how.")

    default_instructions = """You are a sentiment analyzer. Classify the sentiment of the following comment as "positive", "negative", or "neutral"."""

    prompt_instructions = st.text_area(
        "Instructions (required):",
        value=st.session_state.llmData.get('promptInstructions', '') or default_instructions,
        height=300,
        key="llm_prompt_instructions"
    )
    st.session_state.llmData['promptInstructions'] = prompt_instructions


    st.markdown("#### Classification Labels (Optional)")
    st.markdown("If you want to restrict the output to specific labels, list them here. Leave empty for free-form classification.")

    default_labels = "positive, negative, neutral"

    prompt_labels = st.text_input(
        "Labels (comma-separated, optional):",
        value=st.session_state.llmData.get('promptLabels', '') or default_labels,
        help="The model will be instructed to choose one of these labels",
        key="llm_prompt_labels"
    )
    st.session_state.llmData['promptLabels'] = prompt_labels

    # Test prompt with single text
    if step1_complete and prompt_instructions.strip():
        st.markdown("---")

        with st.expander("🧪 Test Prompt", expanded=False):
            st.markdown("Test your prompt configuration with a sample text.")

            test_text_prompt = st.text_area(
                "Sample text to classify:",
                placeholder="Enter sample text to test the classification prompt...",
                height=80,
                key="llm_test_prompt_text"
            )

            if st.button("🔍 Test Classification", use_container_width=True, key="test_classification"):
                if test_text_prompt.strip():
                    with st.spinner("Classifying..."):
                        # Build a best-effort preview of the prompt sent to the model
                        sent_prompt_preview = prompt_instructions.strip() + "\n\nText:\n" + test_text_prompt.strip()
                        if prompt_labels and prompt_labels.strip():
                            sent_prompt_preview += "\n\nLabels: " + prompt_labels.strip()

                        result, raw_response, error = classify_single_text_llm(
                            test_text_prompt,
                            prompt_instructions,
                            prompt_labels if prompt_labels.strip() else None,
                            st.session_state.llmData['selectedModel'],
                            st.session_state.llmData['apiKey']
                        )

                        if error:
                            st.error(f"❌ {error}")
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Classification", result['classification'])
                            with col2:
                                st.metric("Confidence", result['confidence'])
                            st.markdown(f"**Reasoning:** {result['reasoning']}")

                            st.markdown("---")
                            st.markdown("**Prompt sent to model (preview):**")
                            st.code(sent_prompt_preview, language='text')

                            st.markdown("**Raw response from model (best-effort):**")
                            # Show raw_response as JSON-like or string
                            try:
                                st.code(raw_response, language='json')
                            except Exception:
                                st.code(str(raw_response), language='text')
                else:
                    st.warning("⚠️ Please enter sample text to test.")

        with st.expander("💰 Estimate Classification Cost", expanded=False):
            st.markdown("Estimate the total cost of classifying all texts in your dataset.")

            # Check if dataset is available
            if st.session_state.globalData['dataset'] is not None and st.session_state.globalData['textColumn'] is not None:
                dataset = st.session_state.globalData['dataset']
                text_column = st.session_state.globalData['textColumn']

                # Calculate estimated tokens
                # Rough estimation: 1 token ≈ 4 characters for English text
                all_texts = dataset[text_column].astype(str).tolist()
                total_text_chars = sum(len(text) for text in all_texts)
                instruction_chars = len(prompt_instructions) * len(all_texts)  # Instructions sent with each text
                labels_chars = len(prompt_labels) * len(all_texts) if prompt_labels.strip() else 0

                total_chars = total_text_chars + instruction_chars + labels_chars
                estimated_input_tokens = total_chars / 4  # Approximate tokens

                # Estimate output tokens (roughly 50 tokens per response for classification + reasoning)
                estimated_output_tokens = len(all_texts) * 50

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Texts", f"{len(all_texts):,}")
                with col2:
                    st.metric("Est. Input Tokens", f"{int(estimated_input_tokens):,}")
                with col3:
                    st.metric("Est. Output Tokens", f"{int(estimated_output_tokens):,}")

                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    input_token_cost = st.number_input(
                        "Cost per 1M input tokens ($):",
                        min_value=0.0,
                        value=0.15,
                        step=0.01,
                        format="%.4f",
                        help="Check your provider's pricing page for the cost per million input tokens",
                        key="input_token_cost"
                    )
                with col2:
                    output_token_cost = st.number_input(
                        "Cost per 1M output tokens ($):",
                        min_value=0.0,
                        value=0.60,
                        step=0.01,
                        format="%.4f",
                        help="Check your provider's pricing page for the cost per million output tokens",
                        key="output_token_cost"
                    )

                # Calculate estimated cost
                input_cost = (estimated_input_tokens / 1_000_000) * input_token_cost
                output_cost = (estimated_output_tokens / 1_000_000) * output_token_cost
                total_cost = input_cost + output_cost

                st.markdown("---")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Input Cost", f"${input_cost:.4f}")
                with col2:
                    st.metric("Output Cost", f"${output_cost:.4f}")
                with col3:
                    st.metric("Total Estimated Cost", f"${total_cost:.4f}")

                st.info("💡 This is an approximation. Actual costs may vary based on tokenization and response length.")
            else:
                st.warning("⚠️ Please load a dataset first to estimate costs.")

    # Step 2 completion indicator
    if step2_complete:
        st.success("✅ **Step 2 completed!** Prompt configured.")

st.markdown("")

# =============================================================================
# Step 3: LLM Classification
# =============================================================================

# Check if step 3 is complete
step3_complete = st.session_state.llmData['classificationResults'] is not None

# Check if ready for classification
can_classify = (
    st.session_state.globalData['dataset'] is not None and
    st.session_state.globalData['textColumn'] is not None and
    st.session_state.llmData['selectedModel'] is not None and
    st.session_state.llmData['apiKeyValidated'] and
    st.session_state.llmData['promptInstructions'] != ''
)

with st.container(border=True):
    st.markdown("### 🚀 Step 3: LLM Classification")

    if can_classify:
        st.markdown("Everything ready for classification!")

        # Show configuration summary
        dataset = st.session_state.globalData['dataset']
        text_column = st.session_state.globalData['textColumn']
        selected_model = st.session_state.llmData['selectedModel']
        model_info = get_available_models()[selected_model]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Texts", len(dataset))
        with col2:
            st.metric("Text Column", text_column)
        with col3:
            st.metric("Model", model_info['name'])

        # Estimated cost warning
        st.warning(f"⚠️ **Note:** This will make {len(dataset)} API calls to {model_info['provider']}. Costs will depend on your provider's pricing.")

        # Classification options: all or top-N by likes
        classify_all = st.checkbox("Classify all comments", value=True, key="llm_classify_all")

        likes_column = None
        top_n = None
        if not classify_all:
            st.markdown("Select the likes column and how many top comments to classify:")
            # Allow user to select likes column
            likes_column = st.selectbox(
                "Likes column:",
                options=list(dataset.columns),
                index=0,
                help="Select the column that contains the number of likes for each comment.",
                key="llm_likes_column_select"
            )

            max_n = len(dataset)
            default_n = min(100, max_n)
            top_n = st.number_input(
                "Number of top comments to classify (N):",
                min_value=1,
                max_value=max_n,
                value=default_n,
                step=1,
                key="llm_top_n"
            )

            st.info(f"Only the top {top_n} comments by `{likes_column}` will be classified.")

        # Classification button
        if st.button("🚀 Start Classification", use_container_width=True, type="primary", key="start_llm_classification"):
            st.session_state.llmData['isExecuting'] = True

            # Progress elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_text_display = st.empty()

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
                        st.session_state.llmData['isExecuting'] = False
                        raise Exception("Likes column not selected")

                    # Build top-N dataframe preserving original indices
                    try:
                        top_n_int = int(top_n)
                    except Exception:
                        top_n_int = int(default_n)

                    working_df = dataset.sort_values(by=likes_column, ascending=False).head(top_n_int)
                    original_indices = working_df.index.tolist()

                # Get texts to classify
                all_texts = working_df[text_column].astype(str).tolist()
                total_texts = len(all_texts)

                status_text.text(f"Classifying {total_texts} texts...")

                # Progress callback
                def update_progress(current, total):
                    progress = current / total * 0.9 if total > 0 else 0
                    progress_bar.progress(progress)
                    status_text.text(f"Classifying... {current}/{total} texts")
                    if current <= total and current > 0:
                        preview_text = all_texts[current - 1][:100] + "..." if len(all_texts[current - 1]) > 100 else all_texts[current - 1]
                        current_text_display.text(f"Current text: {preview_text}")

                # Run classification on working_df texts
                classification_results, error = classify_texts_llm(
                    all_texts,
                    st.session_state.llmData['promptInstructions'],
                    st.session_state.llmData['promptLabels'] if st.session_state.llmData['promptLabels'].strip() else None,
                    selected_model,
                    st.session_state.llmData['apiKey'],
                    progress_callback=update_progress
                )

                if error:
                    st.error(f"❌ Error during classification: {error}")
                else:
                    progress_bar.progress(0.95)
                    status_text.text("Organizing results...")
                    current_text_display.empty()

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

                        # Ensure classification columns exist and default to 'not classified'
                        results_df['llm_classification'] = 'not classified'
                        results_df['llm_confidence'] = 'not classified'
                        results_df['llm_reasoning'] = 'not classified'
                        results_df['llm_error'] = None

                        # Map the classified top-N back to original indices
                        results_df.loc[original_indices, 'llm_classification'] = temp_results_df['llm_classification'].values
                        results_df.loc[original_indices, 'llm_confidence'] = temp_results_df['llm_confidence'].values
                        results_df.loc[original_indices, 'llm_reasoning'] = temp_results_df['llm_reasoning'].values
                        results_df.loc[original_indices, 'llm_error'] = temp_results_df['llm_error'].values

                    # Update global dataset with classification results
                    st.session_state.globalData['dataset'] = results_df
                    st.session_state.llmData['classificationResults'] = results_df

                    progress_bar.progress(0.98)
                    status_text.text("Saving results...")

                    # Auto-save global dataset
                    success, result = save_global_dataset()

                    if success:
                        progress_bar.progress(1.0)
                        status_text.text("✅ Classification completed!")
                        st.success(f"✅ Classification completed! File saved to: `{result}`")
                    else:
                        st.warning(f"⚠️ Classification completed but failed to save file: {result}")

                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
            finally:
                st.session_state.llmData['isExecuting'] = False

    else:
        missing = []
        if st.session_state.globalData['dataset'] is None:
            missing.append("📁 Dataset (upload in Home)")
        if st.session_state.globalData['textColumn'] is None:
            missing.append("📝 Text column (select in Home)")
        if st.session_state.llmData['selectedModel'] is None:
            missing.append("🤖 Selected model")
        if not st.session_state.llmData['apiKeyValidated']:
            missing.append("🔑 Validated API key")
        if st.session_state.llmData['promptInstructions'] == '':
            missing.append("📋 Prompt instructions")

        st.info("💡 Complete the previous steps before classification.")
        st.markdown("**Missing requirements:**")
        for item in missing:
            st.markdown(f"- {item}")

    # Step 3 completion indicator
    if step3_complete:
        st.success("✅ **Step 3 completed!** Classification performed successfully.")

        # Results Preview
        st.markdown("---")
        st.markdown("#### 📊 Results Preview")

        results_df = st.session_state.llmData['classificationResults']

        # Show classification and confidence distribution side by side
        import plotly.express as px

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Classification Distribution:**")
            label_dist = results_df['llm_classification'].value_counts().reset_index()
            label_dist.columns = ['Classification', 'Count']

            fig_class = px.pie(
                label_dist,
                values='Count',
                names='Classification',
                hole=0.4
            )
            fig_class.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            fig_class.update_layout(
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20),
                height=350
            )
            st.plotly_chart(fig_class, use_container_width=True)

        with col2:
            st.markdown("**Confidence Distribution:**")
            # Normalize confidence strings (e.g., 'High' vs 'high') before counting
            conf_series = (
                results_df['llm_confidence']
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
            )
            confidence_dist = conf_series.value_counts().reset_index()
            confidence_dist.columns = ['Confidence', 'Count']

            fig_conf = px.pie(
                confidence_dist,
                values='Count',
                names='Confidence',
                hole=0.4
            )
            fig_conf.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            fig_conf.update_layout(
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20),
                height=350
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # Show error count
        error_count = results_df['llm_error'].notna().sum()
        if error_count > 0:
            st.warning(f"⚠️ {error_count} texts had classification errors. Check the 'llm_error' column for details.")

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
