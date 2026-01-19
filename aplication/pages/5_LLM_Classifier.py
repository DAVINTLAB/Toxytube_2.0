"""
LLM Classification Page
Implements text classification workflow using LiteLLM and DSPy
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime

from components.navigation import render_navigation
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
        'dataset': None,                    # Dataset loaded by user
        'textColumn': None,                 # Column selected for classification
        'selectedProvider': None,           # Selected LLM provider
        'selectedModel': None,              # Selected LLM model ID
        'apiKey': '',                        # API key
        'apiKeyValidated': False,           # API key validation status
        'promptInstructions': '',           # Classification instructions
        'promptLabels': '',                 # Optional: comma-separated labels
        'outputDirectory': os.path.expanduser('~/Downloads'),  # Output directory
        'outputFileName': f"llm_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'outputFormat': 'csv',              # File format
        'classificationResults': None,      # Classification results
        'isExecuting': False,               # Execution status
        'fileSaved': False                  # File saved status
    }

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(page_title="LLM Classifier - Toxytube", page_icon="🧠", layout="wide")

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
# Step 1: Dataset Selection
# =============================================================================

# Check if step 1 is complete
step1_complete = (
    st.session_state.llmData['dataset'] is not None and
    st.session_state.llmData['textColumn'] is not None
)

with st.container(border=True):
    st.markdown("### 📁 Step 1: Dataset Selection")
    st.markdown("Upload a file for LLM-based text classification.")

    uploadedFile = st.file_uploader("Select a file:", type=['csv', 'json', 'xlsx', 'xls'], key="llm_uploader")

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
                st.session_state.llmData['dataset'] = dataset

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    # Preview the dataset if loaded
    if st.session_state.llmData['dataset'] is not None:
        st.markdown("#### Dataset Preview")

        dataset = st.session_state.llmData['dataset']

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
        current_selection = st.session_state.llmData.get('textColumn')
        if current_selection and current_selection in textColumns:
            default_index = options_with_placeholder.index(current_selection)
        else:
            default_index = 0

        selectedTextColumn = st.selectbox(
            "Select the column containing text for classification (must be a text/string column):",
            options=options_with_placeholder,
            index=default_index,
            help="This column will be used as input for the LLM model",
            key="llm_text_column"
        )

        # Only update if a valid column is selected (not placeholder)
        if selectedTextColumn != placeholder:
            st.session_state.llmData['textColumn'] = selectedTextColumn
        else:
            st.session_state.llmData['textColumn'] = None

    # Step 1 completion indicator
    if step1_complete:
        st.success("✅ **Step 1 completed!** Dataset loaded and text column selected.")

st.markdown("")

# =============================================================================
# Step 2: Model and API Key Configuration
# =============================================================================

# Check if step 2 is complete
step2_complete = (
    st.session_state.llmData['selectedModel'] is not None and
    st.session_state.llmData['apiKey'] != '' and
    st.session_state.llmData['apiKeyValidated']
)

with st.container(border=True):
    st.markdown("### 🤖 Step 2: Model and API Key Configuration")
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
            st.warning("No models available for this provider.")
            selected_model = None

    # Show model info
    if selected_model and selected_model in available_models:
        model_info = available_models[selected_model]
        st.info(f"**{model_info['name']}**: {model_info['description']} (Context: {model_info['context_window']:,} tokens)")

    st.markdown("---")

    # API Key input
    st.markdown("#### 🔑 API Key")

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

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Validate API Key", use_container_width=True, key="validate_api_key"):
            if api_key_input:
                with st.spinner("Validating API key..."):
                    is_valid, message = validate_api_key(selected_model, api_key_input)
                    if is_valid:
                        st.session_state.llmData['apiKeyValidated'] = True
                        st.success(message)
                    else:
                        st.session_state.llmData['apiKeyValidated'] = False
                        st.error(message)
            else:
                st.warning("⚠️ Please enter an API key.")

    with col2:
        if st.session_state.llmData['apiKeyValidated']:
            st.success("✅ API Key Validated")
        else:
            st.info("🔐 API Key not validated yet")

    st.markdown("---")

    # Test model section
    st.markdown("#### 🧪 Test Model Connection")
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
                response, error = test_model_connection(selected_model, api_key_input, test_message)

                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Model responded successfully!")
                    st.markdown("**Response:**")
                    st.markdown(f"> {response}")
        elif not api_key_input:
            st.warning("⚠️ Please enter an API key first.")
        else:
            st.warning("⚠️ Please enter a test message.")

    # Step 2 completion indicator
    if step2_complete:
        st.success("✅ **Step 2 completed!** Model configured and API key validated.")

st.markdown("")

# =============================================================================
# Step 3: Prompt Configuration (DSPy Style)
# =============================================================================

# Check if step 3 is complete
step3_complete = st.session_state.llmData['promptInstructions'] != ''

with st.container(border=True):
    st.markdown("### 📝 Step 3: Prompt Configuration")
    st.markdown("Configure the classification prompt following DSPy patterns.")

    st.markdown("#### Classification Instructions")
    st.markdown("Define the instructions for the LLM classifier. Be specific about what the model should classify and how.")

    prompt_instructions = st.text_area(
        "Instructions (required):",
        value=st.session_state.llmData.get('promptInstructions', ''),
        placeholder="""Example instructions:

You are a sentiment classifier. Analyze the given text and classify it based on the emotional tone.

Consider the following:
- Positive: Texts expressing happiness, satisfaction, excitement, or praise
- Negative: Texts expressing anger, disappointment, sadness, or criticism
- Neutral: Texts that are factual, objective, or without clear emotional tone

Classify each text into one of these categories based on the overall sentiment.""",
        height=200,
        key="llm_prompt_instructions"
    )
    st.session_state.llmData['promptInstructions'] = prompt_instructions

    st.markdown("---")

    st.markdown("#### Classification Labels (Optional)")
    st.markdown("If you want to restrict the output to specific labels, list them here. Leave empty for free-form classification.")

    prompt_labels = st.text_input(
        "Labels (comma-separated, optional):",
        value=st.session_state.llmData.get('promptLabels', ''),
        placeholder="Example: positive, negative, neutral",
        help="The model will be instructed to choose one of these labels",
        key="llm_prompt_labels"
    )
    st.session_state.llmData['promptLabels'] = prompt_labels

    # Preview the DSPy signature
    st.markdown("---")
    st.markdown("#### 📋 DSPy Signature Preview")

    if prompt_labels.strip():
        st.code("""
class TextClassifierWithLabels(dspy.Signature):
    \"\"\"Classify text according to the provided instructions. You must choose one of the provided labels.\"\"\"
    
    classification_instructions: str = dspy.InputField(desc="Classification instructions")
    labels: str = dspy.InputField(desc="Available classification labels (comma-separated)")
    text: str = dspy.InputField(desc="Text to be classified")
    classification: str = dspy.OutputField(desc="The classification label (must be one of the provided labels)")
    confidence: str = dspy.OutputField(desc="Confidence level: high, medium, or low")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for the classification")
""", language="python")
    else:
        st.code("""
class TextClassifier(dspy.Signature):
    \"\"\"Classify text according to the provided instructions.\"\"\"
    
    classification_instructions: str = dspy.InputField(desc="Classification instructions and label definitions")
    text: str = dspy.InputField(desc="Text to be classified")
    classification: str = dspy.OutputField(desc="The classification label for the text")
    confidence: str = dspy.OutputField(desc="Confidence level: high, medium, or low")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for the classification")
""", language="python")

    # Test prompt with single text
    if step2_complete and prompt_instructions.strip():
        st.markdown("---")
        st.markdown("#### 🧪 Test Prompt")
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
                    result, error = classify_single_text_llm(
                        test_text_prompt,
                        prompt_instructions,
                        prompt_labels if prompt_labels.strip() else None,
                        st.session_state.llmData['selectedModel'],
                        st.session_state.llmData['apiKey']
                    )

                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success("✅ Classification successful!")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Classification", result['classification'])
                        with col2:
                            st.metric("Confidence", result['confidence'])
                        with col3:
                            st.metric("Reasoning", "See below")
                        st.markdown(f"**Reasoning:** {result['reasoning']}")
            else:
                st.warning("⚠️ Please enter sample text to test.")

    # Step 3 completion indicator
    if step3_complete:
        st.success("✅ **Step 3 completed!** Prompt configured.")

st.markdown("")

# =============================================================================
# Step 4: Output Configuration
# =============================================================================

# Check if step 4 is complete
step4_complete = (
    st.session_state.llmData['outputDirectory'] != '' and
    st.session_state.llmData['outputFileName'] != ''
)

with st.container(border=True):
    st.markdown("### 📂 Step 4: Output Configuration")
    st.markdown("Configure where and how to save the classification results.")

    col1, col2 = st.columns(2)

    with col1:
        output_format = st.selectbox(
            "File format:",
            options=['csv', 'xlsx', 'json'],
            index=['csv', 'xlsx', 'json'].index(st.session_state.llmData.get('outputFormat', 'csv')),
            key="llm_output_format"
        )
        st.session_state.llmData['outputFormat'] = output_format

    with col2:
        output_filename = st.text_input(
            "File name:",
            value=st.session_state.llmData.get('outputFileName', f"llm_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            help="Do not include file extension",
            key="llm_output_filename"
        )
        st.session_state.llmData['outputFileName'] = output_filename

    output_directory = st.text_input(
        "Output directory:",
        value=st.session_state.llmData.get('outputDirectory', os.path.expanduser("~/Downloads")),
        help="Full path to the directory where the file will be saved",
        key="llm_output_directory"
    )
    st.session_state.llmData['outputDirectory'] = output_directory

    # Show preview of full path
    if output_filename and output_directory:
        full_path_preview = os.path.join(output_directory, f"{output_filename}.{output_format}")
        st.info(f"📁 File will be saved as: `{full_path_preview}`")

    # Step 4 completion indicator
    if step4_complete:
        st.success("✅ **Step 4 completed!** Output configuration ready.")

st.markdown("")

# =============================================================================
# Step 5: LLM Classification
# =============================================================================

# Check if step 5 is complete
step5_complete = st.session_state.llmData['classificationResults'] is not None

# Check if ready for classification
can_classify = (
    st.session_state.llmData['dataset'] is not None and
    st.session_state.llmData['textColumn'] is not None and
    st.session_state.llmData['selectedModel'] is not None and
    st.session_state.llmData['apiKeyValidated'] and
    st.session_state.llmData['promptInstructions'] != ''
)

with st.container(border=True):
    st.markdown("### 🚀 Step 5: LLM Classification")

    if can_classify:
        st.markdown("✅ Everything ready for classification!")

        # Show configuration summary
        dataset = st.session_state.llmData['dataset']
        text_column = st.session_state.llmData['textColumn']
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

        # Classification button
        if st.button("🚀 Start Classification", use_container_width=True, type="primary", key="start_llm_classification"):
            st.session_state.llmData['isExecuting'] = True

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_text_display = st.empty()

            try:
                status_text.text("Preparing data...")

                # Get all texts
                all_texts = dataset[text_column].astype(str).tolist()
                total_texts = len(all_texts)

                status_text.text(f"Classifying {total_texts} texts...")

                # Progress callback
                def update_progress(current, total):
                    progress = current / total * 0.9  # 90% for classification
                    progress_bar.progress(progress)
                    status_text.text(f"Classifying... {current}/{total} texts")
                    if current <= total and current > 0:
                        preview_text = all_texts[current - 1][:100] + "..." if len(all_texts[current - 1]) > 100 else all_texts[current - 1]
                        current_text_display.text(f"Current text: {preview_text}")

                # Classify
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

                    # Create results dataframe
                    results_df = create_results_dataframe(
                        dataset,
                        text_column,
                        classification_results
                    )

                    # Save results
                    st.session_state.llmData['classificationResults'] = results_df

                    progress_bar.progress(0.98)
                    status_text.text("Saving results...")

                    # Auto-save file
                    output_format = st.session_state.llmData['outputFormat']
                    output_filename = st.session_state.llmData['outputFileName']
                    output_directory = st.session_state.llmData['outputDirectory']

                    filename = f"{output_filename}.{output_format}"
                    full_path = os.path.join(output_directory, filename)

                    save_result = save_classification_results(results_df, full_path, output_format)

                    if save_result['success']:
                        st.session_state.llmData['fileSaved'] = True
                        progress_bar.progress(1.0)
                        status_text.text("✅ Classification completed!")
                        st.success(f"✅ Classification completed! File saved to: `{full_path}`")
                    else:
                        st.warning(f"⚠️ Classification completed but failed to save file: {save_result['error']}")

                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
            finally:
                st.session_state.llmData['isExecuting'] = False

    else:
        missing = []
        if st.session_state.llmData['dataset'] is None:
            missing.append("📁 Dataset")
        if st.session_state.llmData['textColumn'] is None:
            missing.append("📝 Text column")
        if st.session_state.llmData['selectedModel'] is None:
            missing.append("🤖 Selected model")
        if not st.session_state.llmData['apiKeyValidated']:
            missing.append("🔑 Validated API key")
        if st.session_state.llmData['promptInstructions'] == '':
            missing.append("📋 Prompt instructions")

        st.info("⚠️ Complete the previous steps before classification.")
        st.markdown("**Missing requirements:**")
        for item in missing:
            st.markdown(f"- {item}")

    # Step 5 completion indicator
    if step5_complete:
        st.success("✅ **Step 5 completed!** Classification performed successfully.")

        # Results Preview
        st.markdown("---")
        st.markdown("#### 📊 Results Preview")

        results_df = st.session_state.llmData['classificationResults']

        # Show classification distribution
        st.markdown("**Classification Distribution:**")
        label_dist = results_df['llm_classification'].value_counts()
        st.bar_chart(label_dist)

        # Show confidence distribution
        st.markdown("**Confidence Distribution:**")
        confidence_dist = results_df['llm_confidence'].value_counts()
        st.bar_chart(confidence_dist)

        # Show error count
        error_count = results_df['llm_error'].notna().sum()
        if error_count > 0:
            st.warning(f"⚠️ {error_count} texts had classification errors. Check the 'llm_error' column for details.")

        # Show preview
        st.markdown("**Classified Dataset Preview:**")
        st.dataframe(results_df.head(20), use_container_width=True)

        # Show file location if saved
        if st.session_state.llmData.get('fileSaved', False):
            output_format = st.session_state.llmData['outputFormat']
            output_filename = st.session_state.llmData['outputFileName']
            output_directory = st.session_state.llmData['outputDirectory']
            full_path = os.path.join(output_directory, f"{output_filename}.{output_format}")

            st.success(f"💾 File saved at: `{full_path}`")
        else:
            st.info("⏳ File will be saved automatically after classification.")
