"""
Detoxify Classifier Backend
Contains all business logic for text classification using Detoxify library
"""
import torch
import gc
from detoxify import Detoxify


# =============================================================================
# Available Models
# =============================================================================

DETOXIFY_MODELS = {
    'original': {
        'name': 'Original',
        'description': 'Trained on the original Toxic Comment Classification Challenge data',
        'labels': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    },
    'unbiased': {
        'name': 'Unbiased',
        'description': 'Trained on the Unintended Bias in Toxicity Classification data',
        'labels': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']
    },
    'multilingual': {
        'name': 'Multilingual',
        'description': 'Multilingual model supporting multiple languages (XLM-RoBERTa based)',
        'labels': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_attack']
    }
}


def get_available_models():
    """
    Returns the available Detoxify models.
    
    Returns:
        dict: Dictionary with model information
    """
    return DETOXIFY_MODELS


# =============================================================================
# Device Management
# =============================================================================

def get_available_device():
    """
    Detects and returns the best available device.
    
    Returns:
        tuple: (device, device_name, device_type)
    """
    if torch.cuda.is_available():
        return "cuda", "🚀 GPU (CUDA)", "success"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", "⚡ GPU (MPS - Apple Silicon)", "success"
    else:
        return "cpu", "🐌 CPU (Slow Processing)", "warning"


# =============================================================================
# Model Loading and Management
# =============================================================================

def load_detoxify_model(model_name, force_cpu=False):
    """
    Loads a Detoxify model.
    
    Args:
        model_name: Name of the model ('original', 'unbiased', 'multilingual')
        force_cpu: Forces CPU usage even if GPU is available
        
    Returns:
        tuple: (model, model_info, error)
    """
    try:
        # Get best available device or force CPU
        if force_cpu:
            device = "cpu"
            device_name = "🐌 CPU (Forced)"
            device_type = "warning"
        else:
            device, device_name, device_type = get_available_device()

        # Load model
        model = Detoxify(model_name, device=device)

        # Get model info
        model_info = {
            'model_name': model_name,
            'model_display_name': DETOXIFY_MODELS[model_name]['name'],
            'description': DETOXIFY_MODELS[model_name]['description'],
            'labels': DETOXIFY_MODELS[model_name]['labels'],
            'num_labels': len(DETOXIFY_MODELS[model_name]['labels']),
            'device': device,
            'device_name': device_name,
            'device_type': device_type
        }

        return model, model_info, None
    except Exception as e:
        return None, None, str(e)


def unload_detoxify_model(model=None):
    """
    Removes the model from memory and frees resources.
    
    Args:
        model: Model to be removed
    """
    if model is not None:
        del model

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Text Classification
# =============================================================================

def classify_single_text(text, model):
    """
    Classifies a single text and returns the scores.
    
    Args:
        text: Text to be classified
        model: Loaded Detoxify model
        
    Returns:
        dict: Dictionary with label scores
    """
    try:
        results = model.predict(text)
        return results, None
    except Exception as e:
        return None, str(e)


def classify_texts(texts, model, batch_size=32, progress_callback=None):
    """
    Classifies a list of texts using the loaded model.
    
    Args:
        texts: List of texts to be classified
        model: Loaded Detoxify model
        batch_size: Batch size for processing
        progress_callback: Callback function to update progress (current, total)
        
    Returns:
        tuple: (results_list, error)
    """
    all_results = []
    total_texts = len(texts)

    try:
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Predict batch
            batch_results = model.predict(batch_texts)

            # Convert batch results to list of dicts (one per text)
            num_texts_in_batch = len(batch_texts)
            for j in range(num_texts_in_batch):
                text_result = {}
                for label, scores in batch_results.items():
                    text_result[label] = scores[j]
                all_results.append(text_result)

            # Update progress
            if progress_callback:
                current = min(i + batch_size, total_texts)
                progress_callback(current, total_texts)

        return all_results, None

    except Exception as e:
        return None, str(e)


# =============================================================================
# Results Processing
# =============================================================================

def create_results_dataframe(original_df, text_column, classification_results, labels, threshold=0.5):
    """
    Creates DataFrame with classification results.
    
    Args:
        original_df: Original DataFrame
        text_column: Name of the text column
        classification_results: List of classification result dicts
        labels: List of label names
        threshold: Threshold to consider a label as positive
        
    Returns:
        pd.DataFrame: DataFrame with results
    """
    import pandas as pd

    results_df = original_df.copy()

    # Add score columns for each label
    for label in labels:
        results_df[f'score_{label}'] = [result[label] for result in classification_results]

    # Add binary columns for each label (above threshold)
    for label in labels:
        results_df[f'is_{label}'] = [result[label] >= threshold for result in classification_results]

    # Add dominant label (highest score)
    def get_dominant_label(result):
        max_label = max(result.items(), key=lambda x: x[1])
        return max_label[0]

    results_df['dominant_label'] = [get_dominant_label(result) for result in classification_results]

    # Add maximum score
    results_df['max_score'] = [max(result.values()) for result in classification_results]

    # Add toxic flag (if any toxic-related label is above threshold)
    toxic_labels = ['toxic', 'severe_toxic']
    def is_toxic(result):
        for label in toxic_labels:
            if label in result and result[label] >= threshold:
                return True
        return False

    results_df['is_toxic_any'] = [is_toxic(result) for result in classification_results]

    return results_df


def save_classification_results(results_df, output_path, output_format='csv'):
    """
    Saves classification results to a file.
    
    Args:
        results_df: DataFrame with results
        output_path: Full file path
        output_format: Output format ('csv', 'xlsx', 'json')
        
    Returns:
        dict: {'success': bool, 'path': str, 'error': str}
    """
    import os

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save based on format
        if output_format == 'csv':
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_format == 'xlsx':
            results_df.to_excel(output_path, index=False, engine='openpyxl')
        elif output_format == 'json':
            results_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        else:
            return {
                'success': False,
                'error': f'Unsupported format: {output_format}'
            }

        return {
            'success': True,
            'path': output_path
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
