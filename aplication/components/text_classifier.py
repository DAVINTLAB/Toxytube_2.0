"""
BERT Classifier Backend
Contains all business logic for text classification using BERT-based transformer models
"""
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


# =============================================================================
# Device Management
# =============================================================================

def get_available_device():
    """
    Detecta e retorna o melhor dispositivo disponível.
    
    Returns:
        tuple: (device, device_name, device_type)
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "🚀 GPU (CUDA)", "info"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps"), "🚀 GPU (MPS - Apple Silicon)", "info"
    else:
        return torch.device("cpu"), "🐌 CPU", "info"


# =============================================================================
# Model Validation
# =============================================================================

def is_bert_classifier(model_id):
    """
    Verifica se o modelo é um BERT classificador válido.
    
    Args:
        model_id: ID do modelo no Hugging Face
        
    Returns:
        tuple: (is_valid, config)
    """
    try:
        config = AutoConfig.from_pretrained(model_id)

        # Verifica se o modelo é baseado em BERT
        model_type = config.model_type.lower()
        is_bert = 'bert' in model_type

        # Verifica se tem labels (classificador)
        has_labels = hasattr(config, 'num_labels') and config.num_labels > 0

        return is_bert and has_labels, config
    except Exception as e:
        return False, None


# =============================================================================
# Model Loading and Management
# =============================================================================

def load_model(model_id, force_cpu=False):
    """
    Carrega o modelo e tokenizer do Hugging Face.
    
    Args:
        model_id: ID do modelo no Hugging Face
        force_cpu: Força o uso de CPU mesmo se GPU estiver disponível
        
    Returns:
        tuple: (model, tokenizer, model_info, error)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

        # Get best available device or force CPU
        if force_cpu:
            device = torch.device("cpu")
            device_name = "🐌 CPU (Forçado)"
            device_type = "warning"
        else:
            device, device_name, device_type = get_available_device()

        # Move model to device with error handling
        try:
            model.to(device)
            model.eval()
        except Exception as device_error:
            # If GPU fails, fallback to CPU
            device = torch.device("cpu")
            device_name = "🐌 CPU (Fallback)"
            device_type = "warning"
            model.to(device)
            model.eval()

        # Get model info
        config = model.config
        model_info = {
            'model_type': config.model_type,
            'num_labels': config.num_labels,
            'max_position_embeddings': config.max_position_embeddings,
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'num_hidden_layers': config.num_hidden_layers,
            'device': str(device),
            'device_name': device_name,
            'device_type': device_type,
            'labels': config.id2label if hasattr(config, 'id2label') else {}
        }

        return model, tokenizer, model_info, None
    except Exception as e:
        return None, None, None, str(e)


def unload_model(model=None, tokenizer=None):
    """
    Remove o modelo da memória e libera recursos.
    
    Args:
        model: Modelo a ser removido
        tokenizer: Tokenizer a ser removido
    """
    if model is not None:
        del model

    if tokenizer is not None:
        del tokenizer

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Text Classification
# =============================================================================

def classify_single_text(text, model, tokenizer, max_length):
    """
    Classifica um único texto e retorna as probabilidades.
    
    Args:
        text: Texto a ser classificado
        model: Modelo carregado
        tokenizer: Tokenizer do modelo
        max_length: Comprimento máximo de tokens
        
    Returns:
        tuple: (prediction, probabilities)
    """
    device = next(model.parameters()).device

    try:
        # Tokenize
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)

        return prediction.cpu().numpy()[0], probabilities.cpu().numpy()[0]

    except RuntimeError as e:
        # If CUDA error, fallback to CPU
        if 'CUDA' in str(e) or 'cuda' in str(e):
            # Move model to CPU
            model.to(torch.device('cpu'))
            device = torch.device('cpu')

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Retry on CPU
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1)

            return prediction.cpu().numpy()[0], probabilities.cpu().numpy()[0]
        else:
            raise


def classify_texts(texts, model, tokenizer, max_length, batch_size=8, progress_callback=None):
    """
    Classifica uma lista de textos usando o modelo carregado.
    
    Args:
        texts: Lista de textos a serem classificados
        model: Modelo carregado
        tokenizer: Tokenizer do modelo
        max_length: Comprimento máximo de tokens
        batch_size: Tamanho do batch para processamento
        progress_callback: Função callback para atualizar progresso (current, total)
        
    Returns:
        tuple: (predictions, probabilities)
    """
    device = next(model.parameters()).device
    vocab_size = getattr(model.config, 'vocab_size', None)
    max_pos_emb = getattr(model.config, 'max_position_embeddings', 512)
    # RoBERTa/XLM-RoBERTa: position_ids = incremental_indices + padding_idx → max id = padding_idx + seq_len
    # Para position_ids < max_position_embeddings: seq_len <= max_position_embeddings - padding_idx - 1
    padding_idx = getattr(model.config, 'pad_token_id', 0)
    max_safe_seq = max(1, max_pos_emb - padding_idx - 1)
    max_length_safe = min(max_length, max_safe_seq)
    all_predictions = []
    all_probabilities = []
    cuda_error_handled = False
    total_texts = len(texts)

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        try:
            # Tokenize (usar max_length_safe para não exceder position_embeddings)
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length_safe,
                return_tensors="pt"
            ).to(device)

            # Truncar ao máximo suportado (position_embeddings tem tamanho fixo; position_ids = padding_idx + pos)
            seq_len = inputs["input_ids"].size(1)
            if seq_len > max_safe_seq:
                for key in list(inputs.keys()):
                    if inputs[key].dim() >= 2:
                        inputs[key] = inputs[key][:, :max_safe_seq].contiguous()

            # Evitar "index out of range in self" na embedding: garantir que input_ids estão no vocabulário
            if vocab_size is not None and "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].clamp(0, vocab_size - 1)

            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

        except (RuntimeError, IndexError) as e:
            # If CUDA error and not handled yet, fallback to CPU
            if ('CUDA' in str(e) or 'cuda' in str(e)) and not cuda_error_handled:
                # Move model to CPU
                model.to(torch.device('cpu'))
                device = torch.device('cpu')
                cuda_error_handled = True

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Retry current batch on CPU (com truncamento e clamp)
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length_safe,
                    return_tensors="pt"
                ).to(device)
                if inputs["input_ids"].size(1) > max_safe_seq:
                    for key in list(inputs.keys()):
                        if inputs[key].dim() >= 2:
                            inputs[key] = inputs[key][:, :max_safe_seq].contiguous()
                if vocab_size is not None and "input_ids" in inputs:
                    inputs["input_ids"] = inputs["input_ids"].clamp(0, vocab_size - 1)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                if progress_callback:
                    progress_callback(len(all_predictions), total_texts)
            elif isinstance(e, IndexError):
                # IndexError "index out of range in self" (position_embeddings): retentar com truncamento mais agressivo
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length_safe,
                    return_tensors="pt"
                ).to(device)
                if inputs["input_ids"].size(1) > max_safe_seq:
                    for key in list(inputs.keys()):
                        if inputs[key].dim() >= 2:
                            inputs[key] = inputs[key][:, :max_safe_seq].contiguous()
                if vocab_size is not None and "input_ids" in inputs:
                    inputs["input_ids"] = inputs["input_ids"].clamp(0, vocab_size - 1)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                if progress_callback:
                    progress_callback(len(all_predictions), total_texts)
            else:
                raise

        # Atualizar progresso com quantidade já processada (para barra e contagem na interface)
        if progress_callback:
            current = len(all_predictions)
            progress_callback(current, total_texts)

    return all_predictions, all_probabilities


# =============================================================================
# Text Analysis
# =============================================================================

def calculate_text_lengths(texts, tokenizer, sample_size=100):
    """
    Calcula estatísticas de comprimento de textos.
    
    Args:
        texts: Lista de textos
        tokenizer: Tokenizer do modelo
        sample_size: Número de textos a amostrar (None para todos)
        
    Returns:
        dict: {'avg_length': float, 'max_length': int, 'min_length': int}
    """
    sample = texts[:sample_size] if sample_size else texts

    text_lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in sample]

    return {
        'avg_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        'max_length': max(text_lengths) if text_lengths else 0,
        'min_length': min(text_lengths) if text_lengths else 0
    }


# =============================================================================
# Results Processing
# =============================================================================

def create_results_dataframe(original_df, text_column, predictions, probabilities, labels):
    """
    Cria DataFrame com resultados da classificação.
    
    Args:
        original_df: DataFrame original
        text_column: Nome da coluna de texto
        predictions: Lista de predições
        probabilities: Lista de probabilidades
        labels: Dicionário de labels {id: name}
        
    Returns:
        pd.DataFrame: DataFrame com resultados
    """
    import pandas as pd

    results_df = original_df.copy()

    num_labels = len(probabilities[0]) if probabilities else 0
    # Add probability columns for each label (garantir label_id inteiro e dentro do range)
    for label_id, label_name in labels.items():
        idx = int(label_id)
        if idx < 0 or idx >= num_labels:
            continue
        results_df[f'bert_prob_{label_name}'] = [prob[idx] for prob in probabilities]

    return results_df


def save_classification_results(results_df, output_path, output_format='csv'):
    """
    Salva resultados da classificação em arquivo.
    
    Args:
        results_df: DataFrame com resultados
        output_path: Caminho completo do arquivo
        output_format: Formato de saída ('csv', 'xlsx', 'json')
        
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
                'error': f'Formato não suportado: {output_format}'
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
