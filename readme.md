# 🚀 Toxicytube 2.0

A complete and intuitive platform for conducting content moderation studies without writing code. Analyze text toxicity using multiple machine learning classifiers.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-red.svg)

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Available Classifiers](#-available-classifiers)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)

## 🎯 About the Project

Toxicytube 2.0 is a web application developed with Streamlit that enables text toxicity analysis using different classification approaches. The project was created to facilitate content moderation studies, especially focused on YouTube comments, but compatible with any text dataset.

## ✨ Features

### 📁 Dataset Upload
- Support for multiple formats: CSV, Excel (XLSX/XLS), JSON, Parquet
- Text column configuration for classification
- Export results in different formats

### 🎥 YouTube Comments Collection
- Automated collection via YouTube Data API v3
- Detailed video information (views, likes, duration)
- Export collected comments

### 🤖 BERT Classifier
- Compatible with any BERT model from Hugging Face
- Support for BERT, RoBERTa, DistilBERT, XLNet, etc.
- Automatic GPU/CPU detection
- Batch classification with progress bar

### 🛡️ Detoxify Classifier
- Three available models:
  - **Original**: Trained on Toxic Comment Classification Challenge
  - **Unbiased**: Trained on unbiased data
  - **Multilingual**: Multi-language support (XLM-RoBERTa)
- Detects: toxicity, obscenity, threats, insults, hate speech

### 🧠 LLM Classifier
- Integration with multiple providers via LiteLLM:
  - **OpenAI**: GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo
  - **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus
  - **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash
  - **Groq**: Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B
  - **Mistral AI**: Mistral Large, Mistral Small
  - **DeepSeek**: DeepSeek Chat, DeepSeek Reasoner
- Customizable prompts with DSPy
- Custom label system

### 📊 Classified Data Analysis
- Class distribution visualization
- Threshold charts for probabilities
- Time series analysis
- Word cloud
- Top comments table

## 💻 Requirements

- Python 3.10 or higher
- pip (Python package manager)
- Internet connection (to download models and use APIs)
- GPU (optional, but recommended for BERT/Detoxify classifiers)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Toxytube_2.0.git
   cd Toxytube_2.0
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

1. **Navigate to the application folder**
   ```bash
   cd aplication
   ```

2. **Run Streamlit**
   ```bash
   streamlit run Home.py
   ```

3. **Access in browser**
   ```
   http://localhost:8501
   ```

## 📂 Project Structure

```
Toxytube_2.0/
├── readme.md                    # This file
├── requirements.txt             # Project dependencies
└── aplication/
    ├── Home.py                  # Home page
    ├── assets/                  # Static resources
    ├── components/              # Reusable components
    │   ├── detoxify_classifier.py   # Detoxify backend
    │   ├── llm_classifier.py        # LLM backend
    │   ├── navigation.py            # Application navigation
    │   ├── text_classifier.py       # BERT backend
    │   └── youtube_collector.py     # YouTube collector
    └── pages/                   # Application pages
        ├── 0_Dataset_Upload.py      # Dataset upload
        ├── 1_YouTube_Comments.py    # YouTube collection
        ├── 2_Bert_Classifier.py     # BERT classifier
        ├── 3_Detoxify_Classifier.py # Detoxify classifier
        ├── 4_LLM_Classifier.py      # LLM classifier
        └── 5_Classified_Data_Analysis.py # Data analysis
```

## 🔬 Available Classifiers

### BERT Classifier
Uses transformer-based models from Hugging Face. Recommendations:
- `unitary/toxic-bert` - Toxicity classification
- `cardiffnlp/twitter-roberta-base-sentiment` - Sentiment analysis
- `facebook/roberta-hate-speech-dynabench-r4-target` - Hate speech detection

### Detoxify
| Model | Description | Labels |
|-------|-------------|--------|
| Original | Toxicity Challenge | toxic, severe_toxic, obscene, threat, insult, identity_hate |
| Unbiased | Unbiased data | toxic, severe_toxic, obscene, threat, insult, identity_attack, sexual_explicit |
| Multilingual | Multi-language | toxic, severe_toxic, obscene, threat, insult, identity_attack |

### LLM Classifier
Requires API key from the chosen provider:
- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Anthropic models
- `GEMINI_API_KEY` for Google models
- `GROQ_API_KEY` for Groq models
- `MISTRAL_API_KEY` for Mistral models
- `DEEPSEEK_API_KEY` for DeepSeek models

## 📖 Usage

### Basic Workflow

1. **Dataset Upload**
   - Access "Dataset Upload" in the sidebar menu
   - Upload file (CSV, Excel, JSON, or Parquet)
   - Select text column for classification
   - Configure output file name and format

2. **Classification**
   - Choose one of the classifiers (BERT, Detoxify, or LLM)
   - Configure the desired model
   - Run classification
   - Results are automatically added to the dataset

3. **Analysis**
   - Access "Classified Data Analysis"
   - Visualize distribution charts
   - Compare results between classifiers
   - Export classified data

### YouTube Collection

1. Obtain a [YouTube Data API v3](https://console.cloud.google.com/) key
2. Access "YouTube Comments"
3. Enter video URL and API key
4. Configure number of comments and output format
5. Execute collection

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web framework
- **[Transformers](https://huggingface.co/transformers/)** - BERT models
- **[Detoxify](https://github.com/unitaryai/detoxify)** - Toxicity classification
- **[LiteLLM](https://github.com/BerriAI/litellm)** - Unified API for LLMs
- **[DSPy](https://github.com/stanfordnlp/dspy)** - Structured prompting
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation
- **[Plotly](https://plotly.com/)** - Interactive visualizations
- **[WordCloud](https://github.com/amueller/word_cloud)** - Word cloud
