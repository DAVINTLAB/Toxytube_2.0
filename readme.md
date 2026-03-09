# Toxytube 2.0

Web application for collecting, classifying and analyzing YouTube comments using multiple text classification models.

## Description

Toxytube 2.0 is a tool developed in Streamlit that allows you to collect YouTube comments and automatically classify them using three different machine learning approaches:

- **BERT Classifier**: Transformer-based model for text classification
- **Detoxify Classifier**: Model specialized in toxicity detection
- **LLM Classifier**: Classification using Large Language Models via DSPy

The application also offers comparative visualizations of results, including agreement analysis between models, time series and class distributions.

## Features

- YouTube comments collection via Google API
- Custom dataset upload in CSV format
- Text classification with BERT, Detoxify and LLM
- Comparative analysis of results between the three classifiers
- Interactive visualizations:
  - Class distribution by model
  - Agreement between BERT and LLM
  - Detoxify threshold analysis
  - Time series of classifications
  - Word cloud
- Export of classified results

## Requirements

- Python 3.11 or higher
- Google API key (for YouTube comments collection)
- LLM API key compatible with LiteLLM (for LLM classifier)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Toxytube_2.0
```

### 2. Set up virtual environment

Install python3-venv package (if necessary):

```bash
sudo apt install python3.12-venv
```

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK resources

On first run, the application will automatically download the necessary NLTK resources (stopwords).

## Running the application

With the virtual environment active:

```bash
cd aplication
streamlit run Home.py
```

The application will be available at `http://localhost:8501`

## Project structure

```
Toxytube_2.0/
├── aplication/
│   ├── Home.py                          # Main page
│   ├── components/
│   │   ├── analysis_backend.py          # Analysis logic and visualizations
│   │   ├── detoxify_classifier.py       # Detoxify classifier
│   │   ├── llm_classifier.py            # LLM classifier
│   │   ├── navigation.py                # Navigation system
│   │   ├── text_classifier.py           # BERT classifier
│   │   └── youtube_collector.py         # YouTube comments collection
│   └── pages/
│       ├── 0_Dataset_Upload.py          # Dataset upload
│       ├── 1_YouTube_Comments.py        # Comments collection
│       ├── 2_Bert_Classifier.py         # BERT interface
│       ├── 3_Detoxify_Classifier.py     # Detoxify interface
│       ├── 4_LLM_Classifier.py          # LLM interface
│       └── 5_Classified_Data_Analysis.py # Comparative analysis
├── requirements.txt                     # Project dependencies
├── keys.txt                             # API keys (not versioned)
└── .gitignore
```

## Recommended workflow

1. **Initial setup**: Configure the dataset and text column on the Home page
2. **Data collection**: Use the "YouTube Comments" page to collect comments OR upload a CSV dataset
3. **Classification**: Run the three classifiers (BERT, Detoxify and LLM) on the texts
4. **Analysis**: View and compare results on the "Classified Data Analysis" page
5. **Export**: Download the classified results for further analysis
