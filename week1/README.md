# Week 1: AI Infrastructure - Data Pipeline Fundamentals

## Overview
This week focused on building foundational data infrastructure skills for AI/ML projects. I learned how to create reproducible data pipelines using modern tools and best practices for data versioning, processing, and management.

## 🎯 Learning Objectives Achieved
- **Data Versioning**: Implemented DVC (Data Version Control) for reproducible data pipelines
- **Web Scraping**: Built robust article scraping from multiple sources
- **Data Cleaning**: Applied text preprocessing and normalization techniques
- **Dataset Creation**: Generated HuggingFace-compatible datasets for ML workflows
- **Tokenization**: Implemented BERT-based tokenization for NLP preprocessing

## 🏗️ Project Structure
```
├── data/
│   ├── raw/                    # Raw scraped articles
│   │   └── raw_articles.csv
│   └── processed/              # Cleaned and processed data
│       ├── clean_articles.csv
│       ├── tokenized.csv
│       └── hf_dataset/         # HuggingFace dataset format
├── scripts/
│   ├── scrape_articles.py      # Web scraping pipeline
│   └── clean_articles.py       # Data cleaning and tokenization
├── manifests/
│   └── scheme.md              # Data schema documentation
├── dvc.yaml                   # DVC pipeline configuration
├── dvc.lock                   # DVC lock file for reproducibility
└── myenv/                     # Python virtual environment
```

## 🔧 Technologies & Tools Used
- **DVC (Data Version Control)**: Pipeline orchestration and data versioning
- **Python**: Core programming language
- **BeautifulSoup**: Web scraping and HTML parsing
- **Pandas**: Data manipulation and analysis
- **HuggingFace Datasets**: Modern dataset management
- **Transformers**: BERT tokenization
- **Rich**: Enhanced terminal output
- **Virtual Environment**: Isolated dependency management

## 📊 Data Pipeline Stages

### Stage 1: Data Scraping (`scrape`)
- **Script**: `scripts/scrape_articles.py`
- **Function**: Scrapes articles from various web sources
- **Sources**: Finextra, Medium articles
- **Output**: `data/raw/raw_articles.csv`
- **Features**:
  - Robust error handling with timeouts
  - Beautiful terminal output with Rich
  - Structured data extraction

### Stage 2: Data Cleaning & Processing (`clean`)
- **Script**: `scripts/clean_articles.py`
- **Dependencies**: Raw articles, cleaning script
- **Outputs**:
  - `data/processed/clean_articles.csv` - Cleaned text data
  - `data/processed/tokenized.csv` - BERT-tokenized data
  - `data/processed/hf_dataset/` - HuggingFace dataset format
- **Processing Steps**:
  1. Text normalization (whitespace, special characters)
  2. Case standardization
  3. HuggingFace dataset creation
  4. BERT tokenization (max 128 tokens)

## 🚀 Running the Pipeline

### Prerequisites
```powershell
# Activate virtual environment
.\myenv\Scripts\Activate.ps1

# Install dependencies (already done in myenv)
pip install requests beautifulsoup4 pandas datasets transformers rich dvc
```

### Execute Complete Pipeline
```powershell
# Run the entire DVC pipeline
dvc repro

# Or run individual stages
dvc repro scrape    # Scraping only
dvc repro clean     # Cleaning only
```

### Manual Execution
```powershell
# Run scraping
python scripts/scrape_articles.py

# Run cleaning and processing
python scripts/clean_articles.py
```

## 📈 Key Learnings

### 1. **Data Version Control (DVC)**
- Learned to create reproducible ML pipelines
- Understanding of stage dependencies and outputs
- Automatic pipeline execution with `dvc repro`

### 2. **Web Scraping Best Practices**
- Robust error handling and timeouts
- Respectful scraping with proper delays
- Structured data extraction from HTML

### 3. **Text Preprocessing for NLP**
- Text normalization techniques
- Tokenization strategies for transformer models
- Dataset format standardization

### 4. **Modern Data Formats**
- HuggingFace dataset integration
- Arrow format for efficient data storage
- Metadata preservation and schema documentation

### 5. **Development Environment Management**
- Virtual environment best practices
- Dependency isolation
- Reproducible development setup

## 🎉 Outcomes
- **Reproducible Pipeline**: Complete DVC-managed workflow
- **Clean Dataset**: Processed 3 articles with proper text cleaning
- **ML-Ready Data**: Tokenized data ready for transformer models
- **Scalable Architecture**: Easily extensible for more data sources
- **Version Control**: All data transformations tracked and reproducible

## 🔜 Next Steps
- Scale to larger datasets
- Add data validation and quality checks
- Implement automated testing for data pipelines
- Explore advanced preprocessing techniques
- Integration with cloud storage solutions

## 📁 Data Schema
Reference the complete data schema documentation in `manifests/scheme.md` for detailed field descriptions and data validation rules.

---
*Week 1 of AI Infrastructure Learning Journey - Building the foundation for scalable ML data pipelines*
