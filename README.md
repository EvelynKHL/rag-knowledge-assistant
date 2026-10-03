# RAG-Based AI Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application for turning unstructured documents into a searchable, grounded knowledge system.

## What it demonstrates

### Data Engineering
- File ingestion
- Text extraction
- Cleaning
- Deduplication
- Chunking
- Metadata management
- Pipeline statistics

### AI / ML
- Sentence-transformer embeddings
- Vector representations
- FAISS similarity search
- Top-K retrieval
- Retrieval evaluation

### GenAI
- RAG architecture
- Context-grounded prompting
- Source attribution
- Optional OpenAI LLM generation
- Fallback behavior when an API key is absent

### Application Engineering
- Streamlit UI
- Modular Python architecture
- Configuration through environment variables
- Evaluation framework

## Architecture

```text
PDF / DOCX / TXT
       |
       v
Document Extraction
       |
       v
Cleaning + Deduplication
       |
       v
Chunking + Metadata
       |
       v
Sentence Transformer
       |
       v
FAISS Vector Index
       |
       v
User Question
       |
       v
Question Embedding
       |
       v
Top-K Semantic Retrieval
       |
       v
Context Construction
       |
       v
LLM
       |
       v
Grounded Answer + Sources
```

## Run locally

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## LLM configuration

The application can run retrieval without an API key.

For generated answers, create `.env` or set environment variables:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
```

The current implementation reads environment variables. If using a `.env` file, export/source it in your shell or add `python-dotenv` and load it at startup.

## How to demonstrate the project

1. Upload 3–5 PDFs totaling 100+ pages.
2. Process the documents.
3. Show extracted chunks and metadata.
4. Ask factual questions.
5. Show the top 3–5 retrieved chunks.
6. Show source/page attribution.
7. Run the evaluation dataset.
8. Analyze failed retrieval cases.
9. Explain how chunk size, overlap, embeddings and Top-K affect retrieval.

## Evaluation

The evaluation framework reads:

`data/evaluation/questions.json`

Example:

```json
[
  {
    "question": "What was revenue in 2025?",
    "keywords": ["revenue"],
    "source": "annual_report.pdf"
  }
]
```

For a serious portfolio result, create at least 50 test questions and report the actual measured retrieval success rate.

## Resume positioning

### AI/ML Engineer
Emphasize:
- embeddings
- semantic search
- vector databases
- retrieval evaluation
- Python
- model integration

### Data Engineer
Emphasize:
- ingestion
- transformation
- data quality
- metadata
- scalable indexing
- modular pipelines

### Data Analyst
Emphasize:
- document data extraction
- quality analysis
- evaluation metrics
- structured reporting

### GenAI Engineer
Emphasize:
- RAG
- LLMs
- prompt design
- grounding
- source attribution
- evaluation

## Production extensions

For a production-grade system, add:
- PostgreSQL for metadata
- object storage for documents
- background processing with Celery/RQ
- Airflow/Dagster orchestration
- hybrid BM25 + vector retrieval
- reranking
- document versioning
- authentication/authorization
- observability
- automated evaluation
- Docker
- CI/CD
