# Advanced RAG Q&A Tool with Mistral and Stella

An advanced Retrieval-Augmented Generation (RAG) system for answering questions based on web content. The system uses Mistral for inference via Ollama and Stella 400M for embeddings.

## Features

- **Fully Asynchronous**: All components run asynchronously for better performance
- **Advanced RAG Techniques**:
  - Semantic reranking of retrieved documents
  - Query expansion for better recall
  - Hypothetical document embeddings
  - Configurable chunking and retrieval parameters
- **Local Inference**: Uses Mistral via Ollama for 100% local inference
- **Stella 400M Embeddings**: High-quality sentence embeddings
- **User-Friendly Interface**: Built with Streamlit

## Architecture

- **Web Scraping**: Extracts clean content from URLs using readability and BeautifulSoup
- **Embeddings**: Creates semantic embeddings using Stella 400M
- **Vector Storage**: Indexes documents with FAISS
- **Retrieval**: Advanced techniques including query expansion and semantic reranking
- **Inference**: Local inference with Mistral via Ollama

## Setup Instructions

### Prerequisites
- Docker and Docker Compose (for easy setup)
- OR Python 3.9+ with the required packages

### Docker Setup (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/advanced-rag-qa.git
cd advanced-rag-qa
```

2. Start with Docker Compose:
```bash
docker-compose up -d
```

3. Access the application at: http://localhost:8501

### Manual Setup

1. Install Ollama:
   - Follow instructions at [ollama.ai](https://ollama.ai)
   - Pull the Mistral model: `ollama pull mistral`

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter one or more URLs in the interface
2. Configure RAG parameters in the sidebar if needed
3. Ask questions about the content
4. View answers and optionally see the retrieved chunks

## Advanced Configuration

The system can be fine-tuned with these parameters:

- **Retrieval K**: Number of chunks to retrieve (2-10)
- **Chunk Size**: Size of text chunks (500-2000)
- **Chunk Overlap**: Overlap between chunks (0-500)
- **Semantic Reranking**: Enable/disable reranking of chunks

## Project Structure

- `app.py`: Main Streamlit application
- `agent.py`: RAG agent implementation
- `embedding.py`: Stella 400M embeddings wrapper
- `reranker.py`: Semantic reranking module
- `query_expansion.py`: Query expansion techniques
- `prompts.py`: Advanced RAG prompts
- `scraper.py`: Asynchronous web scraper
- `utils.py`: Utility functions

## License

MIT