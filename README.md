# Mini RAG System

A professional Retrieval-Augmented Generation (RAG) system built with modern components for document-based question answering. Features a user-friendly Gradio interface, FastAPI backend, and integration with leading AI services.

# System Architecture
text
graph TD
    A[User Interface - Gradio] --> B[FastAPI Backend]
    B --> C[Document Processing]
    C --> D[Text Chunking]
    D --> E[Embeddings - Google text-embedding-004]
    E --> F[Vector Database - FAISS]
    
    G[User Query] --> H[Query Embedding]
    H --> I[Similarity Search - FAISS]
    I --> J[Document Retrieval]
    J --> K[Reranking - Cohere rerank-v3.0]
    K --> L[Context Formation]
    L --> M[Answer Generation - Gemini 2.0 Flash]
    M --> N[Response with Citations]
    
    F --> I
    
# Core Components
Frontend: Gradio web interface with professional styling

Backend: FastAPI for API endpoints and async processing

Document Processing: PyMuPDF for PDF parsing with intelligent text chunking

Vector Storage: FAISS for efficient similarity search

Embeddings: Google's text-embedding-004 model

Reranking: Cohere's rerank-english-v3.0 for relevance refinement

Generation: Gemini 2.0 Flash for answer synthesis

 **Configuration Parameters**
Document Chunking Settings
python
CHUNK_SIZE = 800          # Words per chunk
OVERLAP = 120            # Word overlap between chunks
EMBEDDING_DIM = 768      # Dimension of embedding vectors
The system uses sentence-aware chunking to maintain semantic boundaries:

Splits text at sentence boundaries (., !, ?)

Maintains context with configurable overlap

Optimized for 800-word chunks for balanced context vs. precision

Retrieval & Reranking Configuration
python
# Retrieval Settings
RETRIEVE_K = 10          # Initial documents retrieved
RERANK_K = 5            # Final documents after reranking
SIMILARITY_METRIC = "cosine"  # Using inner product (normalized cosine)

# Generation Settings
MAX_OUTPUT_TOKENS = 1000
TEMPERATURE = 0.2
TOP_P = 0.8
TOP_K = 40
AI Service Providers
Component	Provider	Model	Purpose
Embeddings	Google AI	text-embedding-004	Document and query vectorization
Reranking	Cohere	rerank-english-v3.0	Relevance score refinement
Generation	Google AI	gemini-2.0-flash-exp	Answer synthesis with citations
🚀 Quick Start
Prerequisites
bash
# Required API Keys (set as environment variables)
export GEMINI_API_KEY="your_gemini_api_key"
export COHERE_API_KEY="your_cohere_api_key"
Installation
Clone the repository:

bash
git clone https://github.com/yourusername/mini_rag.git
cd mini_rag
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
python app.py
The system will start two servers:

Gradio Interface: http://localhost:7860

FastAPI Backend: http://localhost:8001

Usage Workflow
# Knowledge Base Management

Upload PDF or text files

Add manual text content

View and manage document chunks

# Interactive Chat

Ask questions about your documents

Get answers with inline citations

View processing statistics and conversation history

Sample Documents
The system loads three sample documents on startup covering:

Artificial Intelligence fundamentals

Machine Learning principles

Retrieval-Augmented Generation concepts

# Project Structure
text
mini_rag/
├── app.py                 # Main application entry point
├── rag_system.py         # Core RAG system implementation
├── ui_components.py      # Gradio UI components and styling
├── fastapi_backend.py    # FastAPI backend implementation
├── requirements.txt      # Python dependencies
└── README.md            # This file
# Key Features
Professional UI: Clean Gradio interface with custom CSS styling

Dual Architecture: FastAPI backend with Gradio frontend

Intelligent Chunking: Sentence-aware text splitting with overlap

Multi-format Support: PDF and text file processing

Advanced Retrieval: FAISS similarity search with Cohere reranking

Citation System: Inline citations with source tracking

Conversation Memory: Recent question history (configurable limit)

Real-time Stats: Processing metrics and database statistics

# Advanced Configuration
Customizing Chunk Parameters
python
# In DocumentProcessor.chunk_text()
chunks = DocumentProcessor.chunk_text(
    text=document_text,
    chunk_size=800,    # Adjust based on your content
    overlap=120        # Increase for more context retention
)
Modifying Retrieval Settings
python
# In MiniRAGSystem.query()
retrieved_docs = self.retrieve_documents(query, k=10)  # Initial retrieval
reranked_docs = self.rerank_documents(query, retrieved_docs, top_k=5)  # Final selection
Vector Database Options
The system uses FAISS with Inner Product similarity (normalized cosine similarity):

python
# Vector normalization for cosine similarity
normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index = faiss.IndexFlatIP(embedding_dim)  # Inner Product index
# API Endpoints
The FastAPI backend provides RESTful endpoints:

POST /api/query - Process user questions

POST /api/documents/upload - Upload files

POST /api/documents/add-text - Add manual text

GET /api/documents/list - List all documents

DELETE /api/documents/clear - Clear database

# Testing & Development
For development and testing without API keys, the system includes fallback mechanisms:

Mock embeddings for development

Sample document loading

Graceful API failure handling

# License
This project is open source and available under the MIT License.

# Contributing
Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

Built with: Python - Gradio - FastAPI - FAISS - Google AI - Cohere


