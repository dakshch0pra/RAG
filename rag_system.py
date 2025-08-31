import gradio as gr
import numpy as np
import pandas as pd
import google.generativeai as genai
import cohere
import faiss
import os
import json
import time
import logging
from typing import List, Tuple, Dict, Optional
import re
from pathlib import Path
import tempfile
import fitz  # PyMuPDF for PDF processing
from datetime import datetime
import uvicorn
import threading
import requests
from ui_components import UITheme, ConversationHistory, DocumentViewer, ChatInterface, DatabaseMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing including PDF parsing and text chunking"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        """Chunk text with sentence-aware splitting for better semantic boundaries"""
        if not text.strip():
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_words = ' '.join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words + [sentence]
                current_length = len(overlap_words) + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class VectorDatabase:
    """FAISS-based vector database for efficient similarity search"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents = []
        self.metadata = []
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict]):
        """Add multiple documents to the vector database"""
        if len(embeddings) != len(documents) or len(documents) != len(metadata):
            raise ValueError("Embeddings, documents, and metadata must have the same length")
        
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        logger.info(f"Added {len(documents)} documents to vector database")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, Dict, float]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(
            normalized_query.reshape(1, -1).astype('float32'), k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((
                    self.documents[idx],
                    self.metadata[idx],
                    float(score)
                ))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'embedding_dimension': self.embedding_dim
        }
    
    def clear_database(self):
        """Clear all documents from the database"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
        self.metadata = []
        logger.info("Database cleared")


class MiniRAGSystem:
    """Main RAG system orchestrating retrieval and generation"""
    
    def __init__(self):
        """Initialize the RAG system with all components"""
        self.embedding_model = 'text-embedding-004'
        self.llm_model = 'gemini-2.0-flash-exp'
        self.rerank_model = 'rerank-english-v3.0'
        
        # Initialize components
        self.vector_db = VectorDatabase(embedding_dim=768)
        self.doc_processor = DocumentProcessor()
        self.conversation_history = ConversationHistory(max_history=3)
        
        # API clients
        self.gemini_client = None
        self.cohere_client = None
        
        # Setup APIs
        self._setup_apis()
        
        # Load sample data
        self._load_sample_documents()
    
    def _setup_apis(self):
        """Setup API clients with proper error handling"""
        try:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.gemini_client = genai.GenerativeModel(self.llm_model)
                logger.info("Gemini API configured successfully")
            else:
                logger.warning("Gemini API key not found")
        except Exception as e:
            logger.error(f"Failed to setup Gemini API: {e}")
        
        try:
            cohere_key = os.getenv('COHERE_API_KEY')
            if cohere_key:
                self.cohere_client = cohere.Client(cohere_key)
                logger.info("Cohere API configured successfully")
            else:
                logger.warning("Cohere API key not found")
        except Exception as e:
            logger.error(f"Failed to setup Cohere API: {e}")
    
    def _get_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        """Get embeddings using Google's text-embedding-004 model"""
        try:
            if not self.gemini_client:
                return np.random.randn(len(texts), 768).astype('float32')
            
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=f"models/{self.embedding_model}",
                    content=text,
                    task_type=task_type
                )
                embeddings.append(result['embedding'])
            
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.random.randn(len(texts), 768).astype('float32')
    
    def _load_sample_documents(self):
        """Load sample documents for immediate testing"""
        sample_docs = [
            {
                'content': 'Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. The architecture of deep neural networks consists of an input layer, multiple hidden layers, and an output layer. Each layer contains neurons that process information and pass it to the next layer through weighted connections. Convolutional Neural Networks (CNNs) are specifically designed for processing grid-like data such as images. They use convolutional layers that apply filters to detect local features like edges, textures, and patterns. CNNs typically include pooling layers that reduce spatial dimensions and fully connected layers for final classification. Popular CNN architectures include LeNet, AlexNet, VGG, ResNet, and EfficientNet. Recurrent Neural Networks (RNNs) are designed to process sequential data by maintaining hidden states that capture information from previous time steps. Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) are advanced RNN variants that solve the vanishing gradient problem and can learn long-term dependencies in sequences. Transformer architectures have revolutionized natural language processing by using self-attention mechanisms to process sequences in parallel rather than sequentially. The transformer consists of encoder and decoder stacks, each containing multi-head attention layers and feed-forward networks. Key innovations include positional encoding, layer normalization, and residual connections. Training deep neural networks requires careful consideration of optimization algorithms, regularization techniques, and hyperparameter tuning. Common optimization algorithms include Stochastic Gradient Descent (SGD), Adam, and AdamW. Regularization techniques like dropout, batch normalization, and weight decay help prevent overfitting. Learning rate scheduling and early stopping are crucial for achieving optimal performance.',
                'source': 'Deep Learning Handbook',
                'title': 'Deep Learning and Neural Network Architectures',
                'section': 'Architecture Overview',
                'type': 'sample'
            },
            {
                'content': 'Machine learning is a branch of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The field encompasses various algorithms and techniques that allow systems to automatically identify patterns in data and make predictions or decisions. Supervised learning involves training models on labeled datasets where both input features and target outputs are known. Classification algorithms like logistic regression, decision trees, random forests, and support vector machines are used for predicting categorical outcomes. Regression algorithms such as linear regression, polynomial regression, and neural networks predict continuous numerical values. Model evaluation metrics for classification include accuracy, precision, recall, F1-score, and AUC-ROC curves. Unsupervised learning discovers hidden patterns in data without labeled examples. Clustering algorithms like K-means, hierarchical clustering, and DBSCAN group similar data points together. Dimensionality reduction techniques such as Principal Component Analysis (PCA), t-SNE, and UMAP help visualize high-dimensional data and reduce computational complexity. Reinforcement learning involves training agents to make sequential decisions in an environment to maximize cumulative rewards. Key concepts include states, actions, rewards, policies, and value functions. Popular algorithms include Q-learning, Deep Q-Networks (DQN), Policy Gradient methods, and Actor-Critic approaches. Applications include game playing, robotics, and autonomous systems. Feature engineering is crucial for machine learning success and involves selecting, transforming, and creating relevant features from raw data. Techniques include normalization, standardization, encoding categorical variables, handling missing values, and creating polynomial features.',
                'source': 'ML Comprehensive Guide',
                'title': 'Machine Learning Algorithms and Applications',
                'section': 'Core Concepts',
                'type': 'sample'
            },
            {
                'content': 'Natural Language Processing is a field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language. NLP combines computational linguistics with machine learning and deep learning to process and analyze large amounts of natural language data. Text preprocessing is the foundation of NLP and includes tokenization, stemming and lemmatization, part-of-speech tagging, and named entity recognition. Stop word removal and normalization further clean the text for analysis. Traditional NLP approaches relied on statistical methods and feature engineering. Bag-of-words models represent documents as vectors of word frequencies, while TF-IDF weights words based on their importance. N-gram models capture sequential patterns by considering combinations of consecutive words. Modern NLP leverages deep learning architectures for better performance. Word embeddings like Word2Vec, GloVe, and FastText represent words as dense vectors that capture semantic relationships. Contextual embeddings from models like ELMo, BERT, and GPT provide word representations that vary based on context. These pre-trained models can be fine-tuned for specific tasks through transfer learning. Language models have evolved from simple statistical models to sophisticated transformer-based architectures. Large language models like GPT-3, GPT-4, BERT, and T5 demonstrate remarkable capabilities in text generation, question answering, summarization, and translation. Recent advances in prompt engineering and in-context learning have made these models more accessible and versatile. NLP applications span numerous domains including sentiment analysis, chatbots, machine translation, information extraction, and document summarization.',
                'source': 'NLP Complete Reference',
                'title': 'Natural Language Processing Techniques and Applications',
                'section': 'Language Understanding',
                'type': 'sample'
            },
            {
                'content': 'Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world. It combines image processing, pattern recognition, and machine learning to extract meaningful information from digital images and videos. Image processing fundamentals include filtering techniques like Gaussian blur, edge detection using Sobel or Canny operators, and morphological operations. Image transformations such as rotation, scaling, and perspective correction normalize images for analysis. Feature extraction is crucial for traditional computer vision approaches. Local feature detectors like SIFT, SURF, and ORB identify distinctive keypoints in images that are invariant to scale, rotation, and illumination changes. Deep learning has revolutionized computer vision through convolutional neural networks. CNN architectures automatically learn hierarchical features from raw pixels. Early layers detect low-level features like edges and textures, while deeper layers learn complex patterns and object parts. Popular architectures include AlexNet, VGG, ResNet, Inception, and EfficientNet. Object detection extends image classification by localizing objects within images. Two-stage detectors like R-CNN, Fast R-CNN, and Faster R-CNN use region proposals followed by classification. Single-stage detectors like YOLO and SSD perform detection in a single forward pass, achieving real-time performance. Computer vision applications are widespread including medical imaging, autonomous vehicles, retail visual search, security systems with facial recognition, and augmented reality applications that overlay digital content on real-world scenes.',
                'source': 'Computer Vision Textbook',
                'title': 'Computer Vision and Image Recognition Systems',
                'section': 'Visual Intelligence',
                'type': 'sample'
            },
            {
                'content': 'Data science is an interdisciplinary field that combines statistical analysis, machine learning, and domain expertise to extract insights from structured and unstructured data. The data science lifecycle begins with problem formulation and data collection from various sources including databases, APIs, web scraping, sensors, and external datasets. Data quality assessment involves checking for missing values, outliers, inconsistencies, and biases. Exploratory Data Analysis helps understand data characteristics through statistical summaries, correlation analysis, and data visualization using histograms, scatter plots, box plots, and heatmaps. Statistical modeling forms the foundation including descriptive statistics, inferential statistics, probability distributions, and Bayesian methods. Time series analysis handles temporal data with techniques like ARIMA, seasonal decomposition, and forecasting methods. Machine learning integration involves model selection, hyperparameter tuning, and validation using cross-validation techniques. Feature selection and dimensionality reduction improve model interpretability and performance. Ensemble methods like bagging, boosting, and stacking combine multiple models for better predictions. Data visualization and storytelling communicate findings through interactive dashboards using tools like Tableau, Power BI, or custom web applications. Big data technologies handle large-scale datasets using distributed computing frameworks like Apache Spark and Hadoop. NoSQL databases like MongoDB and Cassandra handle unstructured data. Cloud platforms provide scalable infrastructure for data processing and model deployment. MLOps practices streamline the deployment and monitoring of machine learning models in production environments with continuous integration and automated testing.',
                'source': 'Data Science Methodology',
                'title': 'Data Science Methods and Best Practices',
                'section': 'Analytics Framework',
                'type': 'sample'
            }
        ]
        
        texts = [doc['content'] for doc in sample_docs]
        metadata = [
            {
                'source': doc['source'],
                'title': doc['title'],
                'section': doc['section'],
                'type': doc['type'],
                'chunk_id': i,
                'total_chunks': 1,
                'timestamp': datetime.now().isoformat()
            }
            for i, doc in enumerate(sample_docs)
        ]
        
        embeddings = self._get_embeddings(texts)
        self.vector_db.add_documents(embeddings, texts, metadata)
        logger.info(f"Loaded {len(sample_docs)} sample documents")
    
    def add_text_content(self, content: str, title: str = "Manual Entry", source: str = "Text Input") -> Tuple[bool, str, int]:
        """Add text content directly to the knowledge base"""
        try:
            if not content.strip():
                return False, "Content cannot be empty", 0
            
            chunks = self.doc_processor.chunk_text(content)
            
            if not chunks:
                return False, "No chunks created from text", 0
            
            metadata = [
                {
                    'source': source,
                    'title': f'{title} - Chunk {i+1}',
                    'section': f'Part {i+1} of {len(chunks)}',
                    'type': 'manual',
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'timestamp': datetime.now().isoformat()
                }
                for i in range(len(chunks))
            ]
            
            embeddings = self._get_embeddings(chunks)
            self.vector_db.add_documents(embeddings, chunks, metadata)
            
            return True, f"Successfully added text content: {title}", len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding text content: {e}")
            return False, f"Error adding text: {str(e)}", 0
    
    def process_uploaded_file(self, file_path: str) -> Tuple[bool, str, int]:
        """Process uploaded PDF file and add to knowledge base"""
        try:
            if file_path.lower().endswith('.pdf'):
                text = self.doc_processor.extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            if not text.strip():
                return False, "No text content extracted from file", 0
            
            chunks = self.doc_processor.chunk_text(text)
            
            if not chunks:
                return False, "No chunks created from text", 0
            
            filename = Path(file_path).name
            metadata = [
                {
                    'source': filename,
                    'title': f'{filename} - Chunk {i+1}',
                    'section': f'Part {i+1} of {len(chunks)}',
                    'type': 'uploaded',
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'timestamp': datetime.now().isoformat()
                }
                for i in range(len(chunks))
            ]
            
            embeddings = self._get_embeddings(chunks)
            self.vector_db.add_documents(embeddings, chunks, metadata)
            
            return True, f"Successfully processed {filename}", len(chunks)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False, f"Error processing file: {str(e)}", 0
    
    def retrieve_documents(self, query: str, k: int = 10) -> List[Tuple[str, Dict, float]]:
        """Retrieve relevant documents using semantic search"""
        query_embedding = self._get_embeddings([query], task_type="retrieval_query")[0]
        return self.vector_db.search(query_embedding, k=k)
    
    def rerank_documents(self, query: str, documents: List[Tuple[str, Dict, float]], top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Rerank documents using Cohere's rerank model"""
        if not documents:
            return []
        
        try:
            if self.cohere_client and len(documents) > 1:
                texts = [doc[0] for doc in documents]
                
                rerank_response = self.cohere_client.rerank(
                    model=self.rerank_model,
                    query=query,
                    documents=texts,
                    top_n=min(top_k, len(texts)),
                    return_documents=False
                )
                
                reranked_results = []
                for result in rerank_response.results:
                    original_doc = documents[result.index]
                    reranked_results.append((
                        original_doc[0],
                        original_doc[1],
                        result.relevance_score
                    ))
                
                return reranked_results
            else:
                return sorted(documents, key=lambda x: x[2], reverse=True)[:top_k]
                
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return documents[:top_k]
    
    def generate_answer(self, query: str, context_documents: List[Tuple[str, Dict, float]]) -> Tuple[str, List[str], Dict]:
        """Generate answer using Gemini with retrieved context"""
        if not context_documents:
            return "I don't have sufficient information in my knowledge base to answer your question.", [], {}
        
        context_text = ""
        sources_info = []
        
        for i, (doc_text, metadata, score) in enumerate(context_documents, 1):
            context_text += f"[{i}] {metadata.get('title', 'Unknown Document')}\n"
            context_text += f"Source: {metadata.get('source', 'Unknown')}\n"
            context_text += f"Content: {doc_text}\n\n"
            
            sources_info.append({
                'index': i,
                'title': metadata.get('title', 'Unknown'),
                'source': metadata.get('source', 'Unknown'),
                'section': metadata.get('section', 'N/A'),
                'relevance_score': float(score),
                'content_preview': doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            })
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context. Please follow these guidelines:

1. Answer the question using only information from the provided context
2. Include inline citations using [1], [2], [3] etc. to reference the context sources
3. If the context doesn't contain enough information to fully answer the question, state this clearly
4. Provide a comprehensive but concise response
5. Do not make up information not present in the context

Context:
{context_text}

Question: {query}

Please provide a detailed answer with appropriate citations:"""
        
        try:
            if self.gemini_client:
                response = self.gemini_client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1000,
                        temperature=0.2,
                        top_p=0.8,
                        top_k=40
                    )
                )
                answer = response.text
            else:
                answer = self._generate_fallback_answer(query, context_documents)
            
            citations = list(set(re.findall(r'\[(\d+)\]', answer)))
            
            response_metadata = {
                'model_used': self.llm_model if self.gemini_client else 'fallback',
                'context_documents_count': len(context_documents),
                'citations_found': len(citations),
                'total_context_length': len(context_text)
            }
            
            return answer, citations, response_metadata
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            fallback_answer = self._generate_fallback_answer(query, context_documents)
            return fallback_answer, ["1"], {'error': str(e)}
    
    def _generate_fallback_answer(self, query: str, context_documents: List[Tuple[str, Dict, float]]) -> str:
        """Generate a simple fallback answer when APIs are unavailable"""
        if not context_documents:
            return "No relevant information found in the knowledge base."
        
        doc_text, metadata, score = context_documents[0]
        return f"Based on the available information from {metadata.get('source', 'the knowledge base')}, here is what I found: {doc_text[:500]}{'...' if len(doc_text) > 500 else ''} [1]"
    
    def query(self, question: str, retrieve_k: int = 10, rerank_k: int = 5) -> Dict:
        """Complete RAG pipeline: retrieve, rerank, and generate"""
        start_time = time.time()
        
        retrieved_docs = self.retrieve_documents(question, k=retrieve_k)
        
        if not retrieved_docs:
            return {
                'answer': "No relevant documents found in the knowledge base.",
                'sources': [],
                'citations': [],
                'metadata': {
                    'processing_time': round(time.time() - start_time, 3),
                    'retrieved_count': 0,
                    'reranked_count': 0,
                    'database_stats': self.vector_db.get_stats()
                }
            }
        
        reranked_docs = self.rerank_documents(question, retrieved_docs, top_k=rerank_k)
        answer, citations, response_metadata = self.generate_answer(question, reranked_docs)
        
        processing_time = round(time.time() - start_time, 3)
        
        # Add to conversation history
        self.conversation_history.add_conversation(question, answer, citations)
        
        return {
            'answer': answer,
            'sources': [
                {
                    'title': metadata.get('title', 'Unknown'),
                    'source': metadata.get('source', 'Unknown'), 
                    'section': metadata.get('section', 'N/A'),
                    'relevance_score': score,
                    'content_preview': text[:200] + "..." if len(text) > 200 else text
                }
                for text, metadata, score in reranked_docs
            ],
            'citations': citations,
            'metadata': {
                'processing_time': processing_time,
                'retrieved_count': len(retrieved_docs),
                'reranked_count': len(reranked_docs),
                'database_stats': self.vector_db.get_stats(),
                **response_metadata
            }
        }


def create_gradio_interface(rag_system):
    """Create professional Gradio interface with tabs"""
    
    # Interface functions that directly use the rag_system
    def process_query_interface(question: str) -> Tuple[str, str, str, str]:
        """Process user query"""
        if not question.strip():
            return "", "", "", ""
        
        try:
            result = rag_system.query(question)
            
            # Format answer
            answer = ChatInterface.format_answer_display(result['answer'])
            
            # Format citations (only used ones)
            citations = ChatInterface.format_citations_display(result['sources'], result['citations'])
            
            # Format processing stats
            stats = ChatInterface.format_processing_stats(result['metadata'])
            
            # Format conversation history
            history = rag_system.conversation_history.get_history_display()
            
            return answer, citations, stats, history
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            error_msg = f"Error: {str(e)}"
            return error_msg, "", "", ""
    
    def upload_file_interface(file) -> Tuple[str, str]:
        """Handle file upload"""
        if file is None:
            return "", ""
        
        try:
            success, message, chunk_count = rag_system.process_uploaded_file(file.name)
            upload_status = DocumentViewer.format_upload_status(success, message, chunk_count)
            
            # Refresh document viewer
            doc_display = refresh_document_viewer()
            return upload_status, doc_display
                
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            error_status = DocumentViewer.format_upload_status(False, f"Error: {str(e)}")
            return error_status, ""
    
    def add_text_interface(content: str, title: str, source: str) -> Tuple[str, str, str, str, str]:
        """Handle text content addition"""
        if not content.strip():
            return "", "", "", DocumentViewer.format_upload_status(False, "Content cannot be empty"), ""
        
        if not title.strip():
            title = "Manual Entry"
        if not source.strip():
            source = "Text Input"
        
        try:
            success, message, chunk_count = rag_system.add_text_content(content, title, source)
            status = DocumentViewer.format_upload_status(success, message, chunk_count)
            
            # Refresh document viewer
            doc_display = refresh_document_viewer()
            
            # Clear inputs on success
            if success:
                return "", "", "", status, doc_display
            else:
                return content, title, source, status, doc_display
                
        except Exception as e:
            logger.error(f"Add text failed: {e}")
            error_status = DocumentViewer.format_upload_status(False, f"Error: {str(e)}")
            return content, title, source, error_status, ""
    
    def refresh_document_viewer() -> str:
        """Refresh the document viewer"""
        return DocumentViewer.format_document_display(
            rag_system.vector_db.documents, 
            rag_system.vector_db.metadata
        )
    
    def clear_database_interface() -> Tuple[str, str]:
        """Clear the vector database"""
        try:
            rag_system.vector_db.clear_database()
            rag_system.conversation_history.clear_history()
            rag_system._load_sample_documents()
            
            status = DocumentViewer.format_upload_status(True, "Database cleared and sample documents reloaded", 0)
            doc_display = refresh_document_viewer()
            return status, doc_display
            
        except Exception as e:
            logger.error(f"Clear database failed: {e}")
            error_status = DocumentViewer.format_upload_status(False, f"Error: {str(e)}")
            return error_status, ""
    
    def clear_chat_interface() -> Tuple[str, str, str, str]:
        """Clear chat interface"""
        return "", "", "", ""
    
    # Create the interface
    with gr.Blocks(
        title="Mini RAG System - Professional Implementation",
        theme=gr.themes.Soft(),
        css=UITheme.get_custom_css()
    ) as demo:
        
        # Header
        gr.HTML(UITheme.get_header_html())
        
        # Main tabs
        with gr.Tabs(elem_classes="tab-nav"):
            
            # Document Management Tab
            with gr.TabItem("📚 Knowledge Base", elem_id="kb_tab", elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="main-column"):

                        # Manual Text Entry Section
                        with gr.Group(elem_classes="upload-section"):
                            gr.HTML('<div class="section-header"><h3 class="section-title">Add Text Content</h3></div>')
                            
                            text_content = gr.Textbox(
                                label="Text Content",
                                placeholder="Enter your text content here...",
                                lines=8,
                                elem_classes="text-input"
                            )
                            
                            with gr.Row():
                                text_title = gr.Textbox(
                                    label="Title",
                                    placeholder="Document Title",
                                    value="Manual Entry",
                                    elem_classes="text-input"
                                )
                                text_source = gr.Textbox(
                                    label="Source",
                                    placeholder="Source Name",
                                    value="Text Input",
                                    elem_classes="text-input"
                                )
                            
                            add_text_btn = gr.Button("Add Text", variant="primary", elem_classes="primary-button")
                            
                        # File Upload Section
                        with gr.Group(elem_classes="upload-section"):
                            gr.HTML('<div class="section-header"><h3 class="section-title">Upload Documents</h3></div>')
                            
                            file_upload = gr.File(
                                label="Choose PDF or Text File",
                                file_types=[".pdf", ".txt", ".md"],
                                type="filepath"
                            )
                            
                            upload_btn = gr.Button("Process File", variant="primary", elem_classes="primary-button")
                        

                        
                        # Database Actions
                        with gr.Group(elem_classes="content-section"):
                            with gr.Row(elem_classes="button-group"):
                                refresh_btn = gr.Button("Refresh View", variant="secondary")
                                clear_db_btn = gr.Button("Clear Database", variant="stop")
                        
                        # Status Display
                        upload_status = gr.HTML(label="Status")
                    
                    with gr.Column(scale=2, elem_classes="sidebar-column"):
                        gr.HTML('<div class="section-header"><h3 class="section-title">Document Viewer</h3></div>')
                        
                        document_viewer = gr.HTML(
                            value=DocumentViewer.format_document_display(
                                rag_system.vector_db.documents, 
                                rag_system.vector_db.metadata
                            ),
                            label="Knowledge Base Contents"
                        )
            
            # Chat Interface Tab
            with gr.TabItem("💬 Chat", elem_id="chat_tab", elem_classes="tab-content"):
                with gr.Row():
                    with gr.Column(scale=2, elem_classes="main-column"):
                        gr.HTML('<div class="section-header"><h3 class="section-title">Ask Questions</h3></div>')
                        
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about the documents in your knowledge base...",
                            lines=3,
                            elem_classes="text-input"
                        )
                        
                        with gr.Row(elem_classes="button-group"):
                            ask_btn = gr.Button("Ask Question", variant="primary", elem_classes="primary-button")
                            clear_chat_btn = gr.Button("Clear Chat", variant="secondary")
                        
                        # Answer display
                        answer_output = gr.HTML(
                            label="Answer",
                            elem_classes="answer-display"
                        )
                        
                        # Processing statistics
                        processing_stats = gr.HTML(
                            label="Processing Statistics"
                        )
                    
                    with gr.Column(scale=1, elem_classes="sidebar-column"):
                        # Citations box
                        gr.HTML('<div class="section-header"><h3 class="section-title">Citations Used</h3></div>')
                        citations_output = gr.HTML(
                            label="Active Citations",
                            elem_classes="citations-display"
                        )
                        
                        # Conversation history
                        gr.HTML('<div class="section-header"><h3 class="section-title">Recent Questions</h3></div>')
                        history_output = gr.HTML(
                            value=rag_system.conversation_history.get_history_display(),
                            label="Question History",
                            elem_classes="history-display"
                        )
        
        # Event handlers
        
        # Document management events
        upload_btn.click(
            fn=upload_file_interface,
            inputs=[file_upload],
            outputs=[upload_status, document_viewer]
        )
        
        add_text_btn.click(
            fn=add_text_interface,
            inputs=[text_content, text_title, text_source],
            outputs=[text_content, text_title, text_source, upload_status, document_viewer]
        )
        
        refresh_btn.click(
            fn=refresh_document_viewer,
            outputs=[document_viewer]
        )
        
        clear_db_btn.click(
            fn=clear_database_interface,
            outputs=[upload_status, document_viewer]
        )
        
        # Chat events
        ask_btn.click(
            fn=process_query_interface,
            inputs=[question_input],
            outputs=[answer_output, citations_output, processing_stats, history_output]
        )
        
        question_input.submit(
            fn=process_query_interface,
            inputs=[question_input],
            outputs=[answer_output, citations_output, processing_stats, history_output]
        )
        
        clear_chat_btn.click(
            fn=clear_chat_interface,
            outputs=[answer_output, citations_output, processing_stats, question_input]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued);">
            <p><strong>Mini RAG System</strong> - Professional Implementation</p>
            <p>Built with Gradio, FAISS, Google AI, and Cohere APIs</p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    rag_system = MiniRAGSystem()
    demo = create_gradio_interface(rag_system)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )