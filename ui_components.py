import gradio as gr
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime


class UITheme:
    """Custom theme and styling configuration"""
    
    @staticmethod
    def get_custom_css():
        """Return custom CSS for professional styling"""
        return """
        /* Main container styling */
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Consistent tab content sizing */
        .tab-content {
            min-height: 600px;
        }
        
        /* Consistent column heights */
        .main-column {
            min-height: 500px;
        }
        
        .sidebar-column {
            min-height: 500px;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        /* Tab styling */
        .tab-nav {
            background: var(--background-fill-secondary);
            border-radius: 8px;
            padding: 0.25rem;
            margin-bottom: 1.5rem;
        }
        
        /* Section headers */
        .section-header {
            background: var(--background-fill-secondary);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            border-left: 4px solid #667eea;
        }
        
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--body-text-color);
            margin: 0;
        }
        
        /* Card styling */
        .info-card {
            background: var(--background-fill-primary);
            border: 1px solid var(--border-color-primary);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            color: var(--body-text-color);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
            margin: 0;
        }
        
        /* Button styling */
        .primary-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            padding: 0.75rem 2rem;
            border-radius: 6px;
            font-weight: 600;
            transition: transform 0.2s;
            min-width: 140px;
        }
        
        .primary-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Input styling */
        .text-input {
            border-radius: 6px;
            border: 1px solid var(--border-color-primary);
            padding: 0.75rem;
        }
        
        /* Consistent spacing */
        .content-section {
            margin-bottom: 2rem;
        }
        
        .button-group {
            display: flex;
            gap: 0.75rem;
            margin: 1rem 0;
        }
        
        .upload-section {
            background: var(--background-fill-secondary);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color-primary);
        }
        
        /* Citation box styling */
        .citation-box {
            background: var(--background-fill-secondary);
            border-left: 4px solid #4299e1;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
            color: var(--body-text-color);
        }
        
        .citation-header {
            font-weight: 600;
            color: var(--body-text-color);
            margin-bottom: 0.5rem;
        }
        
        /* History styling */
        .history-item {
            background: var(--background-fill-primary);
            border: 1px solid var(--border-color-primary);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: var(--body-text-color);
        }
        
        .history-question {
            font-weight: 600;
            color: var(--body-text-color);
            margin-bottom: 0.5rem;
        }
        
        .history-timestamp {
            font-size: 0.8rem;
            color: var(--body-text-color-subdued);
        }
        
        /* Document viewer styling */
        .document-item {
            background: var(--background-fill-primary);
            border: 1px solid var(--border-color-primary);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            transition: border-color 0.2s;
            color: var(--body-text-color);
        }
        
        .document-item:hover {
            border-color: var(--border-color-accent);
        }
        
        .document-title {
            font-weight: 600;
            color: var(--body-text-color);
            margin-bottom: 0.25rem;
        }
        
        .document-meta {
            font-size: 0.9rem;
            color: var(--body-text-color-subdued);
            margin-bottom: 0.5rem;
        }
        
        /* Status indicators */
        .status-success {
            color: #38a169;
            font-weight: 600;
        }
        
        .status-error {
            color: #e53e3e;
            font-weight: 600;
        }
        
        .status-warning {
            color: #d69e2e;
            font-weight: 600;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .gradio-container {
                padding: 1rem;
            }
        }
        """
    
    @staticmethod
    def get_header_html():
        """Return header HTML"""
        return """
        <div class="main-header">
            <h1>Mini RAG System</h1>
            <p>Professional Retrieval-Augmented Generation Platform</p>
        </div>
        """


class ConversationHistory:
    """Manages conversation history with limited storage"""
    
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.conversations = []
    
    def add_conversation(self, question: str, answer: str, citations: List[str]):
        """Add a new conversation to history"""
        conversation = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': question,
            'answer': answer,
            'citations': citations
        }
        
        self.conversations.insert(0, conversation)
        
        # Keep only the most recent conversations
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[:self.max_history]
    
    def get_history_display(self) -> str:
        """Format conversation history for display"""
        if not self.conversations:
            return "No conversation history yet."
        
        history_html = ""
        for i, conv in enumerate(self.conversations, 1):
            history_html += f"""
            <div class="history-item">
                <div class="history-question">Q{i}: {conv['question'][:100]}{'...' if len(conv['question']) > 100 else ''}</div>
                <div class="history-timestamp">{conv['timestamp']} | Citations: {', '.join(conv['citations']) if conv['citations'] else 'None'}</div>
            </div>
            """
        
        return history_html
    
    def clear_history(self):
        """Clear all conversation history"""
        self.conversations = []


class DocumentViewer:
    """Handles document viewing and management interface"""
    
    @staticmethod
    def format_document_display(documents: List[str], metadata: List[Dict]) -> str:
        """Format documents for display in the viewer"""
        if not documents:
            return "<div class='info-card'>No documents in the knowledge base yet. Upload some documents to get started.</div>"
        
        html_content = f"""
        <div class="info-card">
            <h3>Knowledge Base Contents</h3>
            <p><strong>Total Documents:</strong> {len(documents)} chunks</p>
        </div>
        """
        
        # Group documents by source
        sources = {}
        for doc, meta in zip(documents, metadata):
            source = meta.get('source', 'Unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append((doc, meta))
        
        for source, doc_list in sources.items():
            html_content += f"""
            <div class="document-item">
                <div class="document-title">{source}</div>
                <div class="document-meta">
                    Chunks: {len(doc_list)} | 
                    Type: {doc_list[0][1].get('type', 'Unknown')} |
                    Added: {doc_list[0][1].get('timestamp', 'Unknown')[:19] if doc_list[0][1].get('timestamp') else 'Unknown'}
                </div>
                <details>
                    <summary>View Chunks ({len(doc_list)})</summary>
                    <div style="margin-top: 1rem;">
            """
            
            for i, (doc, meta) in enumerate(doc_list[:5]):  # Show only first 5 chunks
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                html_content += f"""
                        <div style="background: var(--background-fill-secondary); padding: 0.5rem; margin: 0.25rem 0; border-radius: 4px; color: var(--body-text-color);">
                            <strong>Chunk {meta.get('chunk_id', i+1)}:</strong> {preview}
                        </div>
                """
            
            if len(doc_list) > 5:
                html_content += f"<p><em>... and {len(doc_list) - 5} more chunks</em></p>"
            
            html_content += """
                    </div>
                </details>
            </div>
            """
        
        return html_content
    
    @staticmethod
    def format_upload_status(success: bool, message: str, chunk_count: int = 0) -> str:
        """Format upload status message"""
        if success:
            return f"""
            <div class="info-card">
                <p class="status-success">✓ Success</p>
                <p>{message}</p>
                {f'<p><strong>Chunks Added:</strong> {chunk_count}</p>' if chunk_count > 0 else ''}
            </div>
            """
        else:
            return f"""
            <div class="info-card">
                <p class="status-error">✗ Error</p>
                <p>{message}</p>
            </div>
            """


class ChatInterface:
    """Handles the chat interface components"""
    
    @staticmethod
    def format_answer_display(answer: str) -> str:
        """Format the answer with better styling"""
        if not answer:
            return ""
        
        # Add paragraph breaks for better readability
        formatted_answer = answer.replace('\n\n', '</p><p>').replace('\n', '<br>')
        
        return f"""
        <div class="info-card">
            <p>{formatted_answer}</p>
        </div>
        """
    
    @staticmethod
    def format_citations_display(sources: List[Dict], used_citations: List[str]) -> str:
        """Format only the citations that were actually used in the answer"""
        if not sources or not used_citations:
            return "<div class='citation-box'><div class='citation-header'>No citations used in this response.</div></div>"
        
        html_content = """
        <div class="citation-box">
            <div class="citation-header">Citations Used in Response</div>
        """
        
        # Filter sources to only show those that were cited
        for citation_num in used_citations:
            try:
                idx = int(citation_num) - 1
                if 0 <= idx < len(sources):
                    source = sources[idx]
                    html_content += f"""
                    <div style="margin: 1rem 0; padding: 0.75rem; background: var(--background-fill-secondary); border-radius: 6px; border: 1px solid var(--border-color-primary);">
                        <div style="font-weight: 600; color: var(--body-text-color); margin-bottom: 0.25rem;">
                            [{citation_num}] {source.get('title', 'Unknown Document')}
                        </div>
                        <div style="font-size: 0.9rem; color: var(--body-text-color-subdued); margin-bottom: 0.5rem;">
                            Source: {source.get('source', 'Unknown')} | 
                            Section: {source.get('section', 'N/A')} | 
                            Relevance: {source.get('relevance_score', 0):.3f}
                        </div>
                        <div style="font-size: 0.9rem; color: var(--body-text-color);">
                            {source.get('content_preview', 'No preview available')}
                        </div>
                    </div>
                    """
            except (ValueError, IndexError):
                continue
        
        html_content += "</div>"
        return html_content
    
    @staticmethod
    def format_processing_stats(metadata: Dict) -> str:
        """Format processing statistics"""
        if not metadata:
            return ""
        
        return f"""
        <div class="info-card">
            <h4>Processing Statistics</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div class="metric-card">
                    <p class="metric-value">{metadata.get('processing_time', 0):.2f}s</p>
                    <p class="metric-label">Processing Time</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{metadata.get('retrieved_count', 0)}</p>
                    <p class="metric-label">Retrieved</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{metadata.get('reranked_count', 0)}</p>
                    <p class="metric-label">Reranked</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{metadata.get('citations_found', 0)}</p>
                    <p class="metric-label">Citations</p>
                </div>
            </div>
        </div>
        """


class DatabaseMetrics:
    """Handles database metrics and statistics display"""
    
    @staticmethod
    def format_database_stats(stats: Dict) -> str:
        """Format database statistics for display"""
        return f"""
        <div class="info-card">
            <h4>Vector Database Status</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div>
                    <strong>Total Documents:</strong> {stats.get('total_documents', 0)}
                </div>
                <div>
                    <strong>Index Size:</strong> {stats.get('index_size', 0)}
                </div>
                <div>
                    <strong>Embedding Dimension:</strong> {stats.get('embedding_dimension', 0)}
                </div>
                <div>
                    <strong>Model Stack:</strong> Gemini + text-embedding-004 + rerank-v3.0
                </div>
            </div>
        </div>
        """