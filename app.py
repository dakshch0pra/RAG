import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import fastapi_backend
from rag_system import MiniRAGSystem, create_gradio_interface

# Initialize RAG system first
rag_system = MiniRAGSystem()

# Initialize the backend with RAG system
fastapi_backend.init_rag_system(rag_system)

# Create FastAPI app for API routes
api_app = FastAPI(
    title="Mini RAG System API",
    description="API endpoints for the RAG system",
    version="1.0.0"
)

api_app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Include all the API routes from fastapi_backend
api_app.mount("/", fastapi_backend.app)

# Create Gradio interface
demo = create_gradio_interface(rag_system)

# For HF Spaces, we need to launch Gradio directly
# The FastAPI routes will be available but secondary
if __name__ == "__main__":
    if os.getenv("SPACE_ID"):
        # On HF Spaces - launch Gradio only
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    else:
        # Local development - you can choose to run either Gradio or FastAPI
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "api":
            # Run FastAPI only
            import uvicorn
            uvicorn.run(api_app, host="0.0.0.0", port=8000)
        else:
            # Run Gradio (default)
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True
            )