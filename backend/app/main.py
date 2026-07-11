"""
FastAPI Backend for CocoCastAI
Provides REST API endpoints for mobile and web clients
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
import logging

# Import RAG engine
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from step2_rag_engine import load_rag_chain, get_answer, get_plain_answer, translate_text

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ Globals for RAG Chain ============

rag_chain = None
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG chain when server starts"""
    global rag_chain, retriever
    try:
        logger.info("Loading RAG chain...")
        rag_chain, retriever = load_rag_chain()
        logger.info("RAG chain loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load RAG chain: {str(e)}")
        raise
    yield


# Initialize FastAPI app
app = FastAPI(
    title="CocoCastAI API",
    description="RAG-based advisory system for coconut farming in Sri Lanka",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for mobile and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", tags=["Health"])
async def root():
    """Serve the web interface"""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "status": "running",
        "service": "CocoCastAI",
        "version": "1.0.0",
        "message": "Web interface not found at /static/index.html"
    }

class QuestionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the best practices for coconut tree maintenance?",
                "context": "Optional context",
                "language": "en"
            }
        }
    )
    question: str
    context: Optional[str] = None
    language: Optional[str] = 'en'


class SourceDocument(BaseModel):
    title: str
    content: str
    metadata: Optional[dict] = None


class AnswerResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: List[SourceDocument]
    confidence: Optional[float] = None
    context_used: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    code: Optional[str] = None

class TranslateItem(BaseModel):
    id: str
    text: str

class TranslateBatchRequest(BaseModel):
    messages: List[TranslateItem]
    target_lang: str

class TranslateItemResponse(BaseModel):
    id: str
    translated_text: str

class TranslateBatchResponse(BaseModel):
    success: bool
    translations: List[TranslateItemResponse]


# ============ API Endpoints ============


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_chain_loaded": rag_chain is not None,
        "retriever_loaded": retriever is not None
    }


@app.post("/ask", response_model=AnswerResponse, tags=["Advisory"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the CocoCastAI system
    
    Returns:
        - question: The question asked
        - answer: The AI-generated answer
        - sources: Source documents used for the answer
    """
    if not rag_chain or not retriever:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not loaded. Please try again later."
        )
    
    question = request.question.strip()
    user_lang = request.language.strip() if request.language else 'en'
    
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing question: {question} (Lang: {user_lang})")
        
        # Helper to check for Sinhala characters
        def is_sinhala(text: str) -> bool:
            return any('\u0d80' <= char <= '\u0dff' for char in text)
        
        # Translate question to English if it is in Sinhala
        rag_question = question
        if is_sinhala(question):
            logger.info("Sinhala question detected. Translating to English for RAG...")
            try:
                rag_question = translate_text(question, "en")
                logger.info(f"Translated question: {rag_question}")
            except Exception as e:
                logger.error(f"Error translating question to English: {str(e)}")
                # Fallback to original question
                rag_question = question
        
        # Query the RAG engine using the English query
        result = get_answer(rag_question, rag_chain, retriever, user_context=request.context)
        
        # Translate the answer back to Sinhala if Sinhala is requested
        answer = result["answer"]
        if user_lang == "si":
            logger.info("Translating answer to Sinhala...")
            try:
                answer = translate_text(answer, "si")
                logger.info("Answer successfully translated to Sinhala.")
            except Exception as e:
                logger.error(f"Error translating answer to Sinhala: {str(e)}")
                # Fallback to original English answer
        
        # Format sources
        sources = [
            SourceDocument(
                title=source.get("title", "Document"),
                content=source.get("content", ""),
                metadata=source.get("metadata")
            )
            for source in result.get("sources", [])
        ]
        
        return AnswerResponse(
            success=True,
            question=question, # Return the original question asked by the user
            answer=answer,     # Return translated or English answer
            sources=sources,
            confidence=result.get("confidence"),
            context_used=result.get("context_used")
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/translate-batch", response_model=TranslateBatchResponse, tags=["Advisory"])
async def translate_batch(request: TranslateBatchRequest):
    """
    Translates a list of chat messages to the target language
    """
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        target_lang = request.target_lang.strip()
        if target_lang not in ["en", "si"]:
            raise HTTPException(status_code=400, detail="Invalid target language. Must be 'en' or 'si'")
            
        logger.info(f"Batch translating {len(request.messages)} messages to {target_lang}")
        
        # Run translations in parallel using ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(
                    executor, 
                    translate_text, 
                    msg.text, 
                    target_lang
                )
                for msg in request.messages
            ]
            translated_texts = await asyncio.gather(*tasks)
            
        translations = [
            TranslateItemResponse(id=msg.id, translated_text=translated_text)
            for msg, translated_text in zip(request.messages, translated_texts)
        ]
        
        return TranslateBatchResponse(
            success=True,
            translations=translations
        )
    except Exception as e:
        logger.error(f"Error in batch translation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", tags=["Advisory"])
async def compare_answers(request: QuestionRequest):
    """
    Compare Plain LLM vs RAG system
    """
    if not rag_chain or not retriever:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not loaded. Please try again later."
        )
    
    question = request.question.strip()
    
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Comparing question: {question}")
        
        # Get Plain LLM Answer
        plain_result = get_plain_answer(question, user_context=request.context)
        
        # Get RAG Answer
        rag_result = get_answer(question, rag_chain, retriever, user_context=request.context)
        
        # Format sources
        sources = [
            SourceDocument(
                title=source.get("title", "Document"),
                content=source.get("content", ""),
                metadata=source.get("metadata")
            )
            for source in rag_result.get("sources", [])
        ]
        
        return {
            "success": True,
            "question": question,
            "plain_llm": {
                "answer": plain_result["answer"]
            },
            "rag_system": {
                "answer": rag_result["answer"],
                "sources": sources,
                "confidence": rag_result.get("confidence"),
                "context_used": rag_result.get("context_used")
            }
        }
        
    except Exception as e:
        logger.error(f"Error comparing answers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error comparing answers: {str(e)}"
        )


@app.get("/info", tags=["Info"])
async def get_info():
    """Get system information"""
    return {
        "service": "CocoCastAI",
        "version": "1.0.0",
        "description": "RAG-based advisory system for coconut farming in Sri Lanka",
        "endpoints": {
            "ask": "/ask (POST)",
            "health": "/health (GET)"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "success": False,
        "error": "Internal server error",
        "code": "INTERNAL_ERROR"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
