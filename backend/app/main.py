"""
FastAPI Backend for SaruPol
Provides REST API endpoints for mobile and web clients
Updated with 82 CRI Reference Images
"""

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import os
import time
import asyncio
from dotenv import load_dotenv
import logging

import uuid

# Import RAG engine
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from step2_rag_engine import load_rag_chain, get_answer, get_answer_with_memory, translate_text, get_multi_llm_answer, translate_multi_llm_payload, find_relevant_images

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
    title="SaruPol API",
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

# Mount static files (support both /static and /api/static prefixes)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/api/static", StaticFiles(directory=static_dir), name="api_static")

@app.get("/", tags=["Health"])
async def root():
    """Serve the web interface"""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "status": "running",
        "service": "SaruPol",
        "version": "1.0.0",
        "message": "Web interface not found at /static/index.html"
    }

class QuestionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the best practices for coconut tree maintenance?",
                "context": "Optional context",
                "language": "en",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "latitude": 6.9271,
                "longitude": 79.8612
            }
        }
    )
    question: str
    context: Optional[str] = None
    language: Optional[str] = 'en'
    session_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class SourceDocument(BaseModel):
    title: str
    content: str
    metadata: Optional[dict] = None


class ImageReference(BaseModel):
    url: str
    caption: str
    source: str


class AnswerResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: List[SourceDocument]
    images: Optional[List[ImageReference]] = []
    zone: Optional[str] = None
    season: Optional[str] = None
    confidence: Optional[float] = None
    context_used: Optional[str] = None
    session_id: Optional[str] = None

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


# Multi-LLM Validation models
class MultiLLMRequest(BaseModel):
    question: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    language: Optional[str] = 'en'

class MultiLLMResponse(BaseModel):
    success: bool
    best_answer: str
    best_model: str
    reason: str
    consensus_score: int
    llama_answer: str
    llama8b_answer: str
    gemma_answer: Optional[str] = None
    qwen_answer: Optional[str] = None
    sources: List[SourceDocument]
    images: Optional[List[ImageReference]] = []
    zone: Optional[str] = None
    season: Optional[str] = None
    early_exit: bool = False
    similarity_score: Optional[float] = None
    latency_ms: Optional[int] = None


# ============ Helper: Server-side zone/season detection ============

def _determine_zone(lat: float, lon: float) -> str:
    """Replicate the frontend zone detection logic server-side."""
    if 5.9 <= lat <= 7.5 and 79.8 <= lon <= 80.6:
        return 'Wet Zone'
    if 5.9 <= lat <= 8.0 and 79.8 <= lon <= 81.2:
        return 'Intermediate Zone'
    if 5.5 <= lat <= 10.0 and 79.5 <= lon <= 82.0:
        return 'Dry Zone'
    return 'Unknown Zone'

def _determine_season() -> str:
    """Determine current agricultural season from system date."""
    from datetime import datetime
    month = datetime.now().month
    return 'Yala' if 5 <= month <= 9 else 'Maha'

def _get_month_name() -> str:
    """Get current month name."""
    from datetime import datetime
    return datetime.now().strftime('%B')


# ============ API Endpoints ============

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_chain_loaded": rag_chain is not None,
        "retriever_loaded": retriever is not None
    }


@router.post("/ask", response_model=AnswerResponse, tags=["Advisory"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the SaruPol system with conversation memory support.
    
    Returns:
        - question: The question asked
        - answer: The AI-generated answer
        - sources: Source documents used for the answer
        - session_id: Unique conversation session ID
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

    # Manage session_id (generate new UUID if not provided)
    session_id = request.session_id.strip() if (request.session_id and request.session_id.strip()) else str(uuid.uuid4())
    
    try:
        logger.info(f"Processing question for session [{session_id}]: {question} (Lang: {user_lang})")
        
        # Helper to check for Sinhala characters
        def is_sinhala(text: str) -> bool:
            return any('\u0d80' <= char <= '\u0dff' for char in text)
        
        # Translate question to English if it is in Sinhala
        rag_question = question
        if is_sinhala(question):
            logger.info("Sinhala question detected. Translating to English for RAG...")
            try:
                rag_question = await asyncio.to_thread(translate_text, question, "en")
                logger.info(f"Translated question: {rag_question}")
            except Exception as e:
                logger.error(f"Error translating question to English: {str(e)}")
                # Fallback to original question
                rag_question = question
        
        # Calculate context / zone
        user_context = request.context
        zone = "Wet Zone"
        if request.latitude is not None and request.longitude is not None:
            zone = _determine_zone(request.latitude, request.longitude)
            season_name = _determine_season()
            month_name = _get_month_name()
            user_context = f"{zone} | {season_name} Season ({month_name})"
        elif request.context and "|" in request.context:
            zone = request.context.split("|")[0].strip()

        # Query the RAG engine with memory using English query & session_id
        result = await asyncio.to_thread(
            get_answer_with_memory,
            rag_question,
            session_id,
            rag_chain,
            retriever,
            user_context=user_context
        )
        
        # Translate the answer back to Sinhala if Sinhala is requested
        answer = result["answer"]
        display_question = question
        if user_lang == "si":
            logger.info("Translating answer to Sinhala...")
            try:
                answer = await asyncio.to_thread(translate_text, answer, "si")
                if not is_sinhala(question):
                    display_question = await asyncio.to_thread(translate_text, question, "si")
                logger.info("Answer and question successfully translated to Sinhala.")
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

        # Find semantically relevant CRI reference images using question + answer context
        raw_images = await asyncio.to_thread(find_relevant_images, rag_question, result.get("answer", ""), top_k=2)
        images = [
            ImageReference(
                url=img["url"],
                caption=img["caption"],
                source=img["source"]
            )
            for img in raw_images
        ]
        
        # Calculate season
        season = _determine_season()

        return AnswerResponse(
            success=True,
            question=display_question, # Return Sinhala translated or original question
            answer=answer,     # Return translated or English answer
            sources=sources,
            images=images,
            zone=zone,
            season=season,
            confidence=result.get("confidence"),
            context_used=result.get("context_used"),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@router.post("/translate-batch", response_model=TranslateBatchResponse, tags=["Advisory"])
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


# Multi-LLM Validation endpoint


@router.post("/ask-multi", response_model=MultiLLMResponse, tags=["Advisory"])
async def ask_multi_llm(request: MultiLLMRequest):
    """
    Multi-LLM Validation: Send the same question to 3 LLMs in parallel,
    then use a judge LLM to select the best answer.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not loaded. Please try again later.")

    question = request.question.strip()
    user_lang = request.language.strip() if request.language else 'en'
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Helper to check for Sinhala characters
        def is_sinhala(text: str) -> bool:
            return any('\u0d80' <= char <= '\u0dff' for char in text)

        # Pre-translate query to English if Sinhala
        rag_question = question
        if is_sinhala(question):
            logger.info("Sinhala multi-LLM question detected. Translating to English for RAG...")
            try:
                rag_question = await asyncio.to_thread(translate_text, question, "en")
                logger.info(f"Translated multi-LLM question: {rag_question}")
            except Exception as e:
                logger.error(f"Error translating multi-LLM question to English: {str(e)}")
                rag_question = question

        # Determine zone and season context
        zone = None
        season = _determine_season()
        month = _get_month_name()

        if request.latitude is not None and request.longitude is not None:
            zone = _determine_zone(request.latitude, request.longitude)

        context_parts = []
        if zone:
            context_parts.append(zone)
        context_parts.append(f"{season} Season ({month})")
        user_context = " | ".join(context_parts)

        logger.info(f"Multi-LLM query: {rag_question} (Lang: {user_lang}) | Context: {user_context}")

        # Run multi-LLM validation (parallel execution inside)
        start_time = time.time()
        result = await asyncio.to_thread(
            get_multi_llm_answer, rag_question, retriever, user_context
        )
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract fields
        best_answer = result["best_answer"]
        llama_answer = result["llama_answer"]
        llama8b_answer = result["llama8b_answer"]
        gemma_answer = result.get("gemma_answer") or result.get("qwen_answer", "")
        reason = result["reason"]
        early_exit = result.get("early_exit", False)
        similarity_score = result.get("similarity_score", None)

        # Post-translate answers back to Sinhala if requested
        if user_lang == "si":
            logger.info("Translating all Multi-LLM response fields to Sinhala...")
            payload_to_translate = {
                "best_answer": best_answer,
                "reason": reason,
                "llama_answer": llama_answer,
                "llama8b_answer": llama8b_answer,
                "gemma_answer": gemma_answer
            }
            try:
                translated_dict = await asyncio.to_thread(translate_multi_llm_payload, payload_to_translate, target_lang="si")
                best_answer = translated_dict.get("best_answer", best_answer)
                reason = translated_dict.get("reason", reason)
                llama_answer = translated_dict.get("llama_answer", llama_answer)
                llama8b_answer = translated_dict.get("llama8b_answer", llama8b_answer)
                gemma_answer = translated_dict.get("gemma_answer", gemma_answer)
                logger.info("Multi-LLM response fields successfully translated to Sinhala.")
            except Exception as e:
                logger.error(f"Error translating Multi-LLM payload to Sinhala: {e}")

        # Format sources
        sources = [
            SourceDocument(
                title=s.get("title", "Document"),
                content=s.get("content", ""),
                metadata=s.get("metadata")
            )
            for s in result.get("sources", [])
        ]

        # Find semantically relevant CRI reference images using question + best_answer context
        raw_images = await asyncio.to_thread(find_relevant_images, rag_question, result.get("best_answer", ""), top_k=2)
        images = [
            ImageReference(
                url=img["url"],
                caption=img["caption"],
                source=img["source"]
            )
            for img in raw_images
        ]

        return MultiLLMResponse(
            success=True,
            best_answer=best_answer,
            best_model=result["best_model"],
            reason=reason,
            consensus_score=result["consensus_score"],
            llama_answer=llama_answer,
            llama8b_answer=llama8b_answer,
            gemma_answer=gemma_answer,
            qwen_answer=gemma_answer,
            sources=sources,
            images=images,
            zone=zone,
            season=f"{season} ({month})",
            early_exit=early_exit,
            similarity_score=similarity_score,
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Error in multi-LLM query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing multi-LLM query: {str(e)}")


@router.get("/tts", tags=["TTS"])
async def text_to_speech(text: str, lang: str = "en"):
    """
    Generate Text-to-Speech audio stream for a given text and language.
    Includes text preprocessing for cleaner Sinhala pronunciation.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    import re
    
    def clean_text_for_tts(raw_text: str) -> str:
        """Remove markdown formatting and special characters that confuse TTS."""
        cleaned = raw_text
        # Remove markdown bold/italic markers
        cleaned = re.sub(r'\*{1,3}', '', cleaned)
        # Remove markdown headers (###, ##, #)
        cleaned = re.sub(r'^#{1,6}\s*', '', cleaned, flags=re.MULTILINE)
        # Convert bullet points to natural pause
        cleaned = re.sub(r'^[\s]*[-•‣]\s*', '', cleaned, flags=re.MULTILINE)
        # Convert numbered lists to just the content
        cleaned = re.sub(r'^\s*\d+[.)]\s*', '', cleaned, flags=re.MULTILINE)
        # Remove horizontal rules
        cleaned = re.sub(r'^-{3,}$', '', cleaned, flags=re.MULTILINE)
        # Remove URLs
        cleaned = re.sub(r'https?://\S+', '', cleaned)
        # Normalize multiple newlines to single pause
        cleaned = re.sub(r'\n{2,}', '. ', cleaned)
        cleaned = re.sub(r'\n', ', ', cleaned)
        # Remove extra whitespace
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        return cleaned.strip()

    def add_sinhala_pronunciation_hints(raw_text: str) -> str:
        """
        Spell out English abbreviations letter-by-letter so the Sinhala
        TTS engine pronounces them clearly instead of garbling them.
        """
        # Spell out common fertilizer codes letter by letter
        def spell_out(match):
            code = match.group(0)
            # Add spaces between letters/digits for TTS to pronounce individually
            return ' '.join(code)
        
        # Match uppercase letter+digit codes like YPM-W, APM-D, NPK, etc.
        processed = re.sub(r'\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b', spell_out, raw_text)
        return processed

    def normalize_sinhala_for_tts(raw_text: str) -> str:
        """
        Normalize Sinhala Unicode for better TTS pronunciation.
        
        Preserves Zero Width Joiner (U+200D) so conjunct consonants (e.g. ප්‍ර, ක්‍ර, ත්‍ර)
        are pronounced properly as consonant blends.
        
        Maps retroflex characters (ළ, ණ, etc.) to their dental counterparts (ල, න) because
        TTS voices often mispronounce or skip retroflex characters entirely.
        """
        processed = raw_text
        
        # Replace retroflex vowel forms and characters with dental counterparts
        replacements = [
            ('\u0dc5\u0dd6', '\u0dbd\u0dd6'),  # ළූ -> ලූ
            ('\u0dc5\u0dd4', '\u0dbd\u0dd4'),  # ළු -> ලු
            ('\u0dab\u0dd6', '\u0db1\u0dd6'),  # ණූ -> නූ
            ('\u0dab\u0dd4', '\u0db1\u0dd4'),  # ණු -> නු
            ('\u0dab\u0dca', '\u0db1\u0dca'),  # ණ් -> න්
            ('\u0dc5', '\u0dbd'),              # ළ -> ල
            ('\u0dab', '\u0db1'),              # ණ -> න
        ]
        
        for search, replace in replacements:
            processed = processed.replace(search, replace)
            
        # Clean up other invisible formatting codes except ZWJ (\u200d)
        processed = processed.replace('\u200c', '')  # Remove ZWNJ
        processed = processed.replace('\u200b', '')  # Remove ZWSP
        processed = processed.replace('\ufeff', '')  # Remove BOM
        processed = processed.replace('\u00a0', ' ') # Replace non-breaking space
        
        return processed

    # Clean the text
    cleaned_text = clean_text_for_tts(text)
    
    # Map languages to Edge TTS neural voices and rate settings
    if lang.lower() == "si":
        voice = "si-LK-SameeraNeural"  # Revert to SameeraNeural as default for wider baseline
        rate = "-10%"   # Slower rate for clear Sinhala articulation
        # Normalize Sinhala Unicode conjuncts and retroflex characters
        cleaned_text = normalize_sinhala_for_tts(cleaned_text)
        # Add pronunciation hints for English terms in Sinhala text
        cleaned_text = add_sinhala_pronunciation_hints(cleaned_text)
    else:
        voice = "en-US-AriaNeural"
        rate = "+0%"
        
    logger.info(f"Generating TTS for text length {len(cleaned_text)} in voice {voice} at rate {rate}")
    
    try:
        import edge_tts
        
        async def audio_generator():
            communicate = edge_tts.Communicate(cleaned_text, voice, rate=rate)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                    
        return StreamingResponse(audio_generator(), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Error generating TTS audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")


@router.get("/info", tags=["Info"])
async def get_info():
    """Get system information"""
    return {
        "service": "SaruPol",
        "version": "1.0.0",
        "description": "RAG-based advisory system for coconut farming in Sri Lanka",
        "endpoints": {
            "ask": "/ask (POST)",
            "health": "/health (GET)"
        }
    }


# Include router with and without /api prefix for maximum compatibility
app.include_router(router)
app.include_router(router, prefix="/api")


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
