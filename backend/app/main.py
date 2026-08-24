"""
FastAPI Backend for SaruPol
Provides REST API endpoints for mobile and web clients
Updated with 82 CRI Reference Images
"""

from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File, Form
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
import io
import uuid

# Import RAG engine
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from step2_rag_engine import load_rag_chain, get_answer, get_answer_with_memory, translate_text, get_multi_llm_answer, translate_multi_llm_payload, find_relevant_images, get_language, is_sinhala, is_tamil, calculate_combined_reliability, refine_speech_transcription

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
    # Startup: Load RAG chain once
    global rag_chain, retriever
    logger.info("Loading RAG chain...")
    try:
        rag_chain, retriever = load_rag_chain()
        logger.info("RAG chain loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load RAG chain: {str(e)}")
        rag_chain = None
        retriever = None
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="SaruPol API",
    description="Backend API for SaruPol Coconut Advisory System with Multi-LLM Consensus Validation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (support /static, /api/static, and /api/v1/static prefixes)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/api/static", StaticFiles(directory=static_dir), name="api_static")
    app.mount("/api/v1/static", StaticFiles(directory=static_dir), name="api_v1_static")

# API Router with prefix
router = APIRouter(prefix="/api/v1")

# Root endpoints (without prefix) for compatibility
@app.get("/")
async def root():
    index_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
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
    language: Optional[str] = 'en'  # 'en', 'si', or 'ta'
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
    retrieval_confidence: float = 0.0
    combined_reliability: float = 0.0
    reliability_level: str = 'Moderate'
    context_used: Optional[str] = None
    session_id: Optional[str] = None
    model_used: Optional[str] = "openai/gpt-4o-mini"

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
    session_id: Optional[str] = None

class MultiLLMResponse(BaseModel):
    success: bool
    best_answer: str
    best_model: str
    reason: str
    consensus_score: int
    retrieval_confidence: float = 0.0
    combined_reliability: float = 0.0
    reliability_level: str = 'Moderate'
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
    session_id: Optional[str] = None


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
        
        # Detect input language using unified detection
        detected_lang = get_language(question)
        
        # Translate question to English if it is in Sinhala or Tamil
        rag_question = question
        if detected_lang in ('si', 'ta'):
            lang_name = 'Sinhala' if detected_lang == 'si' else 'Tamil'
            logger.info(f"{lang_name} question detected. Translating to English for RAG...")
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

        # Determine target response language
        target_lang = user_lang if user_lang in ("si", "ta") else (detected_lang if detected_lang in ("si", "ta") else "en")

        # Query the RAG engine with memory using English query & session_id
        result = await asyncio.to_thread(
            get_answer_with_memory,
            rag_question,
            session_id,
            rag_chain,
            retriever,
            user_context=user_context
        )
        
        answer = result["answer"]
        display_question = question

        # Translate answer into farmer-friendly Sinhala / Tamil using CRI domain rules
        if target_lang in ("si", "ta"):
            lang_name = 'Sinhala' if target_lang == 'si' else 'Tamil'
            logger.info(f"Translating answer to {lang_name} using CRI vocabulary engine...")
            try:
                answer = await asyncio.to_thread(translate_text, answer, target_lang)
                if detected_lang == 'en' and user_lang in ('si', 'ta'):
                    display_question = await asyncio.to_thread(translate_text, question, user_lang)
                logger.info(f"Answer successfully translated to {lang_name}.")
            except Exception as e:
                logger.error(f"Error translating answer to {lang_name}: {str(e)}")
        
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

        # Calculate combined reliability
        retrieval_conf = result.get("retrieval_confidence", 0.85)
        combined_rel = result.get("combined_reliability")
        rel_level = result.get("reliability_level")
        if combined_rel is None:
            combined_rel, rel_level = calculate_combined_reliability(retrieval_conf, 80.0)

        return AnswerResponse(
            success=True,
            question=display_question, # Return Sinhala translated or original question
            answer=answer,     # Return translated or English answer
            sources=sources,
            images=images,
            zone=zone,
            season=season,
            confidence=result.get("confidence"),
            retrieval_confidence=retrieval_conf,
            combined_reliability=combined_rel,
            reliability_level=rel_level,
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
        if target_lang not in ["en", "si", "ta"]:
            raise HTTPException(status_code=400, detail="Invalid target language. Must be 'en', 'si', or 'ta'")
            
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
        # Detect input language using unified detection
        detected_lang = get_language(question)

        # Pre-translate query to English if Sinhala or Tamil
        rag_question = question
        if detected_lang in ('si', 'ta'):
            lang_name = 'Sinhala' if detected_lang == 'si' else 'Tamil'
            logger.info(f"{lang_name} multi-LLM question detected. Translating to English for RAG...")
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
        session_id = request.session_id or str(uuid.uuid4())

        # Run multi-LLM validation (parallel execution inside) with conversational memory
        start_time = time.time()
        result = await asyncio.to_thread(
            get_multi_llm_answer, rag_question, retriever, user_context, session_id
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

        # Post-translate answers back to Sinhala or Tamil if requested
        if user_lang in ("si", "ta"):
            lang_name = 'Sinhala' if user_lang == 'si' else 'Tamil'
            logger.info(f"Translating all Multi-LLM response fields to {lang_name}...")
            payload_to_translate = {
                "best_answer": best_answer,
                "reason": reason,
                "llama_answer": llama_answer,
                "llama8b_answer": llama8b_answer,
                "gemma_answer": gemma_answer
            }
            try:
                translated_dict = await asyncio.to_thread(translate_multi_llm_payload, payload_to_translate, target_lang=user_lang)
                best_answer = translated_dict.get("best_answer", best_answer)
                reason = translated_dict.get("reason", reason)
                llama_answer = translated_dict.get("llama_answer", llama_answer)
                llama8b_answer = translated_dict.get("llama8b_answer", llama8b_answer)
                gemma_answer = translated_dict.get("gemma_answer", gemma_answer)
                logger.info(f"Multi-LLM response fields successfully translated to {lang_name}.")
            except Exception as e:
                logger.error(f"Error translating Multi-LLM payload to {lang_name}: {e}")

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

        # Calculate combined reliability
        retrieval_conf = result.get("retrieval_confidence", 0.85)
        combined_rel = result.get("combined_reliability")
        rel_level = result.get("reliability_level")
        if combined_rel is None:
            combined_rel, rel_level = calculate_combined_reliability(
                retrieval_confidence=retrieval_conf,
                consensus_score=result.get("consensus_score", 50)
            )

        return MultiLLMResponse(
            success=True,
            best_answer=best_answer,
            best_model=result["best_model"],
            reason=reason,
            consensus_score=result["consensus_score"],
            retrieval_confidence=retrieval_conf,
            combined_reliability=combined_rel,
            reliability_level=rel_level,
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
            latency_ms=latency_ms,
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Error in multi-LLM query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing multi-LLM query: {str(e)}")


@router.get("/tts", tags=["TTS"])
async def text_to_speech(text: str, lang: str = "en"):
    """
    Generate Text-to-Speech audio stream for a given text and language.
    - Sinhala ('si'): Uses Google Voice (gTTS) for natural, fast Sinhala speech.
    - Tamil ('ta'): Uses Edge TTS Neural (ta-LK-KumarNeural).
    - English ('en'): Uses Edge TTS Neural (en-US-AriaNeural).
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    import re
    import hashlib
    
    def clean_text_for_tts(raw_text: str) -> str:
        """Remove markdown formatting, bold/italics, headers and symbols for clean speech."""
        cleaned = raw_text
        cleaned = re.sub(r'[*_~`]', '', cleaned)
        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
        cleaned = re.sub(r'^#{1,6}\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^[\s]*[-•‣]\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*\d+[.)]\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^-{3,}$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'https?://\S+', '', cleaned)
        cleaned = re.sub(r'\n+', '. ', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        return cleaned.strip()

    cleaned_text = clean_text_for_tts(text)
    target_lang = "si" if lang.lower() == "si" else ("ta" if lang.lower() == "ta" else "en")
    
    # In-memory cache for sub-millisecond instant audio playback
    if not hasattr(app.state, "tts_cache"):
        app.state.tts_cache = {}
    
    cache_key = hashlib.md5(f"{target_lang}:{cleaned_text}".encode('utf-8')).hexdigest()
    if cache_key in app.state.tts_cache:
        cached_data = app.state.tts_cache[cache_key]
        return StreamingResponse(io.BytesIO(cached_data), media_type="audio/mpeg")

    logger.info(f"Generating TTS for lang={target_lang}, length={len(cleaned_text)}")

    # 1. Sinhala: Google Voice (gTTS) - Female voice, fast and natural
    if target_lang == "si":
        try:
            from gtts import gTTS
            def _generate_gtts():
                tts = gTTS(text=cleaned_text, lang="si", slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()

            audio_bytes = await asyncio.to_thread(_generate_gtts)
            app.state.tts_cache[cache_key] = audio_bytes
            return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
        except Exception as gtts_err:
            logger.warning(f"Google Voice TTS failed: {gtts_err}. Falling back to Edge TTS...")

    # 2. Tamil & English (and fallback): Edge TTS Neural
    try:
        import edge_tts
        voice = "ta-LK-KumarNeural" if target_lang == "ta" else ("en-US-AriaNeural" if target_lang == "en" else "si-LK-ThiliniNeural")
        rate = "-10%" if target_lang == "ta" else "+0%"
        
        async def audio_generator():
            communicate = edge_tts.Communicate(cleaned_text, voice, rate=rate)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                    
        return StreamingResponse(audio_generator(), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")


def _convert_audio_to_wav(audio_bytes: bytes) -> io.BytesIO:
    """Converts input audio bytes (e.g. M4A, MP3, WebM) to normalized 16kHz WAV format for speech recognition."""
    import imageio_ffmpeg
    import subprocess
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [
            ffmpeg_path,
            "-i", "pipe:0",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=80,lowpass=f=7500",
            "-f", "wav",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "pipe:1"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    wav_data, _ = process.communicate(input=audio_bytes)
    return io.BytesIO(wav_data)


@router.post("/transcribe", tags=["STT"])
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(default="auto")
):
    """
    High-accuracy Speech-to-Text transcription with domain refinement.
    Accepts multipart/form-data audio file (M4A, WAV, MP3, WebM).
    Supported languages: 'auto', 'en', 'si', 'ta'
    """
    import time as _time
    start = _time.time()

    audio_data = await audio.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    target_lang = language if language in ("si", "ta", "en") else "si"
    raw_text = ""
    engine_used = "google"

    # Stage 1: Try Native Google Speech Recognition (Optimized for Sinhala si-LK, Tamil ta-LK, English en-US)
    try:
        import speech_recognition as sr
        lang_map = {"si": "si-LK", "ta": "ta-LK", "en": "en-US", "auto": "si-LK"}
        google_lang = lang_map.get(language, "si-LK")

        def _recognize_google_sync():
            wav_io = _convert_audio_to_wav(audio_data)
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio_recorded = recognizer.record(source)
                return recognizer.recognize_google(audio_recorded, language=google_lang)

        raw_text = await asyncio.to_thread(_recognize_google_sync)
    except Exception as g_err:
        logger.info(f"Google STT fallback to Groq Whisper: {g_err}")
        engine_used = "groq-whisper"

    # Stage 2: Fallback to Groq Whisper Large V3 if needed
    if not raw_text or not raw_text.strip():
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                from groq import Groq as GroqClient
                client = GroqClient(api_key=groq_api_key)
                filename = audio.filename or "recording.m4a"
                audio_tuple = (filename, audio_data, audio.content_type or "audio/m4a")
                whisper_lang = "ta" if language == "ta" else ("en" if language == "en" else None)
                whisper_prompt = (
                    "පොල් වගාව, පොල් පැළ සඳහා පොහොර, රෝග පාලනය, කළු කුරුමිණියා, CRI Sri Lanka advisory."
                    if target_lang == "si"
                    else ("தென்னை பயிர்ச்செய்கை, உரம், பூச்சி கட்டுப்பாடு, CRI Sri Lanka advisory." if target_lang == "ta" else "Coconut farming in Sri Lanka, fertilization, pest control, CRI advisory.")
                )

                transcription = await asyncio.to_thread(
                    lambda: client.audio.transcriptions.create(
                        file=audio_tuple,
                        model="whisper-large-v3",
                        language=whisper_lang,
                        prompt=whisper_prompt,
                        temperature=0.0,
                        response_format="verbose_json",
                    )
                )
                raw_text = getattr(transcription, "text", "").strip()
            except Exception as w_err:
                logger.error(f"Groq Whisper error: {w_err}")

    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=500, detail="Could not recognize speech from audio. Please try speaking closer to the microphone.")

    # Stage 3: Agricultural Domain Refinement
    try:
        refined_text = await asyncio.to_thread(refine_speech_transcription, raw_text, target_lang)
        final_text = refined_text.strip() if refined_text and refined_text.strip() else raw_text
    except Exception as ref_err:
        logger.warning(f"Speech refinement fallback: {ref_err}")
        final_text = raw_text

    duration_ms = int((_time.time() - start) * 1000)
    logger.info(f"Transcription complete: engine={engine_used}, lang={target_lang}, chars={len(final_text)}, time={duration_ms}ms")

    return {
        "success": True,
        "transcribed_text": final_text,
        "detected_language": target_lang,
        "duration_ms": duration_ms,
    }



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
    port = int(os.getenv("API_PORT", 5002))
    reload_flag = os.getenv("RELOAD", "True").lower() == "true"
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(backend_dir, "../.."))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_flag,
        reload_dirs=[backend_dir, root_dir] if reload_flag else None,
        log_level="info"
    )
