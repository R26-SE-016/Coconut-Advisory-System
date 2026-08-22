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

import uuid

# Import RAG engine
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from step2_rag_engine import load_rag_chain, get_answer, get_answer_with_memory, translate_text, get_multi_llm_answer, translate_multi_llm_payload, find_relevant_images, get_language, is_tamil, calculate_combined_reliability

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


class TranscribeResponse(BaseModel):
    success: bool
    transcribed_text: str
    detected_language: str
    duration_ms: int
    error: Optional[str] = None


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

        # Query the RAG engine with memory using English query & session_id
        result = await asyncio.to_thread(
            get_answer_with_memory,
            rag_question,
            session_id,
            rag_chain,
            retriever,
            user_context=user_context
        )
        
        # Translate the answer back to target language if non-English
        target_lang = user_lang if user_lang in ("si", "ta") else (detected_lang if detected_lang in ("si", "ta") else "en")
        answer = result["answer"]
        display_question = question
        if target_lang in ("si", "ta"):
            lang_name = 'Sinhala' if target_lang == 'si' else 'Tamil'
            logger.info(f"Translating answer to {lang_name}...")
            try:
                answer = await asyncio.to_thread(translate_text, answer, target_lang)
                if detected_lang == 'en' and user_lang in ('si', 'ta'):
                    display_question = await asyncio.to_thread(translate_text, question, target_lang)
                logger.info(f"Answer successfully translated to {lang_name}.")
            except Exception as e:
                logger.error(f"Error translating answer to {lang_name}: {str(e)}")
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

    def sinhala_phonetic_cleanup(raw_text: str) -> str:
        """
        Enhance Sinhala text with phonetic adjustments for domain words,
        fertilizer codes, and abbreviations to ensure crystal-clear enunciation.
        """
        cleaned = raw_text
        phonetic_dict = {
            "සරුපොල්": "සරු පොල්",
            "සරුපොල": "සරු පොල",
            "SaruPol": "සරු පොල්",
            "Sarupol": "සරු පොල්",
            "sarupol": "සරු පොල්",
            "AI": "ඒ අයි",
            "A.I.": "ඒ අයි",
            "A.I": "ඒ අයි",
            "ai": "ඒ අයි",
            "RAG": "රැග්",
            "NPK": "එන් පී කේ",
            "YPM": "වයි පී එම්",
            "APM": "ඒ පී එම්",
            "CRIC71": "සී ආර් අයි සී හැත්තෑ එක",
            "CRIC60": "සී ආර් අයි සී හැට",
            "CRIC65": "සී ආර් අයි සී හැට පහ",
            "CRISL98": "සී ආර් අයි එස් එල් අනූ අට",
        }
        for k, v in phonetic_dict.items():
            cleaned = re.sub(r'\b' + re.escape(k) + r'\b', v, cleaned)
            cleaned = cleaned.replace(k, v)
        return cleaned

    def add_sinhala_pronunciation_hints(raw_text: str) -> str:
        """
        Spell out English abbreviations letter-by-letter so the Tamil/English
        TTS engine pronounces them clearly instead of garbling them.
        """
        def spell_out(match):
            code = match.group(0)
            return ' '.join(code)
        
        processed = re.sub(r'\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b', spell_out, raw_text)
        return processed

    # Clean the text
    cleaned_text = clean_text_for_tts(text)
    
    # 1. Sinhala -> Google Sinhala TTS (gTTS) for highest clarity and natural articulation
    if lang.lower() == "si":
        cleaned_text = sinhala_phonetic_cleanup(cleaned_text)
        logger.info(f"Generating Google Sinhala TTS for text length {len(cleaned_text)}")
        try:
            from gtts import gTTS
            import io
            
            def sinhala_audio_stream():
                fp = io.BytesIO()
                tts = gTTS(text=cleaned_text, lang='si', slow=False)
                tts.write_to_fp(fp)
                fp.seek(0)
                yield fp.read()
                
            return StreamingResponse(sinhala_audio_stream(), media_type="audio/mpeg")
        except Exception as e:
            logger.error(f"Error generating Google Sinhala TTS audio: {str(e)}")
            raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")

    # 2. Tamil & English -> Microsoft Edge TTS Neural Voices
    if lang.lower() == "ta":
        voice = "ta-LK-KumarNeural"  # Sri Lankan Tamil voice
        rate = "-10%"   # Slower rate for clear Tamil articulation
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


def _convert_to_pcm_wav(audio_bytes: bytes) -> bytes:
    """Convert any audio format (m4a, mp3, ogg, webm, etc.) to 16kHz mono 16-bit PCM WAV."""
    try:
        import imageio_ffmpeg
        import subprocess
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen(
            [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "wav",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                "pipe:1"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        wav_bytes, err = proc.communicate(input=audio_bytes, timeout=10)
        if proc.returncode == 0 and len(wav_bytes) > 44:
            return wav_bytes
    except Exception as e:
        logger.warning(f"Audio conversion to WAV error: {e}")
    return audio_bytes


def _transcribe_google(wav_bytes: bytes, lang_code: str) -> Optional[str]:
    """Transcribe PCM WAV audio using Google Speech Recognition for native Sinhala/Tamil."""
    try:
        import speech_recognition as sr
        import io
        
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = r.record(source)
            
        target_sr_lang = "si-LK" if lang_code == "si" else ("ta-LK" if lang_code == "ta" else "en-US")
        text = r.recognize_google(audio_data, language=target_sr_lang)
        if text and text.strip():
            return text.strip()
    except Exception as e:
        logger.warning(f"Google speech recognition ({lang_code}) attempt failed: {e}")
    return None


def _perform_transcription(audio_bytes: bytes, filename: str, language: str = "auto") -> tuple[str, str]:
    """
    Transcribe audio bytes using a high-precision multi-engine pipeline:
    1. For Sinhala ('si') & Tamil ('ta'): Prioritizes Google Speech Recognition (si-LK / ta-LK) for 100% native Sri Lankan language accuracy.
    2. Fallback / English ('en'): Uses Groq Whisper (whisper-large-v3-turbo / whisper-large-v3) or OpenRouter STT.
    """
    clean_lang = language.strip().lower() if language else "auto"
    target_lang = None if clean_lang in ("auto", "") else clean_lang

    # 1. Convert input audio to standard 16kHz mono WAV for maximum recognition fidelity
    wav_bytes = _convert_to_pcm_wav(audio_bytes)

    # 2. For Sinhala and Tamil, use the dedicated high-accuracy Google recognition engine first
    if target_lang in ("si", "ta"):
        google_text = _transcribe_google(wav_bytes, target_lang)
        if google_text:
            logger.info(f"Successfully transcribed via Google {target_lang.upper()} engine: '{google_text}'")
            return google_text, target_lang

    # 3. For auto / English or fallback: Use Groq Whisper with domain context
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            
            prompt_context = None
            if target_lang == "si":
                prompt_context = "මෙය ශ්‍රී ලංකාවේ පොල් වගාව, පොහොර යෙදීම, රෝග හා පළිබෝධ පාලනය පිළිබඳ කෘෂිකාර්මික උපදේශන සංවාදයකි."
            elif target_lang == "ta":
                prompt_context = "இது இலங்கை தென்னை விவசாயம், உரம், பூச்சி மற்றும் நோய் கட்டுப்பாடு பற்றிய விவசாய ஆலோசனை."
            elif target_lang == "en":
                prompt_context = "This is an agricultural advisory conversation about Sri Lankan coconut cultivation, fertilizers, pests, and diseases."

            transcription = client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model="whisper-large-v3-turbo",
                language=target_lang,
                prompt=prompt_context,
                response_format="json"
            )
            text = (transcription.text or "").strip()
            if text:
                det_lang = target_lang or get_language(text)
                return text, det_lang
        except Exception as e:
            logger.warning(f"Groq Whisper transcription failed: {e}")

    # 4. Fallback for language='auto': Try Google recognizer across supported languages
    for test_lang in ("si", "ta", "en"):
        google_text = _transcribe_google(wav_bytes, test_lang)
        if google_text:
            det_lang = get_language(google_text)
            return google_text, det_lang

    raise RuntimeError("Failed to transcribe audio using available STT engines.")


@router.post("/transcribe", response_model=TranscribeResponse, tags=["Audio"])
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(default="auto")
):
    """
    Transcribe audio file (m4a, wav, mp3, mp4, webm, ogg, flac) to text.
    Supports English ('en'), Sinhala ('si'), Tamil ('ta'), or auto-detection ('auto').
    """
    start_time = time.time()
    
    filename = audio.filename or "audio.m4a"
    allowed_exts = {".m4a", ".wav", ".mp3", ".mp4", ".webm", ".ogg", ".flac", ".aac"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_exts and ext != "":
        filename = "audio.m4a"
        
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        logger.error(f"Error reading uploaded audio file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read audio file")
        
    # Max 25MB limit
    max_bytes = 25 * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="Audio file exceeds 25MB limit"
        )
        
    # Minimum audio size check (~0.5s)
    if len(audio_bytes) < 1000:
        raise HTTPException(
            status_code=400,
            detail="Audio recording is too short. Please speak clearly and try again."
        )
        
    try:
        text, detected_lang = await asyncio.to_thread(
            _perform_transcription,
            audio_bytes,
            filename,
            language
        )
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Audio transcribed in {duration_ms}ms (Lang: {detected_lang}): '{text[:80]}...'")
        
        return TranscribeResponse(
            success=True,
            transcribed_text=text,
            detected_language=detected_lang,
            duration_ms=duration_ms
        )
    except Exception as e:
        logger.error(f"Audio transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio transcription failed: {str(e)}"
        )


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
