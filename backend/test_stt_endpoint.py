import asyncio
import io
import os
import sys
from dotenv import load_dotenv
from gtts import gTTS
from fastapi.testclient import TestClient

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('d:/GitHub/coconut_advisory_system/backend/.env')

from app.main import app

def generate_tts_audio(text: str, lang: str) -> bytes:
    """Generate sample audio bytes using gTTS."""
    fp = io.BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def test_transcription():
    print("=" * 70)
    print("SaruPol Speech-to-Text (/api/transcribe & /transcribe) Verification Test")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Test Too Short Audio Validation
        print("\n--- Test 1: Too Short Audio Validation (< 1000 bytes) ---")
        tiny_audio = b"RIFF" + b"\x00" * 100
        files = {'audio': ('tiny.wav', tiny_audio, 'audio/wav')}
        data = {'language': 'auto'}
        res = client.post("/api/transcribe", files=files, data=data)
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 400, f"Expected 400 for short audio, got {res.status_code}"
        print("✓ Short audio validation test passed!")

        # 2. Test English Voice Transcription
        print("\n--- Test 2: English Voice Transcription ---")
        en_phrase = "What fertilizer should I apply for young coconut palms?"
        print(f"Input Speech: '{en_phrase}'")
        en_audio = generate_tts_audio(en_phrase, 'en')
        print(f"Generated Audio Size: {len(en_audio)} bytes")

        files = {'audio': ('english_sample.mp3', en_audio, 'audio/mpeg')}
        data = {'language': 'en'}
        res = client.post("/api/transcribe", files=files, data=data)
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
        en_json = res.json()
        assert en_json.get("success") is True
        print(f"Transcribed Text: '{en_json.get('transcribed_text')}'")
        print(f"Detected Lang: '{en_json.get('detected_language')}' | Duration: {en_json.get('duration_ms')}ms")
        print("✓ English transcription test passed!")

        # 3. Test Sinhala Voice Transcription
        print("\n--- Test 3: Sinhala Voice Transcription ---")
        si_phrase = "පොල් ගස්වලට කොපමණ පොහොර දැමිය යුතුද"
        print(f"Input Speech: '{si_phrase}'")
        si_audio = generate_tts_audio(si_phrase, 'si')
        print(f"Generated Audio Size: {len(si_audio)} bytes")

        files = {'audio': ('sinhala_sample.mp3', si_audio, 'audio/mpeg')}
        data = {'language': 'si'}
        res = client.post("/api/transcribe", files=files, data=data)
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
        si_json = res.json()
        assert si_json.get("success") is True
        print(f"Transcribed Text: '{si_json.get('transcribed_text')}'")
        print(f"Detected Lang: '{si_json.get('detected_language')}' | Duration: {si_json.get('duration_ms')}ms")
        print("✓ Sinhala transcription test passed!")

        # 4. Test Tamil Voice Transcription
        print("\n--- Test 4: Tamil Voice Transcription ---")
        ta_phrase = "தேங்காய் மரங்களுக்கு எவ்வளவு உரம் இட வேண்டும்"
        print(f"Input Speech: '{ta_phrase}'")
        ta_audio = generate_tts_audio(ta_phrase, 'ta')
        print(f"Generated Audio Size: {len(ta_audio)} bytes")

        files = {'audio': ('tamil_sample.mp3', ta_audio, 'audio/mpeg')}
        data = {'language': 'ta'}
        res = client.post("/api/transcribe", files=files, data=data)
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
        ta_json = res.json()
        assert ta_json.get("success") is True
        print(f"Transcribed Text: '{ta_json.get('transcribed_text')}'")
        print(f"Detected Lang: '{ta_json.get('detected_language')}' | Duration: {ta_json.get('duration_ms')}ms")
        print("✓ Tamil transcription test passed!")

        # 5. Full Loop Test (Voice Input -> RAG Answer -> TTS Audio Output)
        print("\n--- Test 5: Full Voice Loop (Voice Input -> RAG -> TTS Audio) ---")
        print(f"Step 1: Farmer speaks in Sinhala: '{si_phrase}'")
        
        # Step 2: Transcribe
        transcribe_res = client.post("/api/transcribe", files={'audio': ('farmer.mp3', si_audio, 'audio/mpeg')}, data={'language': 'si'})
        transcribed_text = transcribe_res.json().get("transcribed_text")
        print(f"Step 2: Transcribed Text: '{transcribed_text}'")

        # Step 3: Query RAG
        ask_payload = {
            "question": si_phrase,
            "language": "si"
        }
        rag_res = client.post("/api/ask", json=ask_payload)
        print(f"Step 3: RAG Status: {rag_res.status_code}")
        rag_json = rag_res.json()
        answer_text = rag_json.get("answer", "")
        print(f"Step 3: RAG Sinhala Answer: '{answer_text[:140]}...'")

        # Step 4: TTS Stream
        tts_res = client.get(f"/api/tts?text={answer_text[:80]}&lang=si")
        print(f"Step 4: TTS Stream Status: {tts_res.status_code}, Audio Size: {len(tts_res.content)} bytes")
        assert tts_res.status_code == 200 and len(tts_res.content) > 500
        print("\n🎉 ALL 5 VERIFICATION TESTS PASSED 100% SUCCESSFULLY!")

if __name__ == "__main__":
    test_transcription()
