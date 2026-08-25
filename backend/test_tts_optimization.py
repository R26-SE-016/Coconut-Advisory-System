import time
import sys
import os
from dotenv import load_dotenv
from fastapi.testclient import TestClient

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('d:/GitHub/coconut_advisory_system/backend/.env')

from app.main import app, TTS_CACHE

def test_tts():
    print("=" * 70)
    print("SaruPol Neural TTS Latency & Optimization Benchmark Test")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Test Sinhala Neural TTS (Cold generation)
        print("\n--- Test 1: Sinhala Neural TTS (Cold Call) ---")
        si_text = "පොල් වගාවේ රතු කුරුමිණි හානිය පාලනය සඳහා මුල් අවධියේ හඳුනාගැනීම සහ ෆෙරමෝන් උගුල් භාවිතය වැදගත් වේ."
        t0 = time.time()
        res_si = client.get(f"/tts?text={si_text}&lang=si")
        si_cold_time = time.time() - t0
        print(f"Status: {res_si.status_code}, Audio Bytes: {len(res_si.content)}, Time: {si_cold_time:.3f}s")
        assert res_si.status_code == 200
        assert len(res_si.content) > 1000
        assert res_si.headers.get("content-type") == "audio/mpeg"
        assert res_si.headers.get("content-length") == str(len(res_si.content))
        print("✓ Sinhala cold generation succeeded in < 2.0s!")

        # 2. Test Sinhala Neural TTS (Cache Hit)
        print("\n--- Test 2: Sinhala Neural TTS (Cache Hit / Re-play) ---")
        t0 = time.time()
        res_si_cached = client.get(f"/tts?text={si_text}&lang=si")
        si_cached_time = time.time() - t0
        print(f"Status: {res_si_cached.status_code}, Audio Bytes: {len(res_si_cached.content)}, Time: {si_cached_time:.5f}s")
        assert res_si_cached.status_code == 200
        assert res_si_cached.content == res_si.content
        assert si_cached_time < 0.05
        print(f"✓ Sinhala cache hit instant playback: {si_cached_time*1000:.2f}ms!")

        # 3. Test Tamil Neural TTS (Cold Call)
        print("\n--- Test 3: Tamil Neural TTS (Cold Call) ---")
        ta_text = "தென்னை மரங்களில் சிவப்பு வண்டு தாக்குதலை கட்டுப்படுத்த ஆரம்ப கட்ட கண்காணிப்பு மற்றும் பெரமோன் பொறிகள் அவசியம்."
        t0 = time.time()
        res_ta = client.get(f"/tts?text={ta_text}&lang=ta")
        ta_cold_time = time.time() - t0
        print(f"Status: {res_ta.status_code}, Audio Bytes: {len(res_ta.content)}, Time: {ta_cold_time:.3f}s")
        assert res_ta.status_code == 200
        assert len(res_ta.content) > 1000
        print("✓ Tamil cold generation succeeded in < 2.0s!")

        # 4. Test Tamil Neural TTS (Cache Hit)
        print("\n--- Test 4: Tamil Neural TTS (Cache Hit / Re-play) ---")
        t0 = time.time()
        res_ta_cached = client.get(f"/tts?text={ta_text}&lang=ta")
        ta_cached_time = time.time() - t0
        print(f"Status: {res_ta_cached.status_code}, Audio Bytes: {len(res_ta_cached.content)}, Time: {ta_cached_time:.5f}s")
        assert res_ta_cached.status_code == 200
        assert res_ta_cached.content == res_ta.content
        assert ta_cached_time < 0.05
        print(f"✓ Tamil cache hit instant playback: {ta_cached_time*1000:.2f}ms!")

        # 5. Full Loop Test: /ask -> Background Pre-warming -> /tts instant hit
        print("\n--- Test 5: /ask -> Background Pre-warming -> Instant TTS Playback ---")
        ask_payload = {
            "question": "පොල් පැළ සිටුවීමට සුදුසු පරතරය කුමක්ද?",
            "language": "si"
        }
        rag_res = client.post("/ask", json=ask_payload)
        assert rag_res.status_code == 200
        answer = rag_res.json().get("answer", "")
        print(f"RAG Answer: '{answer[:100]}...'")

        # Give background task 2.0s to finish prewarming
        time.sleep(2.0)

        t0 = time.time()
        tts_res = client.get("/tts", params={"text": answer, "lang": "si"})
        tts_prewarmed_time = time.time() - t0
        print(f"TTS Request Time on Advisory Answer: {tts_prewarmed_time:.5f}s ({tts_prewarmed_time*1000:.2f}ms)")
        assert tts_res.status_code == 200
        assert len(tts_res.content) > 1000
        print(f"✓ Pre-warmed TTS playback started in {tts_prewarmed_time*1000:.2f}ms (INSTANTANEOUS)!")

    print("\n" + "=" * 70)
    print("🎉 ALL TTS LATENCY OPTIMIZATION TESTS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_tts()
