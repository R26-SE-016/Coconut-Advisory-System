import pytest
import sys
import os

# Add parent directory to path so step2_rag_engine can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from step2_rag_engine import (
    load_rag_chain,
    translate_text,
    get_language,
    find_relevant_images
)

# Attempt to import get_multi_llm_answer, which might be in step2_rag_engine or another module
try:
    from step2_rag_engine import get_multi_llm_answer
except ImportError:
    # If not there, let's hope it's somewhere or maybe we can mock it
    pass

def test_retrieval_integration():
    # Test 1: Retriever returns correct number of chunks
    _, retriever = load_rag_chain()
    docs = retriever.invoke('What fertilizer for coconut?')
    assert len(docs) == 4  # k=4

    # Test 2: Retrieved chunks have required metadata
    for doc in docs:
        assert 'source' in doc.metadata
        assert 'topic' in doc.metadata
        assert len(doc.page_content) > 0

    # Test 3: English query returns English chunks primarily
    english_docs = retriever.invoke('termite control coconut nursery')
    english_count = sum(1 for d in english_docs
                       if d.metadata.get('source', '').endswith('English.pdf'))
    assert english_count >= 2

    # Test 4: Hybrid retrieval — topic filter + semantic
    mother_palm_docs = retriever.invoke('How to select good mother palm?')
    assert len(mother_palm_docs) > 0

def test_translation_integration():
    # Test 1: Sinhala to English translation
    sinhala_input = 'පොල් ගස්වලට කොපමණ පොහොර දැමිය යුතුද?'
    english_output = translate_text(sinhala_input, 'en')
    assert len(english_output) > 0
    assert any(word in english_output.lower()
               for word in ['fertilizer', 'coconut', 'apply', 'how much'])

    # Test 2: English to Sinhala translation
    english_input = 'Apply 250g Urea per planting hole'
    sinhala_output = translate_text(english_input, 'si')
    assert any('\u0D80' <= c <= '\u0DFF' for c in sinhala_output)

    # Test 3: Tamil translation
    english_input = 'Apply fertilizer to coconut palms'
    tamil_output = translate_text(english_input, 'ta')
    assert any('\u0B80' <= c <= '\u0BFF' for c in tamil_output)

    # Test 4: Language detection triggers correct translation
    language = get_language(sinhala_input)
    assert language == 'si'

def test_multi_llm_integration():
    # Test 1: Returns required fields
    chain, retriever = load_rag_chain()
    result = get_multi_llm_answer(
        'What fertilizer for young coconut palms?',
        retriever
    )
    assert 'best_answer' in result
    assert 'consensus_score' in result
    assert 'best_model' in result
    assert 'early_exit' in result
    assert 'llama_answer' in result
    assert 'gpt4omini_answer' in result
    assert 'gemma_answer' in result

    # Test 2: Consensus score in valid range
    assert 0 <= result['consensus_score'] <= 100

    # Test 3: Best model is one of three candidates
    assert result['best_model'] in ['llama', 'gpt4omini', 'gemma']

    # Test 4: Best answer is not empty
    assert len(result['best_answer']) > 50

    # Test 5: Early exit is boolean
    assert isinstance(result['early_exit'], bool)

def test_image_retrieval_integration():
    # Test 1: Returns list
    images = find_relevant_images('termite control coconut nursery')
    assert isinstance(images, list)

    # Test 2: Images have required fields
    if len(images) > 0:
        for img in images:
            assert 'url' in img
            assert 'caption' in img
            assert 'source' in img

    # Test 3: Maximum 2 images returned
    assert len(images) <= 2

    # Test 4: Pest question returns pest image
    pest_images = find_relevant_images('black beetle coconut damage')
    assert len(pest_images) > 0
