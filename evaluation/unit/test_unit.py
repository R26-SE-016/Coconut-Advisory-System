import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from step2_rag_engine import (
    is_sinhala,
    is_tamil,
    get_language,
    detect_topic,
    detect_question_topics,
    calculate_combined_reliability,
    _clean_llm_translation_output,
    _clean_tamil_translation_output,
    _sanitize_sinhala_advisory,
    _sanitize_tamil_advisory,
    _is_translation_valid,
    get_zone,
    get_season,
    classify_consensus,
    _compute_similarity
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Language Detection Utils
@pytest.mark.parametrize("text,expected", [
    ('පොල් ගස්වලට පොහොර', True),
    ('தேங்காய்', False),
    ('English text', False),
    ('', False),
    ('පොල් and English', True),
    ('12345', False),
])
def test_is_sinhala(text, expected):
    assert is_sinhala(text) == expected

@pytest.mark.parametrize("text,expected", [
    ('தேங்காய் மரங்களுக்கு உரம்', True),
    ('පොල්', False),
    ('English text', False),
    ('', False),
    ('தமிழ் and English', True),
    ('12345', False),
])
def test_is_tamil(text, expected):
    assert is_tamil(text) == expected

@pytest.mark.parametrize("text,expected", [
    ('පොල් ගස්වලට පොහොර', 'si'),
    ('தேங்காய் மரங்களுக்கு உரம்', 'ta'),
    ('How to fertilize coconut palms?', 'en'),
    ('පොල් fertilizer application', 'si'),
    ('தேங்காய் and fertilizer', 'ta'),
    ('පොල් and தேங்காய்', 'si'), # Sinhala priority
    ('', 'en'),
    ('   \n\t  ', 'en'),
    ('250 300 500', 'en'),
    ('🥥🌴', 'en'),
])
def test_get_language(text, expected):
    assert get_language(text) == expected

# 2. Topic Detection
@pytest.mark.parametrize("text,expected", [
    ('mother palm selection', 'mother_palm'),
    ('coconut nursery management', 'nursery'),
    ('applying urea and mop', 'fertilizer'),
    ('red palm weevil attack', 'pest_disease'),
    ('spacing for new planting', 'planting'),
    ('CRIC60 hybrid', 'variety'),
    ('general agriculture', 'general'),
    ('', 'general'),
])
def test_detect_topic(text, expected):
    assert detect_topic(text) == expected

@pytest.mark.parametrize("question,expected", [
    ('how to select mother palm and nursery?', ['mother_palm', 'nursery']),
    ('urea dosage for beetle attack', ['fertilizer', 'pest_disease']),
    ('general question about weather', ['general']),
    ('', ['general']),
])
def test_detect_question_topics(question, expected):
    assert detect_question_topics(question) == expected

# 3. Reliability and Consensus
@pytest.mark.parametrize("retrieval_conf,consensus,expected_combined,expected_level", [
    (0.9, 90, 90.0, 'High'),
    (1.0, 100, 100.0, 'High'),
    (0.5, 50, 50.0, 'Low'),
    (0.7, 70, 70.0, 'Moderate'),
    (0.9, 0, 45.0, 'Low'),
    (0.0, 90, 45.0, 'Low'),
    (-0.5, 120, 50.0, 'Low'), # Out of bounds check
    (1.5, -10, 50.0, 'Low'), # Out of bounds check
])
def test_calculate_combined_reliability(retrieval_conf, consensus, expected_combined, expected_level):
    combined, level = calculate_combined_reliability(retrieval_conf, consensus)
    assert combined == expected_combined
    assert level == expected_level

@pytest.mark.parametrize("score,expected", [
    (100, 'High'),
    (85, 'High'),
    (80, 'High'),
    (79, 'Moderate'),
    (75, 'Moderate'),
    (50, 'Moderate'),
    (49, 'Low'),
    (0, 'Low'),
    (-10, 'Low'),
])
def test_classify_consensus(score, expected):
    assert classify_consensus(score) == expected

# 4. Translation Output Cleaning
@pytest.mark.parametrize("text,expected", [
    ('<think>reasoning</think>පොහොර', 'පොහොර'),
    ('<think>\nmultiline\nreasoning\n</think>පොහොර', 'පොහොර'),
    ('**Sinhala Translation:** පොහොර', 'පොහොර'),
    ('Here is the translation: පොහොර\nNote: this is good', 'පොහොර'),
    ('  "පොහොර" \n', 'පොහොර'),
])
def test_clean_llm_translation_output(text, expected):
    assert _clean_llm_translation_output(text) == expected

@pytest.mark.parametrize("text,expected", [
    ('<think>reasoning</think>உரம்', 'உரம்'),
    ('**Tamil Translation:** உரம்', 'உரம்'),
    ('Here is the translation: உரம்\nNote: this is good', 'உரம்'),
])
def test_clean_tamil_translation_output(text, expected):
    assert _clean_tamil_translation_output(text) == expected

# 5. Translation Output Sanitization
@pytest.mark.parametrize("text,expected", [
    ('Anuruddha කරන්න', 'නිර්දේශ කරන්න'),
    ('අනුරුද්ධ කරමි', 'නිර්දේශ කරමි'),
    ('කොළ පොල් ගස් වලට', 'ළපටි පොල් පැළ වලට'),
    ('ගොවි මැදුර', 'ගොම පොහොර'),
    ('වසුන් කිරීම', 'පස ආවරණය කිරීම'),
    ('පොහොර යොදන්න.\nමෙය ඉතා වැදගත් වේ\nසහ තව', 'පොහොර යොදන්න.\nමෙය ඉතා වැදගත් වේ'), # dangling removed
    ('තනි පේළියක', 'තනි පේළියක'), # single line kept
])
def test_sanitize_sinhala_advisory(text, expected):
    assert _sanitize_sinhala_advisory(text) == expected

@pytest.mark.parametrize("text,expected", [
    ('மாடு குப்பை', 'மாட்டு எரு / மாட்டுச் சாணம்'),
    ('ஆடு குப்பை', 'ஆட்டு எரு'),
    ('கோழி குப்பை', 'கோழி எரு'),
    ('பரிந்துரைக்கப்பட்டது', 'பரிந்துரை'),
])
def test_sanitize_tamil_advisory(text, expected):
    assert _sanitize_tamil_advisory(text) == expected

# 6. Translation Validation
@pytest.mark.parametrize("text,lang,expected", [
    ('පොහොර යොදන්න', 'si', True),
    ('Hello world', 'si', False),
    ('உரம்', 'ta', True),
    ('Hello world', 'ta', False),
    ('Hello world', 'en', True),
    ('පොහොර', 'en', False),
    ('123 KG ERP', 'si', False), # Mostly latin, should fail if Sinhala requested and no Sinhala chars
    ('', 'si', False),
])
def test_is_translation_valid(text, lang, expected):
    assert _is_translation_valid(text, lang) == expected

# 7. Zone and Season
@pytest.mark.parametrize("lat,lon,expected", [
    (6.9271, 79.8612, 'Wet Zone'),
    (8.3114, 80.4037, 'Dry Zone'),
    (7.2906, 80.6337, 'Intermediate Zone'),
    (0.0, 0.0, 'Wet Zone'),
    (8.0001, 80.0, 'Dry Zone'),
    (7.0, 81.0, 'Wet Zone'),
    (-5.0, 10.0, 'Wet Zone'),
])
def test_get_zone(lat, lon, expected):
    assert get_zone(lat, lon) == expected

@pytest.mark.parametrize("month,expected", [
    (5, 'Yala'), (7, 'Yala'), (9, 'Yala'),
    (10, 'Maha'), (12, 'Maha'), (1, 'Maha'), (4, 'Maha'),
    (0, 'Maha'), (13, 'Maha'),
])
def test_get_season(month, expected):
    assert get_season(month) == expected

# 8. Text Chunking
@pytest.mark.parametrize("text_len,expected_min_chunks", [
    (2000, 2),
    (100, 1),
])
def test_chunking_limits(text_len, expected_min_chunks):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text = 'A' * text_len
    chunks = splitter.split_text(text)
    for chunk in chunks:
        assert len(chunk) <= 500
    assert len(chunks) >= expected_min_chunks

# 9. Early Exit Similarity
def test_compute_similarity_identical():
    text = 'Apply 250g Urea per planting hole'
    assert _compute_similarity(text, text) >= 0.99

def test_compute_similarity_different():
    assert _compute_similarity('Apply fertilizer', 'The weather is sunny') < 0.5

def test_compute_similarity_similar():
    score = _compute_similarity(
        'Apply 250g Urea per planting hole in wet zone',
        'Use 250 grams of Urea for each planting hole'
    )
    assert score >= 0.75

def test_compute_similarity_case_insensitive():
    assert _compute_similarity('APPLY UREA', 'apply urea') >= 0.99

def test_compute_similarity_empty():
    assert _compute_similarity('', '') == 0.0
