import pytest
from agent_functions.core_functions.text_ops import *

def test_tokenization():
    text = "Hello, world! This is a test."
    tokens = tokenize(text)
    assert len(tokens) == 6
    assert tokens == ['Hello', 'world', 'This', 'is', 'a', 'test']

def test_text_cleaning():
    text = "Hello,   world! \n\tThis is a TEST."
    cleaned = clean_text(text)
    assert cleaned == "hello world this is a test"

def test_stopword_removal():
    text = "This is a test sentence with some stopwords"
    filtered = remove_stopwords(text)
    assert "is" not in filtered
    assert "a" not in filtered
    assert "test" in filtered
    assert "sentence" in filtered

def test_sentiment_analysis():
    positive_text = "I love this amazing product!"
    negative_text = "This is terrible and disappointing."
    neutral_text = "The sky is blue."
    
    assert sentiment_analysis(positive_text)['polarity'] > 0
    assert sentiment_analysis(negative_text)['polarity'] < 0
    assert abs(sentiment_analysis(neutral_text)['polarity']) < 0.2

def test_pos_tagging():
    text = "The quick brown fox jumps"
    tags = pos_tag(text)
    assert len(tags) == 5
    assert isinstance(tags, list)
    assert all(len(tag) == 2 for tag in tags)  # Each tag should be (word, pos) tuple

def test_lemmatization():
    words = ["running", "flies", "better", "studies"]
    lemmas = lemmatize(words)
    assert "run" in lemmas
    assert "fly" in lemmas
    assert "good" in lemmas
    assert "study" in lemmas

def test_input_validation():
    with pytest.raises(ValueError):
        tokenize("")
    with pytest.raises(TypeError):
        clean_text(None)
    with pytest.raises(ValueError):
        pos_tag("")
