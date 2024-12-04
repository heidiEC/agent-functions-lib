import pytest
from agent_functions.core_functions.cognitive_ops import *

def test_entity_extraction():
    text = "Apple Inc. is headquartered in Cupertino, California and Tim Cook is its CEO."
    entities = extract_entities(text)
    
    assert any(e['text'] == 'Apple Inc.' and e['type'] == 'ORG' for e in entities)
    assert any(e['text'] == 'Cupertino' and e['type'] == 'GPE' for e in entities)
    assert any(e['text'] == 'California' and e['type'] == 'GPE' for e in entities)
    assert any(e['text'] == 'Tim Cook' and e['type'] == 'PERSON' for e in entities)

def test_text_summarization():
    long_text = """
    Natural language processing (NLP) is a field of artificial intelligence 
    that focuses on the interaction between computers and human language. 
    It involves the ability of computers to understand, interpret, and 
    generate human language in a way that is both meaningful and useful. 
    NLP combines computational linguistics, machine learning, and deep 
    learning to process and analyze large amounts of natural language data.
    """
    summary = summarize_text(long_text, max_length=50)
    
    assert len(summary) <= 50
    assert isinstance(summary, str)
    assert len(summary.split()) >= 3  # Ensure it's a meaningful summary

def test_question_answering():
    context = """
    The Python programming language was created by Guido van Rossum 
    and was first released in 1991. Python is known for its simple 
    syntax and readability.
    """
    question = "Who created Python?"
    
    answer = answer_question(context, question)
    assert "Guido van Rossum" in answer
    
    question = "When was Python first released?"
    answer = answer_question(context, question)
    assert "1991" in answer

def test_zero_shot_classification():
    text = "This movie was absolutely fantastic, I loved every minute of it!"
    labels = ["positive", "negative", "neutral"]
    
    result = zero_shot_classify(text, labels)
    assert isinstance(result, dict)
    assert "positive" in result
    assert result["positive"] > result["negative"]
    assert result["positive"] > result["neutral"]

def test_semantic_similarity():
    text1 = "The cat sat on the mat"
    text2 = "A feline was resting on the rug"
    text3 = "The weather is nice today"
    
    sim_score1 = semantic_similarity(text1, text2)
    sim_score2 = semantic_similarity(text1, text3)
    
    assert sim_score1 > sim_score2  # Similar sentences should have higher score
    assert 0 <= sim_score1 <= 1  # Scores should be normalized
    assert 0 <= sim_score2 <= 1

def test_input_validation():
    with pytest.raises(ValueError):
        extract_entities("")
    with pytest.raises(ValueError):
        summarize_text("", max_length=50)
    with pytest.raises(ValueError):
        answer_question("", "Who?")
    with pytest.raises(ValueError):
        zero_shot_classify("text", [])
