"""Cognitive operations for advanced text understanding and reasoning."""

from typing import List, Dict, Optional, Union
from ..core import AgentFunction
from transformers import pipeline
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load language models
nlp = spacy.load("en_core_web_sm")

# Initialize transformers
summarizer = pipeline("summarization")
qa_pipeline = pipeline("question-answering")
zero_shot = pipeline("zero-shot-classification")

@AgentFunction(
    category="cognitive.comprehension",
    description="Extract main entities and their relationships from text",
    agent_triggers=["extract_entities", "identify_entities", "find_relationships"],
    examples=[
        {
            "inputs": {
                "text": "Apple CEO Tim Cook announced new iPhone features in California."
            },
            "output": {
                "entities": {
                    "ORG": ["Apple"],
                    "PERSON": ["Tim Cook"],
                    "GPE": ["California"],
                    "PRODUCT": ["iPhone"]
                },
                "relationships": [
                    {
                        "subject": "Tim Cook",
                        "relation": "is CEO of",
                        "object": "Apple"
                    }
                ]
            }
        }
    ]
)
def extract_entities_and_relations(text: str) -> Dict[str, Union[Dict, List]]:
    """Extract entities and their relationships from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with entities and relationships
    """
    doc = nlp(text)
    
    # Extract entities
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    # Extract relationships
    relationships = []
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass'):
            subject = token.text
            verb = token.head.text
            obj = None
            for child in token.head.children:
                if child.dep_ in ('dobj', 'pobj'):
                    obj = child.text
                    break
            if obj:
                relationships.append({
                    "subject": subject,
                    "relation": verb,
                    "object": obj
                })
    
    return {
        "entities": entities,
        "relationships": relationships
    }

@AgentFunction(
    category="cognitive.summarization",
    description="Generate a concise summary of text",
    agent_triggers=["summarize_text", "generate_summary", "create_abstract"],
    examples=[
        {
            "inputs": {
                "text": "The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully in the sun. The fox was practicing its jumping skills.",
                "max_length": 30
            },
            "output": "A fox jumps over a sleeping dog while practicing its skills."
        }
    ]
)
def summarize(text: str, max_length: int = 130) -> str:
    """Generate a concise summary of text.
    
    Args:
        text: Input text
        max_length: Maximum length of summary
        
    Returns:
        Generated summary
    """
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

@AgentFunction(
    category="cognitive.qa",
    description="Answer questions based on provided context",
    agent_triggers=["answer_question", "extract_answer", "find_information"],
    examples=[
        {
            "inputs": {
                "context": "The Eiffel Tower was completed in 1889. It stands 324 meters tall.",
                "question": "How tall is the Eiffel Tower?"
            },
            "output": {
                "answer": "324 meters",
                "confidence": 0.95
            }
        }
    ]
)
def answer_question(context: str, question: str) -> Dict[str, Union[str, float]]:
    """Answer a question based on provided context.
    
    Args:
        context: Text context for answering
        question: Question to answer
        
    Returns:
        Dictionary with answer and confidence score
    """
    result = qa_pipeline(question=question, context=context)
    return {
        "answer": result['answer'],
        "confidence": result['score']
    }

@AgentFunction(
    category="cognitive.classification",
    description="Classify text into given categories without training",
    agent_triggers=["classify_text", "categorize_text", "determine_category"],
    examples=[
        {
            "inputs": {
                "text": "This movie was amazing! The acting was superb.",
                "categories": ["positive", "negative", "neutral"]
            },
            "output": {
                "classification": "positive",
                "scores": {
                    "positive": 0.92,
                    "neutral": 0.06,
                    "negative": 0.02
                }
            }
        }
    ]
)
def zero_shot_classify(
    text: str,
    categories: List[str]
) -> Dict[str, Union[str, Dict[str, float]]]:
    """Classify text into given categories without training.
    
    Args:
        text: Input text
        categories: List of possible categories
        
    Returns:
        Dictionary with classification and confidence scores
    """
    result = zero_shot(text, categories)
    return {
        "classification": result['labels'][0],
        "scores": dict(zip(result['labels'], result['scores']))
    }

@AgentFunction(
    category="cognitive.similarity",
    description="Calculate semantic similarity between texts",
    agent_triggers=["compare_texts", "find_similarity", "measure_relatedness"],
    examples=[
        {
            "inputs": {
                "text1": "The cat sat on the mat",
                "text2": "A kitten was resting on the rug"
            },
            "output": {
                "similarity": 0.85,
                "analysis": "High similarity: both texts describe a feline resting on a floor covering"
            }
        }
    ]
)
def semantic_similarity(text1: str, text2: str) -> Dict[str, Union[float, str]]:
    """Calculate semantic similarity between texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with similarity score and analysis
    """
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    similarity = doc1.similarity(doc2)
    
    # Generate analysis based on similarity score
    if similarity > 0.8:
        analysis = "High similarity: texts are very closely related"
    elif similarity > 0.5:
        analysis = "Moderate similarity: texts share some common elements"
    else:
        analysis = "Low similarity: texts are mostly unrelated"
    
    return {
        "similarity": float(similarity),
        "analysis": analysis
    }
