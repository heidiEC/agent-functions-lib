"""Text processing operations optimized for agent use."""

from typing import List, Dict, Optional, Union
from ..core import AgentFunction
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

@AgentFunction(
    category="text.basic",
    description="Tokenize text into words and sentences",
    agent_triggers=["tokenize_text", "split_text", "get_tokens"],
    examples=[
        {
            "inputs": {"text": "Hello world! This is a test."},
            "output": {
                "words": ["Hello", "world", "!", "This", "is", "a", "test", "."],
                "sentences": ["Hello world!", "This is a test."]
            }
        }
    ]
)
def tokenize(text: str) -> Dict[str, List[str]]:
    """Tokenize text into words and sentences.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        Dictionary with word and sentence tokens
    """
    return {
        "words": word_tokenize(text),
        "sentences": sent_tokenize(text)
    }

@AgentFunction(
    category="text.preprocessing",
    description="Clean and normalize text by removing special characters, converting to lowercase, etc.",
    agent_triggers=["clean_text", "normalize_text", "preprocess_text"],
    examples=[
        {
            "inputs": {
                "text": "Hello, World! This is a TEST...",
                "remove_punctuation": True,
                "lowercase": True
            },
            "output": "hello world this is a test"
        }
    ]
)
def clean_text(
    text: str,
    remove_punctuation: bool = True,
    lowercase: bool = True
) -> str:
    """Clean and normalize text.
    
    Args:
        text: Input text to clean
        remove_punctuation: Whether to remove punctuation
        lowercase: Whether to convert to lowercase
        
    Returns:
        Cleaned text
    """
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

@AgentFunction(
    category="text.preprocessing",
    description="Remove stopwords from text",
    agent_triggers=["remove_stopwords", "filter_stopwords"],
    examples=[
        {
            "inputs": {"text": "this is a test message"},
            "output": ["test", "message"]
        }
    ]
)
def remove_stopwords(text: str, language: str = 'english') -> List[str]:
    """Remove stopwords from text.
    
    Args:
        text: Input text
        language: Language of stopwords
        
    Returns:
        List of words with stopwords removed
    """
    stop_words = set(stopwords.words(language))
    words = word_tokenize(text.lower())
    return [word for word in words if word.lower() not in stop_words]

@AgentFunction(
    category="text.analysis",
    description="Perform sentiment analysis on text",
    agent_triggers=["analyze_sentiment", "get_sentiment", "emotion_analysis"],
    examples=[
        {
            "inputs": {"text": "I love this product! It's amazing."},
            "output": {
                "polarity": 0.8,
                "subjectivity": 0.9,
                "sentiment": "positive"
            }
        }
    ]
)
def analyze_sentiment(text: str) -> Dict[str, Union[float, str]]:
    """Analyze sentiment of text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with sentiment metrics
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Determine sentiment label
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "sentiment": sentiment
    }

@AgentFunction(
    category="text.analysis",
    description="Extract key phrases from text",
    agent_triggers=["extract_phrases", "get_key_phrases", "find_important_phrases"],
    examples=[
        {
            "inputs": {
                "text": "The quick brown fox jumps over the lazy dog.",
                "max_phrases": 2
            },
            "output": ["quick brown fox", "lazy dog"]
        }
    ]
)
def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract key phrases from text.
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    blob = TextBlob(text)
    noun_phrases = blob.noun_phrases
    return list(noun_phrases)[:max_phrases]

@AgentFunction(
    category="text.analysis",
    description="Tag parts of speech in text",
    agent_triggers=["tag_pos", "get_parts_of_speech", "analyze_grammar"],
    examples=[
        {
            "inputs": {"text": "The cat sat on the mat"},
            "output": [
                ("The", "DT"),
                ("cat", "NN"),
                ("sat", "VBD"),
                ("on", "IN"),
                ("the", "DT"),
                ("mat", "NN")
            ]
        }
    ]
)
def tag_parts_of_speech(text: str) -> List[tuple]:
    """Tag parts of speech in text.
    
    Args:
        text: Input text
        
    Returns:
        List of (word, tag) tuples
    """
    blob = TextBlob(text)
    return blob.tags

@AgentFunction(
    category="text.preprocessing",
    description="Lemmatize words to their base form",
    agent_triggers=["lemmatize_text", "get_base_words", "normalize_words"],
    examples=[
        {
            "inputs": {"words": ["running", "cats", "better", "lives"]},
            "output": ["run", "cat", "good", "life"]
        }
    ]
)
def lemmatize(words: List[str]) -> List[str]:
    """Lemmatize words to their base form.
    
    Args:
        words: List of words to lemmatize
        
    Returns:
        List of lemmatized words
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]
