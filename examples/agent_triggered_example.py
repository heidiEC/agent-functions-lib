import json
import logging
from typing import Dict, Any, Optional
from agent_functions import AgentFunction, workflow

# Configure logging
logging.basicConfig(level=logging.INFO)

@AgentFunction(
    category="nlp",
    description="Analyze text sentiment",
    agent_triggers=["sentiment_analysis_requested", "emotion_detection_needed"]
)
def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze text sentiment - this is a simplified example.
    In practice, you might use a more sophisticated model.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing sentiment scores
    """
    # Simplified sentiment scoring
    positive_words = {'good', 'great', 'excellent', 'happy', 'positive'}
    negative_words = {'bad', 'poor', 'terrible', 'sad', 'negative'}
    
    words = text.lower().split()
    pos_score = sum(1 for word in words if word in positive_words) / len(words)
    neg_score = sum(1 for word in words if word in negative_words) / len(words)
    
    return {
        'positive_score': pos_score,
        'negative_score': neg_score,
        'neutral_score': 1 - (pos_score + neg_score)
    }

@AgentFunction(
    category="data_transform",
    description="Extract key entities from text",
    agent_triggers=["entity_extraction_needed", "key_concept_identification"]
)
def extract_entities(text: str) -> Dict[str, list]:
    """
    Extract key entities from text - simplified example.
    
    Args:
        text: Input text for entity extraction
        
    Returns:
        Dictionary containing extracted entities
    """
    # Simplified entity extraction
    words = text.split()
    capitalized = [word for word in words if word[0].isupper()]
    
    return {
        'potential_entities': capitalized,
        'word_count': len(words)
    }

@workflow
def process_text_with_analytics(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process text based on agent-triggered events.
    
    Args:
        event: Lambda event containing text and trigger information
        
    Returns:
        Processing results
    """
    text = event.get('text', '')
    trigger = event.get('trigger', '')
    
    results = {}
    
    if 'sentiment' in trigger.lower():
        results['sentiment'] = analyze_sentiment(text)
    
    if 'entity' in trigger.lower():
        results['entities'] = extract_entities(text)
    
    return results

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        results = process_text_with_analytics(body)
        
        return {
            'statusCode': 200,
            'body': json.dumps(results),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

# Local testing
if __name__ == "__main__":
    # Simulate an agent triggering sentiment analysis
    test_event = {
        'body': {
            'text': 'This is a great example of agent-triggered functions! However, some parts need improvement.',
            'trigger': 'sentiment_analysis_requested'
        }
    }
    
    result = lambda_handler(test_event, None)
    print(f"Test result: {json.dumps(result, indent=2)}")
