import logging
from agent_functions import AgentFunction, workflow

# Configure logging
logging.basicConfig(level=logging.INFO)

@AgentFunction(category="data_transform", description="Clean and normalize text")
def clean_text(text: str) -> str:
    """
    Clean and normalize input text.
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned and normalized text
    """
    return text.strip().lower()

@AgentFunction(category="validation", description="Check text length")
def validate_length(text: str, min_length: int = 3) -> bool:
    """
    Validate text length.
    
    Args:
        text: Text to validate
        min_length: Minimum acceptable text length
    
    Returns:
        True if text meets length requirement, False otherwise
    """
    return len(text) >= min_length

@workflow
def process_text(input_text: str):
    """
    Demonstrate a simple workflow combining multiple agent functions.
    
    Args:
        input_text: Text to process
    
    Returns:
        Processed text or None if validation fails
    """
    # Clean the text
    cleaned_text = clean_text(input_text)
    
    # Validate text length
    if validate_length(cleaned_text):
        return cleaned_text
    
    return None

def main():
    # Example usage
    sample_texts = [
        "  Hello World  ",
        "Hi",
        "Agent Functions are awesome!"
    ]
    
    for text in sample_texts:
        try:
            result = process_text(text)
            print(f"Original: '{text}', Processed: {result}")
        except Exception as e:
            print(f"Error processing '{text}': {e}")

if __name__ == "__main__":
    main()
