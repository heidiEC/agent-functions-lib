import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from agent_functions import AgentFunction, workflow
from agent_functions.exceptions import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class WorkflowState:
    """State management for workflows."""
    context: Dict[str, Any] = None
    history: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.context = self.context or {}
        self.history = self.history or []
        self.metadata = self.metadata or {}
    
    def add_history(self, action: str):
        self.history.append(action)
    
    def update_context(self, **kwargs):
        self.context.update(kwargs)

class ValidationChain:
    """Chain multiple validations together."""
    def __init__(self):
        self.validations = []
    
    def add(self, validation_func):
        self.validations.append(validation_func)
        return self
    
    def validate(self, data: Any) -> bool:
        for validation in self.validations:
            if not validation(data):
                return False
        return True

# Advanced validation functions
@AgentFunction(category="validation", description="Check if text contains required keywords")
def has_keywords(text: str, keywords: List[str]) -> bool:
    """Validate presence of keywords in text."""
    return any(keyword.lower() in text.lower() for keyword in keywords)

@AgentFunction(category="validation", description="Check text complexity")
def check_complexity(text: str, min_words: int = 5) -> bool:
    """Validate text complexity."""
    words = text.split()
    return len(words) >= min_words

# Data transformation functions
@AgentFunction(category="transform", description="Extract key phrases from text")
def extract_key_phrases(text: str) -> List[str]:
    """Extract important phrases from text."""
    # Simple implementation - in practice, you might use NLP
    sentences = text.split('.')
    return [s.strip() for s in sentences if len(s.split()) > 3]

@AgentFunction(category="transform", description="Categorize text content")
def categorize_content(text: str) -> List[str]:
    """Categorize text content."""
    categories = []
    if '?' in text:
        categories.append('question')
    if '!' in text:
        categories.append('exclamation')
    if len(text.split()) > 10:
        categories.append('long_form')
    return categories

# Parallel processing function
@AgentFunction(category="processing", description="Process multiple texts in parallel")
async def parallel_process_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """Process multiple texts in parallel."""
    async def process_single(text: str) -> Dict[str, Any]:
        # Simulate async processing
        await asyncio.sleep(0.1)
        return {
            'original': text,
            'phrases': extract_key_phrases(text),
            'categories': categorize_content(text)
        }
    
    tasks = [process_single(text) for text in texts]
    return await asyncio.gather(*tasks)

# Advanced workflow with state management
@workflow
def advanced_text_processing(texts: List[str]) -> Dict[str, Any]:
    """
    Advanced workflow demonstrating multiple features:
    - State management
    - Validation chains
    - Parallel processing
    - Result aggregation
    """
    # Initialize workflow state
    state = WorkflowState()
    state.add_history("workflow_started")
    
    # Set up validation chain
    validator = ValidationChain()
    validator.add(lambda x: len(x) > 0)
    validator.add(lambda x: has_keywords(x, ['important', 'urgent', 'critical']))
    validator.add(lambda x: check_complexity(x))
    
    # Validate inputs
    valid_texts = [text for text in texts if validator.validate(text)]
    if not valid_texts:
        raise ValidationError("No texts passed validation")
    
    state.update_context(valid_text_count=len(valid_texts))
    state.add_history("validation_completed")
    
    # Process texts in parallel
    async def run_parallel():
        return await parallel_process_texts(valid_texts)
    
    # Run async processing in a new event loop
    with ThreadPoolExecutor() as executor:
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(run_parallel())
        loop.close()
    
    state.add_history("processing_completed")
    
    # Aggregate results
    aggregated_results = {
        'total_processed': len(results),
        'categories': set(),
        'phrase_count': 0,
        'results': results,
        'workflow_history': state.history
    }
    
    for result in results:
        aggregated_results['categories'].update(result['categories'])
        aggregated_results['phrase_count'] += len(result['phrases'])
    
    return aggregated_results

def main():
    # Example usage
    sample_texts = [
        "This is an important message that requires urgent attention!",
        "A critical update is available for your system.",
        "Short text",  # This will fail validation
        "Another important notification that needs critical review.",
    ]
    
    try:
        results = advanced_text_processing(sample_texts)
        print("\nWorkflow Results:")
        print(f"Total Processed: {results['total_processed']}")
        print(f"Unique Categories: {results['categories']}")
        print(f"Total Phrases: {results['phrase_count']}")
        print(f"Workflow History: {results['workflow_history']}")
    except Exception as e:
        print(f"Error in workflow: {e}")

if __name__ == "__main__":
    main()
