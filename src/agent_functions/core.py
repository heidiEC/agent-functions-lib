from typing import Any, Callable, Dict, Optional, TypeVar
from functools import wraps
import logging
from dataclasses import dataclass, field
from .exceptions import AgentFunctionError

T = TypeVar('T')

@dataclass
class FunctionMetadata:
    """Metadata for agent functions."""
    name: Optional[str] = None
    category: str = 'general'
    description: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    version: str = '0.1.0'

class AgentFunction:
    """
    Decorator for creating modular agent functions with rich metadata and validation.
    
    Supports:
    - Function categorization
    - Metadata annotation
    - Basic input validation
    - Logging
    """
    
    def __init__(self, 
                 category: str = 'general', 
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        """
        Initialize an AgentFunction with optional metadata.
        
        Args:
            category: Functional category of the agent function
            name: Custom name for the function
            description: Human-readable description
            **kwargs: Additional metadata tags
        """
        self.metadata = FunctionMetadata(
            name=name, 
            category=category, 
            description=description,
            tags=kwargs
        )
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Wrap the function with metadata and basic validation.
        
        Args:
            func: The function to be decorated
        
        Returns:
            Wrapped function with additional capabilities
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Optional logging
                logging.info(f"Executing {self.metadata.name or func.__name__}")
                
                # Optional input validation could be added here
                result = func(*args, **kwargs)
                
                return result
            
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}")
                raise AgentFunctionError(f"Function {func.__name__} failed") from e
        
        # Attach metadata to the function
        wrapper.metadata = self.metadata
        return wrapper

def workflow(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to mark and potentially enhance workflow functions.
    
    Args:
        func: The workflow function to be decorated
    
    Returns:
        Enhanced workflow function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logging.info(f"Starting workflow: {func.__name__}")
            result = func(*args, **kwargs)
            logging.info(f"Completed workflow: {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Workflow {func.__name__} failed: {str(e)}")
            raise
    
    return wrapper
