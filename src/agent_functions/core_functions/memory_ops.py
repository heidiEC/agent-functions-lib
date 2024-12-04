"""Memory operations for storing and retrieving agent data."""

from typing import Any, Dict, List, Optional
from ..core import AgentFunction
import json
import os
from datetime import datetime

class MemoryStore:
    """Simple key-value store with persistence."""
    
    def __init__(self, storage_path: str = ".agent_memory"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
    def _get_path(self, key: str) -> str:
        return os.path.join(self.storage_path, f"{key}.json")
        
    def save(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        data = {
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        with open(self._get_path(key), 'w') as f:
            json.dump(data, f)
            
    def load(self, key: str) -> Optional[Dict]:
        try:
            with open(self._get_path(key), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
            
    def delete(self, key: str) -> bool:
        try:
            os.remove(self._get_path(key))
            return True
        except FileNotFoundError:
            return False
            
    def list_keys(self) -> List[str]:
        files = os.listdir(self.storage_path)
        return [f[:-5] for f in files if f.endswith('.json')]

# Initialize global memory store
memory_store = MemoryStore()

@AgentFunction(
    category="memory.basic",
    description="Store a value with an associated key",
    agent_triggers=["store_value", "remember_data", "save_for_later"],
    examples=[
        {"inputs": {
            "key": "user_preference",
            "value": {"theme": "dark", "language": "en"},
            "metadata": {"source": "user_settings"}
         },
         "output": True}
    ]
)
def store(key: str, value: Any, metadata: Optional[Dict] = None) -> bool:
    """Store a value with optional metadata.
    
    Args:
        key: Unique identifier for the stored value
        value: Any JSON-serializable value to store
        metadata: Optional metadata about the stored value
        
    Returns:
        True if storage was successful
        
    Raises:
        ValueError: If value is not JSON-serializable
    """
    try:
        memory_store.save(key, value, metadata)
        return True
    except Exception as e:
        raise ValueError(f"Failed to store value: {str(e)}")

@AgentFunction(
    category="memory.basic",
    description="Retrieve a previously stored value by key",
    agent_triggers=["retrieve_value", "recall_data", "get_stored"],
    examples=[
        {"inputs": {"key": "user_preference"},
         "output": {
             "value": {"theme": "dark", "language": "en"},
             "metadata": {"source": "user_settings"},
             "timestamp": "2024-01-20T10:30:00"
         }}
    ]
)
def retrieve(key: str) -> Optional[Dict]:
    """Retrieve a stored value and its metadata.
    
    Args:
        key: Key of the value to retrieve
        
    Returns:
        Dictionary containing value, metadata, and timestamp if found, None otherwise
    """
    return memory_store.load(key)

@AgentFunction(
    category="memory.basic",
    description="Delete a stored value by key",
    agent_triggers=["forget_value", "delete_stored", "remove_data"],
    examples=[
        {"inputs": {"key": "user_preference"},
         "output": True}
    ]
)
def forget(key: str) -> bool:
    """Delete a stored value.
    
    Args:
        key: Key of the value to delete
        
    Returns:
        True if value was deleted, False if key didn't exist
    """
    return memory_store.delete(key)

@AgentFunction(
    category="memory.query",
    description="List all stored keys",
    agent_triggers=["list_stored", "show_memory", "get_keys"],
    examples=[
        {"inputs": {},
         "output": ["user_preference", "calculation_result", "task_status"]}
    ]
)
def list_stored() -> List[str]:
    """List all stored keys.
    
    Returns:
        List of stored keys
    """
    return memory_store.list_keys()
