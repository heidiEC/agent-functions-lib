import pytest
import time
from agent_functions.core_functions.memory_ops import *

@pytest.fixture
def memory_store():
    # Create a fresh memory store for each test
    store = create_memory_store()
    yield store
    # Cleanup after tests
    clear_memory_store(store)

def test_basic_storage(memory_store):
    # Test storing and retrieving
    store_value(memory_store, "test_key", "test_value")
    assert retrieve_value(memory_store, "test_key") == "test_value"
    
    # Test storing different types
    store_value(memory_store, "int_key", 42)
    store_value(memory_store, "list_key", [1, 2, 3])
    store_value(memory_store, "dict_key", {"a": 1, "b": 2})
    
    assert retrieve_value(memory_store, "int_key") == 42
    assert retrieve_value(memory_store, "list_key") == [1, 2, 3]
    assert retrieve_value(memory_store, "dict_key") == {"a": 1, "b": 2}

def test_metadata_tracking(memory_store):
    store_value(memory_store, "key1", "value1")
    metadata = get_metadata(memory_store, "key1")
    
    assert "timestamp" in metadata
    assert "type" in metadata
    assert "size" in metadata
    assert metadata["type"] == "str"
    
    # Test metadata updates
    time.sleep(0.1)  # Ensure timestamp difference
    store_value(memory_store, "key1", "updated_value")
    new_metadata = get_metadata(memory_store, "key1")
    
    assert new_metadata["timestamp"] > metadata["timestamp"]
    assert new_metadata["size"] > metadata["size"]

def test_memory_persistence(tmp_path):
    # Test saving to disk
    store = create_memory_store()
    store_value(store, "persist_key", "persist_value")
    
    save_path = tmp_path / "test_memory.json"
    save_memory_store(store, str(save_path))
    
    # Test loading from disk
    loaded_store = load_memory_store(str(save_path))
    assert retrieve_value(loaded_store, "persist_key") == "persist_value"
    
    # Cleanup
    clear_memory_store(store)
    clear_memory_store(loaded_store)

def test_memory_operations(memory_store):
    # Test exists operation
    store_value(memory_store, "exists_key", "value")
    assert key_exists(memory_store, "exists_key")
    assert not key_exists(memory_store, "nonexistent_key")
    
    # Test delete operation
    delete_value(memory_store, "exists_key")
    assert not key_exists(memory_store, "exists_key")
    
    # Test clear operation
    store_value(memory_store, "key1", "value1")
    store_value(memory_store, "key2", "value2")
    clear_memory_store(memory_store)
    assert not key_exists(memory_store, "key1")
    assert not key_exists(memory_store, "key2")

def test_memory_search(memory_store):
    # Store some test data
    store_value(memory_store, "user1", {"name": "John", "age": 30})
    store_value(memory_store, "user2", {"name": "Jane", "age": 25})
    store_value(memory_store, "product1", {"name": "Laptop", "price": 1000})
    
    # Test searching by pattern
    user_keys = search_keys(memory_store, "user*")
    assert len(user_keys) == 2
    assert "user1" in user_keys
    assert "user2" in user_keys
    
    # Test searching by value condition
    young_users = search_values(memory_store, lambda x: 
        isinstance(x, dict) and x.get("age", 0) < 30)
    assert len(young_users) == 1
    assert young_users[0]["name"] == "Jane"

def test_input_validation(memory_store):
    with pytest.raises(ValueError):
        store_value(memory_store, "", "value")  # Empty key
    
    with pytest.raises(ValueError):
        store_value(memory_store, None, "value")  # None key
    
    with pytest.raises(KeyError):
        retrieve_value(memory_store, "nonexistent_key")
    
    with pytest.raises(ValueError):
        save_memory_store(memory_store, "")  # Empty path
    
    with pytest.raises(ValueError):
        load_memory_store("")  # Empty path
