import pytest
from agent_functions import AgentFunction, workflow
from agent_functions.exceptions import AgentFunctionError

def test_agent_function_metadata():
    @AgentFunction(category="test", name="test_func", description="A test function")
    def sample_func(x: int) -> int:
        return x * 2
    
    assert hasattr(sample_func, 'metadata')
    assert sample_func.metadata.category == "test"
    assert sample_func.metadata.name == "test_func"
    assert sample_func.metadata.description == "A test function"

def test_agent_function_execution():
    @AgentFunction(category="math")
    def add(a: int, b: int) -> int:
        return a + b
    
    result = add(3, 4)
    assert result == 7

def test_agent_function_error_handling():
    @AgentFunction(category="error")
    def divide(a: int, b: int) -> float:
        return a / b
    
    with pytest.raises(AgentFunctionError):
        divide(1, 0)

def test_workflow_decorator():
    @workflow
    def sample_workflow(x: int) -> int:
        return x * 2
    
    result = sample_workflow(5)
    assert result == 10

def test_workflow_error_propagation():
    @workflow
    def failing_workflow():
        raise ValueError("Workflow failed")
    
    with pytest.raises(ValueError):
        failing_workflow()
