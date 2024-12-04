class AgentFunctionError(Exception):
    """Base exception for agent function errors."""
    pass

class ValidationError(AgentFunctionError):
    """Exception raised for function input validation failures."""
    pass

class WorkflowExecutionError(AgentFunctionError):
    """Exception raised when a workflow fails to execute."""
    pass
