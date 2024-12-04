"""Core mathematical operations optimized for agent use."""

from typing import Union, List, Dict, Any
from ..core import AgentFunction
import numpy as np
from scipy import stats

NumericType = Union[int, float]
VectorType = Union[List[NumericType], np.ndarray]
MatrixType = Union[List[List[NumericType]], np.ndarray]

@AgentFunction(
    category="math.basic",
    description="Add two numbers or vectors together",
    agent_triggers=["addition_requested", "sum_needed", "combine_numbers"],
    examples=[
        {"inputs": {"a": 5, "b": 3}, "output": 8},
        {"inputs": {"a": [1, 2], "b": [3, 4]}, "output": [4, 6]}
    ]
)
def add(a: NumericType, b: NumericType) -> NumericType:
    """Add two numbers or vectors.
    
    Args:
        a: First number or vector
        b: Second number or vector
        
    Returns:
        Sum of inputs
        
    Raises:
        TypeError: If inputs are not numeric or compatible vectors
    """
    return np.add(a, b)

@AgentFunction(
    category="math.statistics",
    description="Calculate mean and standard deviation of a numeric sequence",
    agent_triggers=["stats_needed", "distribution_analysis"],
    examples=[
        {"inputs": {"values": [1, 2, 3, 4, 5]}, 
         "output": {"mean": 3.0, "std": 1.5811388300841898}}
    ]
)
def calculate_statistics(values: VectorType) -> Dict[str, float]:
    """Calculate basic statistics of a numeric sequence.
    
    Args:
        values: List of numbers
        
    Returns:
        Dictionary with mean and standard deviation
        
    Raises:
        ValueError: If input is empty
    """
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values))
    }

@AgentFunction(
    category="math.linear_algebra",
    description="Multiply two matrices or perform scalar multiplication",
    agent_triggers=["matrix_multiplication", "scale_matrix"],
    examples=[
        {"inputs": {
            "a": [[1, 2], [3, 4]], 
            "b": [[5, 6], [7, 8]]
         }, 
         "output": [[19, 22], [43, 50]]}
    ]
)
def matrix_multiply(a: MatrixType, b: MatrixType) -> MatrixType:
    """Multiply two matrices or perform scalar multiplication.
    
    Args:
        a: First matrix
        b: Second matrix
        
    Returns:
        Result of matrix multiplication
        
    Raises:
        ValueError: If matrix dimensions are incompatible
    """
    return np.matmul(a, b).tolist()

@AgentFunction(
    category="math.statistics",
    description="Perform correlation analysis between two vectors",
    agent_triggers=["correlation_needed", "relationship_analysis"],
    examples=[
        {"inputs": {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 5, 4, 5]
         },
         "output": {
             "correlation": 0.8164965809277261,
             "p_value": 0.09186468873368339
         }}
    ]
)
def correlation_analysis(x: VectorType, y: VectorType) -> Dict[str, float]:
    """Calculate correlation coefficient and p-value between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Dictionary with correlation coefficient and p-value
        
    Raises:
        ValueError: If vectors have different lengths
    """
    correlation, p_value = stats.pearsonr(x, y)
    return {
        "correlation": float(correlation),
        "p_value": float(p_value)
    }
