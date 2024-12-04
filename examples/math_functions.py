import numpy as np
from typing import List, Union, Tuple, Optional
from dataclasses import dataclass
from agent_functions import AgentFunction, workflow
from agent_functions.exceptions import ValidationError

# Type aliases
Number = Union[int, float]
Vector = List[Number]
Matrix = List[List[Number]]

@dataclass
class StatisticalSummary:
    """Container for statistical results."""
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    quartiles: Tuple[float, float, float]

# Basic Mathematical Operations
@AgentFunction(category="math.basic", description="Perform basic arithmetic operations")
def arithmetic(a: Number, b: Number, operation: str) -> Number:
    """
    Perform basic arithmetic operations.
    
    Args:
        a: First number
        b: Second number
        operation: One of 'add', 'subtract', 'multiply', 'divide', 'power'
    
    Returns:
        Result of the arithmetic operation
    """
    ops = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else float('inf'),
        'power': lambda x, y: x ** y
    }
    
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}")
    
    return ops[operation](a, b)

# Vector Operations
@AgentFunction(category="math.vector", description="Compute vector operations")
def vector_operation(vec1: Vector, vec2: Optional[Vector] = None, operation: str = 'magnitude') -> Union[Number, Vector]:
    """
    Perform vector operations.
    
    Args:
        vec1: First vector
        vec2: Second vector (optional)
        operation: One of 'magnitude', 'normalize', 'dot_product', 'add', 'subtract'
    
    Returns:
        Result of the vector operation
    """
    v1 = np.array(vec1)
    
    ops = {
        'magnitude': lambda: float(np.linalg.norm(v1)),
        'normalize': lambda: (v1 / np.linalg.norm(v1)).tolist(),
        'dot_product': lambda: float(np.dot(v1, np.array(vec2))),
        'add': lambda: (v1 + np.array(vec2)).tolist(),
        'subtract': lambda: (v1 - np.array(vec2)).tolist()
    }
    
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}")
    
    if operation in ['dot_product', 'add', 'subtract'] and vec2 is None:
        raise ValueError(f"Operation {operation} requires two vectors")
    
    return ops[operation]()

# Matrix Operations
@AgentFunction(category="math.matrix", description="Perform matrix operations")
def matrix_operation(matrix1: Matrix, matrix2: Optional[Matrix] = None, operation: str = 'determinant') -> Union[Number, Matrix]:
    """
    Perform matrix operations.
    
    Args:
        matrix1: First matrix
        matrix2: Second matrix (optional)
        operation: One of 'determinant', 'inverse', 'transpose', 'multiply'
    
    Returns:
        Result of the matrix operation
    """
    m1 = np.array(matrix1)
    
    ops = {
        'determinant': lambda: float(np.linalg.det(m1)),
        'inverse': lambda: np.linalg.inv(m1).tolist(),
        'transpose': lambda: m1.T.tolist(),
        'multiply': lambda: (m1 @ np.array(matrix2)).tolist()
    }
    
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}")
    
    if operation == 'multiply' and matrix2 is None:
        raise ValueError("Matrix multiplication requires two matrices")
    
    return ops[operation]()

# Statistical Functions
@AgentFunction(category="math.stats", description="Calculate statistical measures")
def statistical_analysis(data: Vector) -> StatisticalSummary:
    """
    Perform statistical analysis on a dataset.
    
    Args:
        data: List of numbers to analyze
    
    Returns:
        StatisticalSummary object containing various statistical measures
    """
    arr = np.array(data)
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    
    return StatisticalSummary(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std_dev=float(np.std(arr)),
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
        quartiles=(float(q1), float(q2), float(q3))
    )

# Complex Mathematical Workflow
@workflow
def analyze_dataset(data: Vector, reference_vector: Optional[Vector] = None) -> dict:
    """
    Perform comprehensive mathematical analysis on a dataset.
    
    Args:
        data: Main dataset to analyze
        reference_vector: Optional reference vector for comparisons
    
    Returns:
        Dictionary containing various mathematical analyses
    """
    # Validate input
    if not data:
        raise ValidationError("Empty dataset provided")
    
    # Statistical analysis
    stats = statistical_analysis(data)
    
    # Vector analysis
    vec_magnitude = vector_operation(data, operation='magnitude')
    normalized_data = vector_operation(data, operation='normalize')
    
    # Comparison with reference if provided
    comparison = {}
    if reference_vector is not None:
        comparison = {
            'dot_product': vector_operation(data, reference_vector, 'dot_product'),
            'vector_difference': vector_operation(data, reference_vector, 'subtract')
        }
    
    return {
        'statistics': stats,
        'vector_analysis': {
            'magnitude': vec_magnitude,
            'normalized': normalized_data
        },
        'comparison': comparison
    }

def main():
    # Example usage
    print("\nBasic Arithmetic:")
    print(f"3 + 4 = {arithmetic(3, 4, 'add')}")
    print(f"3 * 4 = {arithmetic(3, 4, 'multiply')}")
    
    print("\nVector Operations:")
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    print(f"Vector magnitude: {vector_operation(vec1)}")
    print(f"Dot product: {vector_operation(vec1, vec2, 'dot_product')}")
    
    print("\nMatrix Operations:")
    matrix1 = [[1, 2], [3, 4]]
    print(f"Determinant: {matrix_operation(matrix1)}")
    print(f"Transpose: {matrix_operation(matrix1, operation='transpose')}")
    
    print("\nStatistical Analysis:")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats = statistical_analysis(data)
    print(f"Mean: {stats.mean}")
    print(f"Standard Deviation: {stats.std_dev}")
    
    print("\nComprehensive Analysis:")
    result = analyze_dataset(data, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    print(f"Dataset Analysis: {result}")

if __name__ == "__main__":
    main()
