import pytest
import numpy as np
from agent_functions.core_functions.math_ops import *

def test_basic_arithmetic():
    assert add(2, 3) == 5
    assert subtract(5, 3) == 2
    assert multiply(4, 3) == 12
    assert divide(10, 2) == 5
    with pytest.raises(ZeroDivisionError):
        divide(5, 0)

def test_statistical_operations():
    numbers = [1, 2, 3, 4, 5]
    assert mean(numbers) == 3
    assert median(numbers) == 3
    assert std(numbers) == pytest.approx(1.4142, rel=1e-4)
    assert variance(numbers) == pytest.approx(2.0, rel=1e-4)

def test_matrix_operations():
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    result = matrix_multiply(matrix_a, matrix_b)
    expected = [[19, 22], [43, 50]]
    assert np.array_equal(result, expected)

def test_correlation_analysis():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert correlation(x, y) == pytest.approx(1.0, rel=1e-4)
    
    y_inverse = [10, 8, 6, 4, 2]
    assert correlation(x, y_inverse) == pytest.approx(-1.0, rel=1e-4)

def test_input_validation():
    with pytest.raises(ValueError):
        mean([])
    with pytest.raises(TypeError):
        add("2", 3)
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2]], [[1]])
