import pytest
import pandas as pd
import numpy as np
from agent_functions.core_functions.data_ops import *

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5],
        'D': [True, False, True, False, True]
    })

def test_data_loading(tmp_path):
    # Create test CSV file
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    
    loaded_df = load_data(str(csv_path))
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.shape == (3, 2)
    assert all(loaded_df.columns == ['col1', 'col2'])

def test_data_cleaning(sample_df):
    # Add some null values and duplicates
    dirty_df = sample_df.copy()
    dirty_df.loc[0, 'A'] = None
    dirty_df.loc[2, 'B'] = None
    dirty_df = pd.concat([dirty_df, dirty_df.iloc[[1]]])
    
    clean_df = clean_data(dirty_df)
    
    assert clean_df.isnull().sum().sum() == 0  # No null values
    assert len(clean_df) == len(clean_df.drop_duplicates())  # No duplicates
    assert all(col in clean_df.columns for col in dirty_df.columns)  # All columns preserved

def test_feature_scaling():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Test standardization
    scaled_data = scale_features(data, method='standard')
    assert np.allclose(scaled_data.mean(axis=0), [0, 0])
    assert np.allclose(scaled_data.std(axis=0), [1, 1])
    
    # Test min-max scaling
    scaled_data = scale_features(data, method='minmax')
    assert np.allclose(scaled_data.min(axis=0), [0, 0])
    assert np.allclose(scaled_data.max(axis=0), [1, 1])

def test_dimensionality_reduction():
    data = np.random.rand(100, 10)
    
    # Test PCA
    reduced_data = reduce_dimensions(data, method='pca', n_components=3)
    assert reduced_data.shape == (100, 3)
    
    # Test t-SNE
    reduced_data = reduce_dimensions(data, method='tsne', n_components=2)
    assert reduced_data.shape == (100, 2)

def test_data_summarization(sample_df):
    summary = summarize_data(sample_df)
    
    assert isinstance(summary, dict)
    assert 'shape' in summary
    assert 'dtypes' in summary
    assert 'missing_values' in summary
    assert 'numeric_stats' in summary
    assert summary['shape'] == sample_df.shape

def test_pivot_reshape(sample_df):
    # Test pivoting
    pivot_df = pivot_data(sample_df, 
                         index='A',
                         columns='B',
                         values='C')
    assert isinstance(pivot_df, pd.DataFrame)
    
    # Test reshaping (melt)
    melted_df = reshape_data(sample_df, 
                            id_vars=['A'],
                            value_vars=['B', 'C'])
    assert isinstance(melted_df, pd.DataFrame)
    assert 'variable' in melted_df.columns
    assert 'value' in melted_df.columns

def test_input_validation():
    with pytest.raises(ValueError):
        load_data("nonexistent_file.csv")
    
    with pytest.raises(ValueError):
        scale_features(np.array([]), method='invalid')
    
    with pytest.raises(ValueError):
        reduce_dimensions(np.array([[1, 2]]), method='invalid')
    
    with pytest.raises(ValueError):
        pivot_data(pd.DataFrame(), index='A', columns='B', values='C')
