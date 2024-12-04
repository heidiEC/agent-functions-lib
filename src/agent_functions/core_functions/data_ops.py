"""Data transformation and analysis operations."""

from typing import List, Dict, Optional, Union, Any
from ..core import AgentFunction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import json

@AgentFunction(
    category="data.load",
    description="Load data from various file formats",
    agent_triggers=["load_data", "read_file", "import_data"],
    examples=[
        {
            "inputs": {
                "file_path": "data.csv",
                "file_type": "csv",
                "options": {"index_col": 0}
            },
            "output": "DataFrame with loaded data"
        }
    ]
)
def load_data(
    file_path: str,
    file_type: str = "csv",
    options: Optional[Dict] = None
) -> pd.DataFrame:
    """Load data from file.
    
    Args:
        file_path: Path to data file
        file_type: Type of file (csv, json, excel)
        options: Additional loading options
        
    Returns:
        Loaded DataFrame
    """
    options = options or {}
    
    if file_type == "csv":
        return pd.read_csv(file_path, **options)
    elif file_type == "json":
        return pd.read_json(file_path, **options)
    elif file_type == "excel":
        return pd.read_excel(file_path, **options)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

@AgentFunction(
    category="data.clean",
    description="Clean and preprocess DataFrame",
    agent_triggers=["clean_data", "preprocess_data", "handle_missing"],
    examples=[
        {
            "inputs": {
                "df": "DataFrame",
                "handle_missing": "mean",
                "remove_duplicates": True
            },
            "output": "Cleaned DataFrame"
        }
    ]
)
def clean_data(
    df: pd.DataFrame,
    handle_missing: str = "mean",
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """Clean and preprocess DataFrame.
    
    Args:
        df: Input DataFrame
        handle_missing: How to handle missing values (mean, median, drop)
        remove_duplicates: Whether to remove duplicate rows
        
    Returns:
        Cleaned DataFrame
    """
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Handle missing values
    if handle_missing == "mean":
        df = df.fillna(df.mean())
    elif handle_missing == "median":
        df = df.fillna(df.median())
    elif handle_missing == "drop":
        df = df.dropna()
        
    # Remove duplicates if requested
    if remove_duplicates:
        df = df.drop_duplicates()
        
    return df

@AgentFunction(
    category="data.transform",
    description="Scale numerical features in DataFrame",
    agent_triggers=["scale_data", "normalize_data", "standardize_features"],
    examples=[
        {
            "inputs": {
                "df": "DataFrame",
                "columns": ["height", "weight"],
                "method": "standard"
            },
            "output": "DataFrame with scaled features"
        }
    ]
)
def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard"
) -> pd.DataFrame:
    """Scale numerical features.
    
    Args:
        df: Input DataFrame
        columns: Columns to scale
        method: Scaling method (standard or minmax)
        
    Returns:
        DataFrame with scaled features
    """
    df = df.copy()
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
        
    df[columns] = scaler.fit_transform(df[columns])
    return df

@AgentFunction(
    category="data.transform",
    description="Reduce dimensionality of data using PCA",
    agent_triggers=["reduce_dimensions", "apply_pca", "compress_features"],
    examples=[
        {
            "inputs": {
                "df": "DataFrame",
                "columns": ["x1", "x2", "x3"],
                "n_components": 2
            },
            "output": "DataFrame with reduced dimensions"
        }
    ]
)
def reduce_dimensions(
    df: pd.DataFrame,
    columns: List[str],
    n_components: int
) -> pd.DataFrame:
    """Reduce dimensionality using PCA.
    
    Args:
        df: Input DataFrame
        columns: Columns to transform
        n_components: Number of components to keep
        
    Returns:
        DataFrame with reduced dimensions
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(df[columns])
    
    # Create new DataFrame with PCA results
    pca_df = pd.DataFrame(
        transformed,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df.index
    )
    
    # Add non-transformed columns
    for col in df.columns:
        if col not in columns:
            pca_df[col] = df[col]
            
    return pca_df

@AgentFunction(
    category="data.analyze",
    description="Generate summary statistics for DataFrame",
    agent_triggers=["summarize_data", "get_statistics", "analyze_features"],
    examples=[
        {
            "inputs": {
                "df": "DataFrame",
                "include_correlations": True
            },
            "output": {
                "basic_stats": {
                    "mean": {...},
                    "std": {...}
                },
                "correlations": {...}
            }
        }
    ]
)
def summarize_data(
    df: pd.DataFrame,
    include_correlations: bool = True
) -> Dict[str, Any]:
    """Generate summary statistics.
    
    Args:
        df: Input DataFrame
        include_correlations: Whether to include correlation matrix
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "basic_stats": {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "min": df.min().to_dict(),
            "max": df.max().to_dict()
        },
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }
    
    if include_correlations:
        # Only include numeric columns in correlation
        numeric_df = df.select_dtypes(include=[np.number])
        summary["correlations"] = numeric_df.corr().to_dict()
        
    return summary

@AgentFunction(
    category="data.transform",
    description="Pivot DataFrame for different view of data",
    agent_triggers=["pivot_data", "reshape_data", "aggregate_view"],
    examples=[
        {
            "inputs": {
                "df": "DataFrame",
                "index": "date",
                "columns": "category",
                "values": "sales",
                "aggfunc": "sum"
            },
            "output": "Pivoted DataFrame"
        }
    ]
)
def pivot_data(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "mean"
) -> pd.DataFrame:
    """Pivot DataFrame for different view.
    
    Args:
        df: Input DataFrame
        index: Column to use as index
        columns: Column to use as columns
        values: Column to aggregate
        aggfunc: Aggregation function
        
    Returns:
        Pivoted DataFrame
    """
    return pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc
    )
