import numpy as np
import pandas as pd
from scipy import optimize, signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from agent_functions import AgentFunction, workflow
from agent_functions.exceptions import ValidationError

# Advanced Mathematical Types
@dataclass
class OptimizationResult:
    """Results from optimization problems."""
    solution: np.ndarray
    objective_value: float
    success: bool
    message: str

@dataclass
class SignalAnalysisResult:
    """Results from signal analysis."""
    frequencies: np.ndarray
    power_spectrum: np.ndarray
    filtered_signal: np.ndarray
    peaks: np.ndarray
    properties: Dict[str, Any]

@dataclass
class ClusteringResult:
    """Results from clustering analysis."""
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    silhouette_score: float

# Advanced Mathematical Operations
@AgentFunction(category="math.optimization", description="Solve optimization problems")
def optimize_function(func: callable, 
                     initial_guess: np.ndarray,
                     bounds: Optional[List[Tuple[float, float]]] = None,
                     constraints: Optional[List[Dict[str, Any]]] = None) -> OptimizationResult:
    """
    Solve optimization problems using scipy's minimize function.
    
    Args:
        func: Objective function to minimize
        initial_guess: Initial point for optimization
        bounds: Bounds for variables [(min, max), ...]
        constraints: List of constraint dictionaries
    
    Returns:
        OptimizationResult containing solution and metadata
    """
    result = optimize.minimize(
        func,
        initial_guess,
        bounds=bounds,
        constraints=constraints if constraints else []
    )
    
    return OptimizationResult(
        solution=result.x,
        objective_value=result.fun,
        success=result.success,
        message=result.message
    )

@AgentFunction(category="math.signal", description="Analyze and process signals")
def analyze_signal(signal_data: np.ndarray,
                  sampling_rate: float,
                  filter_freq: Optional[float] = None) -> SignalAnalysisResult:
    """
    Perform signal analysis including FFT and peak detection.
    
    Args:
        signal_data: Time series data
        sampling_rate: Sampling rate in Hz
        filter_freq: Optional cutoff frequency for lowpass filter
    
    Returns:
        SignalAnalysisResult containing frequencies, spectrum, and peaks
    """
    # Compute FFT
    frequencies = np.fft.fftfreq(len(signal_data), 1/sampling_rate)
    spectrum = np.abs(np.fft.fft(signal_data))
    
    # Apply filter if requested
    filtered_signal = signal_data
    if filter_freq:
        nyquist = sampling_rate / 2
        b, a = signal.butter(4, filter_freq/nyquist, 'low')
        filtered_signal = signal.filtfilt(b, a, signal_data)
    
    # Find peaks
    peaks, properties = signal.find_peaks(filtered_signal, 
                                        height=np.mean(filtered_signal),
                                        distance=sampling_rate//10)
    
    return SignalAnalysisResult(
        frequencies=frequencies,
        power_spectrum=spectrum,
        filtered_signal=filtered_signal,
        peaks=peaks,
        properties=properties
    )

@AgentFunction(category="math.ml", description="Perform clustering analysis")
def cluster_data(data: np.ndarray,
                n_clusters: int,
                standardize: bool = True) -> ClusteringResult:
    """
    Perform clustering analysis with preprocessing.
    
    Args:
        data: Input data matrix
        n_clusters: Number of clusters
        standardize: Whether to standardize the data
    
    Returns:
        ClusteringResult containing labels and metadata
    """
    # Preprocess data
    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Calculate silhouette score
    silhouette = silhouette_score(data, labels) if n_clusters > 1 else 0.0
    
    return ClusteringResult(
        labels=labels,
        centroids=kmeans.cluster_centers_,
        inertia=kmeans.inertia_,
        silhouette_score=silhouette
    )

@AgentFunction(category="math.dimensionality", description="Reduce data dimensionality")
def reduce_dimensions(data: np.ndarray,
                     n_components: int,
                     method: str = 'pca') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce data dimensionality using various methods.
    
    Args:
        data: Input data matrix
        n_components: Number of components to keep
        method: Reduction method ('pca' supported)
    
    Returns:
        Tuple of (reduced_data, metadata)
    """
    if method == 'pca':
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        metadata = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'singular_values': pca.singular_values_,
            'components': pca.components_
        }
        return reduced_data, metadata
    else:
        raise ValueError(f"Unsupported method: {method}")

# Advanced Workflow
@workflow
def analyze_complex_dataset(data: np.ndarray,
                          sampling_rate: float = 1.0,
                          n_clusters: int = 3) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a complex dataset.
    
    Args:
        data: Input data matrix
        sampling_rate: Sampling rate for time series analysis
        n_clusters: Number of clusters for clustering analysis
    
    Returns:
        Dictionary containing various analyses results
    """
    results = {}
    
    # Dimensionality reduction
    reduced_data, pca_metadata = reduce_dimensions(data, n_components=2)
    results['dimensionality_reduction'] = {
        'reduced_data': reduced_data,
        'metadata': pca_metadata
    }
    
    # Clustering on reduced data
    clustering = cluster_data(reduced_data, n_clusters=n_clusters)
    results['clustering'] = clustering
    
    # Signal analysis on first principal component
    signal_analysis = analyze_signal(
        reduced_data[:, 0],
        sampling_rate=sampling_rate,
        filter_freq=sampling_rate/4
    )
    results['signal_analysis'] = signal_analysis
    
    # Optimization example: minimize distance to cluster centroids
    def objective(x):
        return np.min(np.linalg.norm(clustering.centroids - x, axis=1))
    
    opt_result = optimize_function(
        objective,
        initial_guess=np.mean(reduced_data, axis=0),
        bounds=[(-10, 10), (-10, 10)]
    )
    results['optimization'] = opt_result
    
    return results

def plot_results(results: Dict[str, Any]):
    """Plot the analysis results."""
    plt.figure(figsize=(15, 10))
    
    # Plot reduced data with clusters
    plt.subplot(2, 2, 1)
    reduced_data = results['dimensionality_reduction']['reduced_data']
    clusters = results['clustering']
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                c=clusters.labels, cmap='viridis')
    plt.scatter(clusters.centroids[:, 0], clusters.centroids[:, 1], 
                marker='x', s=200, linewidths=3, color='r')
    plt.title('Clustered Data in Reduced Space')
    
    # Plot signal analysis
    plt.subplot(2, 2, 2)
    signal_results = results['signal_analysis']
    plt.plot(signal_results.filtered_signal)
    plt.plot(signal_results.peaks, 
             signal_results.filtered_signal[signal_results.peaks], 'x')
    plt.title('Signal Analysis with Peaks')
    
    # Plot power spectrum
    plt.subplot(2, 2, 3)
    plt.plot(signal_results.frequencies[:len(signal_results.frequencies)//2],
             signal_results.power_spectrum[:len(signal_results.frequencies)//2])
    plt.title('Power Spectrum')
    
    # Plot optimization result
    plt.subplot(2, 2, 4)
    opt_point = results['optimization'].solution
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.scatter(opt_point[0], opt_point[1], color='red', s=200, marker='*')
    plt.title('Optimization Result')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate sample data
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    signal1 = np.sin(2 * np.pi * 1.0 * t)
    signal2 = 0.5 * np.sin(2 * np.pi * 2.3 * t)
    noise = np.random.normal(0, 0.2, n_samples)
    
    # Combine signals into a dataset
    data = np.column_stack([
        signal1 + noise,
        signal2 + noise,
        signal1 * signal2 + noise,
        np.exp(-t/5) * signal1 + noise
    ])
    
    try:
        # Analyze the dataset
        results = analyze_complex_dataset(
            data,
            sampling_rate=n_samples/10,
            n_clusters=3
        )
        
        # Print some results
        print("\nDimensionality Reduction:")
        print(f"Explained variance ratios: {results['dimensionality_reduction']['metadata']['explained_variance_ratio']}")
        
        print("\nClustering:")
        print(f"Silhouette score: {results['clustering'].silhouette_score}")
        
        print("\nOptimization:")
        print(f"Optimal point: {results['optimization'].solution}")
        print(f"Objective value: {results['optimization'].objective_value}")
        
        # Plot results
        plot_results(results)
        
    except Exception as e:
        print(f"Error in analysis: {e}")

if __name__ == "__main__":
    main()
