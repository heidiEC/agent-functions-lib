from typing import Dict, Any, List, Type, Optional
from abc import ABC, abstractmethod
import json
import logging
from dataclasses import dataclass, field
from agent_functions import AgentFunction, workflow
from agent_functions.exceptions import AgentFunctionError

# Plugin System
class PluginInterface(ABC):
    """Base interface for plugins."""
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data using the plugin."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

@dataclass
class PluginRegistry:
    """Registry for managing plugins."""
    plugins: Dict[str, Type[PluginInterface]] = field(default_factory=dict)
    
    def register(self, name: str, plugin_class: Type[PluginInterface]) -> None:
        """Register a new plugin."""
        self.plugins[name] = plugin_class
    
    def get_plugin(self, name: str) -> Optional[Type[PluginInterface]]:
        """Get a plugin by name."""
        return self.plugins.get(name)

# Example Plugins
class TextAnalysisPlugin(PluginInterface):
    """Plugin for text analysis."""
    def initialize(self, config: Dict[str, Any]) -> None:
        self.min_length = config.get('min_length', 10)
        self.language = config.get('language', 'en')
    
    def process(self, data: str) -> Dict[str, Any]:
        words = data.split()
        return {
            'word_count': len(words),
            'char_count': len(data),
            'meets_length': len(words) >= self.min_length
        }
    
    def cleanup(self) -> None:
        pass

class JsonFormatterPlugin(PluginInterface):
    """Plugin for JSON formatting."""
    def initialize(self, config: Dict[str, Any]) -> None:
        self.indent = config.get('indent', 2)
        self.sort_keys = config.get('sort_keys', True)
    
    def process(self, data: Any) -> str:
        return json.dumps(data, indent=self.indent, sort_keys=self.sort_keys)
    
    def cleanup(self) -> None:
        pass

# Plugin-aware functions
@AgentFunction(category="plugin", description="Process data through a plugin chain")
def process_with_plugins(data: Any, plugin_chain: List[str], 
                        registry: PluginRegistry, 
                        configs: Dict[str, Dict[str, Any]]) -> Any:
    """
    Process data through a chain of plugins.
    
    Args:
        data: Input data to process
        plugin_chain: List of plugin names to apply in order
        registry: Plugin registry
        configs: Plugin configurations
    
    Returns:
        Processed data
    """
    result = data
    active_plugins = []
    
    try:
        # Initialize plugins
        for plugin_name in plugin_chain:
            plugin_class = registry.get_plugin(plugin_name)
            if not plugin_class:
                raise AgentFunctionError(f"Plugin not found: {plugin_name}")
            
            plugin = plugin_class()
            plugin.initialize(configs.get(plugin_name, {}))
            active_plugins.append(plugin)
        
        # Process data through plugin chain
        for plugin in active_plugins:
            result = plugin.process(result)
        
        return result
    
    finally:
        # Cleanup plugins
        for plugin in active_plugins:
            plugin.cleanup()

# Example workflow using plugins
@workflow
def analyze_and_format_text(text: str) -> str:
    """
    Analyze text and format the results as JSON.
    
    Args:
        text: Input text to analyze
    
    Returns:
        JSON formatted analysis results
    """
    # Set up plugin registry
    registry = PluginRegistry()
    registry.register('text_analysis', TextAnalysisPlugin)
    registry.register('json_formatter', JsonFormatterPlugin)
    
    # Configure plugins
    configs = {
        'text_analysis': {
            'min_length': 5,
            'language': 'en'
        },
        'json_formatter': {
            'indent': 2,
            'sort_keys': True
        }
    }
    
    # Process through plugin chain
    result = process_with_plugins(
        data=text,
        plugin_chain=['text_analysis', 'json_formatter'],
        registry=registry,
        configs=configs
    )
    
    return result

def main():
    # Example usage
    sample_text = "This is a sample text that will be analyzed and formatted using our plugin system."
    
    try:
        result = analyze_and_format_text(sample_text)
        print("\nAnalysis Results:")
        print(result)
    except Exception as e:
        print(f"Error in workflow: {e}")

if __name__ == "__main__":
    main()
