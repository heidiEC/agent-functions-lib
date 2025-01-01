# Agent Functions Library

## Why This Library?
Agent Functions Library is designed for the emerging world of AI-driven software development, where autonomous agents need to trigger and compose functions across different languages and platforms. It stands out by offering:

1. **Native Agent Integration**: Built specifically for AI agents, not adapted after the fact
   - Declarative function triggers that agents can easily understand and use
   - Built-in metadata and documentation that helps agents make decisions
   - Automatic validation and error handling suited for agent interactions

2. **Cross-Language Support**: Seamlessly work with both Python and JavaScript
   - Write functions in either language
   - Call Python functions from JavaScript and vice versa
   - Unified interface across languages

3. **Cloud-Native & Serverless Ready**
   - AWS Lambda integration out of the box
   - FastAPI service for distributed architectures
   - Event-driven architecture support

4. **Enterprise-Grade Features**
   - Type safety and validation
   - Comprehensive logging
   - Plugin system for extensibility
   - Workflow composition
   - Mathematical and data processing capabilities

## Installation

### Python
```bash
pip install agent-functions
```

### JavaScript
```bash
npm install agent-functions
```

## Quick Start

### Python
```python
from agent_functions import AgentFunction, workflow

@AgentFunction(
    category="nlp",
    description="Analyze text sentiment",
    agent_triggers=["sentiment_analysis_requested"]
)
def analyze_sentiment(text: str) -> dict:
    # Your sentiment analysis logic here
    return {"score": 0.8}

@workflow
def process_text(text: str):
    sentiment = analyze_sentiment(text)
    return {"result": sentiment}
```

### JavaScript
```javascript
import { AgentFunction, workflow } from 'agent-functions';

const analyzeSentiment = AgentFunction({
    category: 'nlp',
    description: 'Analyze text sentiment',
    agentTriggers: ['sentiment_analysis_requested']
})(
    async function(text) {
        // Your sentiment analysis logic here
        return { score: 0.8 };
    }
);

const processText = workflow(
    async function(text) {
        const sentiment = await analyzeSentiment(text);
        return { result: sentiment };
    }
);
```

## Key Features
- **Agent-First Design**: Built from the ground up for AI agent integration
- **Cross-Language Support**: Python and JavaScript interoperability
- **Cloud-Ready**: Serverless and microservices support
- **Type-Safe**: Strong typing and validation
- **Modular**: Plugin system for easy extension
- **Workflow Composition**: Chain functions together
- **Rich Analytics**: Mathematical and data processing tools included

## Use Cases
1. **AI Agent Systems**: Create function libraries that agents can discover and use
2. **Microservices**: Build event-driven services that work across languages
3. **Data Processing**: Compose complex data transformation workflows
4. **Machine Learning**: Create modular ML pipelines
5. **API Development**: Build flexible, agent-friendly APIs

## Documentation
For full documentation, visit [our documentation site](https://github.com/heidiEC/agent-functions-lib/wiki)
<img width="723" alt="Screenshot 2024-10-25 at 2 45 40 PM" src="https://github.com/user-attachments/assets/da9d632f-43f9-495b-a7d5-ff74cbe6fa78" />

## Contributing
We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
MIT License - feel free to use this in your projects!
